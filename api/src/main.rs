mod knn;
mod models;
mod predictor;
mod vectorizer;

use mimalloc::MiMalloc;
use monoio::io::{AsyncReadRent, AsyncReadRentExt, AsyncWriteRentExt};
use monoio::net::UnixStream;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use knn::IvfIndex;
use models::{FraudRequest, Normalization};
use predictor::Predictor;
use vectorizer::vectorize;

// ---------------------------------------------------------------------------
// Application state — stored in thread-local, no Arc, no atomics.
// Safe because monoio is single-threaded (threads = 1).
// ---------------------------------------------------------------------------
struct AppState {
    predictor: Predictor,
    ivf_index: IvfIndex,
    norm_config: Normalization,
    mcc_risk: Box<[f32; 10000]>,
}

thread_local! {
    static STATE: std::cell::UnsafeCell<Option<AppState>> = std::cell::UnsafeCell::new(None);
}

#[inline(always)]
fn with_state<F, R>(f: F) -> R
where
    F: FnOnce(&AppState) -> R,
{
    STATE.with(|cell| unsafe {
        let state = &*cell.get();
        f(state.as_ref().unwrap())
    })
}

// ---------------------------------------------------------------------------
// Pre-baked HTTP responses — zero allocation on the response path.
// ---------------------------------------------------------------------------
const RESP_APPROVED: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}";
const RESP_REJECTED: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}";
const RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
const RESP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------
fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> std::io::Result<T> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// HTTP request handler — the hot path
// ---------------------------------------------------------------------------
#[inline(always)]
fn process_fraud_request(body: &mut [u8]) -> &'static [u8] {
    let req: FraudRequest = match simd_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return RESP_APPROVED,
    };

    with_state(|state| {
        let vector = vectorize(&req, &state.norm_config, &state.mcc_risk);

        let fraud_score = match state.predictor.predict(vector) {
            Some(score) => score,
            None => return RESP_APPROVED,
        };

        if fraud_score <= 0.4 {
            RESP_APPROVED
        } else if fraud_score > 0.65 {
            RESP_REJECTED
        } else {
            let knn_score = state.ivf_index.search(&vector);
            if knn_score < 0.6 {
                RESP_APPROVED
            } else {
                RESP_REJECTED
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Connection handler — keep-alive loop over a single Unix socket stream.
//
// Protocol assumptions (we control the LB):
//   - HTTP/1.1 only
//   - Requests always have Content-Length (no chunked)
//   - Headers + request line fit within 2048 bytes
// ---------------------------------------------------------------------------
async fn handle_connection(stream: UnixStream) {
    // 2KB header buffer — typical request headers are ~200-400 bytes.
    // We own this buffer and pass it to monoio's read() which takes ownership.
    let mut hdr_buf = vec![0u8; 2048];
    // Body buffer — reused across keep-alive requests.
    let mut body_buf: Vec<u8> = Vec::with_capacity(1024);

    let mut stream = stream;

    loop {
        // ---- Read headers ------------------------------------------------
        let (res, buf) = stream.read(hdr_buf).await;
        hdr_buf = buf;
        let n = match res {
            Ok(0) | Err(_) => return, // connection closed or error
            Ok(n) => n,
        };

        // ---- Parse HTTP request ------------------------------------------
        let mut headers = [httparse::EMPTY_HEADER; 16];
        let mut req = httparse::Request::new(&mut headers);

        let parse_result = match req.parse(&hdr_buf[..n]) {
            Ok(s) => s,
            Err(_) => {
                // Malformed request → respond approved (same as Actix error handler)
                let (res, _) = stream.write_all(RESP_APPROVED.to_vec()).await;
                if res.is_err() { return; }
                continue;
            }
        };

        let header_len = match parse_result {
            httparse::Status::Complete(len) => len,
            httparse::Status::Partial => {
                // Headers incomplete in 2KB — shouldn't happen for our payloads
                let (res, _) = stream.write_all(RESP_404.to_vec()).await;
                if res.is_err() { return; }
                continue;
            }
        };

        let method = req.method.unwrap_or("");
        let path = req.path.unwrap_or("");

        // ---- Route dispatch ----------------------------------------------
        if method == "GET" && path == "/ready" {
            let (res, _) = stream.write_all(RESP_READY.to_vec()).await;
            if res.is_err() { return; }
            // Drain any leftover bytes (shouldn't be any for GET)
            continue;
        }

        if method != "POST" || path != "/fraud-score" {
            let (res, _) = stream.write_all(RESP_404.to_vec()).await;
            if res.is_err() { return; }
            continue;
        }

        // ---- Extract Content-Length --------------------------------------
        let mut content_length: usize = 0;
        for h in req.headers.iter() {
            if h.name.eq_ignore_ascii_case("content-length") {
                // Fast atoi — content-length is always a small number
                for &b in h.value {
                    content_length = content_length * 10 + (b - b'0') as usize;
                }
                break;
            }
        }

        if content_length == 0 {
            // No body → respond approved
            let (res, _) = stream.write_all(RESP_APPROVED.to_vec()).await;
            if res.is_err() { return; }
            continue;
        }

        // ---- Assemble the body -------------------------------------------
        // Some (or all) of the body may already be in hdr_buf after the headers.
        let body_already_read = n - header_len;
        body_buf.clear();

        if body_already_read >= content_length {
            // Full body already in hdr_buf
            body_buf.extend_from_slice(&hdr_buf[header_len..header_len + content_length]);
        } else {
            // Partial body — copy what we have, then read the rest
            body_buf.extend_from_slice(&hdr_buf[header_len..n]);
            let remaining = content_length - body_already_read;
            let old_len = body_buf.len();
            body_buf.resize(old_len + remaining, 0);

            // Take the tail slice out as a Vec for monoio's ownership model
            let tail = body_buf.split_off(old_len);
            let (res, tail_buf) = stream.read_exact(tail).await;
            match res {
                Ok(_) => {
                    body_buf.extend_from_slice(&tail_buf);
                }
                Err(_) => return,
            }
        }

        // ---- Process and respond -----------------------------------------
        let response = process_fraud_request(&mut body_buf);
        let (res, _) = stream.write_all(response.to_vec()).await;
        if res.is_err() { return; }
    }
}

// ---------------------------------------------------------------------------
// Main — load models, create Unix socket, accept loop.
// ---------------------------------------------------------------------------
#[monoio::main(threads = 1)]
async fn main() -> std::io::Result<()> {
    let model_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/data/model.json".to_string());

    let norm_path = "/data/normalization.json";
    let mcc_path = "/data/mcc_risk.json";
    let centroids_path = "/data/centroids.bin";
    let offsets_path = "/data/ivf_offsets.bin";
    let vectors_path = "/data/ivf_vectors.bin";
    let labels_path = "/data/ivf_labels.bin";

    println!("Loading normalization config...");
    let norm_config: Normalization = load_json(norm_path).expect("Failed to load normalization.json");

    println!("Loading MCC risk map...");
    let raw_mcc: HashMap<String, f64> = load_json(mcc_path).expect("Failed to load mcc_risk.json");

    // Build flat MCC lookup array (40KB, fits in L1 cache) — replaces HashMap
    let mut mcc_risk = Box::new([0.5f32; 10000]);
    for (code, risk) in &raw_mcc {
        if let Ok(idx) = code.parse::<usize>() {
            if idx < 10000 {
                mcc_risk[idx] = *risk as f32;
            }
        }
    }

    println!("Loading XGBoost model...");
    let predictor = Predictor::new(&model_path).expect("Failed to load XGBoost model");

    println!("Loading IVF index...");
    let ivf_index = IvfIndex::load(centroids_path, offsets_path, vectors_path, labels_path)
        .expect("Failed to load IVF index");

    // --- Warmup Sequence ---
    // XGBoost lazily allocates inference buffers on first predict; prime them now.
    // Also warms CPU cache lines for the IVF centroid/vector data.
    println!("Running warmup sequence...");
    let dummy_vector: [f32; 14] = [0.0f32; 14];
    let _ = predictor.predict(dummy_vector);
    let _ = ivf_index.search(&dummy_vector);
    println!("Warmup complete, API is ready!");

    // Store state in thread-local (no Arc needed — single threaded monoio)
    STATE.with(|cell| unsafe {
        *cell.get() = Some(AppState {
            predictor,
            ivf_index,
            norm_config,
            mcc_risk,
        });
    });

    // --- Late-bind Unix Socket ---
    // Socket is created only after warmup so the LB won't route traffic
    // to an instance that is still priming its caches.
    let instance_id = std::env::var("INSTANCE_ID").unwrap_or_else(|_| "1".to_string());
    let sock_path = format!("/tmp/sockets/api{}.sock", instance_id);
    let _ = std::fs::remove_file(&sock_path);

    // Use std UnixListener::bind (standard syscalls) then convert to monoio.
    // bind() runs once at startup — zero perf impact. io_uring benefits are
    // in accept/read/write during request handling, which monoio still handles.
    let std_listener = std::os::unix::net::UnixListener::bind(&sock_path)?;
    std_listener.set_nonblocking(true)?;
    let listener = monoio::net::UnixListener::from_std(std_listener)?;

    // Set permissions to 0777
    let mut perms = std::fs::metadata(&sock_path)?.permissions();
    perms.set_mode(0o777);
    std::fs::set_permissions(&sock_path, perms)?;

    println!("Listening on Unix Socket: {}", sock_path);

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(_) => continue,
        };
        monoio::spawn(handle_connection(stream));
    }
}
