mod knn;
mod models;
mod predictor;
mod vectorizer;

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use mimalloc::MiMalloc;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use knn::IvfIndex;
use models::{FraudRequest, FraudResponse, Normalization};
use predictor::Predictor;
use vectorizer::vectorize;

struct AppState {
    predictor: Predictor,
    ivf_index: IvfIndex,
    norm_config: Normalization,
    mcc_risk_map: HashMap<String, f64>,
}

fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> std::io::Result<T> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

async fn ready_handler() -> impl Responder {
    HttpResponse::Ok().body("OK")
}

const RESP_APPROVED: &[u8] = b"{\"approved\":true,\"fraud_score\":0.0}";
const RESP_REJECTED: &[u8] = b"{\"approved\":false,\"fraud_score\":1.0}";

async fn fraud_score_handler(
    req: web::Json<FraudRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let vector = vectorize(&req, &data.norm_config, &data.mcc_risk_map);

    let fraud_score = match data.predictor.predict(vector) {
        Some(score) => score,
        None => {
            return HttpResponse::Ok().content_type("application/json").body(RESP_APPROVED);
        }
    };

    if fraud_score <= 0.4 {
        HttpResponse::Ok().content_type("application/json").body(RESP_APPROVED)
    } else if fraud_score > 0.65 {
        HttpResponse::Ok().content_type("application/json").body(RESP_REJECTED)
    } else {
        let knn_score = data.ivf_index.search(&vector);
        if knn_score < 0.6 {
            HttpResponse::Ok().content_type("application/json").body(RESP_APPROVED)
        } else {
            HttpResponse::Ok().content_type("application/json").body(RESP_REJECTED)
        }
    }
}

// Fallback for malformed JSON to return approved instead of 400
fn json_error_handler(
    err: actix_web::error::JsonPayloadError,
    _req: &actix_web::HttpRequest,
) -> actix_web::Error {
    actix_web::error::InternalError::from_response(
        err,
        HttpResponse::Ok().content_type("application/json").body(RESP_APPROVED),
    )
    .into()
}

#[actix_web::main]
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
    let mcc_risk_map: HashMap<String, f64> = load_json(mcc_path).expect("Failed to load mcc_risk.json");

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

    // --- Late-bind Unix Socket ---
    // Socket is created only after warmup so HAProxy won't route traffic
    // to an instance that is still priming its caches.
    let instance_id = std::env::var("INSTANCE_ID").unwrap_or_else(|_| "1".to_string());
    let sock_path = format!("/tmp/sockets/api{}.sock", instance_id);
    let _ = std::fs::remove_file(&sock_path);

    let state = web::Data::new(AppState {
        predictor,
        ivf_index,
        norm_config,
        mcc_risk_map,
    });

    let server = HttpServer::new(move || {
        let json_config = web::JsonConfig::default()
            .limit(4096)
            .error_handler(json_error_handler);

        App::new()
            .app_data(state.clone())
            .app_data(json_config)
            .route("/ready", web::get().to(ready_handler))
            .route("/fraud-score", web::post().to(fraud_score_handler))
    })
    .workers(1)
    .bind_uds(&sock_path)?;

    // Set permissions to 0777
    let mut perms = std::fs::metadata(&sock_path)?.permissions();
    perms.set_mode(0o777);
    std::fs::set_permissions(&sock_path, perms)?;

    println!("Listening on Unix Socket: {}", sock_path);

    server.run().await
}
