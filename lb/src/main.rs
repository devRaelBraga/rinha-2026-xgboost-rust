use mimalloc::MiMalloc;
use monoio::io::copy;
use monoio::io::Splitable;
use std::cell::Cell;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    static RR_COUNTER: Cell<usize> = Cell::new(0);
}

#[monoio::main(threads = 1)]
async fn main() -> std::io::Result<()> {
    let listener = monoio::net::TcpListener::bind("0.0.0.0:9999")?;
    println!("L4 Monoio LB listening on :9999");
    
    let backends = ["/sockets/api1.sock", "/sockets/api2.sock"];

    loop {
        let (client_stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = client_stream.set_nodelay(true);

        let idx = RR_COUNTER.get();
        RR_COUNTER.set(idx.wrapping_add(1));
        let target = backends[idx % backends.len()];

        monoio::spawn(async move {
            let start = std::time::Instant::now();
            if let Ok(backend_stream) = monoio::net::UnixStream::connect(target).await {
                let elapsed = start.elapsed();
                if elapsed.as_micros() > 2000 {
                    eprintln!("[SLOW LB CONNECT] UDS connect took {:.2}ms", elapsed.as_secs_f64() * 1000.0);
                }
                let (mut cr, mut cw) = client_stream.into_split();
                let (mut br, mut bw) = backend_stream.into_split();

                let c2b = monoio::spawn(async move {
                    let _ = copy(&mut cr, &mut bw).await;
                });
                
                let b2c = monoio::spawn(async move {
                    let _ = copy(&mut br, &mut cw).await;
                });

                let _ = c2b.await;
                let _ = b2c.await;
            }
        });
    }
}
