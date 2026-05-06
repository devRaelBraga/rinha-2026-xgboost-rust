FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost-dev clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/Cargo.toml ./api/
COPY api/src/ ./api/src/

WORKDIR /app/api
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/api/target/release/api /server

COPY training/output/model.json /data/model.json
COPY resources/mcc_risk.json /data/mcc_risk.json
COPY resources/normalization.json /data/normalization.json
COPY training/output/centroids.bin /data/centroids.bin
COPY training/output/ivf_offsets.bin /data/ivf_offsets.bin
COPY training/output/ivf_vectors.bin /data/ivf_vectors.bin
COPY training/output/ivf_labels.bin /data/ivf_labels.bin

EXPOSE 8080
CMD ["/server", "--model", "/data/model.json", "--port", "8080"]
