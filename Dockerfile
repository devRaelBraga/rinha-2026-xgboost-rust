# ==========================================
# ESTÁGIO 1: BUILDER (O Motor de Otimização)
# ==========================================
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost-dev clang liburing-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/api

# 1. O TRUQUE DO CACHE DE DEPENDÊNCIAS
# Copiamos apenas os arquivos de manifesto primeiro
COPY api/Cargo.toml api/Cargo.lock* ./

# Criamos um arquivo main falso e mandamos compilar. 
# Isso força o Docker a baixar e compilar todas as dependências (actix, serde, etc)
# e salvar essa camada em cache. Se você mudar o seu código, não vai precisar baixar tudo de novo!
RUN mkdir src && \
    echo 'fn main() {}' > src/main.rs && \
    cargo build --release && \
    rm -rf src

# 2. INJEÇÃO DOS ESTEROIDES (SIMD, AVX2, FMA e Strip)
# É aqui que a mágica do processador acontece
# ENV RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma -C link-arg=-s"

# Agora sim copiamos o seu código real
COPY api/src/ ./src/

# Build arg: pass FEATURES=compat for Docker Desktop / Mac.
# Default (empty) = native io_uring bind for the test machine.
ARG FEATURES="compat"

# Atualizamos a data do arquivo pra forçar o cargo a compilar o seu código (e não o falso)
RUN touch src/main.rs && cargo build --release $(if [ -n "$FEATURES" ]; then echo "--features $FEATURES"; fi)

# Removemos qualquer símbolo de debug restante para diminuir a RAM usada
RUN strip target/release/api

# ==========================================
# ESTÁGIO 2: RUNTIME (O Contêiner Leve)
# ==========================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost0 \
    liburing2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiamos o binário otimizado do estágio anterior
COPY --from=builder /app/api/target/release/api /server

# Copiamos seus modelos e matrizes (mantive igual ao seu)
COPY training/output/model.json /data/model.json
COPY resources/mcc_risk.json /data/mcc_risk.json
COPY resources/normalization.json /data/normalization.json
COPY training/output/centroids.bin /data/centroids.bin
COPY training/output/ivf_offsets.bin /data/ivf_offsets.bin
COPY training/output/ivf_vectors.bin /data/ivf_vectors.bin
COPY training/output/ivf_labels.bin /data/ivf_labels.bin

EXPOSE 8080

# Usamos o formato JSON array para o CMD (Melhor prática do Docker)
CMD ["/server", "--model", "/data/model.json", "--port", "8080"]