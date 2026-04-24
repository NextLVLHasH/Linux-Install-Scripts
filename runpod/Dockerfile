# RunPod Serverless worker for llama.cpp.
#
# Extends the official llama.cpp CUDA-server image (which already has
# llama-server compiled against CUDA 12) and layers our Python handler +
# scripts on top. On cold start the handler resolves MODEL_REPO /
# MODEL_FILE from env, downloads to /runpod-volume/models if missing,
# launches start-llama-server.sh as a subprocess, then enters the RunPod
# SDK loop.
#
# Build (from repo root):
#   docker build --platform linux/amd64 -t <user>/llama-runpod:v1 -f runpod/Dockerfile .
# Push:
#   docker push <user>/llama-runpod:v1
# Then create a Serverless endpoint in the RunPod console pointing at
# the image, attach a Network Volume mounted at /runpod-volume, and set
# MODEL_REPO / MODEL_FILE / HF_TOKEN env vars on the endpoint.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Base image's ENTRYPOINT is llama-server; override so our handler runs.
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        iproute2 \
        procps \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

# RunPod SDK + HF download deps. Pinned floors, not exact versions, so
# rebuilds pick up patch fixes.
RUN pip3 install --no-cache-dir --break-system-packages \
        'runpod>=1.7.0' \
        'huggingface_hub>=0.25' \
        'tqdm>=4.66' \
        'requests>=2.31'

WORKDIR /app

# Scripts copied from repo root (build context = repo root).
COPY download_model.py       /app/download_model.py
COPY start-llama-server.sh   /app/start-llama-server.sh
COPY runpod/handler.py       /app/handler.py
RUN chmod +x /app/start-llama-server.sh

# Tell start-llama-server.sh where to find the binary the base image ships.
ENV LLAMA_BIN=/app/llama-server \
    LLAMA_DIR=/app \
    LLAMA_PORT=8080 \
    LLAMA_BIND=127.0.0.1 \
    HF_HOME=/runpod-volume/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python3", "-u", "/app/handler.py"]
