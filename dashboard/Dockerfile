# Dashboard container. Headless: no LM Studio GUI. Includes llama.cpp's
# llama-server so the Server tab can run local GGUF inference without X/VNC.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTO_LAUNCH_LMSTUDIO=0 \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8765 \
    LLAMA_BIND=0.0.0.0 \
    LLAMA_DIR=/opt/llama.cpp-bin \
    LLAMA_BIN=/opt/llama.cpp-bin/llama-server \
    GGUF_MODELS_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates curl \
        build-essential cmake pkg-config libaio-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.1 + the trainer + dashboard stack + multi-GPU runtime
# (NCCL ships with the torch wheel; deepspeed needs build-essential/libaio).
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    pip install \
        "transformers>=4.44" "datasets>=2.20" "peft>=0.12" "trl>=0.11" \
        "accelerate>=0.33" "huggingface_hub[cli]>=0.24" \
        "bitsandbytes>=0.43" "deepspeed>=0.14" \
        scipy evaluate tensorboard PyYAML \
        "fastapi>=0.110" "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9" "sse-starlette>=2.1" && \
    pip install --no-build-isolation "flash-attn>=2.5" \
        || echo "flash-attn build skipped; trainer will use SDPA."

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON && \
    cmake --build /opt/llama.cpp/build --target llama-server -j"$(nproc)" && \
    mkdir -p /opt/llama.cpp-bin && \
    cp -a /opt/llama.cpp/build/bin/. /opt/llama.cpp-bin/

COPY . /app

VOLUME ["/app/models", "/app/data", "/app/runs"]
EXPOSE 8765 1234

CMD ["python", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8765"]
