#!/usr/bin/env bash
# Step 6: install Hugging Face / LoRA fine-tuning stack into the PyTorch venv.
set -euo pipefail

VENV_DIR="${VENV_DIR:-/workspace/venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./04-install-pytorch.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing training/fine-tuning libraries..."
pip install --upgrade \
    "transformers>=4.44" \
    "datasets>=2.20" \
    "peft>=0.12" \
    "trl>=0.11" \
    "accelerate>=0.33" \
    "huggingface_hub[cli]>=0.24" \
    scipy \
    evaluate \
    tensorboard

# bitsandbytes enables 4-bit/8-bit loading. CUDA-only; skip on CPU boxes.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "==> CUDA detected, installing bitsandbytes (4-bit) and deepspeed (ZeRO-3 multi-GPU)..."
    pip install "bitsandbytes>=0.43"

    # PyYAML is needed by gpu_profile to emit accelerate configs as YAML.
    pip install "PyYAML>=6.0"

    # DeepSpeed: needed for ZeRO-3 strategy when sharding huge models across GPUs.
    # Best-effort — if it fails (missing CUDA dev tools), DDP/FSDP still work.
    pip install "deepspeed>=0.14" || echo "WARN: deepspeed install failed — ZeRO-3 strategy will be unavailable."

    # Flash Attention 2: 2-3× faster + lower memory at long sequence lengths.
    # The wheel build needs nvcc and a few minutes; we use --no-build-isolation
    # so it picks up the already-installed torch instead of pulling another one.
    # If this fails, the trainer falls back to PyTorch SDPA automatically.
    echo "==> Installing flash-attn (slow build; falls back to SDPA on failure)..."
    pip install --no-build-isolation "flash-attn>=2.5" \
        || echo "WARN: flash-attn install failed — trainer will use SDPA backend."

    # NCCL ships with PyTorch wheels; nothing extra to install for DDP on Linux.
else
    echo "==> CPU-only system: skipping bitsandbytes/deepspeed."
fi

deactivate
echo "==> Training stack installed."
