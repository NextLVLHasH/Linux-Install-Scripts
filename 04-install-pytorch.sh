#!/usr/bin/env bash
# Step 4: create a Python venv and install PyTorch.
# Picks CUDA 12.1 wheels if an NVIDIA GPU is present, otherwise CPU wheels.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"

echo "==> Creating Python venv at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

if lspci | grep -i -E 'vga|3d|display' | grep -qi nvidia; then
    echo "==> Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "==> Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing common companion libraries..."
pip install numpy transformers accelerate safetensors sentencepiece

echo "==> Verifying PyTorch install..."
python - <<'PY'
import torch
print(f"torch      : {torch.__version__}")
print(f"cuda build : {torch.version.cuda}")
print(f"cuda avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
PY

deactivate
echo "==> PyTorch venv ready at $VENV_DIR"
echo "    Activate with: source $VENV_DIR/bin/activate"
