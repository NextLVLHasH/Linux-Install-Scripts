#!/usr/bin/env bash
# Step 4: create a Python venv and install PyTorch.
# Picks CUDA 12.1 wheels if an NVIDIA GPU is present, otherwise CPU wheels.
set -euo pipefail

VENV_DIR="${VENV_DIR:-/workspace/venv}"

# Idempotent: re-use an existing venv (e.g. one created by 03a-create-venv.sh
# before NVIDIA/CUDA so downstream steps could populate it) instead of failing.
if [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "==> Re-using existing venv at $VENV_DIR"
else
    VENV_PARENT="$(dirname "$VENV_DIR")"
    if [[ ! -d "$VENV_PARENT" ]]; then
        echo "==> Creating parent directory $VENV_PARENT"
        sudo mkdir -p "$VENV_PARENT"
        sudo chown "$(id -u):$(id -g)" "$VENV_PARENT" 2>/dev/null || true
    fi
    echo "==> Creating Python venv at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# Detect NVIDIA GPU: try nvidia-smi first (reliable post-driver-install),
# then fall back to lspci (works before drivers, reads PCI hardware).
_has_nvidia=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --list-gpus 2>/dev/null | grep -qi gpu; then
    _has_nvidia=1
    echo "==> NVIDIA GPU confirmed via nvidia-smi."
elif command -v lspci >/dev/null 2>&1 && lspci | grep -i -E 'vga|3d|display' | grep -qi nvidia; then
    _has_nvidia=1
    echo "==> NVIDIA GPU found via lspci (drivers may not be installed yet)."
fi

if [[ $_has_nvidia -eq 1 ]]; then
    echo "==> Installing PyTorch with CUDA 12.4 support (Ampere/Ada/Hopper)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "==> No NVIDIA GPU detected. Installing CPU-only PyTorch..."
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
