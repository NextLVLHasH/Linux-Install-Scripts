#!/usr/bin/env bash
# Step 3: install NVIDIA driver + CUDA runtime so PyTorch and LM Studio can use the GPU.
# Skips gracefully on systems without an NVIDIA GPU.
set -euo pipefail

if ! lspci | grep -i -E 'vga|3d|display' | grep -qi nvidia; then
    echo "==> No NVIDIA GPU detected. Skipping driver/CUDA install."
    echo "    PyTorch will be installed with CPU-only wheels."
    exit 0
fi

echo "==> NVIDIA GPU detected."

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> nvidia-smi already present:"
    nvidia-smi || true
else
    echo "==> Installing recommended NVIDIA driver via ubuntu-drivers..."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "==> NVIDIA driver installed. A reboot is required before the GPU is usable."
    # Signal to install-all.sh that a reboot is needed before continuing
    sudo mkdir -p /var/lib/ml-stack-install
    sudo touch /var/lib/ml-stack-install/.needs-reboot
fi

echo "==> Installing CUDA toolkit runtime (nvidia-cuda-toolkit) from Ubuntu repos..."
sudo apt-get install -y nvidia-cuda-toolkit || {
    echo "WARN: nvidia-cuda-toolkit failed to install from apt."
    echo "      PyTorch ships its own CUDA runtime, so this is usually fine."
}

echo "==> NVIDIA/CUDA step complete."
