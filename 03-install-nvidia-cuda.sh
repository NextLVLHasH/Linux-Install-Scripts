#!/usr/bin/env bash
# Step 3: install NVIDIA's proprietary driver + CUDA toolkit.
#
# Two paths depending on SecureBoot state:
#
#   SecureBoot ON  →  Ubuntu's pre-signed packages
#                     (`nvidia-driver-<N>` + `linux-modules-nvidia-<N>-generic`
#                      from Ubuntu's restricted repo). Same NVIDIA proprietary
#                      driver code, just compiled and Canonical-signed so the
#                      kernel module loads under SecureBoot with no MOK step.
#
#   SecureBoot OFF →  NVIDIA's official CUDA apt repo
#                     (developer.download.nvidia.com). Latest driver and CUDA
#                     straight from NVIDIA; DKMS builds the unsigned module.
#
# Skips gracefully on systems without an NVIDIA GPU. Requires reboot after
# first-time install.
set -euo pipefail

STATE_DIR="/var/lib/ml-stack-install"

if ! lspci | grep -i -E 'vga|3d|display' | grep -qi nvidia; then
    echo "==> No NVIDIA GPU detected. Skipping driver/CUDA install."
    echo "    PyTorch will be installed with CPU-only wheels."
    exit 0
fi

echo "==> NVIDIA GPU detected:"
lspci | grep -i -E 'vga|3d|display' | grep -i nvidia

# Short-circuit if the driver is already live and happy.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "==> nvidia-smi already working — driver is live:"
    nvidia-smi | head -20
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "==> Installing CUDA toolkit..."
        sudo apt-get install -y nvidia-cuda-toolkit || \
            echo "WARN: cuda-toolkit install failed — PyTorch ships its own CUDA runtime."
    fi
    exit 0
fi

# ── detect SecureBoot to pick the right driver source ────────────────────
SECURE_BOOT=0
if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -qi "SecureBoot enabled"; then
    SECURE_BOOT=1
fi

if (( SECURE_BOOT )); then
    # ─────────────────────────────────────────────────────────────────────
    # SecureBoot ON: use Ubuntu's signed NVIDIA packages.
    # ─────────────────────────────────────────────────────────────────────
    echo "==> SecureBoot is enabled — using Ubuntu's pre-signed NVIDIA packages."
    echo "    (Same proprietary driver code from NVIDIA, compiled + signed by"
    echo "     Canonical so the kernel module loads under SecureBoot.)"

    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common

    # Pick the latest *proprietary* `nvidia-driver-<N>` (no -open, -server, or
    # -server-open suffix). Ubuntu's `-open` variant is NVIDIA's open kernel
    # module (github.com/NVIDIA/open-gpu-kernel-modules) which, while official
    # NVIDIA code, is explicitly off the table per user requirements.
    RECOMMENDED=$(apt-cache search --names-only '^nvidia-driver-[0-9]+$' 2>/dev/null \
        | awk '{print $1}' | sort -V | tail -1)

    if [[ -z "$RECOMMENDED" ]]; then
        echo "ERROR: could not find any nvidia-driver package in the apt archive."
        echo "       Check that Ubuntu's 'restricted' repo is enabled:"
        echo "       sudo add-apt-repository restricted && sudo apt-get update"
        exit 1
    fi

    DRIVER_VER=$(echo "$RECOMMENDED" | grep -oE '[0-9]+$')
    SIGNED_MODULE="linux-modules-nvidia-${DRIVER_VER}-generic"

    echo "==> Installing $RECOMMENDED + $SIGNED_MODULE (Canonical-signed)..."
    # Install the signed module explicitly alongside the driver so apt
    # doesn't try to pull the DKMS variant (which would prompt for MOK).
    sudo apt-get install -y "$RECOMMENDED" "$SIGNED_MODULE"

    # CUDA toolkit from Ubuntu repos — older version, but signed and safe.
    # PyTorch ships its own CUDA runtime so this is optional.
    echo "==> Installing CUDA toolkit from Ubuntu repos..."
    sudo apt-get install -y nvidia-cuda-toolkit || \
        echo "WARN: nvidia-cuda-toolkit install failed — PyTorch ships its own runtime, usually fine."

else
    # ─────────────────────────────────────────────────────────────────────
    # SecureBoot OFF: use NVIDIA's official CUDA apt repo (latest driver).
    # ─────────────────────────────────────────────────────────────────────
    echo "==> SecureBoot is disabled — using NVIDIA's official CUDA apt repo."

    . /etc/os-release
    UBUNTU_VER="${VERSION_ID//./}"           # 24.04 → 2404
    ARCH="$(dpkg --print-architecture)"
    case "$ARCH" in
        amd64) CUDA_ARCH="x86_64" ;;
        arm64) CUDA_ARCH="sbsa"    ;;
        *) echo "ERROR: unsupported arch $ARCH for NVIDIA CUDA repo."; exit 1 ;;
    esac

    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb"
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

    echo "==> Fetching NVIDIA CUDA keyring from $KEYRING_URL"
    if ! wget -q -O "$TMP/cuda-keyring.deb" "$KEYRING_URL"; then
        echo "ERROR: could not download cuda-keyring for Ubuntu $VERSION_ID ($CUDA_ARCH)."
        echo "       Check NVIDIA's repo index at:"
        echo "       https://developer.download.nvidia.com/compute/cuda/repos/"
        exit 1
    fi
    sudo dpkg -i "$TMP/cuda-keyring.deb"
    sudo apt-get update

    # Blacklist nouveau so it can't fight the NVIDIA module after reboot.
    echo "==> Blacklisting nouveau..."
    sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null <<'CONF'
blacklist nouveau
options nouveau modeset=0
CONF
    sudo update-initramfs -u 2>&1 | tail -5 || true

    DRIVER_PKG=$(apt-cache search --names-only '^nvidia-driver-[0-9]+$' 2>/dev/null \
        | awk '{print $1}' | sort -V | tail -1)
    [[ -z "$DRIVER_PKG" ]] && DRIVER_PKG="cuda-drivers"
    echo "==> Installing $DRIVER_PKG + DKMS kernel module..."
    sudo apt-get install -y "$DRIVER_PKG"

    echo "==> Installing CUDA toolkit..."
    sudo apt-get install -y cuda-toolkit || \
        echo "WARN: cuda-toolkit failed — PyTorch ships its own runtime, usually fine."
fi

# ── flag a reboot so install-all.sh pauses, reboots, and resumes ─────────
sudo mkdir -p "$STATE_DIR"
sudo touch "$STATE_DIR/.needs-reboot"

echo "==> NVIDIA driver + CUDA installed."
echo "    A reboot is required before the GPU is usable."
echo "    After reboot, verify with: nvidia-smi"
