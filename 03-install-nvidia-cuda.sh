#!/usr/bin/env bash
# Step 3: install the official NVIDIA proprietary driver + CUDA toolkit from
# NVIDIA's apt repo. No open-source drivers (nouveau gets blacklisted, and
# the proprietary `nvidia-driver-XXX` package is preferred over -open).
#
# Skips gracefully on systems without an NVIDIA GPU. Requires reboot after
# first-time install to switch from nouveau to the NVIDIA kernel module.
set -euo pipefail

STATE_DIR="/var/lib/ml-stack-install"

if ! lspci | grep -i -E 'vga|3d|display' | grep -qi nvidia; then
    echo "==> No NVIDIA GPU detected. Skipping driver/CUDA install."
    echo "    PyTorch will be installed with CPU-only wheels."
    exit 0
fi

echo "==> NVIDIA GPU detected:"
lspci | grep -i -E 'vga|3d|display' | grep -i nvidia

# ── SecureBoot check ──────────────────────────────────────────────────────
# NVIDIA's apt repo ships unsigned kernel module sources; DKMS builds them
# locally. On SecureBoot-enabled systems the install hangs on an interactive
# MOK-password prompt (no TTY under automation) and leaves dpkg broken.
# Fail early with a clear message instead of wedging the box.
if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -qi "SecureBoot enabled"; then
    cat <<'WARN'

╔════════════════════════════════════════════════════════════════════╗
║  SecureBoot is ENABLED — NVIDIA's proprietary driver cannot be     ║
║  installed non-interactively on this system.                       ║
║                                                                    ║
║  NVIDIA's apt repo ships unsigned kernel modules; DKMS tries to    ║
║  sign them during install and prompts for a MOK password. That     ║
║  prompt can't be answered under a headless/automated install and   ║
║  will hang the dpkg transaction.                                   ║
║                                                                    ║
║  FIX: reboot into BIOS/UEFI, set SecureBoot to Disabled, save &    ║
║       exit, then re-run this script.                               ║
║                                                                    ║
║  (After boot, verify with: mokutil --sb-state)                     ║
╚════════════════════════════════════════════════════════════════════╝

WARN
    exit 1
fi

# Short-circuit if the proprietary driver is already live and happy.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "==> nvidia-smi already working — driver is live:"
    nvidia-smi | head -20
    echo "==> Ensuring CUDA toolkit is present..."
    if ! command -v nvcc >/dev/null 2>&1; then
        sudo apt-get install -y cuda-toolkit || \
            echo "WARN: cuda-toolkit install failed — PyTorch ships its own CUDA runtime, so this is usually fine."
    fi
    exit 0
fi

# ── add NVIDIA's official CUDA apt repo ───────────────────────────────────
. /etc/os-release
UBUNTU_VER="${VERSION_ID//./}"           # e.g. 24.04 → 2404
ARCH="$(dpkg --print-architecture)"      # amd64 → we need x86_64 below
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
    echo "       Check NVIDIA's repo for a matching Ubuntu release at:"
    echo "       https://developer.download.nvidia.com/compute/cuda/repos/"
    exit 1
fi
sudo dpkg -i "$TMP/cuda-keyring.deb"
sudo apt-get update

# ── blacklist nouveau so it can't fight the NVIDIA module after reboot ───
echo "==> Blacklisting nouveau..."
sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null <<'CONF'
blacklist nouveau
options nouveau modeset=0
CONF
sudo update-initramfs -u 2>&1 | tail -5 || true

# ── pick the latest *proprietary* (non -open) driver in NVIDIA's repo ────
echo "==> Selecting proprietary driver package..."
DRIVER_PKG=$(apt-cache search --names-only '^nvidia-driver-[0-9]+$' 2>/dev/null \
    | awk '{print $1}' | sort -V | tail -1)
if [[ -z "$DRIVER_PKG" ]]; then
    # Fall back to CUDA's meta-package. On some repos this can pull the open
    # variant; set it to proprietary via apt preference below if possible.
    DRIVER_PKG="cuda-drivers"
fi
echo "    → $DRIVER_PKG"

echo "==> Installing NVIDIA driver + kernel module (DKMS)..."
sudo apt-get install -y "$DRIVER_PKG"

echo "==> Installing CUDA toolkit..."
sudo apt-get install -y cuda-toolkit || \
    echo "WARN: cuda-toolkit failed — PyTorch ships its own runtime, usually fine."

# ── flag a reboot so install-all.sh pauses, reboots, and resumes ─────────
sudo mkdir -p "$STATE_DIR"
sudo touch "$STATE_DIR/.needs-reboot"

echo "==> NVIDIA driver + CUDA installed."
echo "    A reboot is required before the GPU is usable."
echo "    After reboot, verify with: nvidia-smi"
