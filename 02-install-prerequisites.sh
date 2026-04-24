#!/usr/bin/env bash
# Step 2: install common build tools and Python.
set -euo pipefail

echo "==> Installing base tools and libraries..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    cmake \
    dkms \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    gnupg \
    lsb-release \
    pkg-config \
    libgomp1 \
    "linux-headers-$(uname -r)"

echo "==> Installing Python 3 and venv tooling..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

if [[ "${INSTALL_LMSTUDIO_GUI:-0}" == "1" ]]; then
    echo "==> Installing optional AppImage + VNC deps for LM Studio GUI..."
    sudo apt-get install -y libfuse2 || sudo apt-get install -y libfuse2t64
    sudo apt-get install -y \
        libnss3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libgbm1 \
        libgtk-3-0 \
        libasound2t64 2>/dev/null || sudo apt-get install -y libasound2
    sudo apt-get install -y xvfb x11vnc
    sudo apt-get install -y novnc websockify 2>/dev/null \
        || sudo apt-get install -y python3-websockify \
        || echo "WARN: novnc/websockify not in apt; install manually if needed."
else
    echo "==> Skipping LM Studio GUI/Xvfb deps (headless default uses llama-server)."
    echo "    Set INSTALL_LMSTUDIO_GUI=1 if you intentionally want the optional GUI stack."
fi

echo "==> Prerequisites installed."
