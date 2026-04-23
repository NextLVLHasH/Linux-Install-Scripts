#!/usr/bin/env bash
# Step 2: install common build tools, Python, and AppImage runtime deps.
set -euo pipefail

echo "==> Installing base tools and libraries..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    gnupg \
    lsb-release \
    pkg-config

echo "==> Installing Python 3 and venv tooling..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

echo "==> Installing AppImage runtime dependencies (needed by LM Studio)..."
# libfuse2 is required by AppImage on Ubuntu 22.04+
sudo apt-get install -y libfuse2 || sudo apt-get install -y libfuse2t64

# GUI libs typically required by Electron-based AppImages.
sudo apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libasound2t64 2>/dev/null || sudo apt-get install -y libasound2

echo "==> Installing virtual display + VNC (headless LM Studio → browser)..."
# Xvfb: virtual framebuffer X server (fake display for headless machines)
# x11vnc: VNC server that reads from the virtual display
# novnc + websockify: HTML5 browser VNC client
sudo apt-get install -y xvfb x11vnc
sudo apt-get install -y novnc websockify 2>/dev/null \
    || sudo apt-get install -y python3-websockify \
    || echo "WARN: novnc/websockify not in apt — install manually if needed."

echo "==> Prerequisites installed."
