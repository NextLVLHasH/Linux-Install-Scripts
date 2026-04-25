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

echo "==> Prerequisites installed."
