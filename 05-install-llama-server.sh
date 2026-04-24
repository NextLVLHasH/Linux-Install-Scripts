#!/usr/bin/env bash
# Step 5: build/install llama.cpp's llama-server for headless inference.
#
# This is the default local provider for servers. It exposes an OpenAI-compatible
# API on :1234 without Electron, Xvfb, VNC, or any desktop session.
set -euo pipefail

REAL_USER="${REAL_USER:-${SUDO_USER:-${USER:-$(id -un)}}}"
REAL_HOME="${REAL_HOME:-$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)}"
: "${REAL_HOME:=$HOME}"

LLAMA_SRC_DIR="${LLAMA_SRC_DIR:-$REAL_HOME/llama.cpp-src}"
LLAMA_DIR="${LLAMA_DIR:-$REAL_HOME/llama.cpp-bin/current}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_SRC_DIR/build}"

_run_as_user() {
    if [[ -n "${SUDO_USER:-}" && "$(id -u)" == "0" ]]; then
        sudo -u "$REAL_USER" -H "$@"
    else
        "$@"
    fi
}

echo "==> Installing build tools for llama.cpp..."
sudo apt-get update -y
sudo apt-get install -y git cmake build-essential pkg-config curl ca-certificates

if [[ ! -d "$LLAMA_SRC_DIR/.git" ]]; then
    echo "==> Cloning llama.cpp -> $LLAMA_SRC_DIR"
    _run_as_user git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC_DIR"
else
    echo "==> Updating llama.cpp in $LLAMA_SRC_DIR"
    _run_as_user git -C "$LLAMA_SRC_DIR" pull --ff-only
fi

CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
if command -v nvcc >/dev/null 2>&1; then
    echo "==> CUDA compiler detected; building llama.cpp with GGML_CUDA=ON"
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)
else
    echo "==> nvcc not found; building CPU llama-server. Install CUDA toolkit for GPU offload."
fi

echo "==> Configuring llama.cpp..."
_run_as_user cmake -S "$LLAMA_SRC_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"

echo "==> Building llama-server..."
_run_as_user cmake --build "$BUILD_DIR" --config Release --target llama-server -j"$(nproc)"

echo "==> Installing llama-server runtime -> $LLAMA_DIR"
_run_as_user mkdir -p "$LLAMA_DIR"
if [[ -d "$BUILD_DIR/bin" ]]; then
    _run_as_user cp -a "$BUILD_DIR/bin/." "$LLAMA_DIR/"
else
    _run_as_user cp -a "$BUILD_DIR/." "$LLAMA_DIR/"
fi

if [[ ! -x "$LLAMA_DIR/llama-server" ]]; then
    echo "ERROR: llama-server was not produced at $LLAMA_DIR/llama-server" >&2
    exit 1
fi

echo "==> llama-server installed:"
"$LLAMA_DIR/llama-server" --version 2>/dev/null || true
echo
echo "Next:"
echo "  1. Download a GGUF into $REAL_HOME/models or through the dashboard Server tab."
echo "  2. Run ./10-install-systemd.sh to enable llama-server.service."
