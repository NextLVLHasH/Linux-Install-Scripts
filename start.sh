#!/usr/bin/env bash
# Launches LM Studio and opens a shell with the PyTorch venv active.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"
LMSTUDIO_DIR="${LMSTUDIO_DIR:-$HOME/LMStudio}"
APPIMAGE_PATH="$LMSTUDIO_DIR/LMStudio.AppImage"

if [[ ! -x "$APPIMAGE_PATH" ]]; then
    echo "ERROR: LM Studio not found at $APPIMAGE_PATH"
    echo "       Run ./05-install-lmstudio.sh first."
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "WARN: PyTorch venv not found at $VENV_DIR — skipping activation."
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "==> PyTorch venv active: $VENV_DIR"
fi

echo "==> Launching LM Studio..."
nohup "$APPIMAGE_PATH" >/dev/null 2>&1 &
disown
echo "    PID: $!"
echo "    LM Studio is starting in the background."
