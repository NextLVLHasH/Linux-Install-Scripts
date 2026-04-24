#!/usr/bin/env bash
# Step 8: install dashboard backend deps (FastAPI + uvicorn) into the venv.
set -euo pipefail

VENV_DIR="${VENV_DIR:-/workspace/venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./04-install-pytorch.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing dashboard backend deps..."
pip install --upgrade \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9" \
    "sse-starlette>=2.1"

deactivate
echo "==> Dashboard deps installed."
