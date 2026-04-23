#!/usr/bin/env bash
# Launches the Material 3 dashboard.
#
#   Dashboard  → http://<ip>:8765
#
# The dashboard controls LM Studio headlessly through the `lms` CLI and the
# LM Studio HTTP server (:1234). If you want the LM Studio GUI streamed over
# noVNC, run ./start-lmstudio.sh separately.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8765}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./04-install-pytorch.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "════════════════════════════════════════════════"
echo "  Dashboard  → http://${IP}:${PORT}"
echo "════════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR/dashboard"
exec uvicorn app:app --host "$HOST" --port "$PORT"
