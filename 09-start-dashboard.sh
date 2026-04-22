#!/usr/bin/env bash
# Launches the Material 3 dashboard. Auto-launches LM Studio too (set
# AUTO_LAUNCH_LMSTUDIO=0 to disable). Listens on 0.0.0.0:8765 by default
# so it's reachable from other machines on the LAN (e.g. ZimaOS dashboard).
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

cd "$SCRIPT_DIR/dashboard"
echo "==> Dashboard at http://$HOST:$PORT"
exec uvicorn app:app --host "$HOST" --port "$PORT"
