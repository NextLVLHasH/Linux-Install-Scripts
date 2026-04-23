#!/usr/bin/env bash
# Launches the Material 3 dashboard.
#
#   Dashboard  → http://<ip>:8765
#
# The dashboard controls LM Studio headlessly through the `lms` CLI and the
# LM Studio HTTP server (:1234). If you want the LM Studio GUI streamed over
# noVNC, run ./start-lmstudio.sh separately.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Environment — mirror the systemd unit so running this script by hand behaves
# identically to the service. Any value already exported by the caller (or by
# systemd's Environment= directives) wins.
export INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
export VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"
export LMSTUDIO_DIR="${LMSTUDIO_DIR:-$HOME/LMStudio}"
export LMS_MODELS_DIR="${LMS_MODELS_DIR:-$HOME/.lmstudio/models}"
export LMS_API_PORT="${LMS_API_PORT:-1234}"
export AUTO_LAUNCH_LMSTUDIO="${AUTO_LAUNCH_LMSTUDIO:-1}"
export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

HOST="$DASHBOARD_HOST"
PORT="$DASHBOARD_PORT"

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
