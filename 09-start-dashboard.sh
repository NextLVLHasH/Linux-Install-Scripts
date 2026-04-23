#!/usr/bin/env bash
# Launches the Material 3 dashboard and the LM Studio VNC browser stack.
#
#   Dashboard  → http://<ip>:8765
#   LM Studio  → http://<ip>:6080/vnc.html
#
# Flags:
#   --no-lmstudio   skip the LM Studio / VNC stack
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8765}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

LAUNCH_LMSTUDIO=true
for arg in "$@"; do [[ "$arg" == "--no-lmstudio" ]] && LAUNCH_LMSTUDIO=false; done

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./04-install-pytorch.sh first."
    exit 1
fi

# ── LM Studio VNC stack ────────────────────────────────────────────────────
LMSTUDIO_PID=""
if $LAUNCH_LMSTUDIO && [[ -f "$SCRIPT_DIR/start-lmstudio.sh" ]]; then
    APPIMAGE="${LMSTUDIO_DIR:-$HOME/LMStudio}/LMStudio.AppImage"
    if [[ -f "$APPIMAGE" ]]; then
        echo "==> Starting LM Studio VNC stack..."
        bash "$SCRIPT_DIR/start-lmstudio.sh" &
        LMSTUDIO_PID=$!
    else
        echo "==> LM Studio AppImage not found — skipping VNC stack."
        echo "    Run ./05-install-lmstudio.sh to install it."
    fi
fi

_cleanup() {
    [[ -n "$LMSTUDIO_PID" ]] && kill "$LMSTUDIO_PID" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# ── dashboard ──────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "════════════════════════════════════════════════"
echo "  Dashboard  → http://${IP}:${PORT}"
echo "  LM Studio  → http://${IP}:${NOVNC_PORT:-6080}/vnc.html"
echo "════════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR/dashboard"
exec uvicorn app:app --host "$HOST" --port "$PORT"
