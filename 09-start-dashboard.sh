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

# 1) If install-all.sh persisted paths, prefer those — this is the only source
#    of truth that survives sudo, systemd, and different users.
CONFIG_FILE="/var/lib/ml-stack-install/config.env"
if [[ -r "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

# 2) Figure out the real invoking user's home so ~/pytorch-env doesn't resolve
#    to /root when this script is started with sudo.
REAL_USER="${REAL_USER:-${SUDO_USER:-$USER}}"
REAL_HOME="${REAL_HOME:-$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)}"
: "${REAL_HOME:=$HOME}"

# 3) Env — mirror the systemd unit so running this script by hand behaves
#    identically to the service. Values from the config or the caller win.
export INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
export VENV_DIR="${VENV_DIR:-$REAL_HOME/pytorch-env}"
export LMSTUDIO_DIR="${LMSTUDIO_DIR:-$REAL_HOME/LMStudio}"
export LMS_MODELS_DIR="${LMS_MODELS_DIR:-$REAL_HOME/.lmstudio/models}"
export LMS_API_PORT="${LMS_API_PORT:-1234}"
export AUTO_LAUNCH_LMSTUDIO="${AUTO_LAUNCH_LMSTUDIO:-1}"
export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$REAL_HOME/.Xauthority}"

HOST="$DASHBOARD_HOST"
PORT="$DASHBOARD_PORT"

# Bootstrap the venv + dashboard deps if they're missing, so this script can
# be the single entry point after a fresh clone. Each step is idempotent and
# safe to re-run.
_run_as_user() {
    # Run an installer script as the real user (never as root) so the venv
    # ends up under REAL_HOME and is owned by them.
    if [[ -n "${SUDO_USER:-}" && "$(id -u)" == "0" ]]; then
        sudo -u "$REAL_USER" -H "$@"
    else
        "$@"
    fi
}

if [[ ! -d "$VENV_DIR" ]]; then
    # Prereqs (python3-venv, curl, etc.) are required for `python3 -m venv` to
    # work. This step needs root, so we don't drop to REAL_USER for it.
    if ! dpkg -s python3-venv >/dev/null 2>&1; then
        echo "==> Prerequisites missing — running 02-install-prerequisites.sh..."
        sudo "$SCRIPT_DIR/02-install-prerequisites.sh"
    fi
    echo "==> venv missing at $VENV_DIR — running 04-install-pytorch.sh..."
    _run_as_user env VENV_DIR="$VENV_DIR" "$SCRIPT_DIR/04-install-pytorch.sh"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv still not found at $VENV_DIR after bootstrap."
    echo "       Inspect ./04-install-pytorch.sh output above and re-run."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import uvicorn, fastapi" >/dev/null 2>&1; then
    echo "==> Dashboard deps missing in venv — running 08-install-dashboard.sh..."
    deactivate
    _run_as_user env VENV_DIR="$VENV_DIR" "$SCRIPT_DIR/08-install-dashboard.sh"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Start the LM Studio HTTP server before the dashboard so the first request
# doesn't race the server coming up. Skip silently if already running or if
# the CLI isn't installed (dashboard handles both cases gracefully).
LMS_BIN="$REAL_HOME/.lmstudio/bin/lms"
if [[ -x "$LMS_BIN" ]]; then
    if curl -sf "http://127.0.0.1:${LMS_API_PORT}/v1/models" >/dev/null 2>&1; then
        echo "==> LM Studio server already running on :${LMS_API_PORT}"
    else
        echo "==> Starting LM Studio server on :${LMS_API_PORT}..."
        "$LMS_BIN" server start --port "$LMS_API_PORT" >/dev/null 2>&1 || \
            echo "    (lms server start failed — dashboard will still run)"
    fi
else
    echo "==> LM Studio CLI not found at $LMS_BIN — skipping server start."
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "════════════════════════════════════════════════"
echo "  Dashboard  → http://${IP}:${PORT}"
echo "════════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR/dashboard"
exec uvicorn app:app --host "$HOST" --port "$PORT"
