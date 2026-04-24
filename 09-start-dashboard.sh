#!/usr/bin/env bash
# Launches the dashboard.
#
#   Dashboard -> http://<ip>:8765
#
# The dashboard controls llama.cpp's headless llama-server on :1234. LM Studio
# remains available as an optional legacy backend, but it is not the default.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Prefer paths persisted by install-all.sh. This is the source of truth that
# survives sudo, systemd, and different users.
CONFIG_FILE="/var/lib/ml-stack-install/config.env"
if [[ -r "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

# Figure out the invoking user's home so paths like ~/LMStudio or
# ~/llama.cpp-bin don't resolve to /root when this script runs under sudo.
REAL_USER="${REAL_USER:-${SUDO_USER:-${USER:-$(id -un)}}}"
REAL_HOME="${REAL_HOME:-$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)}"
: "${REAL_HOME:=$HOME}"

# Mirror the systemd unit so running this by hand behaves like the service.
# Values from the config or caller win.
export INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
export VENV_DIR="${VENV_DIR:-/workspace/venv}"
export LMSTUDIO_DIR="${LMSTUDIO_DIR:-$REAL_HOME/LMStudio}"
export LLAMA_DIR="${LLAMA_DIR:-$REAL_HOME/llama.cpp-bin/current}"
export LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/llama-server}"
export GGUF_MODELS_DIR="${GGUF_MODELS_DIR:-$REAL_HOME/models}"
export LMS_MODELS_DIR="${LMS_MODELS_DIR:-$GGUF_MODELS_DIR}"
export LMS_API_PORT="${LMS_API_PORT:-1234}"
export AUTO_LAUNCH_LMSTUDIO="${AUTO_LAUNCH_LMSTUDIO:-0}"

# Default: bind to the current LAN IP, not 0.0.0.0. Keeps the dashboard
# reachable from the LAN without exposing it on every interface.
DASHBOARD_HOST_AUTO="$(hostname -I 2>/dev/null | awk '{print $1}')"
export DASHBOARD_HOST="${DASHBOARD_HOST:-${DASHBOARD_HOST_AUTO:-127.0.0.1}}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
HOST="$DASHBOARD_HOST"
PORT="$DASHBOARD_PORT"

_is_interactive() { [[ -t 0 && -t 1 ]] && [[ -z "${INVOCATION_ID:-}" ]]; }
_running_under_systemd() { [[ -n "${INVOCATION_ID:-}" ]]; }

_run_as_user() {
    if [[ -n "${SUDO_USER:-}" && "$(id -u)" == "0" ]]; then
        sudo -u "$REAL_USER" -H "$@"
    else
        "$@"
    fi
}

_need_bootstrap() {
    [[ ! -d "$VENV_DIR" ]] && return 0
    source "$VENV_DIR/bin/activate" 2>/dev/null || return 0
    python -c "import uvicorn, fastapi" >/dev/null 2>&1 || {
        deactivate 2>/dev/null
        return 0
    }
    deactivate 2>/dev/null
    return 1
}

if _need_bootstrap; then
    if ! _is_interactive; then
        # Do not attempt sudo/apt without a TTY. Exit 0 so systemd does not
        # restart us into a loop; the next boot after install will run normally.
        echo "==> Dashboard not yet installed (venv or deps missing)."
        echo "    Run ./install-all.sh from a terminal to complete the install."
        _running_under_systemd && echo "    Service will remain idle until then."
        exit 0
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        if ! dpkg -s python3-venv >/dev/null 2>&1; then
            echo "==> Prerequisites missing; running 02-install-prerequisites.sh..."
            sudo "$SCRIPT_DIR/02-install-prerequisites.sh"
        fi
        echo "==> venv missing at $VENV_DIR; running 04-install-pytorch.sh..."
        _run_as_user env VENV_DIR="$VENV_DIR" "$SCRIPT_DIR/04-install-pytorch.sh"
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "ERROR: venv still not found at $VENV_DIR after bootstrap."
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    if ! python -c "import uvicorn, fastapi" >/dev/null 2>&1; then
        echo "==> Dashboard deps missing in venv; running 08-install-dashboard.sh..."
        deactivate
        _run_as_user env VENV_DIR="$VENV_DIR" "$SCRIPT_DIR/08-install-dashboard.sh"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# If the llama-server systemd unit exists, make a best-effort start so the
# dashboard's first status poll has a server to talk to. Without systemd, the
# Server tab can start/stop a direct llama-server process.
if systemctl list-unit-files llama-server.service >/dev/null 2>&1; then
    if systemctl is-active --quiet llama-server.service 2>/dev/null; then
        echo "==> llama-server.service already running on :${LMS_API_PORT}"
    else
        echo "==> Starting llama-server.service..."
        sudo -n systemctl reset-failed llama-server.service >/dev/null 2>&1 || true
        sudo -n systemctl start llama-server.service >/dev/null 2>&1 || \
            echo "    Could not start via sudo -n; use the Server tab or run ./10-install-systemd.sh."
    fi
elif [[ -x "$LLAMA_BIN" ]]; then
    echo "==> llama-server binary found at $LLAMA_BIN"
    echo "    Start it from the Server tab, or install the unit with ./10-install-systemd.sh."
else
    echo "==> llama-server not found at $LLAMA_BIN"
    echo "    Run ./05-install-llama-server.sh, then ./10-install-systemd.sh."
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

# Port-conflict handling. Under systemd we kill any orphan uvicorn still holding
# the port after a bad restart. When run interactively, we keep the old
# "do not double-start" behavior and print a clear message.
if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; then
    HOLDER_PIDS=$(ss -tlnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p' \
                    | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)

    if [[ -n "${INVOCATION_ID:-}" ]]; then
        echo "==> Port ${PORT} held by orphan pid(s): ${HOLDER_PIDS:-?}; killing..."
        for pid in $HOLDER_PIDS; do
            [[ "$pid" == "$$" ]] && continue
            kill -TERM "$pid" 2>/dev/null || true
        done

        for _ in 1 2 3; do
            sleep 1
            ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$" || break
        done

        for pid in $HOLDER_PIDS; do
            [[ "$pid" == "$$" ]] && continue
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        done
    else
        echo ""
        echo "================================================"
        echo "  Port ${PORT} already in use; not starting another dashboard."
        echo "  Existing listener: pid(s) ${HOLDER_PIDS:-unknown}"
        echo ""
        if systemctl is-active --quiet lmstudio-dashboard.service 2>/dev/null; then
            echo "  lmstudio-dashboard.service is running it; that is expected."
            echo "  Open: http://${IP}:${PORT}"
            echo "  To restart it: sudo systemctl restart lmstudio-dashboard.service"
            echo "  To run it by hand instead:"
            echo "    sudo systemctl stop lmstudio-dashboard.service && ./09-start-dashboard.sh"
        else
            echo "  Stop whatever is using the port, or set DASHBOARD_PORT=<other>."
        fi
        echo "================================================"
        echo ""
        exit 0
    fi
fi

echo ""
echo "================================================"
echo "  Dashboard -> http://${IP}:${PORT}"
echo "================================================"
echo ""

cd "$SCRIPT_DIR/dashboard"
exec uvicorn app:app --host "$HOST" --port "$PORT"
