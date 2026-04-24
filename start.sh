#!/usr/bin/env bash
# Convenience launcher for a headless box: activate the venv and start the dashboard.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-bin/current}"

if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "==> Python venv active: $VENV_DIR"
else
    echo "WARN: venv not found at $VENV_DIR; dashboard script may bootstrap it."
fi

if [[ -x "$LLAMA_DIR/llama-server" ]]; then
    echo "==> llama-server available: $LLAMA_DIR/llama-server"
else
    echo "WARN: llama-server not found. Run ./05-install-llama-server.sh."
fi

exec "$SCRIPT_DIR/09-start-dashboard.sh"
