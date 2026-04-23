#!/usr/bin/env bash
# Step 10: install systemd units so the dashboard AND the llama.cpp API
# server auto-start on boot.
#
#   lmstudio-dashboard.service  — the Python/FastAPI dashboard on :8765
#   llama-server.service        — llama.cpp OpenAI-compatible API on :1234
#                                 (auto-sizes context to free VRAM at boot)
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

USER_NAME="${SUDO_USER:-$USER}"
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
VENV_DIR="${VENV_DIR:-$USER_HOME/pytorch-env}"
LMSTUDIO_DIR="${LMSTUDIO_DIR:-$USER_HOME/LMStudio}"
LLAMA_DIR="${LLAMA_DIR:-$USER_HOME/llama.cpp-bin/llama-b8893}"
MODEL="${MODEL:-$USER_HOME/models/qwen25-coder-7b-q3.gguf}"

_render_and_install() {
    local name=$1 template="$SCRIPT_DIR/systemd/${name}"
    local target="/etc/systemd/system/${name}"
    [[ -f "$template" ]] || { echo "WARN: template $template missing — skipping"; return 1; }
    echo "==> Rendering $name (user=$USER_NAME, install=$SCRIPT_DIR)"
    sudo sed \
        -e "s|__USER__|$USER_NAME|g" \
        -e "s|__INSTALL_DIR__|$SCRIPT_DIR|g" \
        -e "s|__VENV_DIR__|$VENV_DIR|g" \
        -e "s|__LMSTUDIO_DIR__|$LMSTUDIO_DIR|g" \
        -e "s|__LLAMA_DIR__|$LLAMA_DIR|g" \
        -e "s|__MODEL__|$MODEL|g" \
        -e "s|__HOME__|$USER_HOME|g" \
        "$template" | sudo tee "$target" >/dev/null
}

_render_and_install "lmstudio-dashboard.service" || true
_render_and_install "llama-server.service"       || true

echo "==> Reloading systemd..."
sudo systemctl daemon-reload

# Dashboard: start only if venv is present (avoids a restart loop while the
# install is still mid-flight).
if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    echo "==> Enabling + starting lmstudio-dashboard.service..."
    sudo systemctl enable --now lmstudio-dashboard.service
else
    echo "==> venv not ready — enabling dashboard for next boot only."
    sudo systemctl enable lmstudio-dashboard.service
fi

# llama-server: start only if both the binary and the model exist.
if [[ -x "$LLAMA_DIR/llama-server" && -f "$MODEL" ]]; then
    echo "==> Enabling + starting llama-server.service..."
    sudo systemctl enable --now llama-server.service
else
    echo "==> llama-server binary or model not present — enabling for next boot only."
    echo "    binary: $LLAMA_DIR/llama-server"
    echo "    model : $MODEL"
    sudo systemctl enable llama-server.service 2>/dev/null || true
fi

echo "==> Done. Status:"
sudo systemctl --no-pager status lmstudio-dashboard.service 2>&1 | head -8 || true
echo
sudo systemctl --no-pager status llama-server.service 2>&1 | head -8 || true
echo
IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo "Dashboard     → http://$IP:8765"
echo "llama-server  → http://$IP:1234/v1"
