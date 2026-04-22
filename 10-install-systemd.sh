#!/usr/bin/env bash
# Step 10: install systemd unit so the dashboard auto-starts on boot.
# Works on Ubuntu and ZimaOS (Debian-based, systemd).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TEMPLATE="$SCRIPT_DIR/systemd/lmstudio-dashboard.service"
TARGET="/etc/systemd/system/lmstudio-dashboard.service"

USER_NAME="${SUDO_USER:-$USER}"
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
VENV_DIR="${VENV_DIR:-$USER_HOME/pytorch-env}"
LMSTUDIO_DIR="${LMSTUDIO_DIR:-$USER_HOME/LMStudio}"

echo "==> Rendering service file for user=$USER_NAME, install=$SCRIPT_DIR"
sudo sed \
    -e "s|__USER__|$USER_NAME|g" \
    -e "s|__INSTALL_DIR__|$SCRIPT_DIR|g" \
    -e "s|__VENV_DIR__|$VENV_DIR|g" \
    -e "s|__LMSTUDIO_DIR__|$LMSTUDIO_DIR|g" \
    -e "s|__HOME__|$USER_HOME|g" \
    "$TEMPLATE" | sudo tee "$TARGET" >/dev/null

echo "==> Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable --now lmstudio-dashboard.service

echo "==> Done. Status:"
sudo systemctl --no-pager status lmstudio-dashboard.service || true
echo ""
echo "Dashboard will start on every boot. Visit: http://$(hostname -I | awk '{print $1}'):8765"
