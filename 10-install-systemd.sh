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

echo "==> Reloading systemd..."
sudo systemctl daemon-reload

# Only start the service if the venv is already in place — starting it against
# a missing venv previously put 09-start-dashboard.sh into a 5s sudo-prompt
# restart loop (see restart-limit fields in the unit file for a safety net).
if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "==> venv present — enabling and starting service..."
    sudo systemctl enable --now lmstudio-dashboard.service
else
    echo "==> venv not found at $VENV_DIR — enabling for next boot only."
    echo "    (Start manually with: sudo systemctl start lmstudio-dashboard.service)"
    sudo systemctl enable lmstudio-dashboard.service
fi

echo "==> Done. Status:"
sudo systemctl --no-pager status lmstudio-dashboard.service || true
echo ""
echo "Dashboard will start on every boot. Visit: http://$(hostname -I | awk '{print $1}'):8765"
