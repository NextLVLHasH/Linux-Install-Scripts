#!/usr/bin/env bash
# Legacy optional: download the LM Studio AppImage and set it up as a launchable app.
# The default provider is now 05-install-llama-server.sh.
set -euo pipefail

LMSTUDIO_DIR="${LMSTUDIO_DIR:-$HOME/LMStudio}"
LMSTUDIO_URL="${LMSTUDIO_URL:-https://installers.lmstudio.ai/linux/x64/0.3.9-6/LM-Studio-0.3.9-6-x64.AppImage}"
APPIMAGE_PATH="$LMSTUDIO_DIR/LMStudio.AppImage"

mkdir -p "$LMSTUDIO_DIR"

echo "==> Downloading LM Studio AppImage..."
echo "    URL: $LMSTUDIO_URL"
echo "    (Override by exporting LMSTUDIO_URL before running.)"
if ! curl -L --fail -o "$APPIMAGE_PATH" "$LMSTUDIO_URL"; then
    echo "ERROR: download failed. Visit https://lmstudio.ai/ to get the current Linux URL"
    echo "       then re-run: LMSTUDIO_URL=<new url> ./05-install-lmstudio.sh"
    exit 1
fi

chmod +x "$APPIMAGE_PATH"

echo "==> Creating desktop launcher..."
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/lmstudio.desktop" <<EOF
[Desktop Entry]
Name=LM Studio
Exec=$APPIMAGE_PATH %U
Terminal=false
Type=Application
Icon=applications-science
Categories=Development;Science;
Comment=Local LLM runner
EOF

echo "==> LM Studio installed at: $APPIMAGE_PATH"
