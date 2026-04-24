#!/usr/bin/env bash
# Legacy optional: start LM Studio in a virtual display and stream the GUI to a browser via noVNC.
# The default headless provider is start-llama-server.sh.
#
# Access LM Studio at:  http://<server-ip>:6080/vnc.html
#
# Environment overrides:
#   LMSTUDIO_DIR   path to the directory containing LMStudio.AppImage
#   VNC_DISPLAY    X display number (default: 99)
#   VNC_PORT       raw VNC port   (default: 5900)
#   NOVNC_PORT     browser port   (default: 6080)
#   VNC_PASSWORD   set to protect the VNC session (default: no password)
#   RESOLUTION     virtual screen resolution (default: 1920x1080)
set -euo pipefail

LMSTUDIO_DIR="${LMSTUDIO_DIR:-/workspace/LMStudio}"
APPIMAGE="$LMSTUDIO_DIR/LMStudio.AppImage"

VNC_DISPLAY="${VNC_DISPLAY:-99}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
RESOLUTION="${RESOLUTION:-1920x1080}"

export DISPLAY=":${VNC_DISPLAY}"

# ── sanity checks ──────────────────────────────────────────────────────────
if [[ ! -f "$APPIMAGE" ]]; then
    echo "ERROR: LM Studio not found at $APPIMAGE"
    echo "       Run ./05-install-lmstudio.sh first, or set LMSTUDIO_DIR."
    exit 1
fi

for cmd in Xvfb x11vnc websockify; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found. Run ./02-install-prerequisites.sh first."
        exit 1
    fi
done

# Find noVNC web root
NOVNC_WEB=""
for candidate in /usr/share/novnc /usr/share/noVNC /opt/noVNC; do
    if [[ -f "$candidate/vnc.html" ]]; then
        NOVNC_WEB="$candidate"
        break
    fi
done
if [[ -z "$NOVNC_WEB" ]]; then
    echo "ERROR: noVNC web root not found. Run ./02-install-prerequisites.sh first."
    exit 1
fi

# ── kill any stale processes on our display ────────────────────────────────
pkill -f "Xvfb :${VNC_DISPLAY}" 2>/dev/null || true
pkill -f "x11vnc.*:${VNC_DISPLAY}" 2>/dev/null || true
pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
sleep 0.5

# ── start virtual display ──────────────────────────────────────────────────
echo "==> Starting virtual display :${VNC_DISPLAY} (${RESOLUTION})..."
Xvfb ":${VNC_DISPLAY}" -screen 0 "${RESOLUTION}x24" -ac +extension GLX &
XVFB_PID=$!
sleep 1

# ── start VNC server ───────────────────────────────────────────────────────
echo "==> Starting x11vnc on port ${VNC_PORT}..."
X11VNC_ARGS=(-display ":${VNC_DISPLAY}" -forever -shared -quiet -rfbport "$VNC_PORT")
if [[ -n "${VNC_PASSWORD:-}" ]]; then
    X11VNC_ARGS+=(-passwd "$VNC_PASSWORD")
else
    X11VNC_ARGS+=(-nopw)
fi
x11vnc "${X11VNC_ARGS[@]}" &
X11VNC_PID=$!
sleep 1

# ── start noVNC (browser client) ───────────────────────────────────────────
echo "==> Starting noVNC on port ${NOVNC_PORT}..."
websockify --web "$NOVNC_WEB" "$NOVNC_PORT" "localhost:${VNC_PORT}" &
NOVNC_PID=$!
sleep 1

# ── launch LM Studio ───────────────────────────────────────────────────────
echo "==> Launching LM Studio..."
DISPLAY=":${VNC_DISPLAY}" "$APPIMAGE" --no-sandbox &
LMSTUDIO_PID=$!

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "════════════════════════════════════════════════"
echo "  LM Studio  →  http://${IP}:${NOVNC_PORT}/vnc.html"
echo "  (open that URL in any browser on your network)"
echo "════════════════════════════════════════════════"
echo ""

# ── wait and clean up ─────────────────────────────────────────────────────
_cleanup() {
    echo "==> Shutting down LM Studio VNC stack..."
    kill "$LMSTUDIO_PID" "$NOVNC_PID" "$X11VNC_PID" "$XVFB_PID" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# Stay alive as long as LM Studio is running
wait "$LMSTUDIO_PID" || true
