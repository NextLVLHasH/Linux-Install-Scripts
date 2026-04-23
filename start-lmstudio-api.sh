#!/usr/bin/env bash
# Clean-restart the LM Studio OpenAI-compatible API using only the `lms` CLI.
# No GUI, no Xvfb — just the command-line interface.
#
# Env knobs:
#   LMS_API_PORT      port for the API            (default 1234)
#   LMS_BIND          interface to listen on      (default first LAN IP)
#   LMS_DEFAULT_MODEL model identifier to load    (default: none)
#   LMS_GPU           GPU offload                 (default max)
#   LMS_CTX           context length              (default 4096)
set -euo pipefail

LMS_BIN="$HOME/.lmstudio/bin/lms"
LMS_API_PORT="${LMS_API_PORT:-1234}"
LMS_BIND="${LMS_BIND:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
LMS_BIND="${LMS_BIND:-127.0.0.1}"
LMS_DEFAULT_MODEL="${LMS_DEFAULT_MODEL:-}"
LMS_GPU="${LMS_GPU:-max}"
LMS_CTX="${LMS_CTX:-4096}"

if [[ ! -x "$LMS_BIN" ]]; then
    echo "ERROR: lms CLI not found at $LMS_BIN."
    echo "       Run ./05-install-lmstudio.sh and then bootstrap once."
    exit 1
fi

# ── kill: stop any running server + drop loaded models ───────────────────
echo "==> Stopping any existing lms server..."
"$LMS_BIN" server stop 2>&1 | tail -2 || true

echo "==> Unloading all models..."
"$LMS_BIN" unload --all 2>&1 | tail -2 || true

# ── pre-write the HTTP config so the server binds to the right interface ─
CFG="$HOME/.lmstudio/.internal/http-server-config.json"
if [[ -r "$CFG" ]] && command -v python3 >/dev/null 2>&1; then
    python3 - "$CFG" "$LMS_BIND" "$LMS_API_PORT" <<'PY'
import json, sys, pathlib
cfg_path, bind, port = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
try:
    cfg = json.loads(cfg_path.read_text())
except Exception:
    cfg = {}
cfg.update({
    "networkInterface": bind,
    "port": port,
    "cors": False,
})
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"    bind {bind}:{port}")
PY
fi

# ── start: bring the API server up ──────────────────────────────────────
echo "==> Starting lms server on port ${LMS_API_PORT}..."
"$LMS_BIN" server start --port "$LMS_API_PORT" 2>&1 | tail -5

# Verify the API is actually listening.
for _ in $(seq 1 20); do
    if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LMS_API_PORT}\$"; then
        break
    fi
    sleep 1
done
if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LMS_API_PORT}\$"; then
    echo "ERROR: lms server did not bind to port ${LMS_API_PORT}."
    echo "       Check: $LMS_BIN status"
    exit 1
fi

# ── optionally load a default model ─────────────────────────────────────
if [[ -n "$LMS_DEFAULT_MODEL" ]]; then
    echo "==> Loading model: $LMS_DEFAULT_MODEL (gpu=$LMS_GPU ctx=$LMS_CTX)"
    "$LMS_BIN" load "$LMS_DEFAULT_MODEL" --gpu "$LMS_GPU" --context-length "$LMS_CTX" 2>&1 | tail -3 || \
        echo "WARN: model load failed — API is still up, load manually later."
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "════════════════════════════════════════════════"
echo "  LM Studio API → http://${IP}:${LMS_API_PORT}/v1"
echo "════════════════════════════════════════════════"

# ── status summary ───────────────────────────────────────────────────────
"$LMS_BIN" status 2>&1 | tail -15 || true
