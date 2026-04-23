#!/usr/bin/env bash
# Launch llama.cpp's llama-server (OpenAI-compatible API, truly CLI, no X/Electron).
# Auto-sizes the context window to the largest that fits in free GPU VRAM.
#
# Env knobs (all optional):
#   MODEL          GGUF file to serve          (default: Qwen Q3 GGUF we downloaded)
#   LLAMA_BIN      llama-server binary path    (default: ~/llama.cpp-bin/llama-b8893/llama-server)
#   LLAMA_PORT     API port                    (default: 1234)
#   LLAMA_BIND     interface IP to listen on   (default: first LAN IP)
#   LLAMA_NGL      layers offloaded to GPU     (default: 99 = all)
#   LLAMA_CTX      context length              (default: auto — largest that fits in VRAM)
#   LLAMA_THREADS  CPU threads                 (default: nproc)
#   KV_BYTES_PER_TOKEN  override KV-cache estimate per token, bytes (default: 65536)
#   OVERHEAD_MB    VRAM held back for compute buffers / OS (default: 1024)
set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen25-coder-7b-q3.gguf}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-bin/llama-b8893}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/llama-server}"
LLAMA_PORT="${LLAMA_PORT:-1234}"
LLAMA_BIND="${LLAMA_BIND:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
LLAMA_BIND="${LLAMA_BIND:-127.0.0.1}"
LLAMA_NGL="${LLAMA_NGL:-99}"
LLAMA_THREADS="${LLAMA_THREADS:-$(nproc)}"
# KV cache per token — *measured* on this box with flash-attn on and the
# Vulkan backend comes in at ≈34 KB/tok for Qwen 2.5 7B, not the
# formula's 57 KB. llama.cpp packs K/V tighter than the naive
# 2×kv_heads×head_dim×layers×2 calculation suggests when FA is enabled.
# Tune up (50000–65000) if you disable FA or switch to a model with
# more KV heads.
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-35000}"
# llama-server's non-KV overhead is ~300 MiB idle; prefill batch buffers
# eat another chunk when LLAMA_BATCH is large. 256 fills VRAM hardest
# without bumping into OOM at load.
OVERHEAD_MB="${OVERHEAD_MB:-256}"

# Context ceiling — llama.cpp caps at 128K internally but some builds
# permit higher. Set LLAMA_CTX_MAX to override the 131072 clamp.
LLAMA_CTX_MAX="${LLAMA_CTX_MAX:-131072}"

# ── throughput knobs ──────────────────────────────────────────────────────
# Flash Attention: 20–40 % prefill + a bit of decode speedup, costs no extra
# VRAM on llama.cpp's Vulkan/CUDA backends. `auto` lets llama.cpp pick on
# backends that can't safely enable it; `on` forces it.
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
# Batch sizes control prompt-prefill parallelism. Bigger = faster prefill,
# more VRAM during prefill. 8192/2048 roughly doubles the prompt
# throughput vs. llama-server's 2048/512 defaults on the 3060, and the
# VRAM spike only shows up while prefill is running.
LLAMA_BATCH="${LLAMA_BATCH:-8192}"
LLAMA_UBATCH="${LLAMA_UBATCH:-2048}"
# Single-user boxes should only reserve one KV slot; default 4 would split
# VRAM into 4 parallel sequences and leave ctx/4 per query.
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
# Use mlock to keep the model pages in RAM (prevents paging to disk under
# pressure). Harmless on GPU-offloaded setups too.
LLAMA_MLOCK="${LLAMA_MLOCK:-1}"

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-server not found at $LLAMA_BIN"
    echo "       Download a prebuilt llama.cpp release and extract into $LLAMA_DIR,"
    echo "       or set LLAMA_BIN / LLAMA_DIR."
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL"
    exit 1
fi

# ── auto-size context window ─────────────────────────────────────────────
# Strategy:
#   1. Start with the llama.cpp ceiling (LLAMA_CTX_MAX, default 131072).
#   2. Cap that by what the static VRAM-budget formula says can fit:
#        usable_kv_mb = free_vram - 1.1 × model_weights - overhead
#        max_ctx      = usable_kv_mb × 1024² / KV_BYTES_PER_TOKEN
#      (KV_BYTES_PER_TOKEN is empirical — with flash-attn on the Vulkan
#      backend it's ≈34 KB for Qwen 2.5 7B, not the naive 57 KB.)
#   3. Round down to the nearest 1024.
#
# If this overshoots in reality (OOM at load), the launch loop further
# down halves the ctx and retries — so "smart 131072" means we aim for
# the ceiling and let the retry loop back off until it fits.
_auto_ctx() {
    local free_mb total_mb model_bytes model_mb usable ctx="$LLAMA_CTX_MAX"
    if command -v nvidia-smi >/dev/null 2>&1; then
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    fi
    if [[ -n "${free_mb:-}" && "$free_mb" -gt 0 ]]; then
        model_bytes=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
        model_mb=$(( model_bytes / 1048576 ))
        model_mb=$(( model_mb + model_mb / 10 ))
        usable=$(( free_mb - model_mb - OVERHEAD_MB ))
        if (( usable >= 256 )); then
            local budget=$(( usable * 1048576 / KV_BYTES_PER_TOKEN ))
            (( budget < ctx )) && ctx=$budget
        else
            ctx=2048
        fi
        printf "    VRAM: %d / %d MiB free · model ~%d MiB · overhead %d MiB · KV %d B/token → aim ctx=%d (ceiling %d)\n" \
            "$free_mb" "$total_mb" "$model_mb" "$OVERHEAD_MB" "$KV_BYTES_PER_TOKEN" "$ctx" "$LLAMA_CTX_MAX" >&2
    fi
    (( ctx > LLAMA_CTX_MAX )) && ctx=$LLAMA_CTX_MAX
    (( ctx <  2048 ))         && ctx=2048
    ctx=$(( (ctx / 1024) * 1024 ))
    echo "$ctx"
}

if [[ -z "${LLAMA_CTX:-}" ]]; then
    echo "==> Computing max context for available VRAM..."
    LLAMA_CTX=$(_auto_ctx)
fi

echo "==> llama-server args:"
echo "    model       : $MODEL"
echo "    bind        : ${LLAMA_BIND}:${LLAMA_PORT}"
echo "    ctx         : $LLAMA_CTX"
echo "    gpu-layers  : $LLAMA_NGL"
echo "    threads     : $LLAMA_THREADS"
echo "    flash-attn  : $LLAMA_FLASH_ATTN"
echo "    batch/ubatch: ${LLAMA_BATCH}/${LLAMA_UBATCH}"
echo "    parallel    : $LLAMA_PARALLEL"
echo "    mlock       : $LLAMA_MLOCK"

# ── check port is free ──────────────────────────────────────────────────
if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LLAMA_PORT}\$"; then
    HOLDER=$(ss -tlnp 2>/dev/null | awk -v p=":${LLAMA_PORT}" '$4 ~ p {print $NF}' | head -1)
    echo "ERROR: port ${LLAMA_PORT} already in use: ${HOLDER:-unknown}"
    echo "       stop it first or set LLAMA_PORT=<other>"
    exit 1
fi

# ── exec llama-server ───────────────────────────────────────────────────
cd "$LLAMA_DIR"   # so the backend plugin libs resolve
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"

echo ""
echo "════════════════════════════════════════════════"
echo "  llama-server → http://${LLAMA_BIND}:${LLAMA_PORT}/v1"
echo "════════════════════════════════════════════════"
echo ""

EXTRA_ARGS=()
[[ "$LLAMA_MLOCK" == "1" ]] && EXTRA_ARGS+=(--mlock)

# Self-tuning launch loop: aim high, back off on early crash.
# llama-server either (a) binds the port within ~30 s of a successful load,
# (b) exits within ~30 s with an OOM / "failed to allocate" from the GPU
# backend, or (c) keeps running while it builds compute graphs.
#
# We check (a) and (b); if the process dies before binding we halve the
# context size and try again (down to 2048 minimum, max 4 attempts). On
# success we `wait` on the child so systemd sees our lifecycle and
# SIGTERM propagates cleanly.
_launch_with_ctx() {
    local ctx=$1
    echo "==> Launching llama-server with -c $ctx ..."
    "$LLAMA_BIN" \
        -m "$MODEL" \
        --host "$LLAMA_BIND" \
        --port "$LLAMA_PORT" \
        -c "$ctx" \
        -ngl "$LLAMA_NGL" \
        -t "$LLAMA_THREADS" \
        -b "$LLAMA_BATCH" \
        -ub "$LLAMA_UBATCH" \
        -fa "$LLAMA_FLASH_ATTN" \
        --parallel "$LLAMA_PARALLEL" \
        "${EXTRA_ARGS[@]}" &
    LLAMA_PID=$!

    # Give it up to 45 s to either bind the port or die.
    for _ in $(seq 1 45); do
        if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LLAMA_PORT}\$"; then
            return 0   # bound — happy path
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            return 1   # died before binding
        fi
        sleep 1
    done
    # Timed out waiting for bind but still alive; treat as success so systemd
    # doesn't flip-flop and we don't kill a slow-loader.
    return 0
}

CUR_CTX="$LLAMA_CTX"
for attempt in 1 2 3 4; do
    if _launch_with_ctx "$CUR_CTX"; then
        # Clean handoff: forward signals so systemd stop works as expected.
        _fwd() { kill -"$1" "$LLAMA_PID" 2>/dev/null || true; wait "$LLAMA_PID"; exit $?; }
        trap '_fwd TERM' TERM
        trap '_fwd INT'  INT
        wait "$LLAMA_PID"
        exit $?
    fi
    NEXT=$(( CUR_CTX / 2 ))
    if (( NEXT < 2048 )); then
        echo "ERROR: llama-server wouldn't even load at ctx=$CUR_CTX — giving up."
        exit 1
    fi
    echo "==> llama-server died at ctx=$CUR_CTX (likely OOM); retrying at ctx=$NEXT"
    CUR_CTX=$NEXT
done
echo "ERROR: exhausted retry attempts."
exit 1
