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
# KV cache per token ≈ 2 (K+V) × num_kv_heads × head_dim × num_layers × 2 bytes.
# Qwen 2.5 7B: 2×4×128×28×2 = 57344 B/tok. Default tuned to that; override for
# other architectures (e.g. MoE / 70B models use much more).
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-57344}"
# Tight overhead budget: llama-server itself reserves ~300–400 MiB for compute
# buffers on top of KV + model weights. 512 fills VRAM to ~95 %; leave more
# headroom if you hit OOM at load or during long prompts.
OVERHEAD_MB="${OVERHEAD_MB:-512}"

# ── throughput knobs ──────────────────────────────────────────────────────
# Flash Attention: 20–40 % prefill + a bit of decode speedup, costs no extra
# VRAM on llama.cpp's Vulkan/CUDA backends. `auto` lets llama.cpp pick on
# backends that can't safely enable it; `on` forces it.
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
# Batch sizes control prompt-prefill parallelism. Bigger = faster prefill,
# more VRAM. Defaults here push the 3060 harder than llama-server's default
# 2048/512 while staying well inside the budget at ctx≈130K.
LLAMA_BATCH="${LLAMA_BATCH:-4096}"
LLAMA_UBATCH="${LLAMA_UBATCH:-1024}"
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
# Formula:
#   usable_kv_mb = free_vram_mb - model_weights_mb - overhead_mb
#   max_ctx     = usable_kv_mb * 1024 * 1024 / KV_BYTES_PER_TOKEN
# Clamp to sensible bounds: >= 2048, <= 131072, rounded down to nearest 1024.
_auto_ctx() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "4096"   # no GPU info — sane CPU fallback
        return
    fi
    local free_mb total_mb model_bytes model_mb usable ctx
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$free_mb" || "$free_mb" -le 0 ]]; then
        echo "4096"; return
    fi
    model_bytes=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
    model_mb=$(( model_bytes / 1048576 ))
    # Add ~10% for load-time expansion of quantized weights in VRAM
    model_mb=$(( model_mb + model_mb / 10 ))
    usable=$(( free_mb - model_mb - OVERHEAD_MB ))
    if (( usable < 256 )); then
        # Not enough for a real context — user should pick a smaller quant
        echo "2048"
        return
    fi
    ctx=$(( usable * 1048576 / KV_BYTES_PER_TOKEN ))
    (( ctx > 131072 )) && ctx=131072
    (( ctx <  2048 )) && ctx=2048
    ctx=$(( (ctx / 1024) * 1024 ))
    echo "$ctx"
    # Emit sizing detail to stderr so the caller can see it without muddling stdout
    printf "    VRAM: %d / %d MiB free · model ~%d MiB · overhead %d MiB · KV %d B/token → ctx=%d\n" \
        "$free_mb" "$total_mb" "$model_mb" "$OVERHEAD_MB" "$KV_BYTES_PER_TOKEN" "$ctx" >&2
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

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host "$LLAMA_BIND" \
    --port "$LLAMA_PORT" \
    -c "$LLAMA_CTX" \
    -ngl "$LLAMA_NGL" \
    -t "$LLAMA_THREADS" \
    -b "$LLAMA_BATCH" \
    -ub "$LLAMA_UBATCH" \
    -fa "$LLAMA_FLASH_ATTN" \
    --parallel "$LLAMA_PARALLEL" \
    "${EXTRA_ARGS[@]}"
