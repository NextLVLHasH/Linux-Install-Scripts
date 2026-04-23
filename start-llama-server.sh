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
# KV cache per token is derived from the gguf metadata below (via
# _probe_kv_per_token) so the script adapts to whatever model is loaded
# without hand-tuning. You can still pin a value with KV_BYTES_PER_TOKEN=.
# Fallback (if the probe fails) matches a Qwen 2.5 7B with flash-attn.
KV_BYTES_PER_TOKEN_FALLBACK=35000
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
# Batch sizes control prompt-prefill parallelism. Bigger = faster
# prefill, but also a bigger compute-buffer allocation that scales
# with batch × embedding_length × layers — so 8192/2048 that worked
# fine for a 4B model can fail 'graph_reserve: failed to allocate
# compute buffers' on a 7-14B. Start at 4096/1024 (still double the
# llama-server defaults), the retry loop shrinks further on OOM.
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

# ── probe gguf for KV cache size per token ───────────────────────────────
# Use llama-cli --show-metadata to dump the gguf header, then compute:
#   bytes_per_token = 2 (K+V) × n_layers × head_count_kv × head_dim × 2 (fp16)
# where head_dim prefers attention.key_length if present, otherwise falls
# back to embedding_length / head_count. Halve for flash-attn on
# backends that can share a single cache (best-effort — if unsure we
# keep the full size and let the retry loop back off).
_probe_kv_per_token() {
    # Caller-supplied override wins.
    if [[ -n "${KV_BYTES_PER_TOKEN:-}" ]]; then
        echo "$KV_BYTES_PER_TOKEN"; return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "$KV_BYTES_PER_TOKEN_FALLBACK"; return
    fi
    # Parse the gguf header directly. llama.cpp's own CLIs don't expose a
    # stable metadata-dump flag across versions, so do it ourselves — the
    # format is documented + simple. Prints the per-token KV size to stdout
    # and a human summary to stderr; falls back silently to the default on
    # any parse problem.
    python3 - "$MODEL" "$KV_BYTES_PER_TOKEN_FALLBACK" <<'PY' 2>/dev/null
import struct, sys
path, fallback = sys.argv[1], int(sys.argv[2])

def read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, vt):
    scalar = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',10:'<Q',11:'<q',12:'<d'}
    sizes  = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
    if vt in scalar:
        return struct.unpack(scalar[vt], f.read(sizes[vt]))[0]
    if vt == 7:                                    # bool
        return bool(f.read(1)[0])
    if vt == 8:                                    # string
        return read_str(f)
    if vt == 9:                                    # array
        (at,) = struct.unpack('<I', f.read(4))
        (n,)  = struct.unpack('<Q', f.read(8))
        return [read_value(f, at) for _ in range(n)]
    raise ValueError(f'unknown gguf type {vt}')

try:
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            print(fallback); sys.exit(0)
        struct.unpack('<I', f.read(4))             # version
        (_, n_kv) = struct.unpack('<QQ', f.read(16))
        kv = {}
        for _ in range(n_kv):
            k = read_str(f)
            (vt,) = struct.unpack('<I', f.read(4))
            kv[k] = read_value(f, vt)

    arch   = kv.get('general.architecture', '')
    P      = lambda s: f'{arch}.{s}'
    layers = kv.get(P('block_count'))
    khv    = kv.get(P('attention.head_count_kv'))
    nh     = kv.get(P('attention.head_count'))
    kl     = kv.get(P('attention.key_length'))
    vl     = kv.get(P('attention.value_length'))
    emb    = kv.get(P('embedding_length'))

    # Hybrid architectures (Qwen 3.5 Gated Delta Net, Mamba/Jamba, ...)
    # only put full attention on every Nth layer; the rest use recurrent
    # state whose memory footprint is fixed, not per-token. If the gguf
    # header tells us that interval, scale the attention-layer count
    # accordingly — otherwise we over-allocate and undersize the ctx.
    full_interval = kv.get(P('full_attention_interval'))
    has_ssm       = any(k.startswith(f'{arch}.ssm.') for k in kv)

    if layers is None or khv is None:
        print(fallback); sys.exit(0)
    if kl is None and nh and emb:
        kl = emb // nh
    kl = kl or 128
    vl = vl or kl

    attn_layers = layers
    hybrid_note = ''
    if full_interval and full_interval > 1:
        attn_layers = max(1, layers // full_interval)
        hybrid_note = f' (hybrid: {attn_layers}/{layers} attn layers, interval={full_interval})'
    elif has_ssm:
        # SSM models with no explicit interval key — rough default: assume
        # ~1/4 of layers are attention. Override with KV_BYTES_PER_TOKEN=
        # if this heuristic is wrong for your model.
        attn_layers = max(1, layers // 4)
        hybrid_note = f' (ssm hybrid: estimated {attn_layers}/{layers} attn layers)'

    # K cache + V cache, fp16, one entry per (layer, kv_head) per token.
    bpt = attn_layers * khv * (kl + vl) * 2
    print(bpt)
    print(f'    KV from gguf: arch={arch} layers={layers} kv_heads={khv} '
          f'key_len={kl} val_len={vl}{hybrid_note} → {bpt} B/token',
          file=sys.stderr)
except Exception as e:
    print(fallback)
    print(f'    gguf probe failed: {e}', file=sys.stderr)
PY
}

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
    echo "==> Probing gguf for KV cache footprint..."
    KV_BYTES_PER_TOKEN=$(_probe_kv_per_token)
    export KV_BYTES_PER_TOKEN
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
# Two failure modes to back off from separately:
#   - KV OOM: 'Vulkan0 KV buffer size …' followed by allocation failure
#     → halve ctx.
#   - Compute-buffer OOM: 'graph_reserve: failed to allocate compute
#     buffers' → halve batch/ubatch (compute buf scales with batch).
# We don't parse the log in-process; instead, on each failure we
# shrink whichever is bigger, in a round-robin-ish pattern:
#   attempt 1:  ctx ,     batch / ubatch
#   attempt 2:  ctx ,     batch/2 / ubatch/2
#   attempt 3:  ctx/2 ,   batch/2 / ubatch/2
#   attempt 4:  ctx/2 ,   batch/4 / ubatch/4
# On success we `wait` on the child so systemd sees our lifecycle and
# SIGTERM propagates cleanly.
_launch_once() {
    local ctx=$1 batch=$2 ubatch=$3
    echo "==> Launching llama-server: -c $ctx -b $batch -ub $ubatch ..."
    "$LLAMA_BIN" \
        -m "$MODEL" \
        --host "$LLAMA_BIND" \
        --port "$LLAMA_PORT" \
        -c "$ctx" \
        -ngl "$LLAMA_NGL" \
        -t "$LLAMA_THREADS" \
        -b "$batch" \
        -ub "$ubatch" \
        -fa "$LLAMA_FLASH_ATTN" \
        --parallel "$LLAMA_PARALLEL" \
        "${EXTRA_ARGS[@]}" &
    LLAMA_PID=$!

    # Up to 45 s for bind-or-die.
    for _ in $(seq 1 45); do
        if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LLAMA_PORT}\$"; then
            return 0
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done
    # Still alive but no bind yet — assume slow loader, hand off to wait.
    return 0
}

CUR_CTX="$LLAMA_CTX"
CUR_BATCH="$LLAMA_BATCH"
CUR_UBATCH="$LLAMA_UBATCH"
for attempt in 1 2 3 4 5; do
    if _launch_once "$CUR_CTX" "$CUR_BATCH" "$CUR_UBATCH"; then
        _fwd() { kill -"$1" "$LLAMA_PID" 2>/dev/null || true; wait "$LLAMA_PID"; exit $?; }
        trap '_fwd TERM' TERM
        trap '_fwd INT'  INT
        wait "$LLAMA_PID"
        exit $?
    fi
    case "$attempt" in
        1)  CUR_BATCH=$(( CUR_BATCH / 2 )); CUR_UBATCH=$(( CUR_UBATCH / 2 )) ;;
        2)  CUR_CTX=$(( CUR_CTX / 2 )) ;;
        3)  CUR_BATCH=$(( CUR_BATCH / 2 )); CUR_UBATCH=$(( CUR_UBATCH / 2 )) ;;
        4)  CUR_CTX=$(( CUR_CTX / 2 )) ;;
    esac
    # Floors — below these, llama.cpp's own minima kick in.
    (( CUR_BATCH  <  512 )) && CUR_BATCH=512
    (( CUR_UBATCH <  128 )) && CUR_UBATCH=128
    (( CUR_CTX    < 2048 )) && CUR_CTX=2048
    echo "==> Retry ${attempt}: ctx=$CUR_CTX batch=$CUR_BATCH ubatch=$CUR_UBATCH"
done
echo "ERROR: exhausted retry attempts — check 'journalctl -u llama-server.service'."
exit 1
