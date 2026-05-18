#!/bin/bash
# installstart.sh — One-shot RunPod install + start script.
#
# Stands up DavidAU's Qwen3.6-40B-Deck-Opus (uncensored, NEO-CODE Q8_0) and starts
# llama-server with 1,010,000-token context via YaRN rope scaling + Q8 KV cache
# quantization. Endpoint lands at http://0.0.0.0:1234/v1 — point HasH AI there.
#
# PERSISTENCE MODEL — RunPod deletes container storage when a pod terminates;
# only /workspace (the template / network-volume drive) survives. This script
# redirects every install location to /workspace BEFORE running any installer
# so a stop/start cycle doesn't wipe LM Studio, the HF cache, the pip packages,
# or the login token. On a fresh pod, just re-run the script — it detects
# everything still on /workspace and skips the slow parts (download, bootstrap)
# while redoing the ephemeral pieces (home-dir symlinks, ~/.bashrc, venv
# activation). First run ≈ 15 min (mostly model download). Restart re-run ≈ 30s.
#
# Usage:
#   chmod +x installstart.sh
#   ./installstart.sh
#   # Foreground — run inside tmux/screen if you'll disconnect:
#   tmux new -s llama
#   ./installstart.sh
#   # Ctrl-b d to detach; reattach with: tmux attach -t llama

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — overridable via env vars; otherwise prompted for at runtime
# ---------------------------------------------------------------------------
PERSIST="${PERSIST:-/workspace}"        # RunPod's persistent drive
PORT="${LLAMA_PORT:-1234}"

# YaRN rope-scaling parameters. These defaults extend Qwen3.6's native 262144
# context to ~1.01M tokens via factor=4. Override via env vars for other models:
#   ROPE_FREQ_BASE — leave at the model's native value unless the card says otherwise
#   ROPE_SCALE     — multiplier; 1 disables YaRN (uses native context)
#   YARN_ORIG_CTX  — the model's TRAINED context length (not the scaled target)
#   TARGET_CONTEXT — what you want to actually run at; must equal YARN_ORIG_CTX*ROPE_SCALE
ROPE_FREQ_BASE="${ROPE_FREQ_BASE:-10000000}"
ROPE_SCALE="${ROPE_SCALE:-4}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-262144}"
TARGET_CONTEXT="${TARGET_CONTEXT:-1010000}"

# KV cache quantization. FP16 KV at 1M would need ~96 GB on its own — mandatory
# to quantize on a 96 GB card. q8_0 = 48 GB at 1M, q4_0 = 24 GB. Quality drop
# from q8_0 is negligible; q4_0 starts to show on long-context recall tasks.
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "==> $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [ ! -d "$PERSIST" ]; then
  echo "ERROR: \$PERSIST=$PERSIST does not exist. Set PERSIST to your template's"
  echo "       persistent mount (/workspace on most RunPod templates, sometimes"
  echo "       /runpod-volume or /persistent). Run 'df -h' to find it."
  exit 1
fi

# ---------------------------------------------------------------------------
# 0. Resolve the Hugging Face model. Three input shapes accepted:
#
#    (a) Full URL with the file picker query — what you get by clicking a file
#        on the repo page and copying the URL:
#        https://huggingface.co/OWNER/REPO?show_file_info=PATH/TO/FILE.gguf
#
#    (b) Direct file URL — works with either `/blob/main/` or `/resolve/main/`:
#        https://huggingface.co/OWNER/REPO/blob/main/PATH/TO/FILE.gguf
#        https://huggingface.co/OWNER/REPO/resolve/main/PATH/TO/FILE.gguf
#
#    (c) Repo URL only — downloads the whole repo (every quant); script will
#        prompt for which file to launch with:
#        https://huggingface.co/OWNER/REPO
#
# Input source priority: $HF_URL env var → first positional arg → interactive
# prompt. The script is re-runnable; for a fresh pod, set HF_URL and skip the
# prompt so unattended re-runs after `tmux new` work the same way.
# ---------------------------------------------------------------------------
step "0/10  Hugging Face model"
HF_INPUT="${HF_URL:-${1:-}}"
if [ -z "$HF_INPUT" ]; then
  echo "    Paste the HuggingFace URL (with ?show_file_info=… or /blob/main/…)"
  echo "    Example: https://huggingface.co/DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF?show_file_info=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf"
  printf "    HF URL: "
  read -r HF_INPUT
fi
if [ -z "$HF_INPUT" ]; then
  echo "ERROR: no URL provided. Re-run with HF_URL=… or paste a URL when prompted."
  exit 1
fi

# Parser. Strip the protocol + host, then peel off either the ?show_file_info=
# query, the /blob/<branch>/ segment, or the /resolve/<branch>/ segment to
# recover REPO (owner/name) and FILE (path within the repo).
PARSE="${HF_INPUT#https://}"
PARSE="${PARSE#http://}"
PARSE="${PARSE#huggingface.co/}"
PARSE="${PARSE#hf.co/}"

MODEL_FILE=""
case "$PARSE" in
  *\?show_file_info=*)
    MODEL_REPO="${PARSE%%\?*}"
    MODEL_FILE="${PARSE#*\?show_file_info=}"
    # Strip any trailing query params after the filename (rare but possible)
    MODEL_FILE="${MODEL_FILE%%&*}"
    ;;
  */blob/*/*|*/resolve/*/*)
    # Format: OWNER/REPO/blob/BRANCH/FILE  →  OWNER/REPO + FILE
    OWNER_REPO="${PARSE%%/blob/*}"
    OWNER_REPO="${OWNER_REPO%%/resolve/*}"
    MODEL_REPO="$OWNER_REPO"
    REST="${PARSE#"$OWNER_REPO/"}"
    REST="${REST#blob/}"
    REST="${REST#resolve/}"
    # Strip the branch (first path segment of REST)
    MODEL_FILE="${REST#*/}"
    ;;
  *)
    # No file specifier — repo-only URL. Will download the whole repo and
    # prompt for which file to launch.
    MODEL_REPO="$PARSE"
    MODEL_FILE=""
    ;;
esac

# Trim any trailing slashes / fragments
MODEL_REPO="${MODEL_REPO%/}"
MODEL_REPO="${MODEL_REPO%%\#*}"
MODEL_FILE="${MODEL_FILE%%\#*}"

if [ -z "$MODEL_REPO" ] || [[ "$MODEL_REPO" != */* ]]; then
  echo "ERROR: couldn't parse OWNER/REPO from '$HF_INPUT'"
  exit 1
fi

# Derive a directory slug from the repo name for $PERSIST/models/<slug>/
SLUG=$(echo "$MODEL_REPO" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9.-' | cut -c1-80)
MODEL_DIR="$PERSIST/models/$SLUG"

echo "    repo: $MODEL_REPO"
echo "    file: ${MODEL_FILE:-<whole repo>}"
echo "    dest: $MODEL_DIR"

# ---------------------------------------------------------------------------
# 1. Persistent directories — every place an installer might write
# ---------------------------------------------------------------------------
step "1/10  Persistent directories on $PERSIST"
mkdir -p \
  "$PERSIST/.lmstudio" \
  "$PERSIST/.cache/huggingface" \
  "$PERSIST/.cache/lm-studio" \
  "$PERSIST/.cache/pip" \
  "$PERSIST/models" \
  "$PERSIST/venv-parent" \
  "$PERSIST/bin"
echo "    ok"

# ---------------------------------------------------------------------------
# 2. Symlink home-directory locations to /workspace — so the LM Studio installer,
#    huggingface_hub, and pip all write straight to persistent storage. We
#    recreate these on every run because /root is ephemeral and a pod restart
#    wipes the previous symlinks even though their targets survive.
# ---------------------------------------------------------------------------
step "2/10  Symlink ~/.lmstudio, ~/.cache/* → $PERSIST"
relink() {
  local target="$1" link="$2"
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    echo "    ok: $link → $target"
    return
  fi
  # If a real directory sits at $link (first-ever run on a pod that wrote to
  # ephemeral storage before this script touched it), MOVE its contents into
  # $target so we don't lose anything, then replace with symlink.
  if [ -d "$link" ] && [ ! -L "$link" ]; then
    echo "    migrating existing $link into $target"
    cp -an "$link"/. "$target"/ 2>/dev/null || true
    rm -rf "$link"
  fi
  rm -f "$link"
  ln -s "$target" "$link"
  echo "    linked: $link → $target"
}
relink "$PERSIST/.lmstudio"          "$HOME/.lmstudio"
relink "$PERSIST/.cache/huggingface" "$HOME/.cache/huggingface"
relink "$PERSIST/.cache/lm-studio"   "$HOME/.cache/lm-studio"
relink "$PERSIST/.cache/pip"         "$HOME/.cache/pip"

# ---------------------------------------------------------------------------
# 3. Write a persistent env-init file at $PERSIST/init.sh and source it from
#    ~/.bashrc. After a pod restart you can also `source $PERSIST/init.sh`
#    manually to get the same shell setup without re-running this script.
# ---------------------------------------------------------------------------
step "3/10  Persistent shell init"
cat > "$PERSIST/init.sh" <<EOF
# Auto-generated by installstart.sh — re-sourced on every shell start
export PATH="\$HOME/.lmstudio/bin:$PERSIST/bin:\$PATH"
export HF_HOME="$PERSIST/.cache/huggingface"
export PIP_CACHE_DIR="$PERSIST/.cache/pip"
# Activate the persistent Python venv if it exists
if [ -f "$PERSIST/venv-parent/llama/bin/activate" ]; then
  source "$PERSIST/venv-parent/llama/bin/activate"
fi
EOF
if ! grep -q "source $PERSIST/init.sh" "$HOME/.bashrc" 2>/dev/null; then
  echo "source $PERSIST/init.sh" >> "$HOME/.bashrc"
fi
# Load it in THIS shell too
source "$PERSIST/init.sh"
echo "    $PERSIST/init.sh written; sourced from ~/.bashrc"

# ---------------------------------------------------------------------------
# 4. Persistent Python venv for huggingface_hub and any future tooling.
#    Lives on /workspace so pip install survives pod restart.
# ---------------------------------------------------------------------------
step "4/10  Python venv at $PERSIST/venv-parent/llama"
if [ ! -f "$PERSIST/venv-parent/llama/bin/activate" ]; then
  python3 -m venv "$PERSIST/venv-parent/llama"
  echo "    created"
else
  echo "    already exists"
fi
source "$PERSIST/venv-parent/llama/bin/activate"

# ---------------------------------------------------------------------------
# 5. Install LM Studio CLI. Because ~/.lmstudio is now a symlink to /workspace,
#    the installer writes binaries + runtimes straight to persistent storage.
# ---------------------------------------------------------------------------
step "5/10  LM Studio CLI"
if have lms && [ -x "$HOME/.lmstudio/bin/lms" ]; then
  echo "    already installed: $(lms version 2>/dev/null | head -1)"
else
  curl -fsSL https://lmstudio.ai/install.sh | bash
fi
# Make sure the binary is on PATH for THIS shell (init.sh did it but only if
# the file existed when init.sh sourced; the installer may have created it
# AFTER that point on the first run)
export PATH="$HOME/.lmstudio/bin:$PATH"

# ---------------------------------------------------------------------------
# 6. LM Studio runtimes (llama.cpp + CUDA backend). Skipped if already present
#    under ~/.lmstudio/extensions/backends (persistent via the symlink).
# ---------------------------------------------------------------------------
step "6/10  LM Studio runtimes"
if [ -d "$HOME/.lmstudio/extensions/backends" ] && \
   find "$HOME/.lmstudio/extensions/backends" -name "llama-server" -type f 2>/dev/null | grep -q .; then
  echo "    already bootstrapped (llama-server found)"
else
  lms bootstrap
fi

# ---------------------------------------------------------------------------
# 7. huggingface_hub CLI inside the persistent venv. pip cache + token both
#    live on /workspace via the symlinks set up above.
# ---------------------------------------------------------------------------
step "7/10  huggingface_hub"
if pip show huggingface_hub >/dev/null 2>&1; then
  echo "    already installed in venv: $(pip show huggingface_hub | grep -i ^version)"
else
  pip install -U huggingface_hub
fi

# Optional auth. Use $HF_TOKEN if exported; otherwise just note the login path.
# The token lands at $HF_HOME/token which is on /workspace, so login persists.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "    using HF_TOKEN from env"
  hf auth login --token "$HF_TOKEN" 2>/dev/null || true
elif [ -f "$HF_HOME/token" ]; then
  echo "    already logged in (token at $HF_HOME/token)"
else
  echo "    not logged in. Public models still download (rate-limited)."
  echo "    For higher limits or gated models: hf auth login"
fi

# ---------------------------------------------------------------------------
# 8. Model download. Goes to $PERSIST/models/<slug>/ via --local-dir. hf download
#    is idempotent — re-runs are a no-op once the blob is cached. If the URL was
#    repo-only, pull the whole repo and prompt for which file to launch.
# ---------------------------------------------------------------------------
step "8/10  Model — ${MODEL_FILE:-<repo: $MODEL_REPO>}"
mkdir -p "$MODEL_DIR"
if [ -n "$MODEL_FILE" ]; then
  EXPECTED_PATH="$MODEL_DIR/$MODEL_FILE"
  if [ -f "$EXPECTED_PATH" ] || [ -L "$EXPECTED_PATH" ]; then
    echo "    already at $EXPECTED_PATH ($(du -h "$EXPECTED_PATH" 2>/dev/null | awk '{print $1}' || echo '?'))"
  else
    echo "    downloading — first run pulls the full GGUF (10s of GB)"
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  fi
else
  echo "    URL didn't specify a file — pulling the whole repo (can be large)"
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  # Pick which GGUF to launch. If exactly one is present, use it; otherwise list
  # them and prompt.
  mapfile -t GGUFS < <(find "$MODEL_DIR" -maxdepth 4 -name "*.gguf" -not -name "mmproj*" | sort)
  if [ "${#GGUFS[@]}" -eq 0 ]; then
    echo "ERROR: no .gguf files found under $MODEL_DIR"
    exit 1
  elif [ "${#GGUFS[@]}" -eq 1 ]; then
    EXPECTED_PATH="${GGUFS[0]}"
    MODEL_FILE="$(basename "$EXPECTED_PATH")"
    echo "    only GGUF in repo: $MODEL_FILE"
  else
    echo "    multiple GGUFs found — pick one:"
    i=0
    for g in "${GGUFS[@]}"; do
      i=$((i+1))
      sz=$(du -h "$g" 2>/dev/null | awk '{print $1}')
      echo "      $i) $(basename "$g")  [$sz]"
    done
    printf "    Number: "
    read -r CHOICE
    EXPECTED_PATH="${GGUFS[$((CHOICE-1))]}"
    MODEL_FILE="$(basename "$EXPECTED_PATH")"
  fi
fi

# Resolve symlinks so we hand llama-server a real path (the HF cache layout
# uses snapshot/<hash> → blobs/<hash> symlinks; llama-server is fine following
# them but logging the resolved path is helpful for debugging).
REAL_MODEL=$(readlink -f "$EXPECTED_PATH")
echo "    resolved: $REAL_MODEL"
ls -lh "$REAL_MODEL" | awk '{print "    size:    "$5}'

# ---------------------------------------------------------------------------
# 9. Locate llama-server (inside the LM Studio runtime, now on /workspace)
# ---------------------------------------------------------------------------
step "9/10  Locate llama-server"
LLAMA_SERVER=$(find "$HOME/.lmstudio/extensions/backends" -name "llama-server" -executable -type f 2>/dev/null | head -1)
if [ -z "$LLAMA_SERVER" ]; then
  echo "ERROR: llama-server not found. Try: lms bootstrap"
  exit 1
fi
echo "    $LLAMA_SERVER"

# ---------------------------------------------------------------------------
# 10. Free the port and exec llama-server with YaRN + Q8 KV + 1M context
# ---------------------------------------------------------------------------
step "10/10 Launch llama-server on :$PORT"
# Stop LM Studio's own daemon if it's holding the port, then kill any leftover
# llama-server from a previous run.
if have lms; then
  lms server stop 2>/dev/null || true
fi
EXISTING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "    killing PIDs on :$PORT — $EXISTING"
  kill -TERM $EXISTING 2>/dev/null || true
  sleep 2
  EXISTING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
  [ -n "$EXISTING" ] && kill -KILL $EXISTING 2>/dev/null || true
fi

echo
echo "    model:       $REAL_MODEL"
echo "    port:        $PORT"
echo "    context:     $TARGET_CONTEXT tokens (YaRN factor=$ROPE_SCALE on $YARN_ORIG_CTX-trained base)"
echo "    KV cache:    $KV_CACHE_TYPE (k and v)"
echo "    GPU offload: all layers"
echo
echo "    OpenAI-compatible endpoint: http://0.0.0.0:$PORT/v1"
echo "    Wait for 'all slots are idle' before pointing HasH AI at it."
echo

exec "$LLAMA_SERVER" \
  --model "$REAL_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$TARGET_CONTEXT" \
  --rope-scaling yarn \
  --rope-scale "$ROPE_SCALE" \
  --rope-freq-base "$ROPE_FREQ_BASE" \
  --yarn-orig-ctx "$YARN_ORIG_CTX" \
  --cache-type-k "$KV_CACHE_TYPE" \
  --cache-type-v "$KV_CACHE_TYPE" \
  --metrics \
  --no-warmup
