#!/bin/bash
# Build/LMStudioServer/installstart.sh — Pod-side installer + launcher.
#
# Downloads the YaRN-capable llama-server release from NextLVLHasH/AgentsRemoteBuild,
# extracts it to $PERSIST/llama.cpp-prebuilt/, fetches the GGUF model you point
# at via HF_URL, and launches llama-server with full YaRN + Q8 KV at 1M context.
#
# No LM Studio install, no source build — just download + run. The release is
# pinned by default to a known-good tag (RELEASE_URL below); set RELEASE_URL to
# override, or set USE_LATEST=1 to auto-discover the newest release.
#
# Usage on a RunPod pod (inside tmux):
#   chmod +x installstart.sh
#   HF_URL="https://huggingface.co/<owner>/<repo>?show_file_info=<file>.gguf" ./installstart.sh
#
# Environment overrides:
#   RELEASE_URL     — exact tarball URL. Default: pinned yarn-45b455e build.
#   USE_LATEST      — set to 1 to query GitHub for the latest release URL instead.
#   PERSIST         — persistent drive root. Default: /workspace
#   LLAMA_PORT      — endpoint port. Default: 1234
#   ROPE_*/YARN_*   — YaRN params (defaults tuned for Qwen3.6).
#   KV_CACHE_TYPE   — k/v cache quant. Default: q8_0.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERSIST="${PERSIST:-/workspace}"
PORT="${LLAMA_PORT:-1234}"

# The release tarball this pod will pull. Default uses GitHub's /releases/latest/
# alias so any future build-and-publish run becomes the new default without
# editing this script — as long as the published filename stays the same
# (`lm-link-yarn-<llama-sha>-linux-x64-cuda12.tar.gz`). Override RELEASE_URL
# to pin a specific tag, or set USE_LATEST=1 to auto-discover the asset name
# via the GitHub API (handles filename changes between releases).
RELEASE_URL="${RELEASE_URL:-https://github.com/NextLVLHasH/AgentsRemoteBuild/releases/latest/download/lm-link-yarn-45b455e-linux-x64-cuda12.tar.gz}"

# YaRN rope-scaling params. Defaults extend Qwen3.6's 262k native context to ~1M.
ROPE_FREQ_BASE="${ROPE_FREQ_BASE:-10000000}"
ROPE_SCALE="${ROPE_SCALE:-4}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-262144}"
TARGET_CONTEXT="${TARGET_CONTEXT:-1010000}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"

step() { echo; echo "==> $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Auto-create PERSIST if it doesn't exist. The previous strict check (error +
# exit) protected against typos where PERSIST pointed at a non-existent mount
# and would have created a wrong-place dir under /. In practice the friction
# of asking users to `mkdir -p` before every run on a pod without /workspace
# was worse than the typo risk. mkdir is bounded; if PERSIST is unwritable
# the mkdir itself fails and the script exits cleanly with a real error.
if [ ! -d "$PERSIST" ]; then
  if ! mkdir -p "$PERSIST" 2>/dev/null; then
    echo "ERROR: \$PERSIST=$PERSIST could not be created. Pick a writable path:"
    echo "       e.g. /workspace (RunPod template volume), /runpod-volume,"
    echo "       /persistent, or /root/lm on pods without a network volume."
    exit 1
  fi
  echo "    created $PERSIST (it didn't exist yet)"
fi

# ---------------------------------------------------------------------------
# 0. Resolve the Hugging Face model URL → REPO + FILE
# ---------------------------------------------------------------------------
step "0/7  Hugging Face model"
HF_INPUT="${HF_URL:-${1:-}}"
if [ -z "$HF_INPUT" ]; then
  echo "    Paste the HuggingFace URL (with ?show_file_info=… or /blob/main/…)"
  printf "    HF URL: "
  read -r HF_INPUT
fi
[ -z "$HF_INPUT" ] && { echo "ERROR: no URL provided"; exit 1; }

PARSE="${HF_INPUT#https://}"
PARSE="${PARSE#http://}"
PARSE="${PARSE#huggingface.co/}"
PARSE="${PARSE#hf.co/}"

MODEL_FILE=""
case "$PARSE" in
  *\?show_file_info=*)
    MODEL_REPO="${PARSE%%\?*}"
    MODEL_FILE="${PARSE#*\?show_file_info=}"
    MODEL_FILE="${MODEL_FILE%%&*}"
    ;;
  */blob/*/*|*/resolve/*/*)
    OWNER_REPO="${PARSE%%/blob/*}"
    OWNER_REPO="${OWNER_REPO%%/resolve/*}"
    MODEL_REPO="$OWNER_REPO"
    REST="${PARSE#"$OWNER_REPO/"}"
    REST="${REST#blob/}"
    REST="${REST#resolve/}"
    MODEL_FILE="${REST#*/}"
    ;;
  *)
    MODEL_REPO="$PARSE"
    MODEL_FILE=""
    ;;
esac
MODEL_REPO="${MODEL_REPO%/}"
MODEL_REPO="${MODEL_REPO%%\#*}"
MODEL_FILE="${MODEL_FILE%%\#*}"

[[ "$MODEL_REPO" == */* ]] || { echo "ERROR: couldn't parse OWNER/REPO from '$HF_INPUT'"; exit 1; }

SLUG=$(echo "$MODEL_REPO" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9.-' | cut -c1-80)
MODEL_DIR="$PERSIST/models/$SLUG"

echo "    repo: $MODEL_REPO"
echo "    file: ${MODEL_FILE:-<whole repo>}"
echo "    dest: $MODEL_DIR"

# ---------------------------------------------------------------------------
# 1. Persistent dirs
# ---------------------------------------------------------------------------
step "1/7  Persistent directories on $PERSIST"
mkdir -p \
  "$PERSIST/.cache/huggingface" \
  "$PERSIST/.cache/pip" \
  "$PERSIST/.lmstudio" \
  "$PERSIST/.cache/lm-studio" \
  "$PERSIST/models" \
  "$PERSIST/venv-parent" \
  "$PERSIST/bin" \
  "$PERSIST/llama.cpp-prebuilt"

# Symlink HF cache home → persistent storage so login + blobs survive restart.
# Also symlinks both possible LM Studio install roots (~/.lmstudio for the
# older layout, ~/.cache/lm-studio for the newer "llmster" rewrite) so the lms
# installer writes straight to the persistent drive regardless of version.
relink() {
  local target="$1" link="$2"
  [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ] && return
  # Make sure the parent dir of the link exists — fresh pods may not yet have
  # /root/.cache, so `ln -s /workspace/.cache/huggingface /root/.cache/huggingface`
  # fails with "No such file or directory" without this.
  mkdir -p "$(dirname "$link")"
  if [ -d "$link" ] && [ ! -L "$link" ]; then
    cp -an "$link"/. "$target"/ 2>/dev/null || true
    rm -rf "$link"
  fi
  rm -f "$link"
  ln -s "$target" "$link"
}
relink "$PERSIST/.cache/huggingface" "$HOME/.cache/huggingface"
relink "$PERSIST/.cache/pip"         "$HOME/.cache/pip"
relink "$PERSIST/.lmstudio"          "$HOME/.lmstudio"
relink "$PERSIST/.cache/lm-studio"   "$HOME/.cache/lm-studio"

# Persistent shell init (PATH, HF_HOME, LD_LIBRARY_PATH for the prebuilt).
# PATH covers both LM Studio install roots ($HOME/.lmstudio/bin for the older
# layout, $HOME/.cache/lm-studio/bin for the newer "llmster" layout) plus our
# prebuilt llama-server, all sourced from /workspace via the symlinks above.
cat > "$PERSIST/init.sh" <<EOF
export PATH="\$HOME/.lmstudio/bin:\$HOME/.cache/lm-studio/bin:$PERSIST/llama.cpp-prebuilt/bin:\$PATH"
export LD_LIBRARY_PATH="$PERSIST/llama.cpp-prebuilt/lib:\${LD_LIBRARY_PATH:-}"
export HF_HOME="$PERSIST/.cache/huggingface"
export PIP_CACHE_DIR="$PERSIST/.cache/pip"
if [ -f "$PERSIST/venv-parent/llama/bin/activate" ]; then
  source "$PERSIST/venv-parent/llama/bin/activate"
fi
EOF
grep -q "source $PERSIST/init.sh" "$HOME/.bashrc" 2>/dev/null || echo "source $PERSIST/init.sh" >> "$HOME/.bashrc"
source "$PERSIST/init.sh"

# ---------------------------------------------------------------------------
# 2. LM Studio CLI — provides `lms login`, `lms link`, `lms ls`, and the
#    model catalog UI. Skipped if already installed. Auto-detects which
#    install layout (~/.lmstudio vs ~/.cache/lm-studio) actually got files.
# ---------------------------------------------------------------------------
step "2/7  LM Studio CLI (lms)"
if have lms; then
  echo "    already installed: $(lms version 2>/dev/null | head -1) (at $(command -v lms))"
elif [ -x "$HOME/.lmstudio/bin/lms" ] || [ -x "$HOME/.cache/lm-studio/bin/lms" ]; then
  echo "    binary present on disk; PATH refreshed from init.sh"
else
  echo "    fetching LM Studio installer"
  curl -fsSL https://lmstudio.ai/install.sh | bash
fi
# Force both candidate bin dirs onto PATH for THIS shell so the rest of the
# script can use `lms` without waiting for a new login.
export PATH="$HOME/.lmstudio/bin:$HOME/.cache/lm-studio/bin:$PATH"

if have lms; then
  echo "    lms ready: $(command -v lms)"
  # Detect which install layout is in use so later steps know where the
  # models/ tree lives.
  if [ -x "$HOME/.lmstudio/bin/lms" ]; then
    LM_HOME="$HOME/.lmstudio"
  elif [ -x "$HOME/.cache/lm-studio/bin/lms" ]; then
    LM_HOME="$HOME/.cache/lm-studio"
  else
    LM_HOME=""
  fi
  [ -n "$LM_HOME" ] && echo "    LM_HOME: $LM_HOME"

  # Interactive login. Skipped on non-TTY runs (e.g. `nohup … &`) or when
  # SKIP_LMS_LOGIN=1. Otherwise we run `lms login` inline — it prints a
  # 3-word pairing code and blocks until you enter it at
  # https://lmstudio.ai/pairing, then returns. Existing tokens persist
  # at $LM_HOME/credentials/ which is symlinked to /workspace, so this
  # only fires on a truly fresh setup.
  LOGGED_IN=0
  if [ -d "$LM_HOME/credentials" ] && find "$LM_HOME/credentials" -name "*.json" -type f 2>/dev/null | grep -q .; then
    LOGGED_IN=1
  fi

  if [ "$LOGGED_IN" = "1" ]; then
    echo "    already paired with LM Studio cloud"
  elif [ "${SKIP_LMS_LOGIN:-0}" = "1" ]; then
    echo "    SKIP_LMS_LOGIN=1 — not pairing. To pair later, run: lms login"
  elif [ -t 0 ] && [ -t 1 ]; then
    echo
    echo "    Pairing this pod with your LM Studio cloud account."
    echo "    Press Enter to start (or Ctrl-C, then re-run with SKIP_LMS_LOGIN=1)."
    read -r _
    lms login || echo "    lms login returned non-zero; continuing anyway"
  else
    echo
    echo "    Non-interactive run — skipping lms login. To pair later:"
    echo "        lms login          # prints a 3-word code"
    echo "        # paste it at:    https://lmstudio.ai/pairing"
  fi
else
  echo "    WARNING: lms still not on PATH after install — check the installer output above"
  LM_HOME=""
fi

# ---------------------------------------------------------------------------
# 3. Python venv + huggingface_hub
# ---------------------------------------------------------------------------
step "3/7  Python venv + huggingface_hub"
if [ ! -f "$PERSIST/venv-parent/llama/bin/activate" ]; then
  python3 -m venv "$PERSIST/venv-parent/llama"
fi
source "$PERSIST/venv-parent/llama/bin/activate"
pip show huggingface_hub >/dev/null 2>&1 || pip install -q -U huggingface_hub
echo "    hf: $(pip show huggingface_hub | grep -i ^version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 3. Download + extract the YaRN llama-server release. Skipped if already
#    present on disk (cached on /workspace across pod restarts).
# ---------------------------------------------------------------------------
step "4/7  llama-server release"
PREBUILT_DIR="$PERSIST/llama.cpp-prebuilt"
LLAMA_SERVER="$PREBUILT_DIR/bin/llama-server"
if [ "${USE_LATEST:-0}" = "1" ]; then
  echo "    USE_LATEST=1 — querying GitHub for latest"
  LATEST=$(curl -fsSL https://api.github.com/repos/NextLVLHasH/AgentsRemoteBuild/releases/latest \
    | grep -oE '"browser_download_url"\s*:\s*"[^"]+"' \
    | sed -E 's/.*"(https[^"]+)".*/\1/' \
    | grep -E 'lm-link-yarn-.*-linux-x64-cuda12\.tar\.gz$' \
    | head -1)
  [ -n "$LATEST" ] && RELEASE_URL="$LATEST"
fi
RELEASE_NAME=$(basename "$RELEASE_URL")
EXTRACTED_DIR="${RELEASE_NAME%.tar.gz}"

if [ -x "$LLAMA_SERVER" ] && [ -d "$PREBUILT_DIR/lib" ]; then
  echo "    already extracted: $LLAMA_SERVER"
else
  echo "    pulling: $RELEASE_URL"
  cd "$PREBUILT_DIR"
  curl -fL --progress-bar -o release.tgz "$RELEASE_URL"
  tar -xzf release.tgz
  # Verify a llama-server is inside; symlink bin/ and lib/ at a stable path so
  # PATH + LD_LIBRARY_PATH from init.sh resolve without knowing the version.
  EXTRACTED=$(find . -maxdepth 2 -type d -name "lm-link-yarn-*linux-x64-cuda12*" | head -1)
  [ -n "$EXTRACTED" ] || { echo "ERROR: no lm-link-yarn-* dir in tarball"; exit 1; }
  ln -sfn "$PREBUILT_DIR/$(basename "$EXTRACTED")/bin" "$PREBUILT_DIR/bin"
  ln -sfn "$PREBUILT_DIR/$(basename "$EXTRACTED")/lib" "$PREBUILT_DIR/lib"
  chmod +x "$LLAMA_SERVER"
  rm -f release.tgz
  cd - >/dev/null
  echo "    installed: $LLAMA_SERVER"
fi

# Force LD path for THIS shell (init.sh did it, but the symlink may have been
# new since init.sh ran)
export PATH="$PREBUILT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$PREBUILT_DIR/lib:${LD_LIBRARY_PATH:-}"

# Quick sanity check the binary actually runs with this CUDA driver
if ! "$LLAMA_SERVER" --version >/dev/null 2>&1; then
  echo "WARNING: llama-server failed --version — likely a CUDA driver mismatch."
  echo "         Confirm nvidia-smi works on this pod and the driver supports CUDA 12."
fi

# ---------------------------------------------------------------------------
# 4.5  GPU arch compatibility check + auto-rebuild.
#
#      The prebuilt llama-server includes CUDA kernels compiled for a fixed
#      set of architectures (sm_75 through sm_90 by default in
#      build-release.sh). If this pod's GPU is newer than what's baked in —
#      most importantly Blackwell (sm_120 on RTX Pro 6000 Blackwell, sm_100
#      on B100/B200) — every CUDA kernel dispatch dies with
#      "no kernel image is available for execution on the device".
#
#      Detection:
#        1. Read GPU compute_cap from nvidia-smi → "12.0" → sm_120.
#        2. List the binary's embedded archs via cuobjdump.
#        3. If GPU's sm_* isn't in the list, rebuild from source on this pod
#           with -DCMAKE_CUDA_ARCHITECTURES=native so kernels match exactly.
# ---------------------------------------------------------------------------
step "4.5/7  GPU arch compatibility check"
if ! have nvidia-smi; then
  echo "    no nvidia-smi — skipping (CPU-only setup?)"
else
  GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  GPU_SM=$(echo "$GPU_COMPUTE_CAP" | awk -F. '{printf "%d%d", $1, $2}')
  echo "    GPU compute capability: $GPU_COMPUTE_CAP (sm_$GPU_SM)"

  NEEDS_REBUILD=0
  CUOBJDUMP=""
  for cand in cuobjdump /usr/local/cuda-12.8/bin/cuobjdump /usr/local/cuda*/bin/cuobjdump; do
    [ -x "$cand" ] && { CUOBJDUMP="$cand"; break; }
  done
  if [ -n "$CUOBJDUMP" ]; then
    # cuobjdump --list-elf prints lines like "ELF file 1: <name>.sm_86.cubin".
    # Grep the binary itself + the cuda shared lib (most kernels live in libggml-cuda.so).
    ARCHS=$("$CUOBJDUMP" --list-elf "$PREBUILT_DIR/lib/libggml-cuda.so" 2>/dev/null \
      | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    if [ -z "$ARCHS" ]; then
      ARCHS=$("$CUOBJDUMP" --list-elf "$LLAMA_SERVER" 2>/dev/null \
        | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    fi
    echo "    prebuilt archs: ${ARCHS:-<none found>}"
    if [ -n "$ARCHS" ] && ! echo " $ARCHS " | grep -q " sm_$GPU_SM "; then
      echo "    MISMATCH: prebuilt has no sm_$GPU_SM kernels for this GPU"
      NEEDS_REBUILD=1
    fi
  else
    # Fallback: pessimistic detection on Blackwell, since that's the most
    # common case where prebuilts trip. sm major >= 10 = Blackwell or newer.
    GPU_MAJOR=$(echo "$GPU_COMPUTE_CAP" | awk -F. '{print $1}')
    if [ "${GPU_MAJOR:-0}" -ge 10 ]; then
      echo "    cuobjdump not available; assuming Blackwell-class GPU needs rebuild"
      NEEDS_REBUILD=1
    else
      echo "    cuobjdump not available; trusting prebuilt"
    fi
  fi

  if [ "$NEEDS_REBUILD" = "1" ]; then
    step "4.6/7  Auto-rebuild llama.cpp for sm_$GPU_SM"
    # Find nvcc — could be on PATH already, or under /usr/local/cuda*/bin
    if ! have nvcc; then
      NVCC_FOUND=""
      for cand in /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda-12/bin/nvcc /usr/local/cuda/bin/nvcc; do
        [ -x "$cand" ] && { NVCC_FOUND="$cand"; export PATH="$(dirname "$cand"):$PATH"; break; }
      done
      [ -z "$NVCC_FOUND" ] && for cand in /usr/local/cuda*/bin/nvcc; do
        [ -x "$cand" ] && { export PATH="$(dirname "$cand"):$PATH"; break; }
      done
    fi
    if ! have nvcc; then
      echo "ERROR: nvcc not found — cannot rebuild. Install CUDA toolkit:"
      echo "       apt-get install -y nvidia-cuda-toolkit"
      exit 1
    fi
    # Make sure build deps are installed
    for pkg in git cmake build-essential; do
      dpkg -s "$pkg" >/dev/null 2>&1 || apt-get install -y "$pkg" 2>&1 | tail -1
    done
    REBUILD_SRC="$PERSIST/llama-build-src"
    if [ ! -d "$REBUILD_SRC/.git" ]; then
      git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$REBUILD_SRC"
    fi
    cd "$REBUILD_SRC"
    rm -rf build
    echo "    configuring with CMAKE_CUDA_ARCHITECTURES=$GPU_SM"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$GPU_SM" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_COMMON=ON \
      >/dev/null
    echo "    compiling (-j $(nproc))"
    cmake --build build --config Release --target llama-server --target llama -j "$(nproc)"
    cd - >/dev/null

    # Swap the broken prebuilt files for the freshly-built ones. Preserve the
    # SONAME chain via cp -a.
    echo "    swapping prebuilt with local build"
    cp -af "$REBUILD_SRC/build/bin/llama-server" "$PREBUILT_DIR/bin/llama-server"
    rm -f "$PREBUILT_DIR/lib"/*.so*
    for pat in 'libllama.so*' 'libggml*.so*' 'libmtmd.so*' 'libllama-common.so*'; do
      while IFS= read -r src; do
        [ -e "$src" ] && cp -a "$src" "$PREBUILT_DIR/lib/"
      done < <(find "$REBUILD_SRC/build" -maxdepth 4 -name "$pat" 2>/dev/null)
    done

    # Re-verify
    if LD_LIBRARY_PATH="$PREBUILT_DIR/lib" "$LLAMA_SERVER" --version >/dev/null 2>&1; then
      echo "    rebuild succeeded; llama-server --version OK"
    else
      echo "    WARNING: --version still fails after rebuild — check $REBUILD_SRC/build for errors"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 4. Model download
# ---------------------------------------------------------------------------
step "5/7  Model — ${MODEL_FILE:-<repo: $MODEL_REPO>}"
mkdir -p "$MODEL_DIR"
if [ -n "$MODEL_FILE" ]; then
  EXPECTED_PATH="$MODEL_DIR/$MODEL_FILE"
  if [ -f "$EXPECTED_PATH" ] || [ -L "$EXPECTED_PATH" ]; then
    echo "    already at $EXPECTED_PATH ($(du -h "$EXPECTED_PATH" | awk '{print $1}'))"
  else
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  fi
else
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  mapfile -t GGUFS < <(find "$MODEL_DIR" -maxdepth 4 -name "*.gguf" -not -name "mmproj*" | sort)
  [ "${#GGUFS[@]}" -gt 0 ] || { echo "ERROR: no .gguf in $MODEL_DIR"; exit 1; }
  if [ "${#GGUFS[@]}" -eq 1 ]; then
    EXPECTED_PATH="${GGUFS[0]}"
  else
    echo "    multiple GGUFs — pick one:"
    i=0; for g in "${GGUFS[@]}"; do i=$((i+1)); echo "      $i) $(basename "$g")"; done
    printf "    Number: "; read -r CHOICE
    EXPECTED_PATH="${GGUFS[$((CHOICE-1))]}"
  fi
  MODEL_FILE="$(basename "$EXPECTED_PATH")"
fi
REAL_MODEL=$(readlink -f "$EXPECTED_PATH")
echo "    resolved: $REAL_MODEL ($(ls -lh "$REAL_MODEL" | awk '{print $5}'))"

# Register the model with LM Studio's catalog so `lms ls` / `lms load` see it
# without you running `lms import` manually (which is interactive and creates
# a broken relative symlink — we use an absolute symlink instead so the
# pointer always resolves regardless of where /workspace gets mounted).
#
# Drops the .gguf at <LM_HOME>/models/<creator>/<repo-name>/<file>.gguf, which
# is the layout LM Studio's scanner expects. We do this for whichever LM
# Studio home is in use, AND for both layouts if both exist (the older
# ~/.lmstudio and the newer ~/.cache/lm-studio) — cheap and idempotent.
LMS_CREATOR="${MODEL_REPO%/*}"
LMS_NAME="${MODEL_REPO#*/}"
LMS_REGISTERED=0
for root in "$HOME/.lmstudio" "$HOME/.cache/lm-studio"; do
  [ -d "$root" ] || continue
  target_dir="$root/models/$LMS_CREATOR/$LMS_NAME"
  target_link="$target_dir/$(basename "$MODEL_FILE")"
  mkdir -p "$target_dir"
  rm -f "$target_link"
  ln -s "$REAL_MODEL" "$target_link"
  echo "    registered: $target_link"
  LMS_REGISTERED=1
done

# Restart the lms daemon so the catalog rescan picks up the new entry.
# Failures here are non-fatal — llama-server can still use $REAL_MODEL
# directly without lms knowing about the model.
if [ "$LMS_REGISTERED" = "1" ] && have lms; then
  lms server stop >/dev/null 2>&1 || true
  lms server start >/dev/null 2>&1 || true
  echo "    lms server restarted; verify with: lms ls"
fi

# ---------------------------------------------------------------------------
# 5. Free port + kill stale launchers
# ---------------------------------------------------------------------------
step "6/7  Free port $PORT"
have lms && lms server stop 2>/dev/null || true
EXISTING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "    killing PIDs on :$PORT — $EXISTING"
  kill -TERM $EXISTING 2>/dev/null || true
  sleep 2
  EXISTING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
  [ -n "$EXISTING" ] && kill -KILL $EXISTING 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 6. Launch llama-server with YaRN + Q8 KV. If the YaRN attempt errors out,
#    retry at native context as fallback.
# ---------------------------------------------------------------------------
step "7/7  Launch llama-server on :$PORT"
echo "    model:    $REAL_MODEL"
echo "    context:  $TARGET_CONTEXT tokens (YaRN factor=$ROPE_SCALE from $YARN_ORIG_CTX-trained)"
echo "    KV cache: $KV_CACHE_TYPE"
echo "    endpoint: http://0.0.0.0:$PORT/v1"
echo

ROPE_FREQ_SCALE=$(awk -v n="$ROPE_SCALE" 'BEGIN{printf "%.6f", 1.0/n}')

# llama-server defaults to 4 parallel slots and clamps each slot's context to
# the model's training context (n_ctx_train, 262144 for Qwen3.6). With 1M
# total split 4 ways that gives 4 concurrent conversations at 256k each —
# fine for serving, useless if you want ONE long-context chat.
#
# Default to --parallel 1 so the full TARGET_CONTEXT goes to a single slot.
# Override PARALLEL=4 (or whatever) if you actually want concurrent slots.
PARALLEL="${PARALLEL:-1}"

echo "    parallel slots: $PARALLEL (each slot up to $TARGET_CONTEXT tokens)"
echo

if "$LLAMA_SERVER" \
    --model "$REAL_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --n-gpu-layers 999 \
    --ctx-size "$TARGET_CONTEXT" \
    --parallel "$PARALLEL" \
    --rope-scaling yarn \
    --rope-scale "$ROPE_SCALE" \
    --rope-freq-scale "$ROPE_FREQ_SCALE" \
    --rope-freq-base "$ROPE_FREQ_BASE" \
    --yarn-orig-ctx "$YARN_ORIG_CTX" \
    --cache-type-k "$KV_CACHE_TYPE" \
    --cache-type-v "$KV_CACHE_TYPE" \
    --metrics --no-warmup --verbose; then
  exit 0
fi

echo
echo "==> YaRN attempt exited non-zero. Retrying at native $YARN_ORIG_CTX context."
echo
exec "$LLAMA_SERVER" \
  --model "$REAL_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$YARN_ORIG_CTX" \
  --cache-type-k "$KV_CACHE_TYPE" \
  --cache-type-v "$KV_CACHE_TYPE" \
  --metrics --no-warmup
