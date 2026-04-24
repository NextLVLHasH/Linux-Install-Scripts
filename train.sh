#!/usr/bin/env bash
# Smart trainer launcher. Picks single-process vs `accelerate launch` based on
# --num-gpus, generates an accelerate config on the fly, and forwards the rest
# of the args to finetune.py.
#
# GPU selection (heterogeneous rigs):
#   --gpu-ids 0,1,2,3        explicitly use only these CUDA indices
#   --exclude-smallest       drop GPUs whose VRAM is far from the median
#                            (useful when mixing e.g. H100 + A100-40GB)
#
# Usage examples:
#   ./train.sh --auto-tune --model ./models/X --data ./data/d.jsonl --output ./runs/r1
#   ./train.sh --num-gpus 4 --strategy ddp --auto-tune --model ... --data ... --output ...
#   ./train.sh --gpu-ids 0,1,2,3,4,5,6,7 --strategy zero3 --auto-tune \
#              --model 70B --data ... --output ...
#   ./train.sh --exclude-smallest --strategy ddp --auto-tune \
#              --model ... --data ... --output ...
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
NUM_GPUS=""
GPU_IDS=""
EXCLUDE_SMALLEST=0
STRATEGY="auto"
MIXED_PRECISION="bf16"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)          GPU_IDS="$2"; shift 2 ;;
        --exclude-smallest) EXCLUDE_SMALLEST=1; shift ;;
        --strategy)         STRATEGY="$2"; shift 2 ;;
        --mixed-precision)  MIXED_PRECISION="$2"; shift 2 ;;
        *)                  PASSTHRU+=("$1"); shift ;;
    esac
done

if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# 1) Resolve the GPU set we'll actually use.
if [[ -z "$GPU_IDS" && "$EXCLUDE_SMALLEST" -eq 1 ]]; then
    GPU_IDS=$(python "$SCRIPT_DIR/gpu_profile.py" detect \
              | python -c 'import sys,json; d=json.load(sys.stdin); print(",".join(str(i) for i in d.get("majority_subset_indices", [])))')
fi

if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    # When CUDA_VISIBLE_DEVICES is set, num_gpus = number of entries in that list.
    NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')
fi

# 2) Detect total visible GPUs if --num-gpus wasn't given.
if [[ -z "$NUM_GPUS" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    else
        NUM_GPUS=0
    fi
fi

if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
if [[ "$NUM_GPUS" -gt 9 ]]; then
    echo "WARN: --num-gpus $NUM_GPUS is above this rig's tested max of 9. Continuing."
fi

if [[ "$STRATEGY" == "auto" ]]; then
    if [[ "$NUM_GPUS" -le 1 ]]; then STRATEGY="single"; else STRATEGY="ddp"; fi
fi

echo "==> train.sh: num_gpus=$NUM_GPUS strategy=$STRATEGY mixed_precision=$MIXED_PRECISION"
[[ -n "$GPU_IDS" ]] && echo "    CUDA_VISIBLE_DEVICES=$GPU_IDS"

if [[ "$STRATEGY" == "single" || "$NUM_GPUS" -eq 1 ]]; then
    exec python "$SCRIPT_DIR/finetune.py" "${PASSTHRU[@]}"
fi

# Multi-GPU: render an accelerate config to a temp file, then launch.
TMP_CFG=$(mktemp -t accelerate_cfg_XXXX.yaml)
trap 'rm -f "$TMP_CFG"' EXIT
python "$SCRIPT_DIR/gpu_profile.py" accelerate \
    --num-gpus "$NUM_GPUS" --strategy "$STRATEGY" --mixed-precision "$MIXED_PRECISION" \
    > "$TMP_CFG"

echo "==> Generated accelerate config:"
cat "$TMP_CFG"

exec accelerate launch \
    --config_file "$TMP_CFG" \
    --num_processes "$NUM_GPUS" \
    "$SCRIPT_DIR/finetune.py" "${PASSTHRU[@]}"
