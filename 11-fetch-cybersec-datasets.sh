#!/usr/bin/env bash
# Step 11: pulls cybersecurity training data into ./data.
#  - Hugging Face: Canstralian/pentesting_dataset (instruction-tuning ready)
#  - GitHub catalog: gfek/Real-CyberSecurity-Datasets (index of public datasets)
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR="$SCRIPT_DIR/data"
CATALOG_DIR="$DATA_DIR/cybersec-catalog"

mkdir -p "$DATA_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./04-install-pytorch.sh first."
    exit 1
fi

echo "==> Cloning gfek/Real-CyberSecurity-Datasets catalog (links to public datasets)..."
if [[ -d "$CATALOG_DIR/.git" ]]; then
    git -C "$CATALOG_DIR" pull --ff-only
else
    git clone --depth 1 https://github.com/gfek/Real-CyberSecurity-Datasets.git "$CATALOG_DIR"
fi
echo "    Browse: $CATALOG_DIR/README.md"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Pulling Canstralian/pentesting_dataset from Hugging Face..."
python "$SCRIPT_DIR/fetch_hf_dataset.py" \
    Canstralian/pentesting_dataset \
    --out "$DATA_DIR/pentesting_dataset.jsonl" \
    --split train

deactivate
echo "==> Cybersec datasets ready in $DATA_DIR"
