#!/usr/bin/env bash
# Step 7: download a Hugging Face model for training.
# Usage: ./07-download-model.sh <repo_id> [--revision X] [--token HF_TOKEN]
# Example: ./07-download-model.sh Qwen/Qwen2.5-0.5B
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf_repo_id> [extra args to download_model.py]"
    echo "Example: $0 Qwen/Qwen2.5-0.5B"
    exit 1
fi

VENV_DIR="${VENV_DIR:-/workspace/venv}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python "$SCRIPT_DIR/download_model.py" "$@"
deactivate
