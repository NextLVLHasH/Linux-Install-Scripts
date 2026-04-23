#!/usr/bin/env bash
# Step 3a: create an empty Python venv so downstream install steps that need
# it (dashboard, cybersec dataset fetch) can run before PyTorch is installed.
# 04-install-pytorch.sh later populates this same venv with torch + friends.
#
# Runs as the invoking user — not via sudo — so the venv lives under the
# correct $HOME and is owned by that user.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pytorch-env}"

if [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "==> venv already exists at $VENV_DIR — nothing to do."
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Run ./02-install-prerequisites.sh first."
    exit 1
fi

echo "==> Creating empty Python venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip setuptools wheel
deactivate

echo "==> Empty venv ready at $VENV_DIR"
