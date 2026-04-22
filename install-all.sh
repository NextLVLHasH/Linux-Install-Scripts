#!/usr/bin/env bash
# Master installer. Runs every step in order.
# Skip individual steps with env vars, e.g.:
#   SKIP_NVIDIA=1 SKIP_SYSTEMD=1 ./install-all.sh
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

chmod +x ./*.sh ./train.sh 2>/dev/null || true

run() {
    local skip_var="$1"
    local script="$2"
    if [[ -n "${!skip_var:-}" ]]; then
        echo "==> Skipping $script ($skip_var set)"
        return
    fi
    echo ""
    echo "============================================="
    echo " Running $script"
    echo "============================================="
    "./$script"
}

run SKIP_UPDATE        01-update-system.sh
run SKIP_PREREQS       02-install-prerequisites.sh
run SKIP_NVIDIA        03-install-nvidia-cuda.sh
run SKIP_PYTORCH       04-install-pytorch.sh
run SKIP_LMSTUDIO      05-install-lmstudio.sh
run SKIP_TRAINING      06-install-training-deps.sh
run SKIP_DASHBOARD     08-install-dashboard.sh
run SKIP_CYBERSEC      11-fetch-cybersec-datasets.sh

if [[ -z "${SKIP_SYSTEMD:-}" ]]; then
    run SKIP_SYSTEMD   10-install-systemd.sh
fi

echo ""
echo "================================================================"
echo " Install complete."
echo " Dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8765"
echo " Manual launch: ./09-start-dashboard.sh"
echo " Manual model fetch: ./07-download-model.sh <hf-repo-id>"
echo " For ZimaOS / CasaOS: see README, or ./install-casaos.sh"
echo "================================================================"
