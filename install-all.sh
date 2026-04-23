#!/usr/bin/env bash
# Master installer — animated progress, background steps, auto-resume after reboot.
#
# First run:  ./install-all.sh
# After reboot the systemd service (ml-stack-resume) resumes automatically.
#
# Skip individual steps manually:
#   SKIP_NVIDIA=1 SKIP_SYSTEMD=1 ./install-all.sh
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
chmod +x ./*.sh ./train.sh ./start-lmstudio.sh 2>/dev/null || true

STATE_DIR="/var/lib/ml-stack-install"
RESUME_FILE="$STATE_DIR/resume"
REBOOT_MARKER="$STATE_DIR/.needs-reboot"
SERVICE_NAME="ml-stack-resume"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"
CONFIG_FILE="$STATE_DIR/config.env"

# Prime sudo up front so that step scripts which call `sudo` internally never
# trigger a password prompt mid-install (a prompt writes to /dev/tty, which
# bypasses our log redirect and corrupts the TUI). Keep the ticket alive in
# the background for the full run.
echo "Enter sudo password if prompted — required once for the whole install."
sudo -v
( while kill -0 $$ 2>/dev/null; do sudo -n -v 2>/dev/null || true; sleep 50; done ) &
SUDO_KEEPALIVE_PID=$!

sudo mkdir -p "$STATE_DIR" "$LOG_DIR"

# Resolve the *real* user's home even if the installer was launched with sudo,
# so paths like ~/pytorch-env don't accidentally land under /root.
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)
: "${REAL_HOME:=$HOME}"

# Export + persist the shared install paths so every child step and any later
# standalone script (e.g. 09-start-dashboard.sh, systemd-launched or not) sees
# the same values regardless of which user or context runs it.
export INSTALL_DIR="$SCRIPT_DIR"
export VENV_DIR="${VENV_DIR:-$REAL_HOME/pytorch-env}"
export LMSTUDIO_DIR="${LMSTUDIO_DIR:-$REAL_HOME/LMStudio}"

# Force apt/dpkg into fully non-interactive mode so no step can hang on a
# debconf prompt or a "keep local config?" dialog (which would crash out
# under </dev/null and cause steps to fail in ~1s — showing up as the bar
# bouncing 11→12→11 as each step flashes by failing).
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export APT_LISTCHANGES_FRONTEND=none

# Drop in an apt config so dpkg keeps existing config files automatically
# during upgrades (equivalent to --force-confdef --force-confold on every
# apt-get call). Written once, harmless to leave in place.
sudo tee /etc/apt/apt.conf.d/99-ml-stack-noninteractive >/dev/null <<'APTCONF'
Dpkg::Options {
    "--force-confdef";
    "--force-confold";
};
APT::Get::Assume-Yes "true";
APTCONF

sudo tee "$CONFIG_FILE" >/dev/null <<CONF
# ml-stack install config — auto-generated, safe to source
INSTALL_DIR=$INSTALL_DIR
VENV_DIR=$VENV_DIR
LMSTUDIO_DIR=$LMSTUDIO_DIR
REAL_USER=$REAL_USER
REAL_HOME=$REAL_HOME
CONF

# ── terminal setup ─────────────────────────────────────────────────────────
_tput() { command tput "$@" 2>/dev/null || true; }

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
    FANCY=true
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; DIM=''; RESET=''
    FANCY=false
fi

SPINNER=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)

# ── step registry ──────────────────────────────────────────────────────────
declare -a S_NAME S_SCRIPT S_SKIP S_STATUS

_add() { S_NAME+=("$1"); S_SCRIPT+=("$2"); S_SKIP+=("$3"); S_STATUS+=(pending); }

_add "System update"       "01-update-system.sh"           "SKIP_UPDATE"
_add "Prerequisites"       "02-install-prerequisites.sh"   "SKIP_PREREQS"
_add "NVIDIA / CUDA"       "03-install-nvidia-cuda.sh"     "SKIP_NVIDIA"   # index 2
_add "PyTorch"             "04-install-pytorch.sh"         "SKIP_PYTORCH"
_add "LM Studio"           "05-install-lmstudio.sh"        "SKIP_LMSTUDIO"
_add "Training deps"       "06-install-training-deps.sh"   "SKIP_TRAINING"
_add "Dashboard"           "08-install-dashboard.sh"       "SKIP_DASHBOARD"
_add "Cybersec datasets"   "11-fetch-cybersec-datasets.sh" "SKIP_CYBERSEC"
_add "Systemd service"     "10-install-systemd.sh"         "SKIP_SYSTEMD"

TOTAL=${#S_NAME[@]}
NVIDIA_IDX=2   # index of the NVIDIA step above

# ── cleanup ────────────────────────────────────────────────────────────────
_tput civis
_cleanup() {
    _tput cnorm
    [[ -n "${SUDO_KEEPALIVE_PID:-}" ]] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    printf '\n'
}
trap _cleanup EXIT INT TERM

# ── drawing ────────────────────────────────────────────────────────────────
# Permille-scaled bar: `filled` is 0..1000 per total slot (i.e. done*1000 plus
# a fractional contribution for the currently running step), `total` is the
# step count. This lets the bar creep forward while a step runs.
_bar() {
    local filled=$1 total=$2 w=${3:-40}
    local denom=$(( total * 1000 ))
    local n=$(( denom > 0 ? w * filled / denom : 0 ))
    (( n < 0 )) && n=0
    (( n > w )) && n=$w
    local s=''
    for (( i=0; i<n; i++ )); do s+='█'; done
    for (( i=n; i<w; i++ )); do s+='░'; done
    printf '%s' "$s"
}

# Fractional progress for the running step, in permille (0..999). Asymptotic
# so it never hits 100% — the step's real completion snaps it to 1000.
# K controls how fast the bar fills: at tick=K, fraction is ~50%.
_running_frac() {
    local elapsed=$1 K=${2:-300}
    (( elapsed < 0 )) && elapsed=0
    printf '%d' $(( 999 * elapsed / (elapsed + K) ))
}

_draw_header() {
    printf '\n'
    printf "${BOLD}${CYAN}  ╔══════════════════════════════════════════╗\n"
    printf             "  ║      Linux ML Stack  ─  Installer        ║\n"
    printf             "  ╚══════════════════════════════════════════╝${RESET}\n"
    printf '\n'
}
HEADER_ROWS=5

BODY_ROWS=$(( TOTAL + 3 ))
STEP_START_TICK=0
_draw_body() {
    local tick=${1:-0} done_count=0 running_idx=-1

    for i in "${!S_NAME[@]}"; do
        local sym label name="${S_NAME[$i]}"
        case "${S_STATUS[$i]}" in
            pending) sym="${DIM}○${RESET}";                                label="${DIM}${name}${RESET}" ;;
            running) sym="${CYAN}${SPINNER[$((tick % 10))]}${RESET}";      label="${CYAN}${name}…${RESET}"; running_idx=$i ;;
            done)    sym="${GREEN}✔${RESET}";                               label="${name}" ;;
            skipped) sym="${DIM}─${RESET}";                                label="${DIM}${name}  (already installed)${RESET}" ;;
            failed)  sym="${RED}✗${RESET}";                                label="${RED}${name}  (failed — see log)${RESET}" ;;
        esac
        printf "  %b  %b\033[K\n" "$sym" "$label"
    done

    for s in "${S_STATUS[@]}"; do
        [[ $s == done || $s == skipped ]] && (( done_count++ )) || true
    done

    # Permille progress: each completed step contributes 1000, the currently
    # running step contributes a time-based fraction (0..999).
    local filled=$(( done_count * 1000 ))
    if (( running_idx >= 0 )); then
        filled=$(( filled + $(_running_frac $(( tick - STEP_START_TICK ))) ))
    fi
    local pct=0
    (( TOTAL > 0 )) && pct=$(( filled / (TOTAL * 10) )) || true

    printf '\n'
    printf "  ${CYAN}[%s]${RESET}  %3d%%  (%d / %d)\033[K\n" \
        "$(_bar "$filled" "$TOTAL")" "$pct" "$done_count" "$TOTAL"
    printf '\n'
}

_move_up() { printf '\033[%dA' "$1"; }

# ── resume helpers ─────────────────────────────────────────────────────────

# Write SKIP_* vars for every completed step, plus metadata, to the resume file.
_save_state() {
    local next_idx=$1
    {
        echo "# ml-stack resume state — sourced by systemd EnvironmentFile"
        echo "INSTALL_DIR=$SCRIPT_DIR"
        echo "LOG_DIR=$LOG_DIR"
        for i in "${!S_NAME[@]}"; do
            if (( i < next_idx )); then
                echo "${S_SKIP[$i]}=1"
            fi
        done
    } | sudo tee "$RESUME_FILE" >/dev/null
}

# Create a one-shot systemd service that re-runs this script after reboot.
_install_resume_service() {
    sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<SERVICE
[Unit]
Description=ML Stack Install Resume (post-driver reboot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
EnvironmentFile=$RESUME_FILE
ExecStart=$SCRIPT_DIR/install-all.sh
ExecStartPost=/bin/bash -c 'systemctl disable ${SERVICE_NAME}.service; rm -f /etc/systemd/system/${SERVICE_NAME}.service; systemctl daemon-reload'
StandardOutput=journal
StandardError=journal
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SERVICE

    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
}

# ── step runners ───────────────────────────────────────────────────────────
FAILED=()
TICK=0

_log_path() {
    local i=$1 safe="${S_NAME[$i]//[^A-Za-z0-9._-]/_}"
    printf '%s/%s_%s.log' "$LOG_DIR" "$i" "$safe"
}

_marker_path() {
    printf '%s/done.%s' "$STATE_DIR" "${S_SCRIPT[$1]%.sh}"
}

# A step is considered already installed if its success marker exists,
# or if the caller explicitly set SKIP_<STEP>=1.
_should_skip() {
    local i=$1 skip_var="${S_SKIP[$i]}"
    [[ -n "${!skip_var:-}" ]] && return 0
    [[ -f "$(_marker_path "$i")" ]] && return 0
    return 1
}

_mark_done() {
    sudo touch "$(_marker_path "$1")" 2>/dev/null || true
}

_run_step() {
    local i=$1
    local log; log=$(_log_path "$i")

    if _should_skip "$i"; then
        S_STATUS[$i]=skipped
        _move_up "$BODY_ROWS"; _draw_body "$TICK"
        return 0
    fi

    S_STATUS[$i]=running
    STEP_START_TICK=$TICK
    _move_up "$BODY_ROWS"; _draw_body "$TICK"

    local xf; xf=$(mktemp)
    # </dev/null guarantees the step script can never block on a terminal read
    # (e.g. an expired sudo prompt or apt's "Do you want to continue?" if a
    # step forgot -y). sudo should already be primed by the keepalive.
    ( "./${S_SCRIPT[$i]}" </dev/null >"$log" 2>&1; echo $? >"$xf" ) &
    local bg=$!

    while kill -0 "$bg" 2>/dev/null; do
        (( TICK++ )) || true
        _move_up "$BODY_ROWS"
        _draw_body "$TICK"
        sleep 0.1
    done
    wait "$bg" 2>/dev/null || true

    local rc; rc=$(cat "$xf" 2>/dev/null || echo 1)
    rm -f "$xf"

    if [[ "$rc" == "0" ]]; then
        S_STATUS[$i]=done
        _mark_done "$i"
    else
        S_STATUS[$i]=failed
        FAILED+=("${S_NAME[$i]}  →  $log")
    fi

    _move_up "$BODY_ROWS"
    _draw_body "$TICK"
}

_plain_run() {
    local i=$1
    local log; log=$(_log_path "$i")
    if _should_skip "$i"; then
        echo "-- Skipping: ${S_NAME[$i]} (already installed)"
        S_STATUS[$i]=skipped; return 0
    fi
    echo "-- Running:  ${S_NAME[$i]} …"
    if "./${S_SCRIPT[$i]}" </dev/null >"$log" 2>&1; then
        S_STATUS[$i]=done;   echo "-- Done:     ${S_NAME[$i]}"
        _mark_done "$i"
    else
        S_STATUS[$i]=failed; echo "-- FAILED:   ${S_NAME[$i]}  (log: $log)"
        FAILED+=("${S_NAME[$i]}  →  $log")
    fi
}

# ── initial paint ──────────────────────────────────────────────────────────
if $FANCY; then
    clear
    _draw_header
    _draw_body 0
fi

# ── main loop ──────────────────────────────────────────────────────────────
for i in "${!S_NAME[@]}"; do

    if $FANCY; then _run_step "$i"; else _plain_run "$i"; fi

    # After the NVIDIA step: if new drivers were installed, resume after reboot
    if (( i == NVIDIA_IDX )) && [[ "${S_STATUS[$i]}" == "done" ]] \
        && [[ -f "$REBOOT_MARKER" ]]; then

        _save_state $(( i + 1 ))
        _install_resume_service
        sudo rm -f "$REBOOT_MARKER"

        _tput cnorm
        printf "\n${YELLOW}${BOLD}  NVIDIA drivers installed — reboot required.${RESET}\n"
        printf "  Remaining steps will resume automatically after reboot.\n"
        printf "  To follow progress after reboot:\n"
        printf "    journalctl -u %s -f\n\n" "$SERVICE_NAME"
        printf "  Rebooting in 10 seconds… (Ctrl+C to cancel)\n"
        sleep 10
        sudo reboot
    fi

done

# ── summary ────────────────────────────────────────────────────────────────
_tput cnorm
IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

if [[ ${#FAILED[@]} -eq 0 ]]; then
    # Clean up resume state now that install is complete
    sudo rm -f "$RESUME_FILE"

    printf "\n${GREEN}${BOLD}  ✔  All steps complete!${RESET}\n\n"
    printf "  Dashboard  →  ${CYAN}http://%s:8765${RESET}\n" "$IP"
    printf "  Logs       →  ${DIM}%s/${RESET}\n\n" "$LOG_DIR"
    printf "${YELLOW}  Rebooting in 10 seconds to apply any remaining kernel changes…${RESET}\n"
    printf "  Press Ctrl+C to cancel.\n"
    sleep 10
    sudo reboot
else
    printf "\n${RED}${BOLD}  ✗  %d step(s) failed:${RESET}\n\n" "${#FAILED[@]}"
    for f in "${FAILED[@]}"; do
        printf "    ${RED}•${RESET}  %s\n" "$f"
    done
    printf "\n  Fix the errors and re-run install-all.sh.\n"
    printf "  Set SKIP_<STEP>=1 to bypass steps that already succeeded.\n\n"
    exit 1
fi
