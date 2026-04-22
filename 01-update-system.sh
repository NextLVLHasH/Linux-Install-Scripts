#!/usr/bin/env bash
# Step 1: update and upgrade Ubuntu.
set -euo pipefail

echo "==> Updating apt package index..."
sudo apt-get update -y

echo "==> Upgrading installed packages..."
sudo apt-get upgrade -y

echo "==> Running dist-upgrade for kernel/core updates..."
sudo apt-get dist-upgrade -y

echo "==> Removing orphaned packages..."
sudo apt-get autoremove -y
sudo apt-get autoclean -y

echo "==> System update complete."
