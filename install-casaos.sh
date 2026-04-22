#!/usr/bin/env bash
# Optional: install CasaOS (the dashboard layer ZimaOS is built on).
# Use this if you want to host the trainer alongside other CasaOS apps on
# Ubuntu/Debian. After install, deploy this project's docker-compose.yml from
# the CasaOS UI ("Custom Install") or via `docker compose up -d` in this dir.
set -euo pipefail

if command -v casaos >/dev/null 2>&1; then
    echo "==> CasaOS already installed."
    exit 0
fi

echo "==> Installing CasaOS via the official one-liner."
echo "    https://get.casaos.io"
curl -fsSL https://get.casaos.io | sudo bash

echo "==> CasaOS installed. UI: http://$(hostname -I | awk '{print $1}')"
echo "    To deploy the trainer dashboard:"
echo "      cd $(pwd) && docker compose up -d --build"
