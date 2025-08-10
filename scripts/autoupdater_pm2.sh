#!/usr/bin/env bash
APP_NAME="sn1"
# CONFIG="config/mainnet.yaml"
CONFIG="config/testnet.yaml"

UV_INSTALL_URL="https://astral.sh/uv/install.sh"

# Ensure common user bin dirs are in PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/bin:$HOME/.npm-global/bin:$PATH"

# 1) Ensure uv exists.
if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found; installing..."
  if ! command -v curl >/dev/null 2>&1; then
    echo "[error] curl is required to install uv." >&2
    exit 1
  fi
  curl -LsSf "$UV_INSTALL_URL" | sh
  hash -r
  if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv installation completed but 'uv' not found in PATH." >&2
    exit 1
  fi
else
  echo "[info] uv found: $(command -v uv)"
fi

# 2) Ensure pm2 exists.
if ! command -v pm2 >/dev/null 2>&1; then
  echo "[info] pm2 not found; installing globally with npm..."
  if ! command -v npm >/dev/null 2>&1; then
    echo "[error] npm is required to install pm2. Please install Node.js first." >&2
    exit 1
  fi
  npm install -g pm2
  hash -r
  if ! command -v pm2 >/dev/null 2>&1; then
    echo "[error] pm2 installation completed but 'pm2' not found in PATH." >&2
    exit 1
  fi
else
  echo "[info] pm2 found: $(command -v pm2)"
fi

pm2 start scripts/autoupdater.py --interpreter .venv/bin/python --name sn1 -- -c config/testnet.yaml
pm2 logs sn1
