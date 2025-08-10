#!/usr/bin/env bash
set -euo pipefail

APP_NAME="sn1"
PY_VERSION_FILE=".python-version"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
CONFIG="config/mainnet.yaml"

# Ensure common user bin dirs are in PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/bin:$HOME/.npm-global/bin:$PATH"

# 1) Ensure uv exists
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

# 2) Ensure pm2 exists
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

# 3) Determine Python version (read from .python-version; fallback 3.11)
PY_VER="3.11"
if [[ -s "$PY_VERSION_FILE" ]]; then
  CANDIDATE="$(tr -d ' \t\r\n' < "$PY_VERSION_FILE")"
  if [[ "$CANDIDATE" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
    PY_VER="$CANDIDATE"
  else
    echo "[warn] Invalid version in $PY_VERSION_FILE: '$CANDIDATE' — using $PY_VER"
  fi
else
  echo "[warn] $PY_VERSION_FILE missing or empty — using $PY_VER"
fi
echo "[info] Using Python version: $PY_VER"

# 4) Create/update venv and install deps
echo "[info] Creating/refreshing .venv with uv…"
uv venv --python="$PY_VER"
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[error] .venv/bin/python not created. Check that uv can resolve Python $PY_VER." >&2
  exit 1
fi

echo "[info] Installing project dependencies…"
uv pip install '.[dev]'

# 5) Start with pm2
pm2 delete "$APP_NAME" >/dev/null 2>&1 || true
pm2 start ".venv/bin/python" --name "$APP_NAME" -- scripts/autoupdater.py -c "$CONFIG"

echo "[done] pm2 process '$APP_NAME' started."
pm2 logs "$APP_NAME"
