#!/usr/bin/env bash
# Autoupdater script for APEX validator
# This script ensures dependencies (uv, pm2) are installed, then starts the
# autoupdater as a pm2 daemon process.

APP_NAME="apex_validator"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

# Ensure common user bin dirs are in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/bin:$HOME/.npm-global/bin:$PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[info] Root directory: $ROOT_DIR"

# Ensure uv exists
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

# Ensure pm2 exists
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

PYTHON_EXEC=$(python3 -c "import sys; print(sys.executable)")
echo "[info] Python executable: $PYTHON_EXEC"

# Start the autoupdater as a pm2 process, managing the validator as a subprocess
echo "[info] Starting validator autoupdater with pm2..."
cd "$ROOT_DIR"
pm2 start "$ROOT_DIR/scripts/start_validator.py" --interpreter "$PYTHON_EXEC" --name "$APP_NAME"

# Show logs
echo "[info] Showing logs (press Ctrl+C to exit log view)..."
pm2 logs "$APP_NAME"
