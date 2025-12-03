#!/usr/bin/env bash
set -e

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Sync the workspace (installs all shared + service packages)
echo "Syncing workspace..."
uv sync

# Install the apex CLI as a tool (adds the 'apex' command globally)
echo "Installing Apex CLI..."
uv tool install src/cli --force

echo "✅ Setup complete! Try running: apex --help"
