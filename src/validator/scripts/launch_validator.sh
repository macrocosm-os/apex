#!/bin/bash

set -e

# Activate virtual environment
source .venv/bin/activate

# Environment variables are injected by Kubernetes via envFrom
# No need to source /secrets/.env anymore

echo "[+] Launching validator"
exec python main.py
