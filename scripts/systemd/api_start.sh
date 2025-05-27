#!/bin/bash
# Create and launch API systemd service.
set -e

# Check if systemd is running.
if [ ! -d /run/systemd/system ]; then
    echo "Error: systemd does not appear to be running. Exiting."
    exit 1
fi

# Adjust project dir as needed.
WORKDIR=/root/prompting

SERVICE_FILE="/etc/systemd/system/sn1api.service"

# Create (or update) the systemd service file.
sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=SN1 API Service
After=network.target

[Service]
Type=simple
# Set the working directory so that poetry finds the correct pyproject.toml
WorkingDirectory=${WORKDIR}
# Use Poetry to run the API script
ExecStart=python3.10 -m poetry run uvicorn validator_api.api:app --host 0.0.0.0 --port 8005 --workers 2
Restart=always
User=root
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd configuration.
sudo systemctl daemon-reload

# Enable the service to start at boot.
sudo systemctl enable sn1api.service

# Restart the service if already running, or start it if not.
sudo systemctl restart sn1api.service

echo "SN1 API service (sn1api) has been started/restarted."
echo "Attaching to API logs. Press Ctrl+C to exit."
sudo journalctl -f -u sn1api.service
