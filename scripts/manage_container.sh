#!/bin/bash

# This script restarts the gpu-app docker container if it's running,
# or starts it if it's not.

# The directory where docker-compose.yml is located
COMPOSE_DIR="gpu_container"

# The name of the service to manage
SERVICE_NAME="gpu-app"

# Change to the compose directory and exit if it fails
cd "$COMPOSE_DIR" || { echo "Directory $COMPOSE_DIR not found." >&2; exit 1; }

# Check if the service is running.
# 'docker compose ps -q' will output container IDs if they are up.
# We check if the output is non-empty.
if [ -n "$(docker compose ps -q "$SERVICE_NAME")" ]; then
    echo "Service '$SERVICE_NAME' is running. Restarting..."
    docker compose restart "$SERVICE_NAME"
else
    # This will handle both 'stopped' and 'not-created' states.
    # The --build flag ensures the image is up-to-date.
    echo "Service '$SERVICE_NAME' is not running. Starting..."
    docker compose up -d --build "$SERVICE_NAME"
fi

# Go back to the original directory
cd - >/dev/null

echo "Script finished." 