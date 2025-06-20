#!/bin/bash

# Initialize variables
script="neurons/validator.py"
autoRunLoc=$(readlink -f "$0")
proc_name="s1_validator_main_process"
update_proc_name="auto_updater"
args=()
version_location="./prompting/__init__.py"
version="__version__"

old_args=$@

bash scripts/install.sh

bash scripts/manage_container.sh

# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  # Check if the argument starts with a hyphen (flag)
  if [[ "$arg" == -* ]]; then
    # Check if the argument has a value
    if [[ $# -gt 1 && "$2" != -* ]]; then
          if [[ "$arg" == "--script" ]]; then
            script="$2";
            shift 2
        else
            # Add '=' sign between flag and value
            args+=("'$arg'");
            args+=("'$2'");
            shift 2
        fi
    else
      # Add '=True' for flags with no value
      args+=("'$arg'");
      shift
    fi
  else
    # Argument is not a flag, add it as it is
    args+=("'$arg '");
    shift
  fi
done

# Check if script is already running with pm2
if pm2 status | grep -q $proc_name; then
    echo "The main is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

# Check if the update check is already running with pm2
if pm2 status | grep -q $update_proc_name; then
    echo "The update check is already running with pm2. Stopping and restarting..."
    pm2 delete $update_proc_name
fi

# Run the Python script with the arguments using pm2
echo "Running $script with the following pm2 config:"

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma
joined_args=${joined_args%,}

# Create the pm2 config file
echo "module.exports = {

  apps: [
    {
      name: '$proc_name',
      script: 'poetry',
      interpreter: 'python3',
      min_uptime: '5m',
      max_restarts: '5',
      args: ['run', 'python', '$script', $joined_args]
    },
    {
      name: 'auto_updater',
      script: './scripts/autoupdater.sh',
      interpreter: '/bin/bash',
      min_uptime: '5m',
      max_restarts: '5',
      env: {
        'UPDATE_CHECK_INTERVAL': '300',
        'GIT_BRANCH': 'main'
      }
    }
  ]
};" > app.config.js

# Print configuration to be used
cat app.config.js

pm2 start app.config.js
