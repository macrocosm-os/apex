#!/usr/bin/env bash
set -euo pipefail

# Configuration with defaults
readonly INTERVAL=${UPDATE_CHECK_INTERVAL:-300}
readonly REMOTE_BRANCH=${GIT_BRANCH:-"main"}
readonly PYPROJECT_PATH="./pyproject.toml"
readonly LOG_FILE="autoupdate.log"
readonly MAX_RETRIES=3
readonly RETRY_DELAY=30

# Logging with ISO 8601 timestamps
log() {
    local level=$1
    shift
    printf '[%s] [%-5s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$level" "$*" | tee -a "$LOG_FILE"
}
compare_semver() {
    local v1=() v2=()
    IFS='.' read -r -a v1 <<< "$1"
    IFS='.' read -r -a v2 <<< "$2"

    # Normalize length (e.g., handle "1.2" as "1.2.0")
    for i in 0 1 2; do
        v1[i]=${v1[i]:-0}
        v2[i]=${v2[i]:-0}
    done

    # Compare each section of MAJOR, MINOR, PATCH
    for i in 0 1 2; do
        if (( v1[i] > v2[i] )); then
            return 1  # v1 is greater
        elif (( v1[i] < v2[i] )); then
            return 2  # v2 is greater
        fi
        # if equal, continue to next
    done

    return 0  # versions are the same
}
get_version() {
    local file=$1
    local version

    if [[ ! -f "$file" ]]; then
        log ERROR "File not found: $file"
        return 1
    fi

    version=$(awk -F'"' '/^version *= *"/ {print $2}' "$file")
    if [[ -z "$version" ]]; then
        log ERROR "Version not found in $file"
        return 1
    fi

    echo "$version"
}

# Retry mechanism for git operations
retry() {
    local cmd=$1
    local attempt=1

    while [[ $attempt -le $MAX_RETRIES ]]; do
        if eval "$cmd"; then
            return 0
        fi

        log WARN "Command failed (attempt $attempt/$MAX_RETRIES): $cmd"

        if [[ $attempt -lt $MAX_RETRIES ]]; then
            sleep "$RETRY_DELAY"
        fi

        ((attempt++))
    done

    return 1
}

# Backup local changes if any exist
backup_changes() {
    if ! git diff --quiet; then
        log WARN "Stashing local changes"
        git stash
    fi
}

# Main update check function
check_for_updates() {
    local local_version remote_version
    local_version=$(get_version "$PYPROJECT_PATH") || return 1

    # Fetch remote updates
    if ! retry "git fetch origin $REMOTE_BRANCH"; then
        log ERROR "Failed to fetch from remote"
        return 1
    fi

    # Get remote version from pyproject.toml in the remote branch
    remote_version=$(git show "origin/$REMOTE_BRANCH:$PYPROJECT_PATH" \
        | awk -F'"' '/^version *= *"/ {print $2}') || {
        log ERROR "Failed to get remote version"
        return 1
    }

    # Semantic version comparison
    compare_semver "$local_version" "$remote_version"
    local cmp=$?

    if [[ $cmp -eq 2 ]]; then
        # Return code 2 => remote_version is greater (update available)
        log INFO "Update available: $local_version → $remote_version"
        return 0
    elif [[ $cmp -eq 0 ]]; then
        # Versions are the same
        log INFO "Already up to date ($local_version)"
        return 1
    else
        # Return code 1 => local is actually ahead or equal in some custom scenario
        # If you only want to *update if remote is strictly greater*, treat this case as no update.
        # Or you could log something else if local is ahead.
        log INFO "Local version ($local_version) is the same or newer than remote ($remote_version)"
        return 1
    fi
}

# Update and restart application
update_and_restart() {
    backup_changes

    if ! retry "git pull origin $REMOTE_BRANCH"; then
        log ERROR "Failed to pull changes"
        return 1
    fi

    if [[ -x "./run.sh" ]]; then
        log INFO "Update successful, restarting validator..."
        pm2 restart s1_validator_main_process
        log INFO "Validator restart initiated via PM2"
        # Let PM2 handle our own restart if needed
        return 0
    else
        log ERROR "run.sh not found or not executable"
        return 1
    fi
}

# Validate git repository
validate_environment() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log ERROR "Not in a git repository"
        exit 1
    fi
}

main() {
    validate_environment
    log INFO "Starting auto-updater (interval: ${INTERVAL}s, branch: $REMOTE_BRANCH)"

    while true; do
        if check_for_updates; then
            update_and_restart
        fi
        sleep "$INTERVAL"
    done
}

main
