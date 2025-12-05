#!/usr/bin/env python3
"""
Apex validator autoupdater.
This script runs the validator and automatically updates it when a new version is released.

Usage:
    python3 scripts/start_validator.py [validator arguments]

When run as a pm2 process, it will:
- Start the validator as a subprocess
- Check for updates every 15 minutes
- Restart the validator when updates are detected
"""
import argparse
import logging
import shutil
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

log = logging.getLogger(__name__)
UPDATES_CHECK_TIME = timedelta(minutes=15)
ROOT_DIR = Path(__file__).parent.parent
VALIDATOR_DIR = ROOT_DIR / "src" / "validator"


def get_version() -> str:
    """Extract the version as current git commit hash"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        cwd=ROOT_DIR,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f"Invalid commit hash: {commit}"
    return commit[:8]


def prepare_validator() -> None:
    """Prepare validator by copying .env file and syncing dependencies"""
    log.info("Preparing validator...")

    # Copy .env file
    env_source = ROOT_DIR / ".env"
    env_dest = VALIDATOR_DIR / ".env"

    if env_source.exists():
        shutil.copy(env_source, env_dest)
        log.info("Copied .env to validator directory")
    else:
        log.warning(".env file not found in root directory")

    # Run uv sync
    log.info("Running uv sync...")
    try:
        subprocess.run(
            ["uv", "sync"],
            check=True,
            cwd=VALIDATOR_DIR,
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to sync dependencies: %s", exc)
        raise


def start_validator_process(args: list[str]) -> subprocess.Popen:
    """
    Start the validator process as a subprocess.
    Returns the process handle so it can be managed/stopped later.
    """
    log.info("Starting validator subprocess...")

    # Prepare the validator before starting
    prepare_validator()

    # Start validator using uv run
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "--package",
            "validator",
            "main.py",
            *args,
        ],
        cwd=VALIDATOR_DIR,
    )

    log.info("Validator started with PID: %s", process.pid)
    return process


def stop_validator_process(process: subprocess.Popen) -> None:
    """Stop the validator process gracefully"""
    if process and process.poll() is None:
        log.info("Stopping validator process (PID: %s)...", process.pid)
        process.terminate()
        try:
            process.wait(timeout=10)
            log.info("Validator stopped.")
        except subprocess.TimeoutExpired:
            log.warning("Validator did not stop gracefully, killing...")
            process.kill()
            process.wait()


def remote_has_updates() -> bool:
    """Check if there are updates available from remote"""
    try:
        subprocess.run(["git", "fetch", "--quiet"], check=True, cwd=ROOT_DIR)
        out = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", "@{u}...HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT_DIR,
        ).strip()
        left, right = map(int, out.split())
        # Remote is ahead
        return left > 0
    except subprocess.CalledProcessError:
        # No upstream or git issue; treat as no updates
        return False


def pull_latest_version() -> None:
    """
    Pull the latest version from git using fast-forward only.
    This prevents merge conflicts by only updating if it's a clean fast-forward.
    """
    try:
        log.info("Pulling latest version...")
        subprocess.run(["git", "pull", "--ff-only"], check=True, cwd=ROOT_DIR)
        log.info("Successfully updated to latest version")
    except subprocess.CalledProcessError as exc:
        log.error("Failed to pull (conflicts or other issues): %s", exc)
        log.error("Staying on the current version")


def main(args: list[str], autoupdate: bool = True) -> None:
    """
    Run the validator process and automatically update it when a new version is released.
    This will check for updates every UPDATES_CHECK_TIME and update the validator
    if a new version is available.
    """
    validator_process = start_validator_process(args)
    current_version = get_version()
    log.info("Current version: %s", current_version)

    def handle_sigint(sig, frame):
        """Handle Ctrl+C gracefully"""
        log.info("Received interrupt signal, stopping validator...")
        stop_validator_process(validator_process)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if autoupdate:
            log.info("Auto-update enabled, checking for updates every %s", UPDATES_CHECK_TIME)
            while True:
                time.sleep(UPDATES_CHECK_TIME.total_seconds())

                # If child exited, propagate its exit code
                if validator_process.poll() is not None:
                    log.error("Validator process exited unexpectedly with code: %s", validator_process.returncode)
                    sys.exit(validator_process.returncode)

                log.info("Checking for updates...")
                if remote_has_updates():
                    log.info("Updates detected: %s -> new version", current_version)

                    stop_validator_process(validator_process)
                    pull_latest_version()

                    validator_process = start_validator_process(args)
                    current_version = get_version()
                    log.info("Restarted with version: %s", current_version)
                else:
                    log.info("No updates available")
        else:
            # If autoupdate is disabled, just wait for the validator to exit
            log.info("Auto-update disabled, validator will run without updates")
            validator_process.wait()

    except KeyboardInterrupt:
        log.info("Received interrupt signal, stopping validator...")
    finally:
        stop_validator_process(validator_process)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Automatically update and restart the validator process when a new version is released.",
        epilog="Example usage: python scripts/start_validator.py",
    )

    parser.add_argument(
        "--no_autoupdate",
        action="store_true",
        help="Disable automatic updates"
    )

    flags, extra_args = parser.parse_known_args()

    main(extra_args, autoupdate=not flags.no_autoupdate)
