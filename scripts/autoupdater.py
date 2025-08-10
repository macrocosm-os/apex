#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# CHECK_INTERVAL = 15 * 60
CHECK_INTERVAL = 15


def venv_python() -> str:
    return os.path.join(".venv", "bin", "python")


def read_python_version() -> str | None:
    try:
        with open(".python-version", encoding="utf-8") as f:
            # Take first non-empty token (pyenv format e.g. "3.11.9").
            return f.read().strip().split()[0]
    except FileNotFoundError:
        return None


def start_proc(config: Path) -> subprocess.Popen:
    py_ver = read_python_version()
    if py_ver:
        subprocess.run(["uv", "venv", "--python", py_ver], check=True)
    else:
        subprocess.run(["uv", "venv"], check=True)

    # Install project in dev mode into the venv.
    subprocess.run(["uv", "pip", "install", ".[dev]"], check=True)

    # Run validator.
    return subprocess.Popen([venv_python(), "validator.py", "-c", str(config)])


def stop_proc(process: subprocess.Popen) -> None:
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def remote_has_updates() -> bool:
    try:
        subprocess.run(["git", "fetch", "--quiet"], check=True)
        out = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", "@{u}...HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
        left, right = map(int, out.split())
        # Remote is ahead.
        return left > 0
    except subprocess.CalledProcessError:
        # No upstream or git issue; treat as no updates.
        return False


def git_pull_ff_only() -> None:
    try:
        subprocess.run(["git", "pull", "--ff-only"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Git pull failed due to conflicts or other issues: {e}", file=sys.stderr)
        print("Staying on the current version.", file=sys.stderr)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apex validator")
    parser.add_argument(
        "-c",
        "--config",
        # default="config/testnet.yaml",
        default="config/mainnet.yaml",
        help="Config file path (e.g. config/mainnet.yaml).",
        type=Path,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = read_args()
    proc = start_proc(config=args.config)

    def handle_sigint(sig, frame):
        stop_proc(proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        time.sleep(CHECK_INTERVAL)
        print("Checking for updates...")

        # If child exited, propagate its code.
        if proc.poll() is not None:
            sys.exit(proc.returncode)

        if remote_has_updates():
            print("Updates detected, restaring process")
            stop_proc(proc)
            git_pull_ff_only()
            proc = start_proc(config=args.config)


if __name__ == "__main__":
    main()
