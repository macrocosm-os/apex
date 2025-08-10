import asyncio
import subprocess
import sys
from pathlib import Path
from shlex import split

from loguru import logger

ROOT_DIR = Path(__file__).parent.parent


def get_version() -> str:
    """Extract the version as current git commit hash"""
    result = subprocess.run(
        split("git rev-parse HEAD"),
        check=True,
        capture_output=True,
        cwd=ROOT_DIR,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f"Invalid commit hash: {commit}"
    return commit[:8]


def pull_latest_version() -> None:
    """Pull the latest version from git.
    This uses `git pull --rebase`, so if any changes were made to the local repository,
    this will try to apply them on top of origin's changes. This is intentional, as we
    don't want to overwrite any local changes. However, if there are any conflicts,
    this will abort the rebase and return to the original state.
    The conflicts are expected to happen rarely since validator is expected
    to be used as-is.
    """
    try:
        subprocess.run(split("git pull --rebase --autostash"), check=True, cwd=ROOT_DIR)
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to pull, reverting: %s", exc)
        subprocess.run(split("git rebase --abort"), check=True, cwd=ROOT_DIR)


def upgrade_packages() -> None:
    """Upgrade python packages by running `pip install --upgrade -r requirements.txt`.
    Notice: this won't work if some package in `requirements.txt` is downgraded.
    Ignored as this is unlikely to happen.
    """
    logger.info("Upgrading packages")
    try:
        subprocess.run(
            split(f"{sys.executable} -m pip install -e ."),
            check=True,
            cwd=ROOT_DIR,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to upgrade packages, proceeding anyway. %s", exc)


async def autoupdate_loop() -> None:
    """Async version of autoupdate that runs alongside the validator.
    Checks for updates every hour and applies them if available.
    """
    current_version = latest_version = get_version()
    logger.info("Current version: %s", current_version)

    try:
        while True:
            await asyncio.sleep(3600)  # Wait 1 hour between checks

            pull_latest_version()
            latest_version = get_version()
            logger.info("Latest version: %s", latest_version)

            if latest_version != current_version:
                logger.info(
                    "Upgraded to latest version: %s -> %s",
                    current_version,
                    latest_version,
                )
                upgrade_packages()
                current_version = latest_version

    except asyncio.CancelledError:
        logger.info("Autoupdate task cancelled")
        raise
