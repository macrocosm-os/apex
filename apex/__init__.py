from importlib.metadata import version
from pathlib import Path
from typing import Any

from loguru import logger

__version__ = version("apex")


def _version_to_int(version_str: str) -> int:
    version_split = version_str.split(".") + ["0", "0"]  # in case a version doesn't have third element, e.g. 3.0
    major = int(version_split[0])
    minor = int(version_split[1])
    patch = int(version_split[2])
    return (10000 * major) + (100 * minor) + patch


__spec_version__ = _version_to_int(__version__)


def setup_logger(log_file_path: str | Path | None = None, level: str = "INFO") -> Any:
    """Set up the loguru logger with optional file logging and specified log level.

    Args:
        log_file_path: Path to the log file. If None, logs won't be saved to a file.
        level: Logging level (e.g., "INFO", "DEBUG", "ERROR"). Defaults to "INFO".
    """
    # Remove existing handlers to avoid duplicate logs.
    logger.remove()

    # Console handler format.
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<cyan>{file}</cyan>:<cyan>{line}</cyan>] <level>{message}</level>"
    )

    # Add console handler.
    logger.add(lambda msg: print(msg), level=level, format=log_format)

    # Add file handler if a path is provided.
    if log_file_path:
        file_log_format = "{time:YYYY-MM-DD HH:mm:ss} [{file}:{line}] {message}"
        logger.add(str(log_file_path), level=level, format=file_log_format, rotation="5 MB", retention="3 days")

    return logger


setup_logger(log_file_path="logs.log", level="DEBUG")
