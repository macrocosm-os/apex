from importlib.metadata import version
from pathlib import Path
from typing import Any

from loguru import logger

__version__ = version("apex")


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
        logger.add(str(log_file_path), level=level, format=file_log_format, rotation="10 MB", retention="7 days")

    return logger


setup_logger(level="DEBUG")
