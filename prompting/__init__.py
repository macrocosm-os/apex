from importlib.metadata import version

from loguru import logger


def _version_to_int(version_str: str) -> int:
    version_split = version_str.split(".") + ["0", "0"]  # in case a version doesn't have third element, e.g. 3.0
    major = int(version_split[0])
    minor = int(version_split[1])
    patch = int(version_split[2])
    return (10000 * major) + (100 * minor) + patch


__version__ = version("prompting")
# Used by W&B logging, which expects an integer version.
__spec_version__ = _version_to_int(__version__)

logger.info(f"Project version: {__version__}")
