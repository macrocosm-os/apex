from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as get_pkg_version


async def _resolve_version_string() -> str:
    """Resolve version string"""
    try:
        return get_pkg_version("apex-monorepo")
    except PackageNotFoundError:
        return "0.0.0"


async def _parse_part_to_int(part: str) -> int:
    """Parse an integer from the beginning of a version segment."""
    digits: list[str] = []
    for ch in part:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits) or 0)


async def _version_to_int(version_str: str) -> int:
    parts: list[str] = (version_str or "0.0.0").split(".") + ["0", "0"]
    major = await _parse_part_to_int(parts[0])
    minor = await _parse_part_to_int(parts[1])
    patch = await _parse_part_to_int(parts[2])
    return (10000 * major) + (100 * minor) + patch


@lru_cache(maxsize=1)
async def spec_version() -> int:
    """Return the integer spec version derived from monorepo version."""
    return await _version_to_int(await _resolve_version_string())


__all__ = ["spec_version"]
