"""Time utilities for dashboard components."""

from datetime import datetime, timezone
from typing import Optional


def utc_to_local(utc_dt: Optional[datetime]) -> Optional[datetime]:
    """Convert UTC datetime to local timezone."""
    if utc_dt is None:
        return None

    # If the datetime is naive (no timezone info), assume it's UTC
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    return utc_dt.astimezone()


def format_datetime(dt: Optional[datetime], include_seconds: bool = False) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"

    local_dt = utc_to_local(dt)
    if include_seconds:
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return local_dt.strftime("%Y-%m-%d %H:%M")


def get_round_progress(start_at: Optional[datetime], end_at: Optional[datetime]) -> tuple[float, str]:
    """
    Calculate round progress as a percentage and status string.

    Returns:
        tuple: (progress_percentage, status_string)
    """
    now = datetime.now(timezone.utc)

    if start_at is None or end_at is None:
        return 0.0, "No schedule"

    # Ensure datetimes are timezone-aware
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)

    if now < start_at:
        return 0.0, "Not started"
    elif now > end_at:
        return 100.0, "Completed"
    else:
        total_duration = (end_at - start_at).total_seconds()
        elapsed = (now - start_at).total_seconds()
        progress = (elapsed / total_duration) * 100
        return progress, "Active"


def _format_time_delta(
    total_seconds: int,
    include_seconds: bool = False,
    always_show_hours: bool = False,
    compact: bool = False,
) -> str:
    """
    Format a time delta (in seconds) as a human-readable string.

    Args:
        total_seconds: Total seconds in the time delta
        include_seconds: Whether to include seconds in the output (default: False)
        always_show_hours: If True, show hours even when 0 if days > 0 (default: False)
        compact: If True, use compact format "XD YH ZM AS" instead of verbose format (default: False)

    Returns:
        str: Formatted time string like "X day(s) Y hr(s) Z min" (and optionally "W sec")
             or compact format like "XD YH ZM AS" when compact=True
    """
    if total_seconds <= 0:
        return "0M" if compact else "0 min"

    # Calculate time components
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if compact:
        # Compact format: XD YH ZM AS
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or (always_show_hours and days > 0):
            parts.append(f"{hours}h")
        if minutes > 0 or days > 0 or hours > 0:
            parts.append(f"{minutes}m")
        if include_seconds or (days == 0 and hours == 0):
            parts.append(f"{seconds}s")

        # If no parts were added (shouldn't happen, but handle edge case)
        if not parts:
            return "0M"

        return " ".join(parts)
    else:
        # Verbose format: X day(s) Y hr(s) Z min (and optionally W sec)
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0 or (always_show_hours and days > 0):
            parts.append(f"{hours} hr{'s' if hours != 1 else ''}")
        if minutes > 0 or days > 0 or hours > 0:
            parts.append(f"{minutes} min")
        if include_seconds or (days == 0 and hours == 0):
            parts.append(f"{seconds} sec")

        # If no parts were added (shouldn't happen, but handle edge case)
        if not parts:
            return "0 min"

        return " ".join(parts)


def get_round_countdown(end_at: Optional[datetime]) -> Optional[str]:
    """
    Get countdown string for round end time in a readable format.

    Returns:
        str: Countdown string or None if round is not active
    """
    if end_at is None:
        return None

    now = datetime.now(timezone.utc)

    # Ensure datetime is timezone-aware
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)

    if now >= end_at:
        return None  # Round is over

    remaining = end_at - now
    total_seconds = int(remaining.total_seconds())

    if total_seconds <= 0:
        return "00:00:00"

    return _format_time_delta(total_seconds, include_seconds=True, always_show_hours=True)


def get_age(dt: Optional[datetime], include_seconds: bool = False, compact: bool = False) -> str:
    """
    Calculate the age of a datetime broken down by days, hours, and minutes.

    Args:
        dt: The datetime to calculate the age of (should be in the past)
        include_seconds: Whether to include seconds in the output (default: False)
        compact: If True, use compact format "XD YH ZM AS" instead of verbose format (default: False)

    Returns:
        str: Age string formatted as "X day(s) Y hr(s) Z min" (and optionally "W sec")
             Returns "N/A" if dt is None or if dt is in the future
    """
    if dt is None:
        return "N/A"

    now = datetime.now(timezone.utc)

    # Ensure datetime is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Calculate the difference
    age = now - dt
    total_seconds = int(age.total_seconds())

    if total_seconds < 0:
        return "N/A"  # Datetime is in the future

    return _format_time_delta(total_seconds, include_seconds=include_seconds, compact=compact)
