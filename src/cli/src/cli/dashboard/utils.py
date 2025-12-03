"""Utility functions for the dashboard."""

from datetime import datetime
from textual.widgets import Log


def log_success(log_widget: Log, message: str) -> None:
    """Log a success message in green."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_widget.write_line(f"\033[32m[{timestamp}] ✓ {message}\033[0m")


def log_info(log_widget: Log, message: str) -> None:
    """Log an info message in white."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_widget.write_line(f"\033[37m[{timestamp}] ℹ {message}\033[0m")


def log_debug(log_widget: Log, message: str) -> None:
    """Log a debug message in blue."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_widget.write_line(f"\033[34m[{timestamp}] 🐛 {message}\033[0m")


def log_error(log_widget: Log, message: str) -> None:
    """Log an error message in red."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_widget.write_line(f"\033[31m[{timestamp}] ✗ {message}\033[0m")


def log_warning(log_widget: Log, message: str) -> None:
    """Log a warning message in yellow."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_widget.write_line(f"\033[33m[{timestamp}] ⚠ {message}\033[0m")


# ANSI color codes for reference:
# 30 = black, 31 = red, 32 = green, 33 = yellow, 34 = blue, 35 = magenta, 36 = cyan, 37 = white
# 0 = reset, 1 = bold, 2 = dim, 3 = italic, 4 = underline


def get_state(state: str, compact: bool = False) -> str:
    mapping = {
        "pending": ("⏳", "[yellow]pending[/yellow]"),
        "active": ("🟢", "[green]active[/green]"),
        "open": ("🟢", "[green]open[/green]"),
        "queued": ("⏳", "[yellow]queued[/yellow]"),
        "evaluating": ("⏳", "[orange]evaluating[/orange]"),
        "scored": ("🟢", "[green]scored[/green]"),
        "completed": ("🏁", "[bold blue]completed[/bold blue]"),
        "stale": ("🟡", "[yellow]stale[/yellow]"),
        "replaced": ("🗑️ ", "[orange]replaced[/orange]"),
        "evaluation": ("⏳", "[orange]evaluation[/orange]"),
    }
    emoji, colored = mapping.get(state, ("🔴", f"[red]{state}[/red]"))
    return emoji if compact else f"{emoji} {colored}"


def get_reveal_status(reveal_at: datetime | None, compact: bool = False) -> str:
    """Get the reveal status indicator based on reveal_at timestamp.

    Returns:
        - Eye emoji + green "Visible" if reveal_at < now (already revealed)
        - Lock emoji + orange "Locked" if reveal_at >= now or None (not yet revealed)
    """
    if reveal_at is None:
        return "🔒" if compact else "🔒 [orange]Locked[/orange]"

    # Handle timezone-aware and naive datetimes
    if reveal_at.tzinfo is not None:
        from datetime import timezone

        now = datetime.now(timezone.utc)
        # Convert reveal_at to UTC if needed for comparison
        reveal_at_utc = reveal_at.astimezone(timezone.utc)
        if reveal_at_utc < now:
            return "👁" if compact else "👁  [green]Visible[/green]"
        else:
            return "🔒" if compact else "🔒 [orange]Locked[/orange]"
    else:
        now = datetime.now()
        if reveal_at < now:
            return "👁" if compact else "👁  [green]Visible[/green]"
        else:
            return "🔒" if compact else "🔒 [orange]Locked[/orange]"


def get_top_score_status(top_score: bool, hotkey: str, top_scorer_hotkey: str, compact: bool = False) -> str:
    """Get the top score status indicator based on top_score and hotkey."""
    if top_score:
        if hotkey == top_scorer_hotkey:
            return "🏆" if compact else "🏆 [green]Current Top Scorer[/green]"
        else:
            return "🥈" if compact else "🥈 [orange]Previous Top Scorer[/orange]"
    else:
        return "[red]✗[/red]" if compact else "[red][bold]✗[/bold] Not a Top Scorer[/red]"
