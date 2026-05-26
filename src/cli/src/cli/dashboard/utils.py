"""Utility functions for the dashboard."""

from datetime import datetime
from textual.widgets import Log

_SCREENER_REJECTION_MARKERS = ("screener validation",)


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


def get_state(state: str, compact: bool = False, eval_error: str | None = None) -> str:
    """Render a state badge. If `eval_error` matches a known sandbox-screener
    rejection marker on a `scored` submission, the badge is overridden to
    `sandbox_rejected` — these submissions technically completed (with a
    0/forfeit score) but were thrown out by the in-sandbox model validator.

    Note: `rejected` and `sandbox_rejected` mean materially different things to
    the miner — only the former (pre-sandbox competition-screener rejection)
    refunds the submission fee. Sandbox-screener rejections are non-refundable.
    """
    if state == "scored" and eval_error and any(m in eval_error.lower() for m in _SCREENER_REJECTION_MARKERS):
        state = "sandbox_rejected"
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
        "rejected": ("🔴", "[red]rejected[/red]"),
        "sandbox_rejected": ("❌", "[red]sandbox rejected[/red]"),
        "evaluation": ("⏳", "[orange]evaluation[/orange]"),
        "partially_scored": ("⏳", "[orange]partially scored[/orange]"),
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

    from datetime import timezone

    # Always use UTC for comparison to ensure consistency
    now = datetime.now(timezone.utc)

    # Handle timezone-aware and naive datetimes
    # If naive, assume it's UTC (consistent with check_code_available)
    if reveal_at.tzinfo is not None:
        # Convert reveal_at to UTC if needed for comparison
        reveal_at_utc = reveal_at.astimezone(timezone.utc)
    else:
        # Treat naive datetime as UTC
        reveal_at_utc = reveal_at.replace(tzinfo=timezone.utc)

    if reveal_at_utc < now:
        return "👁" if compact else "👁 [green]Visible[/green]"
    else:
        return "🔒" if compact else "🔒 [orange]Locked[/orange]"


def get_top_score_status(
    top_score: bool, submission_id: int | None, curr_top_score_id: int | None, compact: bool = False
) -> str:
    """Get the top score status indicator based on top_score and submission ID.

    Only the submission matching curr_top_score_id gets the gold cup.
    Previous top scorers (top_score=True but different ID) get silver medal.
    """
    # Check if this is the current top scorer by comparing submission IDs
    if curr_top_score_id is not None and submission_id == curr_top_score_id:
        return "🏆" if compact else "🏆 [green]Current Top Scorer[/green]"
    # If not current top scorer, check if it was ever a top scorer
    elif top_score:
        return "🥈" if compact else "🥈 [orange]Previous Top Scorer[/orange]"
    else:
        return "[red]✗[/red]" if compact else "[red][bold]✗[/bold] Not a Top Scorer[/red]"
