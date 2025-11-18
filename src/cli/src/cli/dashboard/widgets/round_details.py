from textual.widgets import Static, ProgressBar
from textual.containers import Vertical
from typing import Optional

from cli.dashboard.time_utils import get_round_progress, get_round_countdown, format_datetime
from cli.dashboard.utils import get_state


class RoundDetailsWidget(Vertical):
    """Combined widget showing round details, progress, and countdown."""

    def __init__(self, round_data: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.round_data = round_data or {}
        self.header_label = None
        self.times_label = None
        self.progress_bar = None
        self.countdown_label = None
        self.burn_factor_label = None
        self.update_timer = None

    def compose(self):
        """Compose the widget with child widgets."""
        self.header_label = Static("", id="header_label")
        self.times_label = Static("", id="times_label")
        self.progress_bar = ProgressBar(total=100, show_eta=False, id="progress_bar")
        self.countdown_label = Static("", id="countdown_label")
        self.burn_factor_label = Static("", id="burn_factor_label")

        yield self.header_label
        yield self.times_label
        yield self.progress_bar
        yield self.countdown_label
        yield self.burn_factor_label

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_display()
        # Start a timer to update the display every second
        self.start_update_timer()

    def on_unmount(self) -> None:
        """Called when the widget is unmounted."""
        if self.update_timer:
            self.update_timer.stop()

    def start_update_timer(self) -> None:
        """Start the update timer."""
        if self.update_timer:
            self.update_timer.stop()
        self.update_timer = self.set_timer(1.0, self.timer_callback)

    def timer_callback(self) -> None:
        """Timer callback that updates display and restarts timer."""
        self.update_display()
        # Restart the timer for the next update
        self.start_update_timer()

    def update_round_data(self, round_data: dict) -> None:
        """Update the round data."""
        self.round_data = round_data
        self.update_display()

    def update_display(self) -> None:
        """Update the round details display."""
        # Check if child widgets are available
        if not all(
            [self.header_label, self.times_label, self.progress_bar, self.countdown_label, self.burn_factor_label]
        ):
            return

        round_number = self.round_data.get("round_number", "N/A")
        state = self.round_data.get("state", "unknown")
        start_at = self.round_data.get("start_at")
        end_at = self.round_data.get("end_at")

        # Get progress information
        progress, status = get_round_progress(start_at, end_at)
        countdown = get_round_countdown(end_at)

        # Format times for display
        start_str = format_datetime(start_at)
        end_str = format_datetime(end_at)

        # State formatting
        state_display = get_state(state)

        # Update header
        self.header_label.update(f"[bold]Round #{round_number}[/bold] - {state_display}")

        # Update times
        self.times_label.update(f"[dim]Start:[/dim] {start_str}  [dim]End:[/dim] {end_str}")

        # Update progress bar using advance() method
        # Calculate the difference from current progress and advance by that amount
        # Using advance() instead of setting progress directly avoids resetting the bar
        current_progress_value = self.progress_bar.progress if self.progress_bar.progress is not None else 0.0
        progress_diff = progress - current_progress_value

        if abs(progress_diff) > 0.01:  # Only update if there's a meaningful change
            if progress_diff > 0:
                # Advance forward - this is the key to avoiding resets!
                self.progress_bar.advance(progress_diff)
            else:
                # Progress went backwards (shouldn't happen normally, but handle refresh case)
                self.progress_bar.progress = progress

        # Update progress label - this is a duplicate of the progress bar
        # self.progress_label.update(f"[bold]Progress:[/bold] {progress:.1f}%")

        # Update countdown
        if countdown:
            self.countdown_label.update(f"[bold green]⏰ Time Remaining:[/bold green] [green]{countdown}[/green]")

        burn_factor = self.round_data.get("burn_factor")
        if burn_factor is not None:
            burn_factor_percent = burn_factor * 100
            self.burn_factor_label.update(f"\n[dim]Burn:[/dim] {burn_factor_percent:.1f}% 🔥")
        else:
            self.burn_factor_label.update("")
