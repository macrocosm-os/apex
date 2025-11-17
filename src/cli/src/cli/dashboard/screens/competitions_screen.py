"""Competitions list screen."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Header, Footer, Log, Static
from textual.binding import Binding
from textual.message import Message

from common.models.api.competition import CompetitionResponse
from cli.dashboard.utils import log_success, log_debug, get_state
from cli.dashboard.time_utils import get_round_progress, format_datetime
from cli.dashboard.art import APEX_TITLE


class CompetitionsScreen(Screen):
    """Screen showing the list of competitions."""

    CSS = """
    #competitions_table {
        height: 100%;
    }

    #title_bar {
        height: 8;
        text-align: center;
        color: $primary;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "select_item", "Select"),
        Binding("l", "toggle_log", "Toggle Log"),
    ]

    def __init__(self, competitions: list[CompetitionResponse]) -> None:
        super().__init__()
        self.competitions = competitions

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()
        with Vertical(id="main_container"):
            yield Static(APEX_TITLE, id="title_bar")
            yield Container(DataTable(id="competitions_table"), id="main_container")
            yield Log(id="log")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the competitions table when the screen is mounted."""
        log_widget = self.query_one("#log")
        log_widget.display = False
        log_success(log_widget, "Dashboard started")
        table = self.query_one(DataTable)
        table.add_columns("ID", "Name", "State", "Type", "Package", "Round Details", "Top Score")

        # Enable row selection events
        table.cursor_type = "row"

        for comp in self.competitions:
            # Create enhanced round details
            round_details = self._format_round_details(comp)

            table.add_row(
                str(comp.id),
                comp.name[:30] + "..." if len(comp.name) > 30 else comp.name,
                comp.state,
                comp.ctype,
                comp.pkg,
                round_details,
                f"{comp.top_score_value:.2f}" if comp.top_score_value else "N/A",
            )

        # Table cursor will be at row 0 by default
        table.focus()
        table.cursor_coordinate = (0, 0)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        table = event.data_table
        row_key = event.row_key
        # Get the row index from the row key
        row_index = table.get_row_index(row_key)
        log_widget = self.query_one("#log")
        log_debug(log_widget, f"Row selected: {row_index}")

        # Add this line to also trigger the selection logic
        if 0 <= row_index < len(self.competitions):
            selected_competition = self.competitions[row_index]
            log_success(log_widget, f"Selected competition: {selected_competition.id}")
            self.post_message(CompetitionSelected(selected_competition))

    def action_toggle_log(self) -> None:
        """Toggle the log widget visibility."""
        log_widget = self.query_one("#log")
        if log_widget.display:
            log_widget.display = False
        else:
            log_widget.display = True

    def _format_round_details(self, comp) -> str:
        """Format round details for display in the table."""
        if not comp.curr_round_number:
            return "No rounds"

        # Get round data
        round_number = comp.curr_round_number
        state = comp.curr_round.state if comp.curr_round else "unknown"
        start_at = comp.curr_round.start_at if comp.curr_round else None
        end_at = comp.curr_round.end_at if comp.curr_round else None

        # Get progress information
        progress, status = get_round_progress(start_at, end_at)

        # Format state
        state_display = get_state(state)

        # Format times
        start_str = format_datetime(start_at) if start_at else "N/A"
        end_str = format_datetime(end_at) if end_at else "N/A"

        # Create compact display with simple progress indicator
        progress_indicator = "█" * min(8, int(progress / 12.5)) + "░" * max(0, 8 - int(progress / 12.5))

        details = f"R{round_number} {state_display}\n{progress_indicator} {progress:.0f}%\n{start_str} → {end_str}"

        return details


class CompetitionSelected(Message):
    """Message sent when a competition is selected."""

    def __init__(self, competition: CompetitionResponse) -> None:
        self.competition = competition
        super().__init__()
