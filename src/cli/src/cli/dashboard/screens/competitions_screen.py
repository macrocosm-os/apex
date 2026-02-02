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
        Binding("l", "toggle_log", "Toggle Log", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("c", "toggle_completed", "Show Completed"),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    def __init__(self, competitions: list[CompetitionResponse]) -> None:
        super().__init__()
        self.all_competitions = competitions
        self.show_completed = False
        self.competitions = self._filter_competitions()

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
        table.add_columns(
            "ID",
            "Name",
            "State",
            "Type",
            "Package",
            "Round Details",
            "Top Score",
            "Alloc. Emission",
            "Min Burn",
            "Max Emissions",
        )

        # Enable row selection events
        table.cursor_type = "row"

        # Populate the table with filtered competitions
        self._populate_table()

        # Table cursor will be at row 0 by default
        table.focus()
        table.cursor_coordinate = (0, 0)

    def _populate_table(self) -> None:
        """Populate the table with current filtered competitions."""
        table = self.query_one(DataTable)

        # Calculate max_emissions for each competition and sort by it descending
        # incentive_weight is now a direct decimal allocation (0.0-1.0)
        def calc_max_emissions(comp):
            if comp.state != "active":
                return 0
            emission_allocation = comp.incentive_weight * 100
            return (1 - comp.base_burn_rate) * emission_allocation

        self.competitions = sorted(self.competitions, key=calc_max_emissions, reverse=True)

        for comp in self.competitions:
            # Create enhanced round details
            round_details = self._format_round_details(comp)

            # Calculate emission metrics
            # incentive_weight is directly the allocation (0.0-1.0)
            max_emissions = calc_max_emissions(comp)
            emission_allocation = comp.incentive_weight * 100 if comp.state == "active" else 0
            min_burn = comp.base_burn_rate * 100 if comp.state == "active" else 0

            table.add_row(
                str(comp.id),
                comp.name[:30] + "..." if len(comp.name) > 30 else comp.name,
                get_state(comp.state),
                comp.ctype,
                comp.pkg,
                round_details,
                f"{comp.top_score_value:.2f}" if comp.top_score_value else "N/A",
                f"{emission_allocation:.1f}%",
                f"{min_burn:.0f}%",
                f"{max_emissions:.1f}%",
            )

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

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k)."""
        table = self.query_one(DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down (vim-style j)."""
        table = self.query_one(DataTable)
        table.action_cursor_down()

    def action_toggle_completed(self) -> None:
        """Toggle display of completed competitions."""
        self.show_completed = not self.show_completed
        self.competitions = self._filter_competitions()

        # Refresh the table
        table = self.query_one(DataTable)
        table.clear()
        self._populate_table()
        if self.competitions:
            table.cursor_coordinate = (0, 0)

        log_widget = self.query_one("#log")
        status = "showing" if self.show_completed else "hiding"
        log_success(log_widget, f"Now {status} completed competitions")

    def _filter_competitions(self) -> list:
        """Filter competitions based on show_completed toggle."""
        if self.show_completed:
            return self.all_competitions
        return [c for c in self.all_competitions if c.state != "completed"]

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

    def action_refresh(self) -> None:
        """Refresh competitions list."""
        log_widget = self.query_one("#log")
        log_success(log_widget, "Refreshing competitions...")
        self.post_message(RefreshCompetitions())

    def refresh_data(self, competitions: list[CompetitionResponse]) -> None:
        """Update the screen with fresh competition data."""
        self.all_competitions = competitions
        self.competitions = self._filter_competitions()

        # Refresh the table
        table = self.query_one(DataTable)
        table.clear()
        self._populate_table()
        if self.competitions:
            table.cursor_coordinate = (0, 0)

        log_widget = self.query_one("#log")
        log_success(log_widget, f"Refreshed {len(self.competitions)} competitions")


class RefreshCompetitions(Message):
    """Message sent when user wants to refresh competitions list."""

    pass


class CompetitionSelected(Message):
    """Message sent when a competition is selected."""

    def __init__(self, competition: CompetitionResponse) -> None:
        self.competition = competition
        super().__init__()
