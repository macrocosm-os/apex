from datetime import datetime, timezone
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import DataTable, Static, Header, Footer, Log
from textual.binding import Binding
from textual.message import Message

from common.models.api.competition import CompetitionRecord
from common.models.api.submission import SubmissionRecord, SubmissionPagination
from cli.dashboard.utils import log_success, log_debug, get_state, get_reveal_status, get_top_score_status
from cli.dashboard.time_utils import format_datetime, get_age
from cli.dashboard.widgets.round_details import RoundDetailsWidget
from cli.utils.config import Config
from cli.utils.wallet import load_keypair_from_file


class CompetitionDetailScreen(Screen):
    """Screen showing detailed view of a competition with submissions."""

    CSS = """
    .competition-detail {
        height: 100%;
        padding: 1;
    }

    .detail-title {
        text-style: bold;
        color: $primary;
    }

    .header-section {
        height: 40%;
        margin: 1;
        padding: 0;
    }

    .competition-section {
        height: 100%;
        border: round $primary;
        margin: 0;
        padding: 0 1;
    }

    .round-section {
        height: 70%;
        border: round $secondary;
        margin: 0;
        padding: 0 1;
    }

    .top-score-section {
        height: 30%;
        border: round $secondary;
        margin: 0;
        padding: 0 1;
    }

    .top-score-left, .top-score-right {
        margin: 0;
        padding: 0;
    }

    .submissions-section {
        max-height: 60%;
        border: round $secondary;
        margin: 1;
        padding: 1;
    }

    .submissions-title {
        height: 2;
        margin-bottom: 1;
    }

    .submissions-table {
        /* Table will size to content, no fixed height */
    }

    .pagination-info {
        height: 1;
        margin-top: 1;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("enter", "select_submission", "Select Submission"),
        Binding("l", "toggle_log", "Toggle Log"),
        Binding("r", "refresh", "Refresh"),
        Binding("m", "toggle_filter", "Filter Mine"),
        Binding("t", "filter_top", "Filter Top"),
        Binding("s", "toggle_sort", "Sort"),
        Binding("n", "next_page", "Next Page"),
        Binding("p", "prev_page", "Previous Page"),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    def __init__(
        self,
        competition: CompetitionRecord,
        submissions: list[SubmissionRecord] | None = None,
        pagination: SubmissionPagination = None,
    ) -> None:
        super().__init__()
        self.competition = competition
        self.submissions = submissions or []
        self.pagination = pagination
        self.selected_submission_index = 0
        self.round_widget = None
        self.show_only_mine = False
        self.show_only_top = False
        self.user_hotkey = None
        self.sort_mode = "score"  # "score" or "time"

        # Load user's hotkey from config
        try:
            config = Config.load_config()
            if config.hotkey_file_path:
                keypair = load_keypair_from_file(config.hotkey_file_path)
                self.user_hotkey = keypair.ss58_address
        except Exception:
            # If we can't load the hotkey, filter won't work but app should still function
            pass

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()
        with Vertical(id="main_container"):
            yield Container(id="content_container")
            yield Log(id="log")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the competition detail view when the screen is mounted."""
        log_widget = self.query_one("#log")
        log_widget.display = False
        log_success(log_widget, f"Viewing competition: {self.competition.name}")
        self.show_competition_detail()

    def show_competition_detail(self) -> None:
        """Show detailed view of the competition with submissions."""
        comp = self.competition

        # Competition header content
        competition_content = f"""[bold cyan]{comp.name}[/bold cyan] - [bold]ID:[/bold] {comp.id}

[dim]Description:[/dim] {comp.description}
[dim]State:[/dim] {get_state(comp.state)}
[dim]Package:[/dim] {comp.pkg}
[dim]Process Type:[/dim] {comp.ptype}
[dim]Competition Type:[/dim] {comp.ctype}
[dim]Baseline Score:[/dim] {comp.baseline_score}
[dim]Baseline Raw Score:[/dim] {comp.baseline_raw_score}
[dim]Incentive Weight:[/dim] {comp.incentive_weight}

[bold underline]Timeline:[/bold underline]
[dim]Start:[/dim] {format_datetime(comp.start_at, include_seconds=True) if comp.start_at else "Not started"}
[dim]End:[/dim] {format_datetime(comp.end_at, include_seconds=True) if comp.end_at else "No end date"}
[dim]Created:[/dim] {format_datetime(comp.created_at, include_seconds=True)}"""

        # Create round details widget
        round_data = {}
        if comp.curr_round:
            round_data = {
                "round_number": comp.curr_round.round_number,
                "state": comp.curr_round.state,
                "start_at": comp.curr_round.start_at,
                "end_at": comp.curr_round.end_at,
                "burn_factor": comp.burn_factor,
            }
        else:
            round_data = {
                "round_number": comp.curr_round_number,
                "state": "unknown",
                "start_at": None,
                "end_at": None,
                "burn_factor": comp.burn_factor,
            }

        self.round_widget = RoundDetailsWidget(round_data)

        # The RoundDetailsWidget now handles its own timer updates

        # Find top scorer's submission to get submit_at time
        top_scorer_submission = None
        if comp.top_scorer_hotkey:
            for sub in self.submissions:
                if sub.hotkey == comp.top_scorer_hotkey and sub.top_score:
                    top_scorer_submission = sub
                    break
            # If not found with top_score match, just find by hotkey
            if top_scorer_submission is None:
                for sub in self.submissions:
                    if sub.hotkey == comp.top_scorer_hotkey:
                        top_scorer_submission = sub
                        break

        # Format top score with 0.8 precision
        top_score_str = f"{comp.top_score_value:.7f}" if comp.top_score_value is not None else "N/A"

        # Format score_to_beat with 7 decimal places
        score_to_beat_str = f"{comp.score_to_beat:.7f}" if comp.score_to_beat is not None else "N/A"

        # Format top scorer hotkey (first 8 chars only)
        top_scorer_str = comp.top_scorer_hotkey[:8] if comp.top_scorer_hotkey else "N/A"

        # Calculate age of top scorer
        if top_scorer_submission and top_scorer_submission.submit_at:
            top_scorer_age = get_age(top_scorer_submission.submit_at, include_seconds=True)
        else:
            top_scorer_age = "N/A"

        # Calculate % Emissions (1 - burn_factor)
        if comp.burn_factor is not None:
            emissions_percent = (1 - comp.burn_factor) * 100
            emissions_str = f"{emissions_percent:.1f}%"
        else:
            emissions_str = "0.0%"

        # Left container content
        top_score_left_content = f"""[dim]Top Scorer:[/dim] {top_scorer_str}
[dim]Top Score:[/dim] {top_score_str}
[dim]Score to Beat:[/dim] {score_to_beat_str}"""

        # Right container content
        top_score_right_content = f"""[dim]Age:[/dim] {top_scorer_age}
[dim]% Emissions:[/dim] {emissions_str}"""

        # Replace the content container with competition details and submissions
        container = self.query_one("#content_container")
        container.remove_children()

        # Create all content first
        competition_section = ScrollableContainer(Static(competition_content), classes="competition-section")
        round_section = ScrollableContainer(self.round_widget, classes="round-section")

        # Create two horizontal containers for top score section
        top_score_left_cntr = ScrollableContainer(Static(top_score_left_content), classes="top-score-left")
        top_score_right_cntr = ScrollableContainer(Static(top_score_right_content), classes="top-score-right")
        top_score_section = Horizontal(top_score_left_cntr, top_score_right_cntr, classes="top-score-section")
        header_section = Horizontal(
            competition_section, Vertical(round_section, top_score_section), classes="header-section"
        )

        # Submissions section
        # Submissions are already filtered and sorted by the API, so use them directly
        filtered_submissions = self.submissions

        # Determine current round number and end date
        current_round = comp.curr_round.round_number if comp.curr_round else comp.curr_round_number
        current_round_end_at = comp.curr_round.end_at if comp.curr_round else None

        if filtered_submissions:
            submissions_table = DataTable(id="submissions_table", classes="submissions-table")
            submissions_table.add_columns(
                "ID", "Round", "Hotkey", "Score", "Top", "Version", "Age", "State", "Code", "Log", "Submit Time"
            )

            for sub in filtered_submissions:
                hotkey = (
                    f"[bold green]{sub.hotkey[:8]}[/bold green]"
                    if sub.hotkey == comp.top_scorer_hotkey
                    else sub.hotkey[:8]
                )
                score = f"{sub.eval_score:.7f}" if sub.eval_score else "N/A"
                if comp.top_score_value is not None and sub.eval_score is not None:
                    if sub.eval_score >= comp.top_score_value:
                        score = f"[bold green]{score}[/bold green]"
                    elif sub.eval_score < comp.top_score_value and sub.top_score:
                        score = f"[bold orange]{score}[/bold orange]"

                top_score = get_top_score_status(sub.top_score, sub.id, comp.curr_top_score_id, compact=True)

                # Get reveal status emoji and text
                reveal_status = get_reveal_status(sub.reveal_at, compact=True)

                # Format version as v1, v2, etc.
                version_str = f"v{sub.version}" if sub.version is not None else "N/A"

                # Calculate age using compact format
                age_str = get_age(sub.submit_at, compact=True)

                # Determine log icon based on round end date (if available)
                # Lock if round hasn't ended yet, eye if it has ended or if we can't determine
                log_icon = "👁"  # Default to eye (revealed)
                if sub.round_number == current_round:
                    # This is the current round, check if it has ended
                    if current_round_end_at is not None:
                        now = datetime.now(timezone.utc)
                        # Ensure end_at is timezone-aware for comparison
                        end_at = current_round_end_at
                        if end_at.tzinfo is None:
                            end_at = end_at.replace(tzinfo=timezone.utc)
                        # Lock if round hasn't ended yet
                        if now < end_at:
                            log_icon = "🔒"
                        else:
                            log_icon = "👁"
                    # If end_at is not available, default to eye (assuming we can't determine)
                # If submission round != current round, assume it's a past round (already ended)

                submissions_table.add_row(
                    str(sub.id),
                    str(sub.round_number),
                    hotkey,
                    score,
                    top_score,
                    version_str,
                    age_str,
                    get_state(sub.state, compact=True),
                    reveal_status,
                    log_icon,
                    format_datetime(sub.submit_at, include_seconds=True),
                )

            submissions_table.cursor_type = "row"

            # Ensure the table is properly focused and cursor is at row 0
            submissions_table.focus()

            # Set a timer to focus the table after it's mounted
            self.set_timer(0.1, lambda: submissions_table.focus())

            # Update submissions title to show filter and sort status
            if self.show_only_mine:
                filter_status = " [dim yellow](Filtered: Mine Only)[/dim yellow]"
            elif self.show_only_top:
                filter_status = " [dim yellow](Filtered: Top Scores)[/dim yellow]"
            else:
                filter_status = ""
            sort_status = f" [dim cyan](Sorted: {'Score' if self.sort_mode == 'score' else 'Time'})[/dim cyan]"

            # Calculate pagination info
            pagination_info = ""
            if self.pagination:
                current_page = (self.pagination.start_idx // self.pagination.count) + 1
                total_pages = (
                    (self.pagination.total + self.pagination.count - 1) // self.pagination.count
                    if self.pagination.total > 0
                    else 1
                )
                pagination_info = f" [dim]Page {current_page} of {total_pages}[/dim]"

            # Build container children
            container_children = [
                Static(
                    f"[bold]Submissions[/bold]{filter_status}{sort_status} [dim]Use arrow keys to navigate, Enter to select[/dim]",
                    classes="submissions-title",
                ),
                submissions_table,
            ]
            if pagination_info:
                container_children.append(Static(pagination_info, classes="pagination-info"))

            detail_section = ScrollableContainer(
                *container_children,
                classes="submissions-section",
            )
        else:
            # Show appropriate message based on filter and sort state
            if self.show_only_mine:
                filter_status = " [dim yellow](Filtered: Mine Only)[/dim yellow]"
            elif self.show_only_top:
                filter_status = " [dim yellow](Filtered: Top Score)[/dim yellow]"
            else:
                filter_status = ""
            sort_status = f" [dim cyan](Sorted: {'Score' if self.sort_mode == 'score' else 'Time'})[/dim cyan]"
            if self.show_only_mine and self.user_hotkey:
                message = f"[bold]Submissions[/bold]{filter_status}{sort_status}\n[yellow]No submissions found matching your hotkey[/yellow]"
            elif self.show_only_top:
                message = f"[bold]Submissions[/bold]{filter_status}{sort_status}\n[yellow]No top score submissions found[/yellow]"
            else:
                message = (
                    f"[bold]Submissions[/bold]{sort_status}\n[yellow]No submissions found for this competition[/yellow]"
                )
            detail_section = ScrollableContainer(
                Static(
                    message,
                    classes="submissions-section",
                ),
                classes="submissions-section",
            )

        # Create layout with all content
        layout = Vertical(header_section, detail_section)

        # Mount the complete layout to the main container
        container.mount(layout)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        table = event.data_table
        row_key = event.row_key
        # Get the row index from the row key
        row_index = table.get_row_index(row_key)
        log_widget = self.query_one("#log")
        log_debug(log_widget, f"Row selected: {row_index}")

        # Submissions are already filtered and sorted by the API
        filtered_submissions = self.submissions

        # Add this line to also trigger the selection logic
        if 0 <= row_index < len(filtered_submissions):
            selected_submission = filtered_submissions[row_index]
            log_success(log_widget, f"Selected submission: {selected_submission.id}")
            self.post_message(SubmissionSelected(selected_submission))

    def action_toggle_log(self) -> None:
        """Toggle the log widget visibility."""
        log_widget = self.query_one("#log")
        if log_widget.display:
            log_widget.display = False
        else:
            log_widget.display = True

    def action_back(self) -> None:
        """Go back to competitions list."""
        self.post_message(BackToCompetitions())

    def action_refresh(self) -> None:
        """Refresh competition details and submissions."""
        log_widget = self.query_one("#log")
        log_success(log_widget, "Refreshing competition details...")
        self.post_message(RefreshCompetitionDetail(self.competition.id))

    def action_toggle_filter(self) -> None:
        """Toggle filter to show only user's submissions."""
        if not self.user_hotkey:
            log_widget = self.query_one("#log")
            log_widget.display = True
            log_debug(log_widget, "Cannot filter: No hotkey found. Please run 'apex link' to link your wallet.")
            return

        # Toggle the filter state
        self.show_only_mine = not self.show_only_mine
        # Reset top filter when toggling mine filter
        self.show_only_top = False

        # Determine filter_mode for API call
        filter_mode = "hotkey" if self.show_only_mine else "all"

        log_widget = self.query_one("#log")
        filter_status = "enabled" if self.show_only_mine else "disabled"
        log_success(
            log_widget,
            f"Filter {'enabled' if self.show_only_mine else 'disabled'}: Showing {'only your' if self.show_only_mine else 'all'} submissions",
        )

        # Trigger API call with filter_mode and current sort_mode, resetting to page 1
        self.app.post_message(FilterSortSubmissions(self.competition.id, filter_mode, self.sort_mode))

    def action_filter_top(self) -> None:
        """Toggle filter to show only top score submissions."""
        # Toggle the filter state
        self.show_only_top = not self.show_only_top
        # Reset mine filter when toggling top filter
        self.show_only_mine = False

        # Determine filter_mode for API call
        filter_mode = "top_score" if self.show_only_top else "all"

        log_widget = self.query_one("#log")
        filter_status = "enabled" if self.show_only_top else "disabled"
        log_success(
            log_widget,
            f"Top filter {'enabled' if self.show_only_top else 'disabled'}: Showing {'only top score' if self.show_only_top else 'all'} submissions",
        )

        # Trigger API call with filter_mode and current sort_mode, resetting to page 1
        self.app.post_message(FilterSortSubmissions(self.competition.id, filter_mode, self.sort_mode))

    def _sort_submissions(self, submissions: list[SubmissionRecord]) -> list[SubmissionRecord]:
        """Sort submissions based on the current sort mode.

        Args:
            submissions: List of submissions to sort

        Returns:
            Sorted list of submissions
        """
        if self.sort_mode == "score":
            # Sort by score (descending), then by submit time (most recent first)
            def sort_key(sub: SubmissionRecord) -> tuple:
                # Use a large negative number for None scores so they sort last
                score = sub.eval_score if sub.eval_score is not None else float("-inf")
                # Use epoch time for comparison, None becomes very old timestamp (sorts last)
                submit_time = sub.submit_at.timestamp() if sub.submit_at else float("-inf")
                # Negate score for descending order, negate time for descending (most recent first)
                # None scores become inf (sorts last), None times become inf (sorts last)
                return (-score, -submit_time)

        else:  # sort_mode == "time"
            # Sort by submit time (most recent first)
            def sort_key(sub: SubmissionRecord) -> float:
                # Use epoch time for comparison, None becomes very old timestamp (sorts last)
                submit_time = sub.submit_at.timestamp() if sub.submit_at else float("-inf")
                # Negate for descending order (most recent first)
                # None times become inf (sorts last)
                return -submit_time

        return sorted(submissions, key=sort_key)

    def action_toggle_sort(self) -> None:
        """Toggle sort mode between score and time."""
        self.sort_mode = "time" if self.sort_mode == "score" else "score"
        log_widget = self.query_one("#log")
        sort_mode_name = "Score" if self.sort_mode == "score" else "Time"
        log_success(log_widget, f"Sort mode changed to: {sort_mode_name}")

        # Determine filter_mode for API call
        if self.show_only_mine:
            filter_mode = "hotkey"
        elif self.show_only_top:
            filter_mode = "top_score"
        else:
            filter_mode = "all"

        # Trigger API call with sort_mode and current filter_mode, resetting to page 1
        self.post_message(FilterSortSubmissions(self.competition.id, filter_mode, self.sort_mode))

    def refresh_data(
        self,
        competition: CompetitionRecord,
        submissions: list[SubmissionRecord],
        pagination: SubmissionPagination = None,
    ) -> None:
        """Update the screen with fresh competition and submission data."""
        self.competition = competition
        self.submissions = submissions or []
        self.pagination = pagination
        self.show_competition_detail()
        # The RoundDetailsWidget will automatically update with new data

    def on_refresh_competition_data(self, event: "RefreshCompetitionData") -> None:
        """Handle refresh competition data message."""
        self.refresh_data(event.competition, event.submissions, event.pagination)

    def action_next_page(self) -> None:
        """Load the next page of submissions."""
        if not self.pagination or not self.pagination.has_more:
            log_widget = self.query_one("#log")
            log_debug(log_widget, "No next page available")
            return

        log_widget = self.query_one("#log")
        log_success(log_widget, "Loading next page...")
        # Post message to app level to ensure it's handled
        self.app.post_message(
            LoadSubmissionsPage(self.competition.id, self.pagination.start_idx + self.pagination.count)
        )

    def action_prev_page(self) -> None:
        """Load the previous page of submissions."""
        if not self.pagination or self.pagination.start_idx == 0:
            log_widget = self.query_one("#log")
            log_debug(log_widget, "No previous page available")
            return

        log_widget = self.query_one("#log")
        log_success(log_widget, "Loading previous page...")
        new_start_idx = max(0, self.pagination.start_idx - self.pagination.count)
        # Post message to app level to ensure it's handled
        self.app.post_message(LoadSubmissionsPage(self.competition.id, new_start_idx))

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k)."""
        table = self.query_one("#submissions_table", DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down (vim-style j)."""
        table = self.query_one("#submissions_table", DataTable)
        table.action_cursor_down()


class SubmissionSelected(Message):
    """Message sent when a submission is selected."""

    def __init__(self, submission: SubmissionRecord) -> None:
        self.submission = submission
        super().__init__()


class BackToCompetitions(Message):
    """Message sent when user wants to go back to competitions list."""

    pass


class RefreshCompetitionDetail(Message):
    """Message sent when user wants to refresh competition details."""

    def __init__(self, competition_id: int) -> None:
        self.competition_id = competition_id
        super().__init__()


class RefreshCompetitionData(Message):
    """Message sent to update competition detail screen with fresh data."""

    def __init__(
        self,
        competition: CompetitionRecord,
        submissions: list[SubmissionRecord],
        pagination: SubmissionPagination = None,
    ) -> None:
        self.competition = competition
        self.submissions = submissions
        self.pagination = pagination
        super().__init__()


class LoadSubmissionsPage(Message):
    """Message sent to load a specific page of submissions."""

    def __init__(self, competition_id: int, start_idx: int) -> None:
        self.competition_id = competition_id
        self.start_idx = start_idx
        super().__init__()


class FilterSortSubmissions(Message):
    """Message sent to filter/sort submissions with API call."""

    def __init__(self, competition_id: int, filter_mode: str, sort_mode: str) -> None:
        self.competition_id = competition_id
        self.filter_mode = filter_mode
        self.sort_mode = sort_mode
        super().__init__()
