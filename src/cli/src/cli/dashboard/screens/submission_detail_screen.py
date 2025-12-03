from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Static, Header, Footer, Log, DataTable, RichLog
from textual.binding import Binding
from textual.message import Message
import os
from rich.console import Console
from rich.json import JSON
from rich.syntax import Syntax

from datetime import datetime, timezone
from common.models.api.submission import SubmissionRecord, SubmissionDetail, FileRequest
from common.models.api.code import CodeRequest
from common.models.api.competition import CompetitionRecord
from cli.dashboard.utils import log_success, log_error, get_state, get_reveal_status, get_top_score_status
from cli.dashboard.time_utils import format_datetime
from cli.utils.client import Client
from cli.utils.config import Config
from cli.dashboard.widgets.download import show_download_dialog

console = Console()


class SubmissionDetailScreen(Screen):
    """Screen showing detailed view of a submission."""

    CSS = """


    .left_panel {
        width: 50%;
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .right_panel {
        width: 50%;
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .submission_container {
        height: 80%;
        margin: 0;
        padding: 0;
    }

    .submission_detail {
        /* Remove height constraint to allow scrolling */
        border: round $primary;
        margin: 0;
        padding: 0 1;
    }

    .file_explorer_container {
        height: 8;
        border: round $secondary;
        margin: 0;
        padding: 1;
    }

    #file_explorer {
        height: 100%;
    }

    .metadata_container {
        height: 100%;
        border: round $secondary;
        margin: 0;
        padding: 0;
    }

    .file_display {
        padding: 1;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("l", "toggle_log", "Toggle Log"),
        Binding("enter", "select_file", "Select File"),
        Binding("d", "download_code", "Download Code"),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    def __init__(self, submission: SubmissionRecord) -> None:
        super().__init__()
        self.submission = submission
        self.submission_detail: SubmissionDetail | None = None
        self.competition: CompetitionRecord | None = None
        self.current_round: int | None = None
        self.current_round_end_at: datetime | None = None
        self.is_loading = True
        self.file_paths = {}  # Store file paths for each file type and name

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()
        with Vertical(id="main_container"):
            with Horizontal(id="content_container"):
                with Vertical(classes="left_panel"):
                    yield ScrollableContainer(Static(classes="submission_detail"), classes="submission_container")
                    yield ScrollableContainer(
                        DataTable(id="file_explorer"),
                        id="file_explorer_container",
                        classes="file_explorer",
                    )
                with Vertical(classes="right_panel"):
                    yield ScrollableContainer(
                        RichLog(classes="file_display", markup=True), classes="metadata_container"
                    )
            yield Log(id="log")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the submission detail view when the screen is mounted."""
        log_widget = self.query_one("#log")
        log_widget.display = False
        log_success(log_widget, f"Viewing submission: {self.submission.id}")

        # Show initial content with loading state
        self.show_submission_detail()
        self.show_loading_state()

        # Load detailed data asynchronously
        self.set_timer(0.1, lambda: self.load_submission_detail_async())

    async def load_submission_detail_async(self) -> None:
        """Load submission detail data asynchronously."""
        try:
            config = Config.load_config()
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                self.submission_detail = await client.get_submission_detail(self.submission.id)

                # Fetch competition data to get current round
                competition_response = await client.list_competitions(id=self.submission.competition_id)
                if competition_response.competitions:
                    self.competition = competition_response.competitions[0]
                    # Determine current round number and end date
                    if self.competition.curr_round:
                        self.current_round = self.competition.curr_round.round_number
                        self.current_round_end_at = self.competition.curr_round.end_at
                    else:
                        self.current_round = self.competition.curr_round_number
                        self.current_round_end_at = None

            # Debug logging
            log_widget = self.query_one("#log")
            log_success(log_widget, f"Loaded detailed data for submission: {self.submission.id}")
            if self.submission_detail and self.submission_detail.eval_file_paths:
                log_success(log_widget, f"Found eval_file_paths: {self.submission_detail.eval_file_paths}")
            else:
                log_error(log_widget, "No eval_file_paths found in submission detail")

            # Update the UI with the loaded data
            self.is_loading = False
            self.setup_file_explorer()
            self.show_metadata()
            # Refresh the submission detail to show Log Reveal
            self.show_submission_detail()

            # Focus the file explorer table
            file_table = self.query_one("#file_explorer", DataTable)
            file_table.focus()
            # Set a timer to ensure focus after mount
            self.set_timer(0.1, lambda: file_table.focus())

        except Exception as e:
            self.is_loading = False
            log_widget = self.query_one("#log")
            log_error(log_widget, f"Failed to load submission detail: {e}")
            console.print(f"[red]Failed to load submission detail: {e}[/red]")

            # Show error state in panels
            self.show_error_state()

    def show_loading_state(self) -> None:
        """Show loading state in the panels."""
        # File explorer loading state
        file_table = self.query_one("#file_explorer", DataTable)
        file_table.clear()
        file_table.add_columns("Type", "Name")
        file_table.add_row("Loading...", "Please wait...")

        # Metadata loading state
        metadata_widget = self.query_one(".file_display", RichLog)
        metadata_widget.clear()
        metadata_widget.write("[dim]Loading metadata...[/dim]")

    def show_error_state(self) -> None:
        """Show error state in the panels."""
        # File explorer error state
        file_table = self.query_one("#file_explorer", DataTable)
        file_table.clear()
        file_table.add_columns("Type", "Name")
        file_table.add_row("Error", "Failed to load files")

        # Metadata error state
        metadata_widget = self.query_one(".file_display", RichLog)
        metadata_widget.clear()
        metadata_widget.write("[red]Failed to load metadata[/red]")

    def show_submission_detail(self) -> None:
        """Show detailed view of the submission."""
        sub = self.submission

        top_scorer_hotkey = self.competition.top_scorer_hotkey if self.competition else None
        top_score = get_top_score_status(sub.top_score, sub.hotkey, top_scorer_hotkey or "")

        error_msg = f"[red]{sub.eval_error}[/red]" if sub.eval_error and not sub.eval_error == "Success" else "None"

        # Determine log reveal icon based on round end date (if available)
        # Lock if round hasn't ended yet, eye if it has ended or if we can't determine
        now = datetime.now(timezone.utc)
        log_status = get_reveal_status(now)
        if self.current_round is not None and sub.round_number == self.current_round:
            # This is the current round, check if it has ended
            if self.current_round_end_at is not None:
                # Ensure end_at is timezone-aware for comparison
                end_at = self.current_round_end_at
                if end_at.tzinfo is None:
                    end_at = end_at.replace(tzinfo=timezone.utc)
                # Lock if round hasn't ended yet
                log_status = f"{format_datetime(end_at, include_seconds=True)} {get_reveal_status(end_at)}"
        version = f"v{sub.version}" if sub.version > 0 else f"v{sub.version} [dim](Auto)[/dim]"

        detail_content = f"""[bold cyan]Submission Details[/bold cyan]

[dim]Submission ID:[/dim] {sub.id}
[dim]Competition ID:[/dim] {sub.competition_id}
[dim]Round Number:[/dim] {sub.round_number}
[dim]State:[/dim] {get_state(sub.state)}
[dim]Hotkey:[/dim] {sub.hotkey}
[dim]Version:[/dim] {version}
[dim]Top Score:[/dim] {top_score}

[bold underline]Scores:[/bold underline]
[dim]Raw Score:[/dim] {sub.eval_raw_score if sub.eval_raw_score else "N/A"}
[dim]Final Score:[/dim] {sub.eval_score if sub.eval_score else "N/A"}
[dim]Evaluation Time:[/dim] {f"{sub.eval_time_in_seconds:.2f}s" if sub.eval_time_in_seconds else "N/A"}
[dim]Evaluation Error:[/dim] {error_msg}

[bold underline]Timeline:[/bold underline]
[dim]Submitted:[/dim] {format_datetime(sub.submit_at, include_seconds=True)}
[dim]Evaluated:[/dim] {format_datetime(sub.eval_at, include_seconds=True)}
[dim]Code Reveal:[/dim] {format_datetime(sub.reveal_at, include_seconds=True)} {get_reveal_status(sub.reveal_at)}
[dim]Log Reveal:[/dim] {log_status}

[dim]Press ESC to go back to submissions list[/dim]
        """

        # Update the detail content
        detail_widget = self.query_one(".submission_detail", Static)
        detail_widget.update(detail_content)

    def setup_file_explorer(self) -> None:
        """Set up the file explorer DataTable."""
        file_table = self.query_one("#file_explorer", DataTable)
        file_table.clear()
        file_table.cursor_type = "row"
        self.file_paths = {}  # Reset file paths

        # Debug logging
        log_widget = self.query_one("#log")
        log_success(log_widget, f"Setting up file explorer for submission {self.submission.id}")

        # Add code file first (from submission detail)
        if self.submission_detail and self.submission_detail.code_path:
            filename = os.path.basename(self.submission_detail.code_path)
            file_table.add_row("Code", filename)
            self.file_paths[("Code", filename)] = self.submission_detail.code_path
            log_success(log_widget, f"Added code file: {filename}")

        # Add eval files from submission detail if available
        if self.submission_detail and self.submission_detail.eval_file_paths:
            log_success(
                log_widget, f"Using submission detail eval_file_paths: {self.submission_detail.eval_file_paths}"
            )
            for file_type, file_paths in self.submission_detail.eval_file_paths.items():
                if isinstance(file_paths, list):
                    for file_path in file_paths:
                        filename = os.path.basename(file_path) if isinstance(file_path, str) else str(file_path)
                        file_table.add_row(file_type.title(), filename)
                        self.file_paths[(file_type.title(), filename)] = file_path
                        log_success(log_widget, f"Added {file_type} file: {filename}")
                elif isinstance(file_paths, str):
                    filename = os.path.basename(file_paths)
                    file_table.add_row(file_type.title(), filename)
                    self.file_paths[(file_type.title(), filename)] = file_paths
                    log_success(log_widget, f"Added {file_type} file: {filename}")
        elif hasattr(self.submission, "eval_file_paths") and self.submission.eval_file_paths:
            # Fallback to original submission data if detail not available
            log_success(log_widget, f"Using fallback submission eval_file_paths: {self.submission.eval_file_paths}")
            for file_type, file_paths in self.submission.eval_file_paths.items():
                if isinstance(file_paths, list):
                    for file_path in file_paths:
                        filename = os.path.basename(file_path) if isinstance(file_path, str) else str(file_path)
                        file_table.add_row(file_type.title(), filename)
                        self.file_paths[(file_type.title(), filename)] = file_path
                        log_success(log_widget, f"Added {file_type} file: {filename}")
                elif isinstance(file_paths, str):
                    filename = os.path.basename(file_paths)
                    file_table.add_row(file_type.title(), filename)
                    self.file_paths[(file_type.title(), filename)] = file_paths
                    log_success(log_widget, f"Added {file_type} file: {filename}")
        else:
            log_error(log_widget, "No eval_file_paths found in submission detail or original submission")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle file selection in the file explorer."""
        table = event.data_table
        row_key = event.row_key
        row_index = table.get_row_index(row_key)

        log_widget = self.query_one("#log")
        log_success(log_widget, f"File row selected: {row_index}")

        # Get the file type and name from the selected row
        if row_index < len(table.rows):
            row_data = table.get_row_at(row_index)
            if len(row_data) >= 2:
                file_type, filename = row_data[0], row_data[1]
                file_key = (file_type, filename)

                if file_key in self.file_paths:
                    file_path = self.file_paths[file_key]
                    log_success(log_widget, f"Selected file: {file_type} - {filename} ({file_path})")
                    # Load and display the file content
                    self.set_timer(0.1, lambda: self.load_file_content(file_type, filename, file_path))
                else:
                    log_error(log_widget, f"File path not found for: {file_type} - {filename}")

    async def load_file_content(self, file_type: str, filename: str, file_path: str) -> None:
        """Load and display file content with syntax highlighting."""
        try:
            log_widget = self.query_one("#log")
            log_success(log_widget, f"Loading file content: {filename}")

            config = Config.load_config()
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                # Code files need to use the code endpoint, not the file endpoint
                if file_type.lower() == "code":
                    code_request = CodeRequest(
                        competition_id=self.submission.competition_id,
                        round_number=self.submission.round_number,
                        hotkey=self.submission.hotkey,
                        version=self.submission.version,
                        start_idx=0,
                    )
                    code_response = await client.get_submission_code(code_request)
                    if code_response:
                        self.display_file_content(filename, code_response.code)
                        log_success(log_widget, f"Loaded code content: {filename}")
                    else:
                        log_error(log_widget, f"Failed to load code content: {filename}")
                        self.show_file_error(filename)
                else:
                    # For non-code files, use the file endpoint
                    file_request = FileRequest(
                        submission_id=self.submission.id,
                        file_type=file_type.lower(),
                        file_name=filename,
                        start_idx=0,
                        reverse=False,
                    )
                    file_data = await client.get_file_chunked(file_request)
                    if file_data:
                        self.display_file_content(filename, file_data.data)
                        log_success(log_widget, f"Loaded file content: {filename}")
                    else:
                        log_error(log_widget, f"Failed to load file content: {filename}")
                        self.show_file_error(filename)

        except Exception as e:
            log_widget = self.query_one("#log")
            log_error(log_widget, f"Error loading file {filename}: {e}")
            self.show_file_error(filename)

    def display_file_content(self, filename: str, content: str) -> None:
        """Display file content with syntax highlighting."""
        metadata_widget = self.query_one(".file_display", RichLog)
        metadata_widget.clear()

        # Determine file extension for syntax highlighting
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in [".py"]:
            # Python syntax highlighting
            syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
            metadata_widget.write(syntax)
        elif file_ext in [".json"]:
            # JSON syntax highlighting
            try:
                import json

                json_data = json.loads(content)
                rich_json = JSON.from_data(json_data, indent=2)
                metadata_widget.write(rich_json)
            except json.JSONDecodeError:
                # If not valid JSON, show as plain text with JSON syntax highlighting
                syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
                metadata_widget.write(syntax)
        else:
            # Plain text display
            metadata_widget.write(f"[bold]File: {filename}[/bold]\n\n")
            metadata_widget.write(content)

    def show_file_error(self, filename: str) -> None:
        """Show error message when file loading fails."""
        metadata_widget = self.query_one(".file_display", RichLog)
        metadata_widget.clear()
        metadata_widget.write(f"[red]Error loading file: {filename}[/red]")

    def show_metadata(self) -> None:
        """Show the evaluation metadata in JSON format with syntax highlighting."""
        metadata = None

        # Use submission detail metadata if available
        if self.submission_detail and self.submission_detail.eval_metadata:
            metadata = self.submission_detail.eval_metadata
        elif hasattr(self.submission, "eval_metadata") and self.submission.eval_metadata:
            # Fallback to original submission data
            metadata = self.submission.eval_metadata

        metadata_widget = self.query_one(".file_display", RichLog)
        metadata_widget.clear()

        if metadata:
            # Create Rich JSON object with syntax highlighting
            rich_json = JSON.from_data(metadata, indent=2)
            metadata_widget.write(rich_json)
        else:
            metadata_widget.write("[dim]Select a file from the file explorer to view its contents[/dim]")

    def action_select_file(self) -> None:
        """Handle Enter key press to select the currently highlighted file."""
        file_table = self.query_one("#file_explorer", DataTable)
        if file_table.cursor_row is not None:
            # Get row index from cursor_row
            row_index = file_table.cursor_row
            # Get row data directly
            if row_index < len(file_table.rows):
                row_data = file_table.get_row_at(row_index)
                if len(row_data) >= 2:
                    file_type, filename = row_data[0], row_data[1]
                    file_key = (file_type, filename)

                    log_widget = self.query_one("#log")
                    if file_key in self.file_paths:
                        file_path = self.file_paths[file_key]
                        log_success(log_widget, f"Selected file: {file_type} - {filename} ({file_path})")
                        # Load and display the file content
                        self.set_timer(0.1, lambda: self.load_file_content(file_type, filename, file_path))
                    else:
                        log_error(log_widget, f"File path not found for: {file_type} - {filename}")

    def action_toggle_log(self) -> None:
        """Toggle the log widget visibility."""
        log_widget = self.query_one("#log")
        if log_widget.display:
            log_widget.display = False
        else:
            log_widget.display = True

    def action_back(self) -> None:
        """Go back to competition detail."""
        self.post_message(BackToCompetitionDetail())

    def action_download_code(self) -> None:
        """Download the code file for this submission."""
        log_widget = self.query_one("#log")
        show_download_dialog(
            screen=self,
            submission=self.submission,
            submission_detail=self.submission_detail,
            log_widget=log_widget,
            notify_callback=self.notify,
        )

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k)."""
        table = self.query_one("#file_explorer", DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down (vim-style j)."""
        table = self.query_one("#file_explorer", DataTable)
        table.action_cursor_down()


class BackToCompetitionDetail(Message):
    """Message sent when user wants to go back to competition detail."""

    pass
