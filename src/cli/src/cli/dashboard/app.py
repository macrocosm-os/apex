from importlib.metadata import version
from rich.console import Console
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer
from textual.binding import Binding

from common.models.api.competition import CompetitionResponse
from common.models.api.submission import SubmissionResponse
from cli.utils.config import Config
from cli.utils.client import Client
from cli.dashboard.screens.competitions_screen import CompetitionsScreen, CompetitionSelected
from cli.dashboard.screens.competition_detail_screen import (
    CompetitionDetailScreen,
    SubmissionSelected,
    BackToCompetitions,
    RefreshCompetitionDetail,
    RefreshCompetitionData,
    LoadSubmissionsPage,
    FilterSortSubmissions,
)
from cli.dashboard.screens.loading_modal import LoadingModal
from cli.dashboard.screens.submission_detail_screen import SubmissionDetailScreen, BackToCompetitionDetail

console = Console()


def get_dashboard_version() -> str:
    """Get the dashboard version from package metadata."""
    try:
        return f"Apex Dashboard v{version('cli')}"
    except Exception:
        # Fallback if version cannot be read
        return "Apex Dashboard v0.0.1"


class DashboardApp(App):
    """Main dashboard application with proper screen management."""

    TITLE = get_dashboard_version()

    CSS = """
    .selected {
        background: $primary;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, competitions: list[CompetitionResponse]) -> None:
        super().__init__()
        self.competitions = competitions
        self.current_competition = None
        self.current_submissions: list[SubmissionResponse] = []

    def compose(self) -> ComposeResult:
        """Compose the main app."""
        yield Header()
        yield Container(id="main_container")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the initial screen when the app is mounted."""
        competitions_screen = CompetitionsScreen(self.competitions)
        self.push_screen(competitions_screen)

    def on_competition_selected(self, event: CompetitionSelected) -> None:
        """Handle competition selection."""
        self.current_competition = event.competition
        # Load submissions asynchronously
        self.set_timer(0.1, self.load_submissions_async)

    async def load_submissions_async(self) -> None:
        """Load submissions asynchronously."""
        if not self.current_competition:
            return

        try:
            submissions_response = await self.load_submissions(self.current_competition.id)
            if submissions_response:
                self.current_submissions = submissions_response.submissions
            else:
                self.current_submissions = []

            # Push the competition detail screen
            pagination = submissions_response.pagination if submissions_response else None
            competition_detail_screen = CompetitionDetailScreen(
                self.current_competition, self.current_submissions, pagination
            )
            self.push_screen(competition_detail_screen)
        except Exception as e:
            console.print(f"[red]Failed to load submissions: {e}[/red]")
            # Still show the competition detail screen without submissions
            competition_detail_screen = CompetitionDetailScreen(self.current_competition, [], None)
            self.push_screen(competition_detail_screen)

    async def load_competition(self, competition_id: int) -> CompetitionResponse | None:
        """Load competition details by ID."""
        try:
            config = Config.load_config()
            if not config.hotkey_file_path:
                console.print(
                    "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
                )
                return None
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                from common.models.api.competition import CompetitionRequest

                # Create competition request
                competition_request = CompetitionRequest()

                response = await client._make_request(
                    method="GET",
                    path="/miner/competition",
                    body=competition_request.model_dump(),
                )
                competitions_response = CompetitionResponse.model_validate(response.json())

                # Find the specific competition by ID
                for comp in competitions_response.competitions:
                    if comp.id == competition_id:
                        return comp
                return None
        except Exception as e:
            console.print(f"[red]Failed to load competition: {e}[/red]")
            return None

    async def load_submissions(
        self,
        competition_id: int,
        start_idx: int = 0,
        count: int = 10,
        filter_mode: str = "all",
        sort_mode: str = "score",
    ) -> SubmissionResponse | None:
        """Load submissions for the selected competition."""
        try:
            config = Config.load_config()
            if not config.hotkey_file_path:
                console.print(
                    "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
                )
                return None
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                from common.models.api.submission import SubmissionRequest
                from cli.utils.wallet import load_keypair_from_file

                # Create query parameters
                keypair = load_keypair_from_file(config.hotkey_file_path)
                req = SubmissionRequest(
                    competition_id=competition_id,
                    hotkey=keypair.ss58_address if filter_mode == "hotkey" else None,
                    start_idx=start_idx,
                    count=count,
                    filter_mode=filter_mode,
                    sort_mode=sort_mode,
                )

                response = await client._make_request(
                    method="GET",
                    path="/miner/submission",
                    params=req.model_dump(),
                )
                return SubmissionResponse.model_validate(response.json())
        except Exception as e:
            console.print(f"[red]Failed to load submissions: {e}[/red]")
            return None

    def on_submission_selected(self, event: SubmissionSelected) -> None:
        """Handle submission selection."""
        submission_detail_screen = SubmissionDetailScreen(event.submission)
        self.push_screen(submission_detail_screen)

    def on_back_to_competitions(self, event: BackToCompetitions) -> None:
        """Handle back to competitions navigation."""
        self.pop_screen()

    def on_back_to_competition_detail(self, event: BackToCompetitionDetail) -> None:
        """Handle back to competition detail navigation."""
        self.pop_screen()

    def on_refresh_competition_detail(self, event: RefreshCompetitionDetail) -> None:
        """Handle refresh competition detail request."""
        # Show loading modal
        loading_modal = LoadingModal("Refreshing competition details and submissions...")
        self.push_screen(loading_modal)

        # Load fresh data asynchronously
        self.set_timer(0.1, lambda: self.refresh_competition_async(event.competition_id))

    async def refresh_competition_async(self, competition_id: int) -> None:
        """Refresh competition details and submissions asynchronously."""
        try:
            # Load fresh competition details
            fresh_competition = await self.load_competition(competition_id)
            if not fresh_competition:
                console.print(f"[red]Failed to refresh competition {competition_id}[/red]")
                # Dismiss loading modal
                self.pop_screen()
                return

            # Get current filter and sort mode from the screen
            current_screen = self.screen
            filter_mode = (
                "hotkey"
                if (isinstance(current_screen, CompetitionDetailScreen) and current_screen.show_only_mine)
                else "all"
            )
            sort_mode = current_screen.sort_mode if isinstance(current_screen, CompetitionDetailScreen) else "score"

            # Load fresh submissions with current filter/sort settings
            submissions_response = await self.load_submissions(
                competition_id, filter_mode=filter_mode, sort_mode=sort_mode
            )
            fresh_submissions = submissions_response.submissions if submissions_response else []

            # Update current data
            self.current_competition = fresh_competition
            self.current_submissions = fresh_submissions

            # Dismiss loading modal
            self.pop_screen()

            # Get the current screen (should be CompetitionDetailScreen) and post message to it
            pagination = submissions_response.pagination if submissions_response else None
            if isinstance(current_screen, CompetitionDetailScreen):
                current_screen.post_message(RefreshCompetitionData(fresh_competition, fresh_submissions, pagination))
            else:
                # Fallback: post to app (though this shouldn't happen)
                self.post_message(RefreshCompetitionData(fresh_competition, fresh_submissions, pagination))

            console.print(
                f"[green]Refreshed competition {competition_id} with {len(fresh_submissions)} submissions[/green]"
            )

        except Exception as e:
            console.print(f"[red]Failed to refresh competition: {e}[/red]")
            # Dismiss loading modal on error
            self.pop_screen()

    def on_load_submissions_page(self, event: LoadSubmissionsPage) -> None:
        """Handle load submissions page request."""
        console.print(
            f"[yellow]DEBUG: on_load_submissions_page called with competition_id={event.competition_id}, start_idx={event.start_idx}[/yellow]"
        )

        # Get the current screen before pushing the modal
        current_screen = self.screen
        console.print(f"[yellow]DEBUG: current_screen type: {type(current_screen)}[/yellow]")

        # Show loading modal
        loading_modal = LoadingModal("Loading submissions page...")
        self.push_screen(loading_modal)

        # Load submissions page asynchronously, passing the screen reference
        self.set_timer(
            0.1, lambda: self.load_submissions_page_async(event.competition_id, event.start_idx, current_screen)
        )

    def on_filter_sort_submissions(self, event: FilterSortSubmissions) -> None:
        """Handle filter/sort submissions request."""
        # Get the current screen before pushing the modal
        current_screen = self.screen

        # Show loading modal
        loading_modal = LoadingModal("Loading filtered/sorted submissions...")
        self.push_screen(loading_modal)

        # Load submissions with filter/sort asynchronously (always start at page 1)
        self.set_timer(
            0.1,
            lambda: self.load_filter_sort_submissions_async(
                event.competition_id, event.filter_mode, event.sort_mode, current_screen
            ),
        )

    async def load_submissions_page_async(self, competition_id: int, start_idx: int, target_screen=None) -> None:
        """Load a specific page of submissions asynchronously."""
        try:
            console.print(
                f"[yellow]DEBUG: load_submissions_page_async called with competition_id={competition_id}, start_idx={start_idx}[/yellow]"
            )

            # Get current filter and sort mode from the screen
            # Use target_screen if provided, otherwise get from screen stack after popping modal
            if target_screen is None:
                # Fallback: dismiss loading modal first to get back to the target screen
                self.pop_screen()
                target_screen = self.screen
            else:
                console.print(f"[yellow]DEBUG: Using provided target_screen: {type(target_screen)}[/yellow]")

            if isinstance(target_screen, CompetitionDetailScreen):
                if target_screen.show_only_mine:
                    filter_mode = "hotkey"
                elif target_screen.show_only_top:
                    filter_mode = "top_score"
                else:
                    filter_mode = "all"
            else:
                filter_mode = "all"
            sort_mode = target_screen.sort_mode if isinstance(target_screen, CompetitionDetailScreen) else "score"
            console.print(f"[yellow]DEBUG: filter_mode={filter_mode}, sort_mode={sort_mode}[/yellow]")

            # Load submissions for the requested page
            submissions_response = await self.load_submissions(
                competition_id, start_idx=start_idx, count=10, filter_mode=filter_mode, sort_mode=sort_mode
            )
            fresh_submissions = submissions_response.submissions if submissions_response else []
            pagination = submissions_response.pagination if submissions_response else None
            console.print(
                f"[yellow]DEBUG: Loaded {len(fresh_submissions)} submissions, pagination={pagination}[/yellow]"
            )

            # Dismiss loading modal (if we haven't already in the fallback case above)
            if target_screen is not None:
                self.pop_screen()

            # Post message to the target screen
            if isinstance(target_screen, CompetitionDetailScreen):
                console.print("[yellow]DEBUG: Posting RefreshCompetitionData to CompetitionDetailScreen[/yellow]")
                target_screen.post_message(
                    RefreshCompetitionData(self.current_competition, fresh_submissions, pagination)
                )
            else:
                console.print("[yellow]DEBUG: Fallback: posting RefreshCompetitionData to app[/yellow]")
                # Fallback: post to app (though this shouldn't happen)
                self.post_message(RefreshCompetitionData(self.current_competition, fresh_submissions, pagination))

            console.print(
                f"[green]Loaded page starting at index {start_idx} with {len(fresh_submissions)} submissions[/green]"
            )

        except Exception as e:
            console.print(f"[red]Failed to load submissions page: {e}[/red]")
            import traceback

            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            # Dismiss loading modal on error
            self.pop_screen()

    async def load_filter_sort_submissions_async(
        self, competition_id: int, filter_mode: str, sort_mode: str, target_screen=None
    ) -> None:
        """Load submissions with filter/sort applied, starting at page 1."""
        try:
            # Always start at page 1 (start_idx=0) when filtering or sorting
            submissions_response = await self.load_submissions(
                competition_id, start_idx=0, count=10, filter_mode=filter_mode, sort_mode=sort_mode
            )
            fresh_submissions = submissions_response.submissions if submissions_response else []
            pagination = submissions_response.pagination if submissions_response else None

            # Dismiss loading modal
            self.pop_screen()

            # Use target_screen if provided, otherwise get from current screen
            if target_screen is None:
                target_screen = self.screen

            # Update the screen's filter/sort state to match what we requested
            if isinstance(target_screen, CompetitionDetailScreen):
                target_screen.show_only_mine = filter_mode == "hotkey"
                target_screen.show_only_top = filter_mode == "top_score"
                target_screen.sort_mode = sort_mode
                target_screen.post_message(
                    RefreshCompetitionData(self.current_competition, fresh_submissions, pagination)
                )
            else:
                # Fallback: post to app (though this shouldn't happen)
                self.post_message(RefreshCompetitionData(self.current_competition, fresh_submissions, pagination))

            filter_text = "filtered" if filter_mode == "hotkey" else "all"
            sort_text = sort_mode
            console.print(
                f"[green]Loaded {len(fresh_submissions)} submissions ({filter_text}, sorted by {sort_text})[/green]"
            )

        except Exception as e:
            console.print(f"[red]Failed to load filtered/sorted submissions: {e}[/red]")
            # Dismiss loading modal on error
            self.pop_screen()


def run_dashboard(competitions: list[CompetitionResponse]) -> None:
    """Run the dashboard application."""
    app = DashboardApp(competitions)
    app.run()
