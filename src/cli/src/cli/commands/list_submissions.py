import asyncio
from datetime import datetime, timezone
import typer
from rich.console import Console
from rich.table import Table

from cli.utils.config import Config
from cli.utils.client import Client
from cli.utils.wallet import load_keypair_from_file
from cli.dashboard.time_utils import format_datetime, get_age
from cli.dashboard.utils import get_state, get_reveal_status, get_top_score_status

console = Console()


def list_submissions(
    competition_id: int = typer.Option(..., "-c", help="Competition ID (required)."),
    mine: bool = typer.Option(False, "-m", help="Show only my submissions."),
    top: bool = typer.Option(False, "-t", help="Show only top-score submissions."),
    page: int = typer.Option(1, "-p", help="Page number (10 results per page)."),
    sort: str = typer.Option("score", "-s", help="Sort by 'score' or 'time'."),
):
    """List submissions for a competition."""
    if sort not in ("score", "time"):
        console.print("[red]Sort must be 'score' or 'time'.[/red]")
        raise typer.Exit(code=1)

    if page < 1:
        console.print("[red]Page must be >= 1.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Fetching submissions for competition {competition_id}...")

    try:
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            raise typer.Exit(code=1)

        filter_mode = "all"
        hotkey_for_filter = None

        if mine:
            filter_mode = "hotkey"
            keypair = load_keypair_from_file(config.hotkey_file_path)
            hotkey_for_filter = keypair.ss58_address
        elif top:
            filter_mode = "top_score"

        start_idx = (page - 1) * 10

        async def _load_all():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                from common.models.api.submission import SubmissionRequest, SubmissionResponse

                req = SubmissionRequest(
                    competition_id=competition_id,
                    hotkey=hotkey_for_filter,
                    start_idx=start_idx,
                    count=10,
                    filter_mode=filter_mode,
                    sort_mode=sort,
                )
                sub_resp = await client._make_request(
                    method="GET",
                    path="/miner/submission",
                    params=req.model_dump(),
                )
                submissions_response = SubmissionResponse.model_validate(sub_resp.json())

                comp_response = await client.list_competitions(id=competition_id)
                comp = comp_response.competitions[0] if comp_response.competitions else None

                return submissions_response, comp

        submissions_response, comp = asyncio.run(_load_all())
        submissions = submissions_response.submissions
        pagination = submissions_response.pagination

        if not submissions:
            console.print("[yellow]No submissions found.[/yellow]")
            return

        # Determine current round info from competition
        current_round = None
        current_round_end_at = None
        top_score_value = None
        top_scorer_hotkey = None
        curr_top_score_id = None
        if comp:
            if comp.curr_round:
                current_round = comp.curr_round.round_number
                current_round_end_at = comp.curr_round.end_at
            else:
                current_round = comp.curr_round_number
            top_score_value = comp.top_score_value
            top_scorer_hotkey = comp.top_scorer_hotkey
            curr_top_score_id = comp.curr_top_score_id

        # Build table matching the dashboard's filtered_submissions columns
        table = Table(show_lines=False)
        table.add_column("ID", justify="right", style="bold")
        table.add_column("Round", justify="right")
        table.add_column("Hotkey")
        table.add_column("Score", justify="right")
        table.add_column("Top")
        table.add_column("Ver")
        table.add_column("Age")
        table.add_column("State")
        table.add_column("Code")
        table.add_column("Log")
        table.add_column("Submit Time")

        for sub in submissions:
            hotkey_display = sub.hotkey[:8]
            if top_scorer_hotkey and sub.hotkey == top_scorer_hotkey:
                hotkey_display = f"[bold green]{hotkey_display}[/bold green]"

            score = f"{sub.eval_score:.7f}" if sub.eval_score is not None else "N/A"
            if top_score_value is not None and sub.eval_score is not None:
                if sub.eval_score >= top_score_value:
                    score = f"[bold green]{score}[/bold green]"
                elif sub.eval_score < top_score_value and sub.top_score:
                    score = f"[bold orange]{score}[/bold orange]"

            top_score = get_top_score_status(sub.top_score, sub.id, curr_top_score_id, compact=True)
            reveal_status = get_reveal_status(sub.reveal_at, compact=True)
            version_str = f"v{sub.version}" if sub.version is not None else "N/A"
            age_str = get_age(sub.submit_at, compact=True)

            log_icon = "👁"
            if current_round is not None and sub.round_number == current_round:
                if current_round_end_at is not None:
                    now = datetime.now(timezone.utc)
                    end_at = current_round_end_at
                    if end_at.tzinfo is None:
                        end_at = end_at.replace(tzinfo=timezone.utc)
                    if now < end_at:
                        log_icon = "🔒"

            table.add_row(
                str(sub.id),
                str(sub.round_number),
                hotkey_display,
                score,
                top_score,
                version_str,
                age_str,
                get_state(sub.state, compact=True),
                reveal_status,
                log_icon,
                format_datetime(sub.submit_at, include_seconds=True),
            )

        # Filter/sort status header
        filter_info = ""
        if mine:
            filter_info = " [dim yellow](Mine Only)[/dim yellow]"
        elif top:
            filter_info = " [dim yellow](Top Scores)[/dim yellow]"
        sort_info = f" [dim cyan](Sorted: {'Score' if sort == 'score' else 'Time'})[/dim cyan]"

        current_page = page
        total_pages = (pagination.total + 9) // 10 if pagination.total > 0 else 1
        page_info = f"Page {current_page} of {total_pages} ({pagination.total} total)"

        console.print(f"\n[bold]Submissions — Competition {competition_id}[/bold]{filter_info}{sort_info}")
        console.print(table)
        console.print(f"[dim]{page_info}[/dim]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Failed to fetch submissions: {e}[/red]")
        raise typer.Exit(code=1)
