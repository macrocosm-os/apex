import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.utils.config import Config
from cli.utils.client import Client
from cli.dashboard.time_utils import format_datetime, get_age, get_round_progress, get_round_countdown
from cli.dashboard.utils import get_state

console = Console()


def competitions(
    competition_id: int = typer.Option(None, "-c", help="Show details for a specific competition ID."),
    show_all: bool = typer.Option(False, "-a", "--all", help="Include completed competitions in the list."),
    page: int = typer.Option(1, "-p", "--page", help="Page number (10 results per page)."),
):
    """List available competitions, or show details for a specific one."""
    if page < 1:
        console.print("[red]Page must be >= 1.[/red]")
        raise typer.Exit(code=1)

    console.print("Fetching competitions...")
    try:
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            raise typer.Exit(code=1)

        start_idx = (page - 1) * 10

        async def _load():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                return await client.list_competitions(
                    id=competition_id,
                    show_completed=show_all,
                    start_idx=start_idx,
                    count=10,
                )

        result = asyncio.run(_load())
        competitions_list = result.competitions

        if not competitions_list:
            if competition_id is not None:
                console.print(f"[yellow]Competition {competition_id} not found.[/yellow]")
            elif page > 1:
                console.print(f"[yellow]No competitions found on page {page}.[/yellow]")
            elif not show_all:
                console.print(
                    "[yellow]No active competitions found.[/yellow] "
                    "[dim]Run[/dim] [green]apex competitions --all[/green] [dim]to include completed ones.[/dim]"
                )
            else:
                console.print("[yellow]No competitions found.[/yellow]")
            return

        if competition_id is not None:
            _show_competition_detail(competitions_list[0])
        else:
            _show_competitions_list(competitions_list, page=page, pagination=result.pagination)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Failed to fetch competitions: {e}[/red]")
        raise typer.Exit(code=1)


def _show_competitions_list(competitions_list, page=1, pagination=None):
    """Print a summary table of competitions for the current page."""
    table = Table(title="Competitions", show_lines=True)
    table.add_column("ID", style="bold", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("State")
    table.add_column("Round")
    table.add_column("Top Score", justify="right")
    table.add_column("Submissions", justify="right")

    for comp in competitions_list:
        round_str = str(comp.curr_round_number) if comp.curr_round_number is not None else "N/A"
        top_score_str = f"{comp.top_score_value:.7f}" if comp.top_score_value is not None else "N/A"
        table.add_row(
            str(comp.id),
            comp.name,
            get_state(comp.state),
            round_str,
            top_score_str,
            str(comp.total_submissions),
        )

    console.print(table)

    if pagination is not None and pagination.total > 0:
        total_pages = (pagination.total + 9) // 10
        console.print(f"[dim]Page {page} of {total_pages} ({pagination.total} total)[/dim]")

    console.print(
        "\n[dim]Run[/dim] [green]apex competitions -c <ID>[/green] [dim]for details on a specific competition.[/dim]"
    )
    if pagination is not None and pagination.has_more:
        console.print(f"[dim]Run[/dim] [green]apex competitions -p {page + 1}[/green] [dim]for the next page.[/dim]")
    console.print("[dim]Run[/dim] [green]apex competitions --all[/green] [dim]to include completed competitions.[/dim]")


def _show_competition_detail(comp):
    """Print detailed competition info mirroring the dashboard's competition_content, round_data, and top_score_section."""

    # --- Competition content ---
    competition_lines = [
        f"[bold cyan]{comp.name}[/bold cyan] — ID: {comp.id}",
        comp.description,
        "",
        f"[dim]State:[/dim]           {get_state(comp.state)}",
        f"[dim]Package:[/dim]         {comp.pkg}",
        f"[dim]Process Type:[/dim]    {comp.ptype}",
        f"[dim]Competition Type:[/dim]{comp.ctype}",
        f"[dim]Baseline Score:[/dim]  {comp.baseline_score}",
        f"[dim]Baseline Raw:[/dim]    {comp.baseline_raw_score}",
        "",
        "[bold underline]Timeline[/bold underline]",
        f"[dim]Created:[/dim]  {format_datetime(comp.created_at, include_seconds=True)}",
        f"[dim]Started:[/dim]  {format_datetime(comp.start_at, include_seconds=True) if comp.start_at else 'Not started'}",
        f"[dim]Ended:[/dim]    {format_datetime(comp.end_at, include_seconds=True) if comp.end_at else 'No end date'}",
    ]
    console.print(Panel("\n".join(competition_lines), title="Competition", border_style="cyan"))

    # --- Round data ---
    if comp.curr_round:
        rnd = comp.curr_round
        round_number = rnd.round_number
        state_display = get_state(rnd.state)
        start_str = format_datetime(rnd.start_at)
        end_str = format_datetime(rnd.end_at)
        progress, _ = get_round_progress(rnd.start_at, rnd.end_at)
        countdown = get_round_countdown(rnd.end_at)
    else:
        round_number = comp.curr_round_number if comp.curr_round_number is not None else "N/A"
        state_display = get_state("unknown")
        start_str = "N/A"
        end_str = "N/A"
        progress = 0.0
        countdown = None

    burn_str = ""
    if comp.burn_factor is not None:
        burn_pct = comp.burn_factor * 100
        min_burn = f" (min burn: {comp.base_burn_rate * 100:.0f}%)" if comp.base_burn_rate is not None else ""
        burn_str = f"\n[dim]Burn:[/dim]      {burn_pct:.1f}% {min_burn}"

    round_lines = [
        f"[bold]Round #{round_number}[/bold] — {state_display}",
        f"[dim]Start:[/dim]  {start_str}    [dim]End:[/dim]  {end_str}",
        f"[dim]Progress:[/dim] {progress:.1f}%",
    ]
    if countdown:
        round_lines.append(f"[bold green]Time Remaining:[/bold green] [green]{countdown}[/green]")
    if burn_str:
        round_lines.append(burn_str)

    console.print(Panel("\n".join(round_lines), title="Current Round", border_style="green"))

    # --- Top score section ---
    top_score_str = f"{comp.top_score_value:.7f}" if comp.top_score_value is not None else "N/A"
    score_to_beat_str = f"{comp.score_to_beat:.7f}" if comp.score_to_beat is not None else "N/A"
    top_scorer_str = comp.top_scorer_hotkey[:8] if comp.top_scorer_hotkey else "N/A"

    if comp.burn_factor_reset_at:
        top_scorer_age = get_age(comp.burn_factor_reset_at, include_seconds=True)
    else:
        top_scorer_age = "N/A"

    if comp.burn_factor is not None and comp.state == "active":
        emissions_percent = (1 - comp.burn_factor) * comp.incentive_weight * 100
        emissions_str = f"{emissions_percent:.1f}%"
    else:
        emissions_str = "0.0%"

    score_lines = [
        f"[dim]Top Scorer:[/dim]    {top_scorer_str}",
        f"[dim]Top Score:[/dim]     {top_score_str}",
        f"[dim]Score to Beat:[/dim] {score_to_beat_str}",
        f"[dim]Age:[/dim]           {top_scorer_age}",
        f"[dim]% Emissions:[/dim]   {emissions_str}",
    ]
    console.print(Panel("\n".join(score_lines), title="Top Score", border_style="yellow"))
