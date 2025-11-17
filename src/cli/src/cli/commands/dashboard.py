import asyncio
from rich.console import Console

from cli.utils.config import Config
from cli.utils.client import Client
from common.models.api.competition import CompetitionRequest, CompetitionResponse

from cli.dashboard.app import run_dashboard

console = Console()


def dashboard():
    """Show competitions dashboard."""
    try:
        # Load configuration
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            return False

        console.print("[blue]Loading competitions dashboard...[/blue]")

        async def _load_competitions():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                # Create competition request
                competition_request = CompetitionRequest()

                response = await client._make_request(
                    method="GET",
                    path="/miner/competition",
                    body=competition_request.model_dump(),
                )
                return CompetitionResponse.model_validate(response.json())

        try:
            result = asyncio.run(_load_competitions())
            competitions = result.competitions

            if not competitions:
                console.print("[yellow]No competitions found.[/yellow]")
                return True

            console.print(f"[green]Found {len(competitions)} competitions[/green]")

            # Create and run the dashboard
            run_dashboard(competitions)

            return True

        except Exception as e:
            console.print(f"[red]Failed to load competitions: {e}[/red]")
            return False

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard cancelled by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return False
