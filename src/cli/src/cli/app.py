import asyncio
import typer
from rich.console import Console

from cli.commands.link import link
from cli.commands.submit import submit
from cli.commands.dashboard import dashboard
from cli.commands.version import get_version
from cli.utils.config import Config
from cli.utils.client import Client


console = Console()
app = typer.Typer(help="Apex CLI — interact with competitions and submissions.")
app.command("link")(link)
app.command("submit")(submit)
app.command("dashboard")(dashboard)
app.command("version")(get_version)


@app.command()
def competitions():
    """List available competitions."""
    console.print("🏁 Fetching competitions from the backend...")
    try:
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            return False

        async def _load_competitions():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                from common.models.api.competition import CompetitionRequest, CompetitionResponse

                req = CompetitionRequest()
                response = await client._make_request(
                    method="GET",
                    path="/miner/competition",
                    body=req.model_dump(),
                )
                return CompetitionResponse.model_validate(response.json())

        result = asyncio.run(_load_competitions())
        competitions = result.competitions
        if not competitions:
            console.print("[yellow]No competitions found.[/yellow]")
            return True

        console.print(f"[green]Found {len(competitions)} competitions[/green]")
        for comp in competitions:
            console.print(f"- [bold]{comp.name}[/bold] (id={comp.id}) — state: {comp.state}")
        return True
    except Exception as e:
        console.print(f"[red]Failed to fetch competitions: {e}[/red]")
        return False


if __name__ == "__main__":
    app()
