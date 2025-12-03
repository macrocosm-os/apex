import asyncio
import json
import typer
from pathlib import Path
from typing import Optional

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

from rich.console import Console
from cli.utils.config import Config
from cli.utils.client import Client
from cli.utils.wallet import load_keypair_from_file
from common.models.api.submission import SubmitRequest

console = Console()


def submit(
    file_path: Optional[Path] = typer.Argument(None, help="Path to the solution file"),
    competition_id: Optional[int] = typer.Option(None, "--competition-id", "-c", help="Competition ID"),
):
    """Submit a coding solution to a competition."""
    try:
        # Load configuration
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            return False

        # Get file path if not provided
        if file_path is None:
            try:
                console.print("[blue]Enter path to solution file:[/blue]")
                file_path_str = prompt(
                    "> ",
                    completer=PathCompleter(only_directories=False, expanduser=True),
                    complete_while_typing=True,
                ).strip()

                if not file_path_str:
                    console.print("[yellow]No file path provided. Submission cancelled.[/yellow]")
                    return False
                file_path = Path(file_path_str).expanduser()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Submission cancelled[/yellow]")
                return False

        # Validate file path
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return False

        if not file_path.is_file():
            console.print(f"[red]Path is not a file: {file_path}[/red]")
            return False

        # Read and validate code content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
        except UnicodeDecodeError:
            console.print(f"[red]Unable to read file as text (not UTF-8 encoded): {file_path}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return False

        if not code:
            console.print(f"[red]File is empty: {file_path}[/red]")
            return False

        # Get competition parameters
        if competition_id is None:
            try:
                # Allow showing list on empty input
                while competition_id is None:
                    raw_comp = typer.prompt(
                        "Enter competition ID (press Enter to see list of competitions)",
                        default="",
                        show_default=False,
                    )
                    if raw_comp.strip() == "":
                        # Show competitions
                        async def _list():
                            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                                # Show only active competitions to the user
                                return await client.list_competitions(state="active", start_idx=0, count=50)

                        try:
                            console.print("\n[blue]Fetching competitions...[/blue]")
                            comps = asyncio.run(_list())
                            if comps and comps.competitions:
                                console.print("\n[bold]Available competitions:[/bold]")
                                for c in comps.competitions:
                                    curr_round = c.curr_round_number if c.curr_round_number is not None else "N/A"
                                    console.print(
                                        f"  ID: {c.id} | {c.name} | state={c.state} | round={curr_round} | pkg={c.pkg}"
                                    )
                            else:
                                console.print("[yellow]No competitions available[/yellow]")
                        except Exception as e:
                            console.print(f"[red]Failed to fetch competitions: {e}[/red]")
                        # loop to re-prompt
                        continue
                    try:
                        competition_id = int(raw_comp)
                    except ValueError:
                        console.print("[red]Invalid competition ID[/red]")
                        competition_id = None
            except (KeyboardInterrupt, typer.Abort, typer.Exit):
                console.print("[red]Cancelled[/red]")
                return False

        # Get hotkey from keypair
        try:
            keypair = load_keypair_from_file(config.hotkey_file_path)
            hotkey = keypair.ss58_address
        except Exception as e:
            console.print(f"[red]Error loading hotkey: {e}[/red]")
            return False

        # Create submission request
        submit_request = SubmitRequest(competition_id=competition_id, round_number=-1, raw_code=code)

        # Show submission summary
        console.print("\n[bold]Submission Summary:[/bold]")
        console.print(f"  File: {file_path}")
        console.print(f"  Competition ID: {competition_id}")
        console.print(f"  Hotkey: {hotkey}")
        console.print(f"  Code length: {len(code)} characters")

        if not typer.confirm("Proceed with submission?"):
            console.print("[yellow]Submission cancelled[/yellow]")
            return False

        # Submit solution
        console.print("\n[blue]Submitting solution...[/blue]")

        async def _submit():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                response = await client._make_request(
                    method="POST",
                    path="/miner/submission",
                    body=submit_request.model_dump(),
                )
                return response

        resp = asyncio.run(_submit())
        if resp.status_code == 200:
            console.print("[green]✓ Solution submitted successfully![/green]")
        else:
            console.print(f"\n[red]✗ Submission returned status {resp.status_code}[/red]")
            try:
                console.print(f"[dim]Response: {json.dumps(resp.json(), indent=2)}[/dim]")
            except Exception:
                pass
            return False

    except (KeyboardInterrupt, typer.Abort, typer.Exit):
        console.print("\n[yellow]Submission cancelled by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]Unexpected error during submission: {e}[/red]")
        return False
