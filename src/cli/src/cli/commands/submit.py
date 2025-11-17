import asyncio
import json
import typer
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.status import Status
from cli.utils.config import Config
from cli.utils.client import Client
from cli.utils.wallet import load_keypair_from_file
from common.models.api.submission import SubmitRequest

console = Console()


def submit(
    file_path: Optional[Path] = typer.Argument(None, help="Path to the solution file"),
    competition_id: Optional[int] = typer.Option(None, "--competition-id", "-c", help="Competition ID"),
    round_number: Optional[int] = typer.Option(None, "--round", "-r", help="Round number"),
):
    """Submit a coding solution to a competition."""
    try:
        # Load configuration
        config = Config.load_config()
        poll_for_results = False
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            return False

        # Get file path if not provided
        if file_path is None:
            file_path_str = str(typer.prompt("Enter path to solution file")).strip()
            file_path = Path(file_path_str)

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

        if round_number is None:
            try:
                # Allow showing round info on empty input
                while round_number is None:
                    raw_round = typer.prompt(
                        "Enter round number (press Enter to see list of rounds)",
                        default="",
                        show_default=False,
                    )
                    if raw_round.strip() == "":
                        # Show current round info for selected competition
                        async def _get_comp():
                            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                                return await client.list_competitions(id=competition_id, start_idx=0, count=1)

                        try:
                            comp_resp = asyncio.run(_get_comp())
                            if comp_resp and comp_resp.competitions:
                                comp = comp_resp.competitions[0]
                                curr_round = comp.curr_round_number if comp.curr_round_number is not None else "N/A"
                                console.print(
                                    f"\n[bold]Rounds for competition {competition_id}[/bold]\n  Active round: {curr_round} (only active round is available)\n"
                                )
                            else:
                                console.print("[yellow]Competition not found[/yellow]")
                        except Exception as e:
                            console.print(f"[red]Failed to fetch rounds: {e}[/red]")
                        continue
                    try:
                        round_number = int(raw_round)
                    except ValueError:
                        console.print("[red]Invalid round number[/red]")
                        round_number = None
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
        submit_request = SubmitRequest(competition_id=competition_id, round_number=round_number, raw_code=code)

        # Show submission summary
        console.print("\n[bold]Submission Summary:[/bold]")
        console.print(f"  File: {file_path}")
        console.print(f"  Competition ID: {competition_id}")
        console.print(f"  Round: {round_number}")
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
            console.print("\n[blue]Waiting for evaluation results...[/blue]")
            poll_for_results = True
            poll_result = asyncio.run(
                _poll_eval_results(
                    config=config,
                    hotkey=hotkey,
                    competition_id=competition_id,
                    round_number=round_number,
                    timeout_sec=60,
                )
            )
            if poll_result is None:
                console.print("\n[yellow]Timed out waiting for evaluation score[/yellow]")
            return True
        else:
            console.print(f"\n[red]✗ Submission returned status {resp.status_code}[/red]")
            try:
                console.print(f"[dim]Response: {json.dumps(resp.json(), indent=2)}[/dim]")
            except Exception:
                pass
            return False

    except (KeyboardInterrupt, typer.Abort, typer.Exit):
        if not poll_for_results:
            console.print("\n[yellow]Submission cancelled by user[/yellow]")
            return False
        else:
            console.print("\n[yellow]Polling cancelled by user[/yellow]")
            return True
    except Exception as e:
        if not poll_for_results:
            console.print(f"\n[red]Unexpected error during submission: {e}[/red]")
            return False
        else:
            console.print(f"\n[red]Unexpected error during polling: {e}[/red]")
            return True


async def _poll_eval_results(
    config: Config, hotkey: str, competition_id: int, round_number: int, timeout_sec: int = 30
):
    import time
    from common.models.api.submission import SubmissionRequest

    # Start spinner immediately
    spinner = Status("Waiting for submission...", console=console, spinner="dots")
    spinner.start()

    async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
        # Match miner: wait briefly before polling
        await asyncio.sleep(10)
        deadline = time.time() + timeout_sec
        last_state = None
        dot_count = 0
        while time.time() < deadline:
            req = SubmissionRequest(hotkey=hotkey, start_idx=0, count=5)
            params = req.model_dump()
            resp = await client._make_request(
                method="GET",
                path="/miner/submission",
                params=params,
            )
            data = resp.json()
            submissions = data.get("submissions", [])

            match = next(
                (
                    s
                    for s in submissions
                    if s.get("competition_id") == competition_id and s.get("round_number") == round_number
                ),
                None,
            )
            if match:
                state = match.get("state")
                if state != last_state:
                    last_state = state
                    dot_count = 0
                    spinner.update(f"Submission state: {state}")
                else:
                    dot_count += 1
                    dots = "." * dot_count
                    spinner.update(f"Submission state: {state}{dots}")

                eval_error = match.get("eval_error")
                eval_score = match.get("eval_score")
                if eval_error and eval_error != "Success":
                    spinner.stop()
                    console.print(f"\n[red]Evaluation error: {eval_error}[/red]")
                    return False
                if eval_score is not None:
                    spinner.stop()
                    if isinstance(eval_score, (float, int)):
                        eval_score = f"{eval_score:.2f}"
                    console.print(f"\n[green]Evaluation score: {eval_score}[/green]")
                    return True

            await asyncio.sleep(5)

        # Stop spinner if still running
        spinner.stop()
        return None
