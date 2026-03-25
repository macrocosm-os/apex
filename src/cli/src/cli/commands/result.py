import asyncio
import json
import os
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.json import JSON as RichJSON
from rich.table import Table

from cli.utils.config import Config
from cli.utils.client import Client
from cli.dashboard.time_utils import format_datetime
from cli.dashboard.utils import get_state, get_reveal_status, get_top_score_status

console = Console()


def result(
    submission_id: int = typer.Argument(..., help="Submission ID to view."),
    file: str = typer.Option(None, "-f", help="View a specific file by name. Use '-f list' to list available files."),
):
    """Show details for a submission result."""
    console.print(f"Fetching submission {submission_id}...")

    try:
        config = Config.load_config()
        if not config.hotkey_file_path:
            console.print(
                "[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]"
            )
            raise typer.Exit(code=1)

        async def _load():
            async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
                from common.models.api.submission import SubmissionRequest, SubmissionResponse

                req = SubmissionRequest(submission_id=submission_id)
                resp = await client._make_request(
                    method="GET",
                    path="/miner/submission",
                    params=req.model_dump(),
                )
                sub_resp = SubmissionResponse.model_validate(resp.json())
                if not sub_resp.submissions:
                    return None, None, None

                submission = sub_resp.submissions[0]

                detail = await client.get_submission_detail(submission_id)

                comp_response = await client.list_competitions(id=submission.competition_id)
                comp = comp_response.competitions[0] if comp_response.competitions else None

                return submission, detail, comp

        submission, detail, comp = asyncio.run(_load())

        if submission is None:
            console.print(f"[red]Submission {submission_id} not found.[/red]")
            raise typer.Exit(code=1)

        if file is not None:
            _handle_file_option(file, submission, detail, config)
        else:
            _show_submission_detail(submission, detail, comp)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Failed to fetch submission: {e}[/red]")
        raise typer.Exit(code=1)


def _show_submission_detail(submission, detail, comp):
    """Print submission detail_content mirroring the dashboard."""
    from datetime import datetime, timezone

    sub = submission
    curr_top_score_id = comp.curr_top_score_id if comp else None
    top_score = get_top_score_status(sub.top_score, sub.id, curr_top_score_id)

    error_msg = f"[red]{sub.eval_error}[/red]" if sub.eval_error and sub.eval_error != "Success" else "None"

    now = datetime.now(timezone.utc)
    log_status = get_reveal_status(now)
    current_round = None
    current_round_end_at = None
    if comp:
        if comp.curr_round:
            current_round = comp.curr_round.round_number
            current_round_end_at = comp.curr_round.end_at
        else:
            current_round = comp.curr_round_number

    if current_round is not None and sub.round_number == current_round:
        if current_round_end_at is not None:
            end_at = current_round_end_at
            if end_at.tzinfo is None:
                end_at = end_at.replace(tzinfo=timezone.utc)
            log_status = f"{format_datetime(end_at, include_seconds=True)} {get_reveal_status(end_at)}"

    version = f"v{sub.version}" if sub.version > 0 else f"v{sub.version} [dim](Auto)[/dim]"

    detail_lines = [
        f"[dim]Submission ID:[/dim]    {sub.id}",
        f"[dim]Competition ID:[/dim]   {sub.competition_id}",
        f"[dim]Round Number:[/dim]     {sub.round_number}",
        f"[dim]State:[/dim]            {get_state(sub.state)}",
        f"[dim]Hotkey:[/dim]           {sub.hotkey}",
        f"[dim]Version:[/dim]          {version}",
        f"[dim]Top Score:[/dim]        {top_score}",
        "",
        "[bold underline]Scores[/bold underline]",
        f"[dim]Raw Score:[/dim]        {sub.eval_raw_score if sub.eval_raw_score is not None and sub.eval_raw_score >= 0 else 'N/A'}",
        f"[dim]Final Score:[/dim]      {sub.eval_score if sub.eval_score is not None and sub.eval_score >= 0 else 'N/A'}",
        f"[dim]Evaluation Time:[/dim]  {f'{sub.eval_time_in_seconds:.2f}s' if sub.eval_time_in_seconds else 'N/A'}",
        f"[dim]Evaluation Error:[/dim] {error_msg}",
        "",
        "[bold underline]Timeline[/bold underline]",
        f"[dim]Submitted:[/dim]   {format_datetime(sub.submit_at, include_seconds=True)}",
        f"[dim]Evaluated:[/dim]   {format_datetime(sub.eval_at, include_seconds=True)}",
        f"[dim]Code Reveal:[/dim] {format_datetime(sub.reveal_at, include_seconds=True)} {get_reveal_status(sub.reveal_at)}",
        f"[dim]Log Reveal:[/dim]  {log_status}",
    ]

    console.print(Panel("\n".join(detail_lines), title="Submission Details", border_style="cyan"))

    # Show eval metadata if available
    if detail and detail.eval_metadata:
        console.print(
            Panel(RichJSON.from_data(detail.eval_metadata, indent=2), title="Evaluation Metadata", border_style="green")
        )


def _handle_file_option(file_arg, submission, detail, config):
    """Handle the -f flag: list files or show file content."""
    if not detail:
        console.print("[red]Could not load submission detail.[/red]")
        raise typer.Exit(code=1)

    # Build the file list the same way the dashboard does
    files = []

    # Eval metadata is always available as a virtual file
    files.append(("Eval", "metadata.json", None))

    if detail.code_path:
        filename = os.path.basename(detail.code_path)
        files.append(("Code", filename, detail.code_path))

    if detail.eval_file_paths:
        for file_type, file_paths in detail.eval_file_paths.items():
            if isinstance(file_paths, list):
                for fp in sorted(file_paths):
                    fname = os.path.basename(fp) if isinstance(fp, str) else str(fp)
                    files.append((file_type.title(), fname, fp))
            elif isinstance(file_paths, str):
                fname = os.path.basename(file_paths)
                files.append((file_type.title(), fname, file_paths))

    if file_arg.lower() == "list":
        _list_files(files, submission.id)
        return

    # -f <filename>: show file content
    _show_file_content(file_arg, files, submission, detail, config)


def _list_files(files, submission_id):
    """List available files for the submission."""
    table = Table(title=f"Files for Submission {submission_id}")
    table.add_column("Type", style="bold")
    table.add_column("Name", style="cyan")

    for file_type, fname, _ in files:
        table.add_row(file_type, fname)

    console.print(table)


def _show_file_content(filename, files, submission, detail, config):
    """Fetch and display the content of a specific file."""
    # Find the file in the list
    match = None
    for file_type, fname, file_path in files:
        if fname == filename:
            match = (file_type, fname, file_path)
            break

    if match is None:
        console.print(f"[red]File '{filename}' not found.[/red]")
        console.print("[dim]Available files:[/dim]")
        for file_type, fname, _ in files:
            console.print(f"  {file_type}: {fname}")
        raise typer.Exit(code=1)

    file_type, fname, file_path = match

    # Handle eval metadata specially
    if file_type == "Eval" and fname == "metadata.json":
        if detail and detail.eval_metadata:
            console.print(
                Panel(RichJSON.from_data(detail.eval_metadata, indent=2), title="metadata.json", border_style="green")
            )
        else:
            console.print("[dim]No evaluation metadata available.[/dim]")
        return

    # Fetch file content from the API
    async def _fetch():
        async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
            if file_type.lower() == "code":
                from common.models.api.code import CodeRequest

                code_request = CodeRequest(
                    competition_id=submission.competition_id,
                    round_number=submission.round_number,
                    hotkey=submission.hotkey,
                    version=submission.version,
                    start_idx=0,
                )
                code_response = await client.get_submission_code(code_request)
                return code_response.code if code_response else None
            else:
                from common.models.api.submission import FileRequest

                file_request = FileRequest(
                    submission_id=submission.id,
                    file_type=file_type.lower(),
                    file_name=fname,
                    start_idx=0,
                    reverse=False,
                )
                file_data = await client.get_file_chunked(file_request)
                return file_data.data if file_data else None

    console.print(f"Fetching {fname}...")
    content = asyncio.run(_fetch())

    if content is None:
        console.print(f"[red]Failed to load file: {fname}[/red]")
        raise typer.Exit(code=1)

    file_ext = os.path.splitext(fname)[1].lower()

    if file_ext == ".py":
        console.print(Panel(Syntax(content, "python", theme="monokai", line_numbers=True), title=fname))
    elif file_ext == ".json":
        try:
            parsed = json.loads(content)
            console.print(Panel(RichJSON.from_data(parsed, indent=2), title=fname))
        except json.JSONDecodeError:
            console.print(Panel(Syntax(content, "json", theme="monokai", line_numbers=True), title=fname))
    else:
        console.print(Panel(content, title=fname))
