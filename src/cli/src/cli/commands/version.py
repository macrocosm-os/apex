from importlib.metadata import version
from rich.console import Console

console = Console()


def get_version() -> None:
    """Display the CLI version."""
    try:
        cli_version = version("cli")
        console.print(f"[bold green]Apex CLI v{cli_version}[/bold green]")
    except Exception:
        # Fallback if version cannot be read
        console.print("[bold green]Apex CLI v0.0.1[/bold green]")
