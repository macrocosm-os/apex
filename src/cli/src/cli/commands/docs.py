import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()

DOCS_BASE = "https://docs.macrocosmos.ai"

DOCS_PATHS = {
    "default": "/subnets/subnet-1-apex.md",
    "incentive": "/subnets/subnet-1-apex/incentive-mechanism.md",
    "competitions": "/subnets/subnet-1-apex/subnet-1-current-competitions.md",
    "faq": "/subnets/subnet-1-apex/apex-support-and-faqs.md",
}


def docs(
    incentive: bool = typer.Option(False, "-i", "--incentive", help="Show incentive mechanism documentation."),
    competitions: bool = typer.Option(False, "-c", "--competitions", help="Show current competitions documentation."),
    faq: bool = typer.Option(False, "-f", "--faq", help="Show support and FAQs."),
) -> None:
    """Fetch and display Apex documentation from docs.macrocosmos.ai (Subnet 1 overview)."""
    selected = sum([incentive, competitions, faq])
    if selected > 1:
        console.print("[red]Only one of -i, -c, or -f may be specified at a time.[/red]")
        raise typer.Exit(code=1)

    if incentive:
        path = DOCS_PATHS["incentive"]
    elif competitions:
        path = DOCS_PATHS["competitions"]
    elif faq:
        path = DOCS_PATHS["faq"]
    else:
        path = DOCS_PATHS["default"]

    url = f"{DOCS_BASE}{path}"

    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to fetch documentation: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(Markdown(response.text))
    console.print()
    console.rule("Additional documentation")
    console.print()
    console.print("[bold]Usage:[/bold] apex docs [OPTIONS]")
    console.print()
    for flag, desc, key in [
        ("-i, --incentive", "Show incentive mechanism documentation.", "incentive"),
        ("-c, --competitions", "Show current competitions documentation.", "competitions"),
        ("-f, --faq", "Show support and FAQs.", "faq"),
    ]:
        console.print(f"  [green]{flag:<22}[/green] {desc}")
    console.print()
