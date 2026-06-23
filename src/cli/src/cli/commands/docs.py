import asyncio
import re

import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown

from cli.utils.client import Client
from cli.utils.config import Config

console = Console()

DOCS_BASE = "https://docs.macrocosmos.ai"

_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
_LINK_RE = re.compile(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _link_text(html: str) -> str:
    """Reduce an <a> tag's inner HTML to plain text, keeping <code> as Markdown backticks."""
    text = _CODE_RE.sub(r"`\1`", html)
    text = _TAG_RE.sub("", text)
    return text.strip()


def _resolve_href(href: str) -> str:
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("/"):
        return f"{DOCS_BASE}{href}"
    return href


def _table_to_list(match: "re.Match[str]") -> str:
    """Convert a GitBook card <table> into a Markdown bullet list of its page links.

    Rich's Markdown renderer drops raw HTML blocks, so card tables vanish entirely.
    The cards only carry a name + page link + cover image; the cover-image links
    (/files/...) are noise in a terminal, so we keep just the named page links.
    """
    items = []
    for href, inner in _LINK_RE.findall(match.group(0)):
        if "/files/" in href:
            continue
        label = _link_text(inner)
        if label:
            items.append(f"- [{label}]({_resolve_href(href)})")
    return "\n".join(items)


def _html_tables_to_lists(md_text: str) -> str:
    return _TABLE_RE.sub(_table_to_list, md_text)


def _render_docs_url(url: str) -> None:
    """Fetch a docs page and render it to the terminal as Markdown."""
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to fetch documentation: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(Markdown(_html_tables_to_lists(response.text)))


def _resolve_competition_doc_url(competition_id: int) -> str:
    """Look up a competition's ``doc_url`` from its metadata via the signed API."""
    config = Config.load_config()
    if not config.hotkey_file_path:
        console.print("[red]No hotkey file path found. Please run `apex link` to link your wallet and hotkey.[/red]")
        raise typer.Exit(code=1)

    async def _load():
        async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
            return await client.list_competitions(id=competition_id, show_completed=True)

    try:
        result = asyncio.run(_load())
    except Exception as e:
        console.print(f"[red]Failed to fetch competition {competition_id}: {e}[/red]")
        raise typer.Exit(code=1)

    if not result.competitions:
        console.print(f"[yellow]Competition {competition_id} not found.[/yellow]")
        raise typer.Exit(code=1)

    doc_url = result.competitions[0].doc_url
    if not doc_url:
        console.print(f"[yellow]Competition {competition_id} has no documentation page.[/yellow]")
        raise typer.Exit(code=1)

    return doc_url


DOCS_PATHS = {
    "default": "/subnets/subnet-1-apex.md",
    "incentive": "/subnets/subnet-1-apex/incentive-mechanism.md",
    "faq": "/subnets/subnet-1-apex/apex-support-and-faqs.md",
}


def docs(
    incentive: bool = typer.Option(False, "-i", "--incentive", help="Show incentive mechanism documentation."),
    competition_id: int = typer.Option(
        None, "-c", "--competition", help="Show the documentation page for a specific competition ID."
    ),
    faq: bool = typer.Option(False, "-f", "--faq", help="Show support and FAQs."),
) -> None:
    """Fetch and display Apex documentation from docs.macrocosmos.ai (Subnet 1 overview)."""
    selected = sum([incentive, competition_id is not None, faq])
    if selected > 1:
        console.print("[red]Only one of -i, -c, or -f may be specified at a time.[/red]")
        raise typer.Exit(code=1)

    if competition_id is not None:
        url = _resolve_competition_doc_url(competition_id)
    elif incentive:
        url = f"{DOCS_BASE}{DOCS_PATHS['incentive']}"
    elif faq:
        url = f"{DOCS_BASE}{DOCS_PATHS['faq']}"
    else:
        url = f"{DOCS_BASE}{DOCS_PATHS['default']}"

    _render_docs_url(url)

    console.print()
    console.rule("Additional documentation")
    console.print()
    console.print("[bold]Usage:[/bold] apex docs [OPTIONS]")
    console.print()
    for flag, desc in [
        ("-i, --incentive", "Show incentive mechanism documentation."),
        ("-c, --competition <ID>", "Show the documentation page for a specific competition ID."),
        ("-f, --faq", "Show support and FAQs."),
    ]:
        console.print(f"  [green]{flag:<26}[/green] {desc}")
    console.print()
