"""Loading modal screen for refresh operations."""

from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


class LoadingModal(ModalScreen):
    """Modal screen shown during refresh operations."""

    CSS = """
    LoadingModal {
        align: center middle;
    }

    .loading-container {
        width: 50;
        height: 10;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    .loading-text {
        text-align: center;
        color: $text;
    }
    """

    def __init__(self, message: str = "Loading...") -> None:
        super().__init__()
        self.message = message

    def compose(self):
        """Compose the loading modal."""
        with Container(classes="loading-container"):
            yield Static(
                f"[bold cyan]{self.message}[/bold cyan]\n[yellow]Please wait...[/yellow]", classes="loading-text"
            )
