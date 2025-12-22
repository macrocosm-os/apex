from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static, Button
from textual.events import Key


class AlertModal(ModalScreen[None]):
    """Modal screen for displaying an alert message to the user."""

    CSS = """
    AlertModal {
        align: center middle;
    }

    .alert-container {
        width: 70;
        height: auto;
        border: solid $warning;
        background: $surface;
        padding: 1 2;
    }

    .alert-title {
        text-align: center;
        color: $warning;
        margin-bottom: 1;
    }

    .alert-message {
        text-align: center;
        color: $text;
        margin-bottom: 1;
    }

    .button-container {
        height: auto;
        align: center middle;
    }

    .alert-button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str = "Alert", message: str = "") -> None:
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        """Compose the alert modal."""
        with Container(classes="alert-container"):
            yield Static(f"[bold yellow]{self.title}[/bold yellow]", classes="alert-title")
            yield Static(self.message, classes="alert-message")
            with Container(classes="button-container"):
                yield Button("OK", variant="primary", id="ok_button", classes="alert-button")

    def on_mount(self) -> None:
        """Focus the OK button."""
        ok_button = self.query_one("#ok_button", Button)
        ok_button.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "ok_button":
            self.dismiss(None)

    def on_key(self, event: Key) -> None:
        """Handle key presses."""
        if event.key == "escape" or event.key == "enter":
            event.stop()
            self.dismiss(None)
