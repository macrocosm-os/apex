from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static, Button


class ConfirmModal(ModalScreen[bool]):
    """Modal screen for getting confirmation from the user."""

    CSS = """
    ConfirmModal {
        align: center middle;
    }

    .confirm-container {
        width: 70;
        height: auto;
        border: solid $warning;
        background: $surface;
        padding: 1 2;
    }

    .confirm-title {
        text-align: center;
        color: $warning;
        margin-bottom: 1;
    }

    .confirm-message {
        text-align: center;
        color: $text;
        margin-bottom: 1;
    }

    .button-container {
        height: auto;
        align: center middle;
    }

    .confirm-button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str = "Confirm", message: str = "Are you sure?") -> None:
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        """Compose the confirmation modal."""
        with Container(classes="confirm-container"):
            yield Static(f"[bold yellow]{self.title}[/bold yellow]", classes="confirm-title")
            yield Static(self.message, classes="confirm-message")
            with Container(classes="button-container"):
                yield Button("Yes", variant="warning", id="yes_button", classes="confirm-button")
                yield Button("No", variant="default", id="no_button", classes="confirm-button")

    def on_mount(self) -> None:
        """Focus the No button by default for safety."""
        no_button = self.query_one("#no_button", Button)
        no_button.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "yes_button":
            self.dismiss(True)
        elif event.button.id == "no_button":
            self.dismiss(False)
