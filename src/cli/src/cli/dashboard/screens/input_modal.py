from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static, Input, Button
from textual.events import Key


class InputModal(ModalScreen[str | None]):
    """Modal screen for getting text input from the user."""

    CSS = """
    InputModal {
        align: center middle;
    }

    .input-container {
        width: 80;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1 2;
    }

    .input-title {
        text-align: center;
        color: $text;
        margin-bottom: 1;
    }

    .input-field {
        margin-bottom: 1;
    }

    .button-container {
        height: auto;
        align: center middle;
    }

    .submit-button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str = "Enter value", default_value: str = "", placeholder: str = "") -> None:
        super().__init__()
        self.title = title
        self.default_value = default_value
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        """Compose the input modal."""
        with Container(classes="input-container"):
            yield Static(f"[bold cyan]{self.title}[/bold cyan]", classes="input-title")
            yield Input(
                value=self.default_value,
                placeholder=self.placeholder,
                id="input_field",
                classes="input-field",
            )
            with Container(classes="button-container"):
                yield Button("Submit", variant="primary", id="submit_button", classes="submit-button")
                yield Button("Cancel", variant="default", id="cancel_button", classes="submit-button")

    def on_mount(self) -> None:
        """Focus the input field when mounted."""
        input_field = self.query_one("#input_field", Input)
        input_field.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "submit_button":
            input_field = self.query_one("#input_field", Input)
            self.dismiss(input_field.value)
        elif event.button.id == "cancel_button":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        self.dismiss(event.value)

    def on_key(self, event: Key) -> None:
        """Handle key presses."""
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
