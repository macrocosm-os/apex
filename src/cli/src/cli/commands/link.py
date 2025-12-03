import typer
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from cli.utils.io import get_folder_options
from cli.utils.interface import interactive_select
from cli.utils.config import Config

# Create a sub-app for link-related commands
console = Console()


def link():
    """Link your wallet and hotkey to authenticate API calls."""
    try:
        # show the currently linked wallet if it exists:
        try:
            config = Config.load_config()
            if config.hotkey_file_path:
                with open(config.hotkey_file_path, "r") as f:
                    hotkey = json.load(f)["ss58Address"]
                console.print(
                    Panel(
                        Markdown(f"- Path: `{config.hotkey_file_path}`\n - Hotkey: `{hotkey[:8]}`"),
                        border_style="green",
                        title="Linked Wallet",
                    )
                )
        except Exception:
            console.print("[red]Current linked wallet not found.[/red]")

        # if no wallet, request it
        wallet_location = typer.prompt(
            "Enter your new wallet location", default=str(Path.home() / ".bittensor" / "wallets")
        )

        wallets = get_folder_options(Path(wallet_location))
        if wallets:
            options = {}
            for wallet in wallets:
                try:
                    # read the coldkeypub.txt file and get the first 8 characters of the ss58 address
                    wallet_path = Path(wallet_location) / wallet
                    if not wallet_path.is_dir():
                        continue

                    coldkeypub_path = wallet_path / "coldkeypub.txt"
                    if not coldkeypub_path.exists():
                        continue

                    with open(coldkeypub_path, "r") as f:
                        coldkey = json.load(f)["ss58Address"]
                    key_name = f"{wallet.ljust(20)} [{coldkey[:8]}]"
                    options[key_name] = [wallet, coldkey]
                except Exception:
                    continue
            wallet_choice = interactive_select(options, "Select your coldkey wallet")
            wallet_name, coldkey = wallet_choice
            if not wallet_name:
                console.print("[red]No wallet selected. Exiting.[/red]")
                return
        else:
            wallet_name = typer.prompt("No wallets found. Enter your wallet name")

        hotkeys = get_folder_options(Path(wallet_location) / wallet_name / "hotkeys")
        if hotkeys:
            options = {}
            for hotkey_name in hotkeys:
                # read the hotkey.txt file and get the first 8 characters of the ss58 address
                try:
                    with open(Path(wallet_location) / wallet_name / "hotkeys" / hotkey_name, "r") as f:
                        hotkey = json.load(f)["ss58Address"]
                    key_name = f"{hotkey_name.ljust(20)} [{hotkey[:8]}]"
                    options[key_name] = [hotkey_name, hotkey]
                except Exception:
                    continue
            hotkey_choice = interactive_select(options, "Select which hotkey to use")
            hotkey_name, hotkey = hotkey_choice
            if not hotkey_name:
                console.print("[red]No hotkey selected. Exiting.[/red]")
                return
        else:
            hotkey_name = typer.prompt(f"No hotkeys found for wallet '{wallet_name}'. Enter your hotkey name")

        config = Config.load_config()
        config.hotkey_file_path = str(Path(wallet_location) / wallet_name / "hotkeys" / hotkey_name)
        config.save_config()

        message = Markdown(
            f"""🔗 Linked wallet at **{Path(wallet_location) / wallet_name / "hotkeys" / hotkey_name}**

- **Coldkey:**   `{coldkey}`
- **Hotkey:**    `{hotkey}`
"""
        )

        console.print(Panel(message, border_style="green", title="Wallet linked successfully"))
    except (KeyboardInterrupt, typer.Abort, typer.Exit):
        console.print("\n[yellow]Link cancelled by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]Unexpected error during link: {e}[/red]")
        return False
