import os
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console

from common.models.api.submission import SubmissionRecord, SubmissionDetail
from common.models.api.code import CodeRequest
from cli.dashboard.time_utils import format_datetime
from cli.dashboard.utils import log_success, log_error
from cli.utils.client import Client
from cli.utils.config import Config
from cli.dashboard.screens.input_modal import InputModal
from cli.dashboard.screens.confirm_modal import ConfirmModal

console = Console()


def check_code_available(submission: SubmissionRecord) -> tuple[bool, str | None]:
    """Check if code is available for download.

    Returns:
        tuple: (is_available, error_message)
            - is_available: True if code can be downloaded
            - error_message: Error message if not available, None otherwise
    """
    now = datetime.now(timezone.utc)
    reveal_at = submission.reveal_at
    if reveal_at.tzinfo is None:
        reveal_at = reveal_at.replace(tzinfo=timezone.utc)

    if now < reveal_at:
        error_msg = f"Code not yet available. Will be revealed at {format_datetime(reveal_at, include_seconds=True)}"
        return False, error_msg

    return True, None


def get_default_download_path(submission: SubmissionRecord, submission_detail: SubmissionDetail | None) -> str:
    """Generate the default download path for a submission.

    Args:
        submission: The submission record
        submission_detail: Optional detailed submission information

    Returns:
        str: Default file path for download
    """
    if submission_detail and submission_detail.code_path:
        original_filename = os.path.basename(submission_detail.code_path)
    else:
        original_filename = f"submission_{submission.id}.py"

    return f"submissions/{original_filename}"


async def download_code(submission: SubmissionRecord, file_path: str, log_widget, notify_callback=None) -> bool:
    """Download code for a submission to the specified path.

    Args:
        submission: The submission to download code for
        file_path: Path where the code should be saved
        log_widget: Log widget for logging messages
        notify_callback: Optional callback function for notifications (message, severity, timeout)

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        log_success(log_widget, f"Downloading code to: {file_path}")

        # Get the code content
        config = Config.load_config()
        async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
            code_request = CodeRequest(
                competition_id=submission.competition_id,
                round_number=submission.round_number,
                hotkey=submission.hotkey,
                version=submission.version,
                start_idx=0,
            )
            code_response = await client.get_submission_code(code_request)

            if code_response and code_response.code:
                # Create directory if it doesn't exist
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)

                # Write the code to file
                with open(path, "w") as f:
                    f.write(code_response.code)

                success_msg = f"Code successfully saved to: {file_path}"
                log_success(log_widget, success_msg)
                console.print(f"[green]✓ Code downloaded to: {file_path}[/green]")
                if notify_callback:
                    notify_callback(success_msg, severity="information", timeout=3)
                return True
            else:
                # Check if code hasn't been revealed yet
                now = datetime.now(timezone.utc)
                reveal_at = submission.reveal_at
                if reveal_at.tzinfo is None:
                    reveal_at = reveal_at.replace(tzinfo=timezone.utc)

                if now < reveal_at:
                    error_msg = f"Code not yet available. Will be revealed at {format_datetime(reveal_at, include_seconds=True)}"
                else:
                    error_msg = "No code content received from server"

                log_error(log_widget, f"Failed to download code: {error_msg}")
                console.print(f"[red]Failed to download code: {error_msg}[/red]")
                if notify_callback:
                    notify_callback(error_msg, severity="error", timeout=3)
                return False

    except Exception as e:
        error_str = str(e)

        # Check for common error patterns that indicate code not available
        if "404" in error_str or "not found" in error_str.lower():
            now = datetime.now(timezone.utc)
            reveal_at = submission.reveal_at
            if reveal_at.tzinfo is None:
                reveal_at = reveal_at.replace(tzinfo=timezone.utc)

            if now < reveal_at:
                error_msg = (
                    f"Code not yet available. Will be revealed at {format_datetime(reveal_at, include_seconds=True)}"
                )
            else:
                error_msg = "Code not found on server"
        elif "403" in error_str or "forbidden" in error_str.lower():
            error_msg = "Access denied. You may not have permission to view this code."
        else:
            error_msg = str(e)

        log_error(log_widget, f"Error downloading code: {error_msg}")
        console.print(f"[red]Error downloading code: {error_msg}[/red]")
        if notify_callback:
            notify_callback(f"Error: {error_msg}", severity="error", timeout=3)
        return False


def show_download_dialog(
    screen, submission: SubmissionRecord, submission_detail: SubmissionDetail | None, log_widget, notify_callback=None
):
    """Show the download dialog with path input and handle the download flow.

    Args:
        screen: The screen instance to push modals to
        submission: The submission to download
        submission_detail: Optional detailed submission information
        log_widget: Log widget for logging
        notify_callback: Optional callback for notifications
    """
    # Check if code is available
    is_available, error_msg = check_code_available(submission)
    if not is_available and error_msg:
        log_error(log_widget, error_msg)
        console.print(f"[red]{error_msg}[/red]")
        if notify_callback:
            notify_callback(error_msg, severity="error", timeout=3)
        return

    # Generate default path
    default_path = get_default_download_path(submission, submission_detail)

    # Show input modal for file path
    def handle_path_input(result: str | None) -> None:
        """Handle the result from the input modal."""
        if result:
            # Check if file already exists
            path = Path(result)
            if path.exists():
                # Show confirmation modal
                def handle_overwrite_confirm(confirmed: bool) -> None:
                    """Handle the overwrite confirmation."""
                    if confirmed:
                        screen.set_timer(0.1, lambda: download_code(submission, result, log_widget, notify_callback))

                screen.app.push_screen(
                    ConfirmModal(
                        title="File Already Exists",
                        message=f"[yellow]{result}[/yellow] already exists.\n\nDo you want to overwrite it?",
                    ),
                    handle_overwrite_confirm,
                )
            else:
                screen.set_timer(0.1, lambda: download_code(submission, result, log_widget, notify_callback))

    screen.app.push_screen(
        InputModal(
            title="Download Code - Enter file path", default_value=default_path, placeholder="submissions/code.py"
        ),
        handle_path_input,
    )
