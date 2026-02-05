import base64
import os
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console

from common.models.api.submission import SubmissionRecord, SubmissionDetail, FileRequest
from common.models.api.code import CodeRequest
from cli.dashboard.time_utils import format_datetime
from cli.dashboard.utils import log_success, log_error
from cli.utils.client import Client
from cli.utils.config import Config
from cli.utils.wallet import load_keypair_from_file
from cli.dashboard.modals.input_modal import InputModal
from cli.dashboard.modals.confirm_modal import ConfirmModal

console = Console()


def check_file_available(
    submission: SubmissionRecord,
    file_type: str,
    log_widget=None,
    current_round: int | None = None,
    current_round_end_at: datetime | None = None,
) -> tuple[bool, str | None]:
    """Check if code is available for download.

    Args:
        submission: The submission to check
        file_type: The type of file to check
        log_widget: Optional log widget for debug logging
        current_round: Optional current round number for log visibility check
        current_round_end_at: Optional current round end date for log visibility check

    Returns:
        tuple: (is_available, error_message)
            - is_available: True if code can be downloaded
            - error_message: Error message if not available, None otherwise
    """
    if log_widget:
        log_success(
            log_widget=log_widget,
            message=f"check_file_available called: file_type={file_type}, submission.round_number={submission.round_number}, "
            f"current_round={current_round}, current_round_end_at={current_round_end_at}",
        )
    now = datetime.now(timezone.utc)
    reveal_at = submission.reveal_at
    if reveal_at.tzinfo is None:
        reveal_at = reveal_at.replace(tzinfo=timezone.utc)

    # Get the user's hotkey from config
    user_hotkey = None
    try:
        config = Config.load_config()
        if config.hotkey_file_path:
            keypair = load_keypair_from_file(hotkey_file_path=config.hotkey_file_path)
            user_hotkey = keypair.ss58_address
            if log_widget:
                log_success(log_widget=log_widget, message=f"Loaded user hotkey: {user_hotkey[:8]}...")
        else:
            if log_widget:
                log_error(log_widget=log_widget, message="No hotkey_file_path in config")
    except Exception as e:
        # If we can't load the hotkey, continue without it
        if log_widget:
            log_error(log_widget=log_widget, message=f"Failed to load user hotkey: {e}")

    if log_widget:
        log_success(log_widget=log_widget, message=f"Submission hotkey: {submission.hotkey[:8]}...")
        log_success(log_widget=log_widget, message=f"Current time: {now}, Reveal time: {reveal_at}")

    # Check log visibility if file_type is "log"
    if file_type and file_type.lower() == "log":
        if log_widget:
            log_success(
                log_widget=log_widget,
                message=f"Checking log availability: submission round={submission.round_number} (type: {type(submission.round_number).__name__}), "
                f"Checking log availability: submission round={submission.round_number} (type: {type(submission.round_number).__name__}), "
                f"current_round={current_round} (type: {type(current_round).__name__}), current_round_end_at={current_round_end_at}",
            )
        # Check if this is the current round and if the round has ended
        # Ensure both are int for comparison
        submission_round = int(submission.round_number)
        if current_round is not None:
            current_round_int = int(current_round)
            if submission_round == current_round_int:
                if log_widget:
                    log_success(log_widget=log_widget, message=f"Submission is from current round {current_round_int}")
                if current_round_end_at is not None:
                    # Ensure end_at is timezone-aware for comparison
                    end_at = current_round_end_at
                    if end_at.tzinfo is None:
                        end_at = end_at.replace(tzinfo=timezone.utc)
                    if log_widget:
                        log_success(
                            log_widget=log_widget,
                            message=f"Round end time: {end_at}, Current time: {now}, Comparison: {now < end_at}",
                        )
                    # Lock if round hasn't ended yet
                    if now < end_at:
                        error_msg = (
                            f"Log not yet available. Will be revealed when the round ends at "
                            f"{format_datetime(end_at, include_seconds=True)}"
                        )
                        if log_widget:
                            log_error(log_widget=log_widget, message=f"Log not available: {error_msg}")
                        return False, error_msg
                    else:
                        if log_widget:
                            log_success(log_widget=log_widget, message="Round has ended, log is available")
                else:
                    if log_widget:
                        log_success(log_widget=log_widget, message="No round end date available, allowing download")
            else:
                if log_widget:
                    log_success(
                        log_widget=log_widget,
                        message=f"Submission is not from current round (submission={submission_round}, "
                        f"current={current_round_int}), allowing download",
                    )
        else:
            if log_widget:
                log_success(log_widget=log_widget, message="No current round available, allowing download")
        # If we can't determine (not current round or no end date), allow download
        # (defaults to viewable, similar to submission detail screen logic)

    # Allow download if it's the user's own submission, even if reveal time hasn't passed
    if file_type is None or file_type.lower() == "code":
        if now < reveal_at:
            if submission.hotkey == user_hotkey:
                if log_widget:
                    log_success(
                        log_widget=log_widget, message="Code available: Own submission, reveal time check bypassed"
                    )
                return True, None
            else:
                error_msg = (
                    f"Code not yet available. Will be revealed at {format_datetime(reveal_at, include_seconds=True)}"
                )
                if log_widget:
                    log_error(log_widget=log_widget, message=f"Code not available: {error_msg}")
                return False, error_msg

    return True, None


def get_default_download_path(
    submission: SubmissionRecord,
    submission_detail: SubmissionDetail | None,
    filename: str | None = None,
) -> str:
    """Generate the default download path for a submission.

    Args:
        submission: The submission record
        submission_detail: Optional detailed submission information
        filename: Optional filename to use

    Returns:
        str: Default file path for download
    """
    if filename:
        original_filename = filename
    elif submission_detail and submission_detail.code_path:
        original_filename = os.path.basename(submission_detail.code_path)
    else:
        original_filename = f"submission_{submission.id}.py"

    return f"submissions/{original_filename}"


async def download_file(
    submission: SubmissionRecord,
    file_path: str,
    log_widget,
    file_type: str | None = None,
    filename: str | None = None,
    notify_callback=None,
) -> bool:
    """Download a file for a submission to the specified path.

    Args:
        submission: The submission to download file for
        file_path: Path where the file should be saved
        log_widget: Log widget for logging messages
        file_type: Optional file type (e.g., "Code", "Log", etc.). If None, downloads code.
        filename: Optional filename. Used for non-code files.
        notify_callback: Optional callback function for notifications (message, severity, timeout)

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        file_display_name = filename if filename else "code"
        log_success(log_widget=log_widget, message=f"Downloading {file_display_name} to: {file_path}")

        # Get the file content
        config = Config.load_config()
        async with Client(hotkey_file_path=config.hotkey_file_path, timeout=config.timeout) as client:
            # Code files need to use the code endpoint, not the file endpoint
            if file_type is None or file_type.lower() == "code":
                code_request = CodeRequest(
                    competition_id=submission.competition_id,
                    round_number=submission.round_number,
                    hotkey=submission.hotkey,
                    version=submission.version,
                    start_idx=0,
                )
                code_response = await client.get_submission_code(code_request=code_request)

                if code_response and code_response.code:
                    # Create directory if it doesn't exist
                    path = Path(file_path)
                    path.parent.mkdir(parents=True, exist_ok=True)

                    # Write the code to file
                    if code_response.is_binary:
                        # Binary file
                        binary_data = base64.b64decode(code_response.code)
                        with open(path, "wb") as f:
                            f.write(binary_data)
                    else:
                        # Text file
                        with open(path, "w") as f:
                            f.write(code_response.code)

                    success_msg = f"Code successfully saved to: {file_path}"
                    log_success(log_widget=log_widget, message=success_msg)
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

                    log_error(log_widget=log_widget, message=f"Failed to download code: {error_msg}")
                    console.print(f"[red]Failed to download code: {error_msg}[/red]")
                    if notify_callback:
                        notify_callback(error_msg, severity="error", timeout=3)
                    return False
            else:
                # For non-code files, use the file endpoint
                if not filename:
                    error_msg = "Filename is required for non-code files"
                    log_error(log_widget=log_widget, message=error_msg)
                    console.print(f"[red]{error_msg}[/red]")
                    if notify_callback:
                        notify_callback(error_msg, severity="error", timeout=3)
                    return False

                file_request = FileRequest(
                    submission_id=submission.id,
                    file_type=file_type.lower(),
                    file_name=filename,
                    start_idx=0,
                    reverse=False,
                )
                file_data = await client.get_file_chunked(file_request=file_request)

                if file_data and file_data.data:
                    # Create directory if it doesn't exist
                    path = Path(file_path)
                    path.parent.mkdir(parents=True, exist_ok=True)

                    # Write the file content
                    with open(path, "w") as f:
                        f.write(file_data.data)

                    success_msg = f"{file_display_name} successfully saved to: {file_path}"
                    log_success(log_widget=log_widget, message=success_msg)
                    console.print(f"[green]✓ {file_display_name} downloaded to: {file_path}[/green]")
                    if notify_callback:
                        notify_callback(success_msg, severity="information", timeout=3)
                    return True
                else:
                    error_msg = f"No content received for {file_display_name}"
                    log_error(log_widget=log_widget, message=error_msg)
                    console.print(f"[red]{error_msg}[/red]")
                    if notify_callback:
                        notify_callback(error_msg, severity="error", timeout=3)
                    return False

    except Exception as e:
        error_str = str(e)
        file_display_name = filename if filename else "code"

        # Check for common error patterns
        if "404" in error_str or "not found" in error_str.lower():
            if file_type is None or file_type.lower() == "code":
                now = datetime.now(timezone.utc)
                reveal_at = submission.reveal_at
                if reveal_at.tzinfo is None:
                    reveal_at = reveal_at.replace(tzinfo=timezone.utc)

                if now < reveal_at:
                    error_msg = f"Code not yet available. Will be revealed at {format_datetime(reveal_at, include_seconds=True)}"
                else:
                    error_msg = "Code not found on server"
            else:
                error_msg = f"{file_display_name} not found on server"
        elif "403" in error_str or "forbidden" in error_str.lower():
            error_msg = f"Access denied. You may not have permission to view {file_display_name}."
        else:
            error_msg = str(e)

        log_error(log_widget=log_widget, message=f"Error downloading {file_display_name}: {error_msg}")
        console.print(f"[red]Error downloading {file_display_name}: {error_msg}[/red]")
        if notify_callback:
            notify_callback(f"Error: {error_msg}", severity="error", timeout=3)
        return False


def show_download_dialog(
    screen,
    submission: SubmissionRecord,
    submission_detail: SubmissionDetail | None,
    log_widget,
    file_type: str | None = None,
    filename: str | None = None,
    notify_callback=None,
    current_round: int | None = None,
    current_round_end_at: datetime | None = None,
):
    """Show the download dialog with path input and handle the download flow.

    Args:
        screen: The screen instance to push modals to
        submission: The submission to download
        submission_detail: Optional detailed submission information
        log_widget: Log widget for logging
        file_type: Optional file type (e.g., "Code", "Log", etc.). If None, downloads code.
        filename: Optional filename. Used for non-code files.
        notify_callback: Optional callback for notifications
        current_round: Optional current round number for log visibility check
        current_round_end_at: Optional current round end date for log visibility check
    """
    # Check if file is available (for code and log files)
    # Normalize file_type to lowercase for consistent checking
    normalized_file_type = file_type.lower() if file_type else "code"

    if normalized_file_type in ["code", "log"]:
        if log_widget:
            log_success(
                log_widget=log_widget,
                message=f"Checking file availability: file_type={file_type} (normalized={normalized_file_type}), "
                f"current_round={current_round}, current_round_end_at={current_round_end_at}",
            )
        is_available, error_msg = check_file_available(
            submission=submission,
            file_type=normalized_file_type,
            log_widget=log_widget,
            current_round=current_round,
            current_round_end_at=current_round_end_at,
        )
        if log_widget:
            log_success(
                log_widget=log_widget,
                message=f"File availability check result: is_available={is_available}, error_msg={error_msg}",
            )
        if not is_available:
            if error_msg:
                log_error(log_widget=log_widget, message=error_msg)
                console.print(f"[red]{error_msg}[/red]")
                if notify_callback:
                    notify_callback(error_msg, severity="error", timeout=3)
            else:
                # If not available but no error message, show generic error
                error_msg = "File is not available for download"
                log_error(log_widget=log_widget, message=error_msg)
                console.print(f"[red]{error_msg}[/red]")
                if notify_callback:
                    notify_callback(error_msg, severity="error", timeout=3)
            return

    # Generate default path
    default_path = get_default_download_path(
        submission=submission, submission_detail=submission_detail, filename=filename
    )

    # Determine dialog title
    file_display_name = filename if filename else "code"
    dialog_title = f"Download {file_display_name} - Enter file path"

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
                        screen.set_timer(
                            0.1,
                            lambda: download_file(
                                submission=submission,
                                file_path=result,
                                log_widget=log_widget,
                                file_type=file_type,
                                filename=filename,
                                notify_callback=notify_callback,
                            ),
                        )

                screen.app.push_screen(
                    ConfirmModal(
                        title="File Already Exists",
                        message=f"[yellow]{result}[/yellow] already exists.\n\nDo you want to overwrite it?",
                    ),
                    handle_overwrite_confirm,
                )
            else:
                screen.set_timer(
                    0.1,
                    lambda: download_file(
                        submission=submission,
                        file_path=result,
                        log_widget=log_widget,
                        file_type=file_type,
                        filename=filename,
                        notify_callback=notify_callback,
                    ),
                )

    screen.app.push_screen(
        InputModal(title=dialog_title, default_value=default_path, placeholder="submissions/file.py"),
        handle_path_input,
    )
