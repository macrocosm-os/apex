import os
import subprocess
import sys
from unittest import mock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
import autoupdater  # isort: skip


def test_venv_python(mocker):
    mocker.patch("os.path.join", return_value=".venv/bin/python")
    assert autoupdater.venv_python() == ".venv/bin/python"
    autoupdater.os.path.join.assert_called_once_with(".venv", "bin", "python")


def test_read_python_version_exists(mocker):
    mocker.patch("autoupdater.open", mocker.mock_open(read_data="3.11.9"))
    assert autoupdater.read_python_version() == "3.11.9"
    autoupdater.open.assert_called_once_with(".python-version", encoding="utf-8")


def test_read_python_version_not_found(mocker):
    mocker.patch("autoupdater.open", side_effect=FileNotFoundError)
    assert autoupdater.read_python_version() is None
    autoupdater.open.assert_called_once_with(".python-version", encoding="utf-8")


def test_start_proc_with_version(mocker):
    mocker.patch("autoupdater.read_python_version", return_value="3.11.9")
    mock_run = mocker.patch("subprocess.run")
    mock_popen = mocker.patch("subprocess.Popen", return_value=mocker.Mock())
    mocker.patch("autoupdater.venv_python", return_value="mock_python")

    proc = autoupdater.start_proc()
    autoupdater.read_python_version.assert_called_once()
    mock_run.assert_has_calls(
        [
            mock.call(["uv", "venv", "--python", "3.11.9"], check=True),
            mock.call(["uv", "pip", "install", ".[dev]"], check=True),
        ]
    )
    mock_popen.assert_called_once_with(["mock_python", "validator.py"])
    assert proc is not None


def test_start_proc_without_version(mocker):
    mocker.patch("autoupdater.read_python_version", return_value=None)
    mock_run = mocker.patch("subprocess.run")
    mock_popen = mocker.patch("subprocess.Popen", return_value=mocker.Mock())
    mocker.patch("autoupdater.venv_python", return_value="mock_python")

    proc = autoupdater.start_proc()
    autoupdater.read_python_version.assert_called_once()
    mock_run.assert_has_calls(
        [
            mock.call(["uv", "venv"], check=True),
            mock.call(["uv", "pip", "install", ".[dev]"], check=True),
        ]
    )
    mock_popen.assert_called_once_with(["mock_python", "validator.py"])
    assert proc is not None


def test_stop_proc_running():
    mock_proc = mock.Mock()
    mock_proc.poll.return_value = None  # Process is running
    autoupdater.stop_proc(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=10)
    mock_proc.kill.assert_not_called()


def test_stop_proc_timeout():
    mock_proc = mock.Mock()
    mock_proc.poll.return_value = None  # Process is running
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
    autoupdater.stop_proc(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=10)
    mock_proc.kill.assert_called_once()


def test_stop_proc_already_stopped():
    mock_proc = mock.Mock()
    mock_proc.poll.return_value = 0  # Process already stopped
    autoupdater.stop_proc(mock_proc)
    mock_proc.terminate.assert_not_called()
    mock_proc.wait.assert_not_called()
    mock_proc.kill.assert_not_called()


def test_remote_has_updates_true(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_check_output = mocker.patch("subprocess.check_output", return_value="1\t0")
    assert autoupdater.remote_has_updates() is True
    mock_run.assert_called_once_with(["git", "fetch", "--quiet"], check=True)
    mock_check_output.assert_called_once_with(
        ["git", "rev-list", "--left-right", "--count", "@{u}...HEAD"], stderr=subprocess.STDOUT, text=True
    )


def test_remote_has_updates_false_no_diff(mocker):
    mocker.patch("subprocess.run")
    mocker.patch("subprocess.check_output", return_value="0\t0")
    assert autoupdater.remote_has_updates() is False


def test_remote_has_updates_error(mocker):
    mock_run = mocker.patch("subprocess.run")
    mocker.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd"))
    assert autoupdater.remote_has_updates() is False
    mock_run.assert_called_once_with(["git", "fetch", "--quiet"], check=True)
    autoupdater.subprocess.check_output.assert_called_once()


def test_git_pull_ff_only_success(mocker):
    mock_run = mocker.patch("subprocess.run")
    autoupdater.git_pull_ff_only()
    mock_run.assert_called_once_with(["git", "pull", "--ff-only"], check=True)


def test_git_pull_ff_only_conflict(mocker):
    mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="conflict"))
    mock_stderr = mocker.patch("sys.stderr", new_callable=mock.MagicMock)
    autoupdater.git_pull_ff_only()
    autoupdater.subprocess.run.assert_called_once_with(["git", "pull", "--ff-only"], check=True)
    # Check for individual calls to write, as print adds newlines separately
    mock_stderr.write.assert_any_call(
        "Error: Git pull failed due to conflicts or other issues: Command 'cmd' returned non-zero exit status 1."
    )
    mock_stderr.write.assert_any_call("\n")
    mock_stderr.write.assert_any_call("Staying on the current version.")
    mock_stderr.write.assert_any_call("\n")


def test_main_loop_with_update(mocker):
    mock_start_proc = mocker.patch("autoupdater.start_proc", return_value=mocker.Mock())
    mock_start_proc.return_value.returncode = 0  # Ensure returncode is 0 for sys.exit(0)
    mock_stop_proc = mocker.patch("autoupdater.stop_proc")
    mock_remote_updates = mocker.patch("autoupdater.remote_has_updates", side_effect=[False, True, False])
    mock_git_pull = mocker.patch("autoupdater.git_pull_ff_only")
    mock_sleep = mocker.patch(
        "time.sleep", side_effect=[None, None, Exception("StopLoop")]
    )  # Allow 2 calls to remote_has_updates
    mock_sys_exit = mocker.patch("sys.exit")

    mock_start_proc.return_value.poll.side_effect = [None, 0]  # proc.poll() returns 0 on second iteration

    with pytest.raises(Exception) as cm:
        autoupdater.main()
    assert str(cm.value) == "StopLoop"

    assert mock_sleep.call_count == 3
    mock_remote_updates.assert_has_calls([mock.call(), mock.call()])
    mock_stop_proc.assert_called_once_with(mock_start_proc.return_value)
    mock_git_pull.assert_called_once()
    mock_start_proc.assert_called_with()  # Called initially and after update
    mock_sys_exit.assert_called_once_with(0)


def test_main_loop_no_update(mocker):
    mock_start_proc = mocker.patch("autoupdater.start_proc", return_value=mocker.Mock())
    mock_start_proc.return_value.returncode = 0  # Ensure returncode is 0 for sys.exit(0)
    mock_stop_proc = mocker.patch("autoupdater.stop_proc")
    mock_remote_updates = mocker.patch("autoupdater.remote_has_updates", return_value=False)
    mock_git_pull = mocker.patch("autoupdater.git_pull_ff_only")
    mock_sleep = mocker.patch(
        "time.sleep", side_effect=[None, None, Exception("StopLoop")]
    )  # Allow 2 calls to remote_has_updates
    mock_sys_exit = mocker.patch("sys.exit")

    mock_start_proc.return_value.poll.side_effect = [None, 0]  # proc.poll() returns 0 on second iteration

    with pytest.raises(Exception) as cm:
        autoupdater.main()
    assert str(cm.value) == "StopLoop"

    assert mock_sleep.call_count == 3
    assert mock_remote_updates.call_count == 2
    mock_stop_proc.assert_not_called()
    mock_git_pull.assert_not_called()
    mock_sys_exit.assert_called_once_with(0)
