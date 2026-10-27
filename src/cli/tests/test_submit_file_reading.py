"""Reading a submission file: a unified diff must reach the partner byte-for-byte."""

from pathlib import Path

from cli.commands.submit import PATCH_EXTENSIONS, read_submission_text


def test_patch_keeps_its_trailing_newline(tmp_path: Path):
    """`git apply` rejects a patch whose final line has no newline. Stripping the file on the
    way out of the CLI corrupts every patch submission."""
    patch = tmp_path / "change.patch"
    body = "diff --git a/server/a.go b/server/a.go\n--- a/server/a.go\n+++ b/server/a.go\n@@ -1 +1 @@\n-a\n+b\n"
    patch.write_text(body)

    assert read_submission_text(patch) == body


def test_diff_extension_is_treated_as_a_patch(tmp_path: Path):
    f = tmp_path / "change.DIFF"
    f.write_text("@@ -1 +1 @@\n-a\n+b\n")

    assert read_submission_text(f).endswith("+b\n")
    assert ".diff" in PATCH_EXTENSIONS and ".patch" in PATCH_EXTENSIONS


def test_code_file_is_still_stripped(tmp_path: Path):
    f = tmp_path / "solution.py"
    f.write_text("\n\nprint(1)\n\n")

    assert read_submission_text(f) == "print(1)"
