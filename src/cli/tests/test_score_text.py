from types import SimpleNamespace

from cli.dashboard.utils import get_error_text, get_score_text

PRIVATE = " [dim](private until the round reveals)[/dim]"


def _sub(**kw):
    return SimpleNamespace(**{"raw_score": None, "score": None, "eval_error": None, **kw})


def _detail(**kw):
    return SimpleNamespace(**{"private_score": True, "raw_score": None, "score": None, "eval_error": None, **kw})


def test_private_score_is_labeled():
    assert get_score_text(_sub(), _detail(raw_score=812.0, score=118.2)) == (f"812.0{PRIVATE}", f"118.2{PRIVATE}")


def test_private_failure_code_is_labeled_and_has_no_score():
    detail = _detail(eval_error="run_unstable (submission)")
    assert get_score_text(_sub(), detail) == ("N/A", "N/A")
    assert get_error_text(_sub(), detail) == f"[red]run_unstable (submission)[/red]{PRIVATE}"


def test_error_text_without_private_feedback_uses_the_submission():
    assert get_error_text(_sub(eval_error="build failed")) == "[red]build failed[/red]"
    assert get_error_text(_sub(eval_error="Success")) == "None"
    assert get_error_text(_sub(), _detail(private_score=False, eval_error="x")) == "None"
