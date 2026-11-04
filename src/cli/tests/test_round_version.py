from types import SimpleNamespace

from cli.dashboard.utils import get_round_version


def _comp(curr_round):
    return SimpleNamespace(curr_round=curr_round)


def test_round_version_comes_from_the_current_round():
    assert get_round_version(_comp(SimpleNamespace(spec_version="0.5.2"))) == "0.5.2"


def test_round_version_missing_is_na():
    assert get_round_version(_comp(None)) == "N/A"
    assert get_round_version(_comp(SimpleNamespace(spec_version=None))) == "N/A"
