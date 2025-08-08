from pathlib import Path
from typing import Any

import yaml

from apex.common import config


def _write_yaml(tmp_path: Path, data: dict[str, Any], name: str = "cfg.yaml") -> Path:
    """Serialize *data* to YAML and return the absolute path."""
    file_path = tmp_path / name
    file_path.write_text(yaml.safe_dump(data))
    return file_path


def test_from_file_success(tmp_path: Path) -> None:
    raw = {
        "chain": {
            "kwargs": {
                "netuid": 1,
                "coldkey": "ck",
                "hotkey": "hk",
                "network": ["finney"],
            }
        },
        "websearch": {"key": "abcd"},
    }
    cfg_file = _write_yaml(tmp_path, raw)

    cfg = config.Config.from_file(cfg_file)

    assert isinstance(cfg.chain, config.ConfigClass)
    assert cfg.chain.kwargs["network"] == ["finney"]


def test_chain_network_coercion() -> None:
    chain_list = config.ConfigClass(
        kwargs={"netuid": 0, "coldkey": "ck", "hotkey": "hk", "network": ["ws://endpoint1", "ws://endpoint2"]}
    )
    assert chain_list.kwargs["network"] == ["ws://endpoint1", "ws://endpoint2"]

    chain_scalar = config.ConfigClass(kwargs={"netuid": 0, "coldkey": "ck", "hotkey": "hk", "network": "ws://solo"})
    assert chain_scalar.kwargs["network"] == "ws://solo"
