from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from apex.common.async_chain import AsyncChain  # noqa: E402


class DummyMetagraph:
    """Light replacement for AsyncMetagraph."""

    def __init__(self, hotkeys: Iterable[str] | None = None, uids: Iterable[int] | None = None) -> None:
        self.hotkeys = list(hotkeys or ())
        self.uids = list(uids or ())
        self.n = len(self.uids)


class DummySubtensor:
    """Stand-in for bittensor.core.async_subtensor.AsyncSubtensor.

    Behaves as an async context-manager.
    Returns a pre-cooked metagraph.
    Records parameters passed to set_weights.
    Can be instructed to raise on entry or on set_weights.
    """

    def __init__(
        self,
        *,
        meta: DummyMetagraph | None = None,
        weights_result: bool = True,
        raise_in_enter: bool = False,
    ):
        self.meta = meta or DummyMetagraph()
        self.weights_result = weights_result
        self.raise_in_enter = raise_in_enter
        self.last_set_weights: dict[str, Any] | None = None

    async def __aenter__(self):
        if self.raise_in_enter:
            raise RuntimeError("boom")
        return self

    async def __aexit__(self, *_):
        return False

    async def metagraph(self, _netuid):
        return self.meta

    async def set_weights(
        self,
        *,
        wallet: Any,
        netuid: int,
        uids: Iterable[int],
        weights: Iterable[float],
        version_key: int,
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ) -> tuple[bool, str | None]:
        self.last_set_weights = {
            "wallet": wallet,
            "netuid": netuid,
            "uids": list(uids),
            "weights": list(weights),
            "version_key": version_key,
            "wait_for_inclusion": wait_for_inclusion,
            "wait_for_finalization": wait_for_finalization,
        }
        return self.weights_result, ""


def patch_wallet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace bt.Wallet with a dummy that records nothing."""
    chain_mod = AsyncChain.__module__
    monkeypatch.setattr(f"{chain_mod}.bt.Wallet", lambda *_, **__: SimpleNamespace())


def patch_subtensor(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    """Replace AsyncSubtensor with the supplied factory.

    The factory should return a *new* DummySubtensor each call so that
    each network URL gives its own object (mirroring the real behaviour).
    """
    chain_mod = AsyncChain.__module__
    monkeypatch.setattr(f"{chain_mod}.AsyncSubtensor", factory)
