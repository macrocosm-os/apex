from unittest.mock import AsyncMock

import pytest

from apex.common.async_chain import AsyncChain  # noqa: E402
from tests.common.mock_async_chain import DummyMetagraph, DummySubtensor, patch_subtensor, patch_wallet


@pytest.mark.asyncio
async def test_connect_success(monkeypatch):
    """All endpoints healthy ➔ lists populated and marked alive."""
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: DummySubtensor())

    chain = AsyncChain("cold", "hot", 0, network=["ws://a", "ws://b"])
    await chain.start()

    assert len(chain._subtensors) == 2
    assert chain._subtensor_alive == [True, True]


@pytest.mark.asyncio
async def test_connect_failure(monkeypatch):
    """Every endpoint explodes ➔ ValueError raised."""
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: DummySubtensor(raise_in_enter=True))

    chain = AsyncChain("ck", "hk", 0, network=["ws://badndpoint"])
    with pytest.raises(ValueError):
        await chain.start()


@pytest.mark.asyncio
async def test_metagraph_cached(monkeypatch):
    """Second call should hit the async_cache-d value (same object)."""
    meta = DummyMetagraph(["hk"], [9])

    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: DummySubtensor(meta=meta))

    chain = AsyncChain("ck", "hk", 1, ["ws://goodendpoint"])
    await chain.start()

    m1 = await chain.metagraph()
    # Metagraph cache hit.
    m2 = await chain.metagraph()

    assert m1 is meta and m2 is meta


@pytest.mark.asyncio
async def test_metagraph_all_endpoints_dead(monkeypatch):
    """If every subtensor.metagraph fails ➔ ValueError raised."""

    async def _corrupted(self, _):  # noqa: D401, ANN001
        raise RuntimeError("Failed to connect")

    monkeypatch.setattr(DummySubtensor, "metagraph", _corrupted)
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: DummySubtensor())

    chain = AsyncChain("ck", "hk", 1, ["ws://badendpoint"])
    await chain.start()

    with pytest.raises(ValueError):
        await chain.metagraph()


@pytest.mark.asyncio
async def test_subtensor_first_alive(monkeypatch):
    """subtensor() returns the first still-alive instance."""
    healthy = DummySubtensor()
    dead = DummySubtensor()

    # Make a factory that returns one dead then one healthy subtensor.
    def factory(**_):
        return dead if not hasattr(factory, "made") else healthy

    factory.made = True  # type: ignore[attr-defined]  # noqa: B018

    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, factory)

    chain = AsyncChain("ck", "hk", 0, ["ws://endpoint1", "ws://endpoint2"])
    await chain.start()
    # Manually mark first as dead.
    chain._subtensor_alive[0] = False

    st = await chain.subtensor()
    assert st is healthy


@pytest.mark.asyncio
async def test_subtensor_none_alive(monkeypatch):
    """If every subtensor is dead ➔ ValueError."""
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: DummySubtensor())

    chain = AsyncChain("ck", "hk", 0, ["ws://endpoint"])
    await chain.start()
    chain._subtensor_alive[0] = False

    with pytest.raises(ValueError):
        await chain.subtensor()


@pytest.mark.asyncio
async def test_set_weights_happy_path(monkeypatch):
    """Valid hotkeys → correct uid/weight list → returns True."""
    meta = DummyMetagraph(["hk0", "hk1"], [1, 2])
    stub = DummySubtensor(meta=meta)
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: stub)

    chain = AsyncChain("cold", "hk0", 5, ["ws://goodendpoint"])
    await chain.start()

    ok = await chain.set_weights({"hk1": 0.7, "unknown": 1.0})

    assert ok is True
    assert stub.last_set_weights is not None
    assert stub.last_set_weights["uids"] == [2]
    assert stub.last_set_weights["weights"] == [0.7]
    # ensure we pass spec version as version_key
    from apex import __spec_version__

    assert stub.last_set_weights["version_key"] == __spec_version__


@pytest.mark.asyncio
async def test_set_weights_remote_reject(monkeypatch):
    """Remote returns False ➔ set_weights returns False too."""
    meta = DummyMetagraph(["hk"], [4])
    stub = DummySubtensor(meta=meta, weights_result=False)
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: stub)

    c = AsyncChain("cold", "hk", 3, ["ws://goodendpoint"])
    await c.start()

    assert await c.set_weights({"hk": 0.1}) is False


@pytest.mark.asyncio
async def test_disconnect_cleans(monkeypatch):
    """disconnect() clears internal lists and closes CMs."""
    cm = DummySubtensor()
    cm.__aexit__ = AsyncMock(return_value=None)  # type: ignore[method-assign]
    patch_wallet(monkeypatch)
    patch_subtensor(monkeypatch, lambda **_: cm)

    chain = AsyncChain("ck", "hk", 0, ["ws://goodendpoint"])
    await chain.start()
    await chain.shutdown()

    cm.__aexit__.assert_called_once()
    assert chain._subtensors == [] and chain._subtensor_cm == []


@pytest.mark.asyncio
async def test_mask_network(monkeypatch):
    """Only `ws://` URLs get redacted."""
    patch_wallet(monkeypatch)

    chain = AsyncChain("ck", "hk", 0, ["ws://abcdefg:1234", "finney"])
    masked = await chain.mask_network()

    assert masked[0].startswith("ws://") and "***" in masked[0]
    assert masked[1] == "finney"
