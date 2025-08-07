from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from pytest import MonkeyPatch

from apex.common.models import MinerGeneratorResults
from apex.validator.miner_sampler import MinerInfo, MinerSampler


class MockAxon(BaseModel):
    """A mock for the axon object."""

    ip: str
    port: int


class MockMetagraph(BaseModel):
    """A mock for the metagraph object."""

    n: Any
    stake: list[float]
    axons: list[MockAxon]
    uids: list[int]
    hotkeys: list[str]


class MockAsyncChain:
    """A mock for the AsyncChain."""

    def __init__(self, metagraph: MockMetagraph) -> None:
        self._metagraph = metagraph

    async def metagraph(self) -> MockMetagraph:
        return self._metagraph


@pytest.fixture
def mock_metagraph() -> MockMetagraph:
    """Returns a mock metagraph object."""
    return MockMetagraph(
        n=MagicMock(item=lambda: 5),
        stake=[10000, 20000, 5000, 30000, 15000],
        axons=[
            MockAxon(ip="1.1.1.1", port=8000),
            MockAxon(ip="2.2.2.2", port=8001),
            MockAxon(ip="3.3.3.3", port=8002),
            MockAxon(ip="4.4.4.4", port=8003),
            MockAxon(ip="5.5.5.5", port=8004),
        ],
        uids=[1, 2, 3, 4, 5],
        hotkeys=["key1", "key2", "key3", "key4", "key5"],
    )


@pytest.fixture
def mock_chain(mock_metagraph: MockMetagraph) -> MockAsyncChain:
    """Returns a mock chain object."""
    return MockAsyncChain(mock_metagraph)


@pytest.fixture
def miner_sampler(mock_chain: MockAsyncChain) -> Generator[MinerSampler, None, None]:
    """Returns a miner sampler object."""
    yield MinerSampler(chain=mock_chain, validator_min_stake=16000)  # type: ignore


def test_miner_info_equality() -> None:
    """Tests the equality of two MinerInfo objects."""
    info1 = MinerInfo(hotkey="key1", uid=1, address="addr1")
    info2 = MinerInfo(hotkey="key1", uid=1, address="addr1")
    info3 = MinerInfo(hotkey="key2", uid=2, address="addr2")
    assert info1 == info2
    assert info1 != info3
    assert info1 != "not_a_miner_info"


def test_miner_info_hash() -> None:
    """Tests the hash of a MinerInfo object."""
    info1 = MinerInfo(hotkey="key1", uid=1, address="addr1")
    info2 = MinerInfo(hotkey="key1", uid=1, address="addr1")
    info3 = MinerInfo(hotkey="key2", uid=2, address="addr2")
    s = {info1, info2, info3}
    assert len(s) == 2


@pytest.mark.asyncio
async def test_get_all_miners(miner_sampler: MinerSampler, mock_metagraph: MockMetagraph) -> None:
    """Tests that all miners are returned."""
    miners = await miner_sampler._get_all_miners()
    assert len(miners) == 3
    uids = {m.uid for m in miners}
    assert uids == {1, 3, 5}
    hotkeys = {m.hotkey for m in miners}
    assert hotkeys == {"key1", "key3", "key5"}
    addresses = {m.address for m in miners}
    assert addresses == {"http://1.1.1.1:8000", "http://3.3.3.3:8002", "http://5.5.5.5:8004"}


@pytest.mark.asyncio
async def test_get_all_miners_with_available_uids(mock_chain: MockAsyncChain) -> None:
    """Tests that all miners are returned when a list of available UIDs is provided."""
    # 10 is not in metagraph.
    sampler = MinerSampler(
        chain=mock_chain,  # type: ignore
        available_uids=[1, 5, 10],
        validator_min_stake=16000,
    )
    miners = await sampler._get_all_miners()
    assert len(miners) == 2
    uids = {m.uid for m in miners}
    assert uids == {1, 5}


@pytest.mark.asyncio
async def test_get_all_miners_with_available_uids_and_addresses(mock_chain: MockAsyncChain) -> None:
    """Tests that all miners are returned when a list of available UIDs and addresses is provided."""
    sampler = MinerSampler(
        chain=mock_chain,  # type: ignore
        available_uids=[1, 3],
        available_addresses=["http://localhost:1234", "http://localhost:5678"],
        validator_min_stake=16000,
    )
    miners = await sampler._get_all_miners()
    assert len(miners) == 2
    miner1 = next(m for m in miners if m.uid == 1)
    miner3 = next(m for m in miners if m.uid == 3)
    assert miner1.address == "http://localhost:1234"
    assert miner3.address == "http://localhost:5678"


@pytest.mark.asyncio
async def test_sample_miners_random(miner_sampler: MinerSampler) -> None:
    """Tests that a random sample of miners is returned."""
    miner_sampler._sample_mode = "random"
    miner_sampler._sample_size = 2

    with patch(
        "random.sample",
        return_value=[
            MinerInfo(hotkey="key1", uid=1, address="http://1.1.1.1:8000"),
            MinerInfo(hotkey="key3", uid=3, address="http://3.3.3.3:8002"),
        ],
    ) as mock_random_sample:
        miners = await miner_sampler._sample_miners()
        assert len(miners) == 2
        mock_random_sample.assert_called_once()
        all_miners = await miner_sampler._get_all_miners()
        arg_uids = {m.uid for m in mock_random_sample.call_args[0][0]}
        all_uids = {m.uid for m in all_miners}
        assert arg_uids == all_uids
        assert mock_random_sample.call_args[0][1] == 2


@pytest.mark.asyncio
async def test_sample_miners_sequential(monkeypatch: MagicMock, miner_sampler: MinerSampler) -> None:
    """Tests that a sequential sample of miners is returned."""
    miner_sampler._sample_mode = "sequential"
    miner_sampler._sample_size = 2

    all_miners = await miner_sampler._get_all_miners()
    all_miners.sort(key=lambda m: m.uid)
    monkeypatch.setattr(miner_sampler, "_get_all_miners", AsyncMock(return_value=all_miners))

    # 1st call in epoch.
    with patch("random.sample", return_value=[0, 2]):
        miners1 = await miner_sampler._sample_miners()

    assert len(miners1) == 2
    assert {m.uid for m in miners1} == {all_miners[0].uid, all_miners[2].uid}
    assert len(miner_sampler._remaining_epoch_miners) == 1

    # 2nd call, new epoch starts as remaining (1) < sample_size (2).
    with patch("random.sample", return_value=[1, 2]):
        miners2 = await miner_sampler._sample_miners()

    assert len(miners2) == 2
    assert {m.uid for m in miners2} == {all_miners[1].uid, all_miners[2].uid}
    assert len(miner_sampler._remaining_epoch_miners) == 1


@pytest.mark.asyncio
async def test_query_miners() -> None:
    """Tests that a query to a miner is successful."""
    mock_chain = MagicMock()
    mock_chain.wallet.hotkey.ss58_address = "test_address"
    sampler = MinerSampler(chain=mock_chain)  # type: ignore
    endpoint = "http://test.com"
    body = {"test": "data"}

    mock_resp = AsyncMock()
    mock_resp.text = AsyncMock(return_value='{"response": "ok"}')

    mock_session_post = MagicMock()
    mock_session_post.__aenter__.return_value = mock_resp
    mock_session_post.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.post.return_value = mock_session_post
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    with patch("aiohttp.ClientSession", return_value=mock_session) as mock_client_session, patch(
        "apex.validator.miner_sampler.generate_header"
    ) as mock_generate_header, patch("time.time", return_value=12345):
        mock_generate_header.return_value = {"some": "header"}
        result = await sampler.query_miners(body, endpoint)

        mock_client_session.assert_called_once()
        expected_body = {"test": "data", "signer": "test_address", "signed_for": "http://test.com", "nonce": "12345"}
        mock_session.post.assert_called_with(
            endpoint + "/v1/chat/completions", headers={"some": "header"}, json=expected_body
        )
        mock_generate_header.assert_called_with(mock_chain.wallet.hotkey, expected_body)
        assert result == '{"response": "ok"}'


@pytest.mark.asyncio
async def test_query_generators(monkeypatch: MonkeyPatch, miner_sampler: MinerSampler) -> None:
    """Tests that a query to a generator is successful."""
    monkeypatch.setattr(
        miner_sampler,
        "_sample_miners",
        AsyncMock(
            return_value=[
                MinerInfo(hotkey="key1", uid=1, address="http://1.1.1.1:8000"),
                MinerInfo(hotkey="key3", uid=3, address="http://3.3.3.3:8002"),
            ],
        ),
    )
    query_miners_mock: AsyncMock = AsyncMock(side_effect=["result1", "result2"])
    monkeypatch.setattr(miner_sampler, "query_miners", AsyncMock(side_effect=query_miners_mock))

    query = "test query"
    results = await miner_sampler.query_generators(query)

    assert isinstance(results, MinerGeneratorResults)
    assert results.query == query
    assert results.generator_hotkeys == ["key1", "key3"]
    assert results.generator_results == ["result1", "result2"]

    assert query_miners_mock.call_count == 2  # type: ignore
    query_miners_mock.assert_any_call(body={"step": "generator", "query": query}, endpoint="http://1.1.1.1:8000")
    query_miners_mock.assert_any_call(  # type: ignore
        body={"step": "generator", "query": query}, endpoint="http://3.3.3.3:8002"
    )


@pytest.mark.asyncio
@patch("random.random", return_value=0.4)
@patch("random.choice")
async def test_query_discriminators_selects_generator(
    mock_random_choice: MagicMock, mock_random_random: MagicMock, monkeypatch: MonkeyPatch, miner_sampler: MinerSampler
) -> None:
    """Tests that a query to a discriminator is successful when a generator is selected."""
    mock_random_choice.return_value = ("gen_key1", "gen_result1")

    monkeypatch.setattr(
        miner_sampler,
        "_sample_miners",
        AsyncMock(
            return_value=[
                MinerInfo(hotkey="disc_key1", uid=10, address="http://10.1.1.1:8000"),
                MinerInfo(hotkey="disc_key2", uid=11, address="http://11.1.1.1:8000"),
            ],
        ),
    )

    monkeypatch.setattr(
        miner_sampler,
        "query_miners",
        AsyncMock(
            side_effect=[
                '{"choices": [{"message": {"content": "1"}}]}',
                '{"choices": [{"message": {"content": "0"}}]}',
            ],
        ),
    )

    generator_results = MinerGeneratorResults(
        query="test query", generator_hotkeys=["gen_key1", "gen_key2"], generator_results=["gen_result1", "gen_result2"]
    )
    reference = "reference text"

    results = await miner_sampler.query_discriminators(
        query="test query", generator_results=generator_results, reference=reference, ground_truth=1
    )

    assert results.generator_hotkey == "gen_key1"
    assert results.discriminator_hotkeys == ["disc_key1", "disc_key2"]
    assert results.discriminator_results == ["1", "0"]
    assert results.discriminator_scores == [0.5, 0.0]
    assert results.generator_score == 0.5


@pytest.mark.asyncio
@patch("random.random", return_value=0.6)
async def test_query_discriminators_selects_reference(
    mock_random_random: MagicMock, monkeypatch: MonkeyPatch, miner_sampler: MinerSampler
) -> None:
    """Tests that a query to a discriminator is successful when a reference is selected."""
    monkeypatch.setattr(
        miner_sampler,
        "_sample_miners",
        AsyncMock(
            return_value=[
                MinerInfo(hotkey="disc_key1", uid=10, address="http://10.1.1.1:8000"),
                MinerInfo(hotkey="disc_key2", uid=11, address="http://11.1.1.1:8000"),
            ],
        ),
    )

    monkeypatch.setattr(
        miner_sampler,
        "query_miners",
        AsyncMock(
            side_effect=[
                '{"choices": [{"message": {"content": "0"}}]}',
                '{"choices": [{"message": {"content": "1"}}]}',
            ],
        ),
    )

    generator_results = MinerGeneratorResults(
        query="test query", generator_hotkeys=["gen_key1", "gen_key2"], generator_results=["gen_result1", "gen_result2"]
    )
    reference = "reference text"

    results = await miner_sampler.query_discriminators(
        query="test query", generator_results=generator_results, reference=reference, ground_truth=0
    )

    assert results.generator_hotkey == "Validator"
    assert results.generator_result == reference
    assert results.discriminator_hotkeys == ["disc_key1", "disc_key2"]
    assert results.discriminator_results == ["0", "1"]
    assert results.discriminator_scores == [0.5, 0.0]
    assert results.generator_score == 0.5


@pytest.mark.asyncio
@patch("random.random", return_value=0.6)
@pytest.mark.parametrize(
    "miner_response, expected_content, expected_score",
    [
        ('{"choices": [{"message": {"content": "0"}}]}', "0", 1.0),
        ('{"choices": [{"message": {"content": "1"}}]}', "1", 0.0),
        ("0", "0", 1.0),
        ("1", "1", 0.0),
        ("", "None", 0.0),
        ("invalid json", "invalid json", 0.0),
        (None, "None", 0.0),
    ],
)
async def test_query_discriminators_response_parsing(
    mock_random: MagicMock,
    monkeypatch: MonkeyPatch,
    miner_sampler: MinerSampler,
    miner_response: Any,
    expected_content: str,
    expected_score: float,
) -> None:
    """Tests that a query to a discriminator is successful when the response is parsed."""
    monkeypatch.setattr(
        miner_sampler,
        "_sample_miners",
        AsyncMock(
            return_value=[
                MinerInfo(hotkey="disc_key1", uid=10, address="http://10.1.1.1:8000"),
            ],
        ),
    )
    monkeypatch.setattr(miner_sampler, "query_miners", AsyncMock(return_value=miner_response))

    generator_results = MinerGeneratorResults(
        query="test query", generator_hotkeys=["gen_key1"], generator_results=["gen_result1"]
    )
    reference = "reference text"

    results = await miner_sampler.query_discriminators(
        query="test query", generator_results=generator_results, reference=reference, ground_truth=0
    )

    assert results.discriminator_results == [expected_content]
    assert results.discriminator_scores == [expected_score]


@pytest.mark.asyncio
async def test_query_discriminators_with_db_log(monkeypatch: MonkeyPatch, miner_sampler: MinerSampler) -> None:
    """Tests that a query to a discriminator is successful when a db log is provided."""
    mock_logger_db = AsyncMock()
    miner_sampler._logger_db = mock_logger_db

    monkeypatch.setattr(
        miner_sampler,
        "_sample_miners",
        AsyncMock(
            return_value=[
                MinerInfo(hotkey="disc_key1", uid=10, address="http://10.1.1.1:8000"),
            ],
        ),
    )
    monkeypatch.setattr(
        miner_sampler, "query_miners", AsyncMock(return_value='{"choices": [{"message": {"content": "0"}}]}')
    )

    with patch("random.random", return_value=0.6):
        generator_results = MinerGeneratorResults(
            query="test query", generator_hotkeys=["gen_key1"], generator_results=["gen_result1"]
        )
        reference = "reference text"

        results = await miner_sampler.query_discriminators(
            query="test query", generator_results=generator_results, reference=reference, ground_truth=0
        )

        mock_logger_db.log.assert_called_once_with(results)
