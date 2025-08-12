import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apex.validator.weight_syncer import WeightSyncer
from tests.common.mock_async_chain import DummyMetagraph


class UidList(list):
    """A list that has a .tolist() method for compatibility with torch tensors."""

    def tolist(self):
        return self


@pytest.fixture
def mock_axon():
    """Returns a function to create a mock axon."""

    def _mock_axon(ip, port, is_serving=True):
        axon = MagicMock()
        axon.ip = ip
        axon.port = port
        axon.is_serving = is_serving
        return axon

    return _mock_axon


@pytest.fixture
def mock_metagraph(mock_axon):
    """Returns a mock metagraph based on DummyMetagraph."""
    metagraph = DummyMetagraph(
        hotkeys=["hotkey0_self", "hotkey1_validator", "hotkey2_validator"],
    )
    # Overwrite uids with our special UidList to provide .tolist()
    metagraph.uids = UidList([0, 1, 2])
    metagraph.stake = [np.float32(1000.0), np.float32(2000.0), np.float32(500.0)]
    metagraph.validator_permit = [True, True, False]
    metagraph.axons = [
        mock_axon("1.1.1.1", 8000),
        mock_axon("2.2.2.2", 8001),
        mock_axon("3.3.3.3", 8002),
    ]
    return metagraph


@pytest.fixture
def mock_chain(mock_metagraph):
    """Returns a mock chain with a mock metagraph."""
    chain = MagicMock()
    chain.wallet.hotkey.ss58_address = "hotkey0_self"
    chain.metagraph = AsyncMock(return_value=mock_metagraph)
    return chain


@pytest.fixture
def weight_syncer(mock_chain):
    """Returns a WeightSyncer instance with a mock chain."""
    return WeightSyncer(chain=mock_chain, min_alpha_stake=1000)


@pytest.mark.asyncio
async def test_compute_weighted_rewards_happy_path(weight_syncer, mock_metagraph):
    """Test that weighted rewards are computed correctly in the ideal case."""
    local_rewards = {"miner1": 0.9, "miner2": 0.1}
    validator1_rewards = {"miner1": 0.85, "miner2": 0.82, "miner3": 0.7}

    with patch.object(weight_syncer, "receive_rewards", new_callable=AsyncMock) as mock_receive:
        mock_receive.side_effect = [validator1_rewards, {}]  # UID 2 has low stake

        weighted_rewards = await weight_syncer.compute_weighted_rewards(local_rewards)

        # self (1000) + validator1 (2000) = 3000 total stake
        # miner1: (0.9 * 1000 + 0.85 * 2000) / 3000 = 0.8666
        # miner2: (0.1 * 1000 + 0.82 * 2000) / 3000 = 0.58
        # miner3: (0.0 * 1000 + 0.70 * 2000) / 3000 = 0.4666
        assert mock_receive.call_count == 1
        assert mock_receive.call_args.args[1] == 1  # Called for UID 1
        assert pytest.approx(weighted_rewards["miner1"], 0.001) == 0.8666
        assert pytest.approx(weighted_rewards["miner2"], 0.001) == 0.58
        assert pytest.approx(weighted_rewards["miner3"], 0.001) == 0.4666


@pytest.mark.asyncio
async def test_compute_weighted_rewards_self_not_in_metagraph(weight_syncer, mock_metagraph):
    """Test that local rewards are returned if the validator's hotkey is not in the metagraph."""
    mock_metagraph.hotkeys = ["other1", "other2", "other3"]
    local_rewards = {"miner1": 0.9}
    weighted_rewards = await weight_syncer.compute_weighted_rewards(local_rewards)
    assert weighted_rewards == local_rewards


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_receive_rewards_success(mock_async_client, weight_syncer, mock_metagraph):
    """Test successfully receiving rewards from another validator."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"miner1": 0.9}
    mock_async_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    rewards = await weight_syncer.receive_rewards(mock_metagraph, 1)
    assert rewards == {"miner1": 0.9}


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_receive_rewards_http_error(mock_async_client, weight_syncer, mock_metagraph):
    """Test that an empty dict is returned on HTTP error."""
    mock_async_client.return_value.__aenter__.return_value.post.side_effect = Exception("HTTP Error")
    rewards = await weight_syncer.receive_rewards(mock_metagraph, 1)
    assert rewards == {}


@patch("apex.validator.weight_syncer.verify_validator_signature", new_callable=AsyncMock)
def test_get_rewards_endpoint(mock_verify_signature, weight_syncer):
    """Test the FastAPI endpoint for serving rewards."""
    app = FastAPI()
    app.include_router(weight_syncer.get_router())
    client = TestClient(app)

    # Case 1: No rewards set yet
    response = client.post("/v1/get_rewards")
    assert response.status_code == 503

    # Case 2: Rewards are set and not expired
    weight_syncer.hotkey_rewards = {"miner1": 0.95}
    weight_syncer.last_update_time = time.time()
    response = client.post("/v1/get_rewards")
    assert response.status_code == 200
    assert response.json() == {"miner1": 0.95}

    # Case 3: Rewards are expired
    weight_syncer.last_update_time = time.time() - WeightSyncer.REWARD_EXPIRATION_SEC - 1
    response = client.post("/v1/get_rewards")
    assert response.status_code == 503
