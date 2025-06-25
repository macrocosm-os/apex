# ruff: noqa: E402
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from unittest.mock import AsyncMock, patch
import pytest
from pytest import MonkeyPatch
from types import SimpleNamespace

import numpy as np

from prompting.weight_setting.weight_setter import set_weights

from shared import settings
settings.shared_settings = settings.SharedSettings(mode="mock")

from prompting.weight_setting import weight_setter
from prompting.tasks.inference import InferenceTask
from prompting.tasks.msrv2_task import MSRv2Task
from prompting.tasks.web_retrieval import WebRetrievalTask
from prompting.weight_setting.weight_setter import WeightSetter
from prompting.rewards.reward import WeightedRewardEvent


UIDS: list[int] = list(range(256))


def _make_event(task_cls: type, rewards: list[float]) -> WeightedRewardEvent:
    """Return a fully-populated WeightedRewardEvent for the given task."""
    return WeightedRewardEvent(
        weight=1.0,
        task=task_cls,
        reward_model_name="test",
        rewards=rewards,
        rewards_normalized=rewards,
        timings=[0.0] * len(rewards),
        reward_model_type="reward",
        batch_time=0.0,
        uids=UIDS,
        threshold=None,
        extra_info=None,
        reward_type="reward",
    )


@pytest.mark.asyncio
async def test_merge_task_rewards() -> None:
    negative_uids: set[int] = {0, 1, 2}
    inference_rewards: list[float] = [
        -2.0 if uid in negative_uids else 1.0 for uid in UIDS
    ]
    msrv2_rewards: list[float] = [1.0] * len(UIDS)
    web_rewards: list[float] = [1.0] * len(UIDS)

    events: list[list[WeightedRewardEvent]] = [
        [
            _make_event(InferenceTask(), inference_rewards),
            _make_event(MSRv2Task(), msrv2_rewards),
            _make_event(WebRetrievalTask(), web_rewards),
        ]
    ]

    final_rewards = await WeightSetter.merge_task_rewards(events)

    assert isinstance(final_rewards, np.ndarray)
    assert final_rewards.dtype == np.float32
    assert final_rewards.shape == (len(UIDS),)
    assert int((final_rewards < 0).sum()) == 3


def test_steepness():
    raw_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test result is even returned.
    result = WeightSetter.apply_steepness(raw_rewards)
    assert result is not None, "Result was None"

    # Test with p = 0.5 (no change).
    result = WeightSetter.apply_steepness(raw_rewards, steepness=0.5)
    assert np.allclose(
        result, raw_rewards / np.sum(raw_rewards), atol=1e-6
    ), "Result should be unchanged from raw rewards"

    # Test with p = 0 (more linear).
    result = WeightSetter.apply_steepness(raw_rewards, steepness=0)
    assert np.isclose(np.std(result), 0, atol=1e-6), "All rewards should be equal"

    # Test with p = 1 (more exponential).
    result = WeightSetter.apply_steepness(raw_rewards, steepness=1)
    assert result[-1] > 0.9, "Top miner should take vast majority of reward"

    # Test with negative values.
    raw_rewards = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = WeightSetter.apply_steepness(raw_rewards, steepness=0.5)
    assert result[0] < 0, "Negative reward should remain negative"


def test_run_step_with_reward_events():
    with (
        patch("shared.uids.get_uids") as mock_get_uids,
        patch("prompting.weight_setting.weight_setter.TaskRegistry") as MockTaskRegistry,
        patch("prompting.weight_setting.weight_setter.set_weights") as mock_set_weights,
        patch("prompting.weight_setting.weight_setter.logger") as mock_logger,
    ):
        class MockTask:
            pass

        class TaskConfig:
            def __init__(self, name, probability):
                self.name = name
                self.probability = probability
                self.task = MockTask

        class WeightedRewardEvent:
            def __init__(self, task, uids, rewards, weight):
                self.task = task
                self.uids = uids
                self.rewards = rewards
                self.weight = weight

        mock_uids = [1, 2, 3, 4, 5]
        mock_get_uids.return_value = mock_uids

        # Set up the mock TaskRegistry
        mock_task_registry = MockTaskRegistry
        mock_task_registry.task_configs = [
            TaskConfig(name="Task1", probability=0.5),
        ]
        mock_task_registry.get_task_config = MagicMock(return_value=mock_task_registry.task_configs[0])

        # Set up the mock mutable_globals.

        weight_setter = WeightSetter(reward_history_path=Path("test_validator_rewards.jsonl"))
        reward_events = [
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0],
                    uids=mock_uids,
                    rewards=[1.0, 2.0, 3.0, 4.0, 5.0],
                    weight=1,
                ),
            ],
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0],
                    uids=mock_uids,
                    rewards=[-5.0, -4.0, -3.0, -2.0, -1.0],
                    weight=1,
                ),
            ],
        ]
        weight_setter.reward_events = reward_events
        asyncio.run(weight_setter.run_step())

        mock_set_weights.assert_called_once()
        call_args = mock_set_weights.call_args[0]
        weights = call_args[0]

        assert weights[0] <= 0
        assert weights[1] <= 0
        assert weights[2] == 0
        assert weights[3] >= 0
        assert weights[4] >= 0

        # Weights are re-normalised to 1.
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

        # Check that the warning about empty reward events is not logged.
        mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_set_weights(monkeypatch: MonkeyPatch):
    """`set_weights` calls Subtensor.set_weights with processed vectors."""
    stub_settings = SimpleNamespace(
        NEURON_DISABLE_SET_WEIGHTS=False,
        UID=0,
        NETUID=42,
        WALLET="dummy-wallet",
        METAGRAPH=SimpleNamespace(uids=np.arange(4, dtype=np.uint16)),
    )

    subtensor_mock = MagicMock()
    subtensor_mock.set_weights = MagicMock(return_value=(True, "ok"))
    stub_settings.SUBTENSOR = subtensor_mock
    monkeypatch.setattr(weight_setter, "shared_settings", stub_settings)

    monkeypatch.setattr(
        weight_setter.bt.utils.weight_utils,
        "process_weights_for_netuid",
        lambda *, uids, weights, **_: (uids, weights),
    )
    monkeypatch.setattr(
        weight_setter.bt.utils.weight_utils,
        "convert_weights_and_uids_for_emit",
        lambda uids, weights: (uids.astype(np.uint16), (weights * 65535).astype(np.uint16)),
    )

    class _Syncer:
        async def get_augmented_weights(self, *, weights: np.ndarray, uid: int) -> np.ndarray:  # noqa: D401
            return weights

    raw = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    await weight_setter.set_weights(
        raw,
        subtensor=stub_settings.SUBTENSOR,
        metagraph=stub_settings.METAGRAPH,
        weight_syncer=_Syncer(),
    )

    subtensor_mock.set_weights.assert_called_once()
    call_kwargs = subtensor_mock.set_weights.call_args.kwargs

    expected_uint16 = (raw * 65535).astype(np.uint16)
    assert np.array_equal(call_kwargs["weights"], expected_uint16)
