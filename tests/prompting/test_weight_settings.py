# ruff: noqa: E402
import asyncio

import numpy as np

from shared import settings

settings.shared_settings = settings.SharedSettings(mode="mock")
raw_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
from unittest.mock import MagicMock, patch

from prompting.weight_setting.weight_setter import PAST_WEIGHTS as W_PAST_WEIGHTS
from prompting.weight_setting.weight_setter import WEIGHTS_HISTORY_LENGTH as W_HISTORY_LENGTH
from prompting.weight_setting.weight_setter import WeightSetter, apply_reward_func, compute_averaged_weights


def test_apply_reward_func():
    raw_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # test result is even returned
    result = apply_reward_func(raw_rewards)
    assert result is not None, "Result was None"

    # Test with p = 0.5 (no change)
    result = apply_reward_func(raw_rewards, p=0.5)
    assert np.allclose(
        result, raw_rewards / np.sum(raw_rewards), atol=1e-6
    ), "Result should be unchanged from raw rewards"

    # Test with p = 0 (more linear)
    result = apply_reward_func(raw_rewards, p=0)
    assert np.isclose(np.std(result), 0, atol=1e-6), "All rewards should be equal"

    # Test with p = 1 (more exponential)
    result = apply_reward_func(raw_rewards, p=1)
    assert result[-1] > 0.9, "Top miner should take vast majority of reward"

    # Test with negative values
    raw_rewards = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = apply_reward_func(raw_rewards, p=0.5)
    assert result[0] < 0, "Negative reward should remain negative"


@patch("prompting.weight_setting.weight_setter.save_weights")
def test_compute_averaged_weights(mock_save_weights):
    original_past_weights_content = list(W_PAST_WEIGHTS)
    W_PAST_WEIGHTS.clear()

    try:
        # Scenario 1: PAST_WEIGHTS is empty
        weights1 = np.array([0.1, 0.2, 0.3])
        averaged1 = compute_averaged_weights(weights1)
        assert np.allclose(averaged1, weights1), "Average with no history should be current weights"
        assert len(W_PAST_WEIGHTS) == 1, "Weights should be added to history"
        assert np.allclose(W_PAST_WEIGHTS[0], weights1), "Correct weights should be in history"
        mock_save_weights.assert_called_once()

        # Scenario 2: PAST_WEIGHTS has fewer than WEIGHTS_HISTORY_LENGTH entries
        mock_save_weights.reset_mock()
        weights2 = np.array([0.4, 0.5, 0.6])
        averaged2 = compute_averaged_weights(weights2)
        expected_average2 = np.average(np.array([weights1, weights2]), axis=0)
        assert np.allclose(averaged2, expected_average2), "Average with partial history is incorrect"
        assert len(W_PAST_WEIGHTS) == 2, "Weights should be added to history"
        mock_save_weights.assert_called_once()

        # Scenario 3: PAST_WEIGHTS is full and should pop the oldest
        mock_save_weights.reset_mock()
        W_PAST_WEIGHTS.clear()  # Start fresh for this specific scenario
        # Fill up PAST_WEIGHTS to just below capacity
        for i in range(W_HISTORY_LENGTH - 1):
            W_PAST_WEIGHTS.append(np.array([0.01 * i, 0.02 * i, 0.03 * i]))

        assert len(W_PAST_WEIGHTS) == W_HISTORY_LENGTH - 1

        weights_new = np.array([1.0, 1.0, 1.0])
        # Add one more to make it full
        compute_averaged_weights(weights_new)  # This makes it full
        assert len(W_PAST_WEIGHTS) == W_HISTORY_LENGTH
        mock_save_weights.assert_called_once()

        # Current content of W_PAST_WEIGHTS (oldest to newest):
        # [array([0.00, 0.00, 0.00])] # if W_HISTORY_LENGTH was e.g. 2, this would be the one pushed out
        # ... up to ...
        # [array([0.01*(W_HISTORY_LENGTH-2), 0.02*(W_HISTORY_LENGTH-2), 0.03*(W_HISTORY_LENGTH-2)])]
        # [array([1.0, 1.0, 1.0])]

        # Add another one, which should cause the oldest to be popped
        mock_save_weights.reset_mock()
        first_weights_in_history = W_PAST_WEIGHTS[0].copy()  # what should be popped
        weights_overflow = np.array([2.0, 2.0, 2.0])
        compute_averaged_weights(weights_overflow)

        assert len(W_PAST_WEIGHTS) == W_HISTORY_LENGTH, "History should remain at max length"
        assert not any(
            np.allclose(arr, first_weights_in_history) for arr in W_PAST_WEIGHTS
        ), "Oldest weights should have been popped"
        assert np.allclose(W_PAST_WEIGHTS[-1], weights_overflow), "Newest weights should be at the end"
        mock_save_weights.assert_called_once()

    finally:
        # Restore PAST_WEIGHTS to its original state
        W_PAST_WEIGHTS.clear()
        W_PAST_WEIGHTS.extend(original_past_weights_content)


def test_run_step_with_reward_events():
    with (
        patch("shared.uids.get_uids") as mock_get_uids,
        patch("prompting.weight_setting.weight_setter.TaskRegistry") as MockTaskRegistry,
        # patch("prompting.weight_setting.weight_setter.mutable_globals") as mock_mutable_globals,
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

        # Set up the mock mutable_globals

        weight_setter = WeightSetter()
        reward_events = [
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0], uids=mock_uids, rewards=[1.0, 2.0, 3.0, 4.0, 5.0], weight=1
                ),
            ],
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0], uids=mock_uids, rewards=[5.0, 4.0, 3.0, 2.0, 1.0], weight=1
                ),
            ],
        ]
        weight_setter.reward_events = reward_events
        output = asyncio.run(weight_setter.run_step())

        print(output)
        mock_set_weights.assert_called_once()
        call_args = mock_set_weights.call_args[0]
        assert len([c for c in call_args[0] if c > 0]) == len(mock_uids)
        assert np.isclose(np.sum(call_args[0]), 1, atol=1e-6)

        # Check that the warning about empty reward events is not logged
        mock_logger.warning.assert_not_called()


def test_run_step_ma_before_grading_order():
    manager_mock = MagicMock()

    # Define a specific array to be returned by the mocked compute_averaged_weights
    # to check if it's passed to apply_reward_func
    mock_averaged_rewards_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Ensure mocked functions return a valid numpy array so run_step can proceed
    manager_mock.compute_averaged_weights.return_value = mock_averaged_rewards_array
    manager_mock.apply_reward_func.return_value = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Example return

    # Mock METAGRAPH.n.item() to return a specific value for all_uids initialization
    mock_metagraph_n = 5

    # Minimal TaskConfig and RewardEvent setup
    class MockTask:
        pass

    class TaskConfig:
        def __init__(self, name, probability, task_class=MockTask):  # Added task_class
            self.name = name
            self.probability = probability
            self.task = task_class  # Use the provided task_class

    class WeightedRewardEvent:
        def __init__(self, task, uids, rewards, weight):
            self.task = task  # This should be an instance of the Task (e.g. InferenceTask, not its config)
            self.uids = uids
            self.rewards = rewards
            self.weight = weight
            # Add a name attribute to the task object if TaskRegistry.get_task_config might access it
            if hasattr(self.task, "name"):
                pass  # already has it
            elif hasattr(task, "name"):  # if task is a config object with a name
                self.task.name = task.name
            else:  # if task is a class like MockTask, give its instance a name
                self.task.name = "MockTaskInstance"

    mock_task_instance = MockTask()  # An instance of a task
    # We need to ensure that TaskRegistry.get_task_config(reward_event.task) works.
    # reward_event.task is an actual task instance (like InferenceTask object), not a config.
    # So, TaskRegistry.get_task_config needs to be able to map this instance to a config.
    # For this test, we'll make get_task_config return a simple mock config.

    mock_actual_task_config = TaskConfig(name="TestTask", probability=1.0)

    with patch(
        "prompting.weight_setting.weight_setter.compute_averaged_weights", manager_mock.compute_averaged_weights
    ), patch("prompting.weight_setting.weight_setter.apply_reward_func", manager_mock.apply_reward_func), patch(
        "prompting.weight_setting.weight_setter.set_weights"
    ) as mock_set_weights, patch(
        "prompting.weight_setting.weight_setter.TaskRegistry"
    ) as MockTaskRegistry, patch(
        "prompting.weight_setting.weight_setter.logger"
    ) as mock_logger, patch(
        "prompting.weight_setting.weight_setter.shared_settings.METAGRAPH.n"
    ) as mock_metagraph_n_obj:
        mock_metagraph_n_obj.item.return_value = mock_metagraph_n  # Configure the item method on the mocked 'n' object

        # Ensure TaskRegistry.get_task_config returns our mock_actual_task_config
        # when called with reward_event.task (which is mock_task_instance)
        MockTaskRegistry.get_task_config.return_value = mock_actual_task_config
        MockTaskRegistry.task_configs = [mock_actual_task_config]  # For iterating over task_configs

        weight_setter = WeightSetter()
        # Provide minimal reward events to trigger the logic
        reward_events = [
            [
                WeightedRewardEvent(
                    task=mock_task_instance,  # pass the task instance
                    uids=list(range(mock_metagraph_n)),
                    rewards=np.random.rand(mock_metagraph_n),
                    weight=1.0,
                ),
            ],
        ]
        weight_setter.reward_events = reward_events

        asyncio.run(weight_setter.run_step())

        # Assert that set_weights was called (ensuring the function ran far enough)
        mock_set_weights.assert_called_once()

        # Check the order of calls
        assert (
            manager_mock.method_calls[0][0] == "compute_averaged_weights"
        ), "compute_averaged_weights was not called first"
        assert manager_mock.method_calls[1][0] == "apply_reward_func", "apply_reward_func was not called second"

        # Verify that the output of compute_averaged_weights was passed to apply_reward_func
        # The first argument to apply_reward_func is raw_rewards
        actual_raw_rewards_arg = manager_mock.apply_reward_func.call_args[1]["raw_rewards"]
        assert np.allclose(
            actual_raw_rewards_arg, mock_averaged_rewards_array
        ), "Output of compute_averaged_weights not passed correctly to apply_reward_func"

        # Ensure no unexpected warnings
        mock_logger.warning.assert_not_called()


# def test_run_step_without_reward_events(weight_setter):
#     with (
#         patch("prompting.weight_setter.get_uids") as mock_get_uids,
#         patch("prompting.weight_setter.TaskRegistry.task_configs", new_callable=property) as mock_task_configs,
#         patch("prompting.weight_setter.mutable_globals.reward_events") as mock_reward_events,
#         patch("prompting.weight_setter.set_weights") as mock_set_weights,
#     ):

#         mock_get_uids.return_value = [1, 2, 3, 4, 5]
#         mock_task_configs.return_value = [
#             TaskConfig(name="Task1", probability=0.5),
#             TaskConfig(name="Task2", probability=0.3),
#         ]
#         mock_reward_events.return_value = []

#         weight_setter.run_step()

#         mock_set_weights.assert_not_called()


# def test_set_weights():
#     with (
#         patch("prompting.weight_setter.settings.SUBTENSOR") as mock_subtensor,
#         patch("prompting.weight_setter.settings.WALLET") as mock_wallet,
#         patch("prompting.weight_setter.settings.NETUID") as mock_netuid,
#         patch("prompting.weight_setter.settings.METAGRAPH") as mock_metagraph,
#         patch("prompting.weight_setter.pd.DataFrame.to_csv") as mock_to_csv,
#     ):

#         weights = np.array([0.1, 0.2, 0.3, 0.4])
#         uids = np.array([1, 2, 3, 4])
#         mock_metagraph.uids = uids

#         set_weights(weights)

#         # Check if weights were processed and set
#         mock_subtensor.set_weights.assert_called_once()

#         # Check if weights were logged
#         if settings.LOG_WEIGHTS:
#             mock_to_csv.assert_called_once()
