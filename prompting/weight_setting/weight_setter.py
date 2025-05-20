import asyncio
import os

import bittensor as bt
import numpy as np
import pandas as pd
from loguru import logger

from prompting import __spec_version__
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.inference import InferenceTask
from prompting.tasks.task_registry import TaskConfig, TaskRegistry
from shared import settings
from shared.loop_runner import AsyncLoopRunner
from shared.misc import ttl_get_block

shared_settings = settings.shared_settings

FILENAME = "validator_weights.npz"
WEIGHTS_HISTORY_LENGTH = 24
PAST_WEIGHTS: list[np.ndarray] = []


def apply_reward_func(raw_rewards: np.ndarray, p=0.5):
    """
    Apply the reward function to the raw rewards. P adjusts the steepness of the function - p = 0.5 leaves
    the rewards unchanged, p < 0.5 makes the function more linear (at p=0 all miners with positives reward values get the same reward),
    p > 0.5 makes the function more exponential (winner takes all).
    """
    exponent = (p**6.64385619) * 100  # 6.64385619 = ln(100)/ln(2) -> this way if p=0.5, the exponent is exatly 1
    raw_rewards = np.array(raw_rewards) / max(1, (np.sum(raw_rewards[raw_rewards > 0]) + 1e-10))
    positive_rewards = np.clip(raw_rewards, 1e-10, np.inf)
    normalised_rewards = positive_rewards / np.max(positive_rewards)
    post_func_rewards = normalised_rewards**exponent
    all_rewards = post_func_rewards / (np.sum(post_func_rewards) + 1e-10)
    all_rewards[raw_rewards <= 0] = raw_rewards[raw_rewards <= 0]

    return all_rewards


def save_weights(weights: list[np.ndarray]):
    """Saves the list of numpy arrays to a file."""
    # Save all arrays into a single .npz file
    np.savez_compressed(FILENAME, *weights)


def compute_averaged_weights(weights: np.ndarray) -> np.ndarray:
    """
    Computes the moving average of past weights to smooth fluctuations.
    """
    PAST_WEIGHTS.append(weights)
    if len(PAST_WEIGHTS) > WEIGHTS_HISTORY_LENGTH:
        PAST_WEIGHTS.pop(0)
    averaged_weights = np.average(np.array(PAST_WEIGHTS), axis=0)
    save_weights(PAST_WEIGHTS)

    return averaged_weights


def set_weights(
    weights: np.ndarray, step: int = 0, subtensor: bt.Subtensor | None = None, metagraph: bt.Metagraph | None = None
):
    """
    Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
    """
    # Check if self.scores contains any NaN values and log a warning if it does.
    try:
        if any(np.isnan(weights).flatten()):
            logger.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions. Scores: {weights}"
            )

        # Replace any NaN values with 0.
        weights_for_bt_processing = np.nan_to_num(weights, nan=0.0)

        # Process the weights_for_bt_processing via subtensor limitations.
        (
            processed_weight_uids,
            processed_chain_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=shared_settings.METAGRAPH.uids,
            weights=weights_for_bt_processing,  # Use the already averaged and graded weights
            netuid=shared_settings.NETUID,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_chain_weights
        )
    except Exception as ex:
        logger.exception(f"Issue with setting weights: {ex}")

    # Create a dataframe from weights and uids and save it as a csv file, with the current step as the filename.
    if shared_settings.LOG_WEIGHTS:
        try:
            weights_df = pd.DataFrame(
                {
                    "step": step,
                    "uids": uint_uids,  # UIDs actually set on chain
                    "weights_on_chain": uint_weights.astype(
                        float
                    ).flatten(),  # Weights actually set on chain (converted from uint16)
                    "weights_input_to_set_weights": str(
                        list(weights_for_bt_processing.flatten())
                    ),  # Log the state before bt.utils.process_weights_for_netuid but after MA+Grading
                    "block": ttl_get_block(subtensor=subtensor),
                }
            )
            step_filename = "weights.csv"
            file_exists = os.path.isfile(step_filename)
            # Append to the file if it exists, otherwise write a new file.
            weights_df.to_csv(step_filename, mode="a", index=False, header=not file_exists)
        except Exception as ex:
            logger.exception(f"Couldn't write to df: {ex}")

    if shared_settings.NEURON_DISABLE_SET_WEIGHTS:
        logger.debug(f"Set weights disabled: {shared_settings.NEURON_DISABLE_SET_WEIGHTS}")
        return

    # Set the weights on chain via our subtensor connection.
    result = subtensor.set_weights(
        wallet=shared_settings.WALLET,
        netuid=shared_settings.NETUID,
        uids=uint_uids,
        weights=uint_weights,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        version_key=__spec_version__,
    )

    if result[0]:
        logger.info("Successfully set weights on chain")
    else:
        logger.error(f"Failed to set weights on chain: {result}")


class WeightSetter(AsyncLoopRunner):
    """The weight setter looks at RewardEvents in the reward_events queue and sets the weights of the miners accordingly."""

    sync: bool = True
    interval: int = 60 * 25  # set rewards every 25 minutes
    reward_events: list[list[WeightedRewardEvent]] | None = None
    subtensor: bt.Subtensor | None = None
    metagraph: bt.Metagraph | None = None

    class Config:
        arbitrary_types_allowed = True

    async def start(self, reward_events, name: str | None = None, **kwargs):
        self.reward_events = reward_events
        global PAST_WEIGHTS  # Keeps the past 8 hours of weights (WEIGHTS_HISTORY_LENGTH)

        try:
            with np.load(FILENAME) as data:
                PAST_WEIGHTS = [data[key] for key in data.files]
        except FileNotFoundError:
            logger.info("No weights file found - this is expected on a new validator, starting with empty weights")
            PAST_WEIGHTS = []
        except Exception as ex:
            logger.error(f"Couldn't load weights from file: {ex}")
        return await super().start(name=name, **kwargs)

    async def run_step(self):
        await asyncio.sleep(0.01)

        all_uids = list(range(shared_settings.METAGRAPH.n.item()))
        # empty weights for fallback in case of exceptions
        final_weights_for_chain = np.zeros(len(all_uids), dtype=float)

        try:
            if not self.reward_events:
                logger.warning("No reward events in queue, skipping weight setting...")
                # In this case, final_weights_for_chain remains all zeros, which will be set on chain.
                set_weights(
                    final_weights_for_chain,
                    step=self.step,
                    subtensor=shared_settings.SUBTENSOR,
                    metagraph=shared_settings.METAGRAPH,
                )
                self.reward_events = []  # Ensure it's an empty
                await asyncio.sleep(0.01)
                return final_weights_for_chain

            # reward_dict = {uid: 0 for uid in get_uids(sampling_mode="all")}
            logger.info(f"Setting weights for {len(all_uids)} uids")
            # miner_rewards is a dictionary that separates each task config into a dictionary of uids with their rewards
            miner_rewards: dict[TaskConfig, dict[int, dict[str, float]]] = {
                config: {uid: {"reward": 0.0, "count": 0.0} for uid in all_uids} for config in TaskRegistry.task_configs
            }

            # Process all reward events
            for reward_event_list in self.reward_events:
                await asyncio.sleep(0.01)
                for reward_event in reward_event_list:
                    if np.sum(reward_event.rewards) > 0:
                        # Consider making this log more specific or less frequent if too noisy
                        logger.debug(
                            f"Processing positive reward event for task: {reward_event.task.name if reward_event.task else 'Unknown'}"
                        )
                    task_config = TaskRegistry.get_task_config(reward_event.task)

                    # Accumulate rewards for each UID, weighted by event weight
                    for uid, reward_value in zip(reward_event.uids, reward_event.rewards):
                        if uid in miner_rewards[task_config]:  # Ensure UID is valid
                            miner_rewards[task_config][uid]["reward"] += reward_value * reward_event.weight
                            miner_rewards[task_config][uid]["count"] += 1 * reward_event.weight
                        else:
                            logger.warning(f"UID {uid} from reward event not in all_uids. Skipping this reward entry.")

            # No separate processing for inference_events needed here;
            # all events are accumulated in the loop above.
            # Specific normalization for inference tasks will happen during the
            # calculation of r_values_for_task.

            raw_aggregated_rewards_array = np.zeros(len(all_uids), dtype=float)

            for task_config, per_task_rewards_map in miner_rewards.items():
                task_uid_rewards = np.array([per_task_rewards_map[uid]["reward"] for uid in all_uids])
                task_uid_counts = np.array([per_task_rewards_map[uid]["count"] for uid in all_uids])

                r_values_for_task = np.zeros_like(task_uid_rewards)
                valid_counts_mask = task_uid_counts > 0
                r_values_for_task[valid_counts_mask] = (
                    task_uid_rewards[valid_counts_mask] / task_uid_counts[valid_counts_mask]
                )

                task_processed_rewards_for_aggregation: np.ndarray
                if task_config.task == InferenceTask:
                    sum_positive_r_inference = np.sum(r_values_for_task[r_values_for_task > 0])
                    task_processed_rewards_for_aggregation = r_values_for_task / max(
                        1, (sum_positive_r_inference + 1e-10)
                    )
                else:
                    task_processed_rewards_for_aggregation = (
                        r_values_for_task  # Raw rewards for non-inference before MA & grading
                    )

                raw_aggregated_rewards_array += task_processed_rewards_for_aggregation * task_config.probability

            # 1. Apply Moving Average to the raw aggregated final rewards
            averaged_raw_final_rewards = compute_averaged_weights(raw_aggregated_rewards_array)

            # 2. Apply Grading Curve to the averaged raw final rewards
            graded_rewards = apply_reward_func(
                raw_rewards=averaged_raw_final_rewards, p=shared_settings.REWARD_STEEPNESS
            )

            # 3. Final normalization (ensure non-negative and sum to 1)
            final_weights_for_chain = np.array(graded_rewards).astype(float)  # Ensure it's a new array
            final_weights_for_chain[final_weights_for_chain < 0] = 0

            sum_final_weights = np.sum(final_weights_for_chain)
            if sum_final_weights > 0:
                final_weights_for_chain /= sum_final_weights + 1e-10
            else:
                logger.warning("Sum of final weights is zero or negative after grading. All weights will be zero.")
                final_weights_for_chain = np.zeros(len(all_uids), dtype=float)

        except Exception as ex:
            logger.exception(f"Exception in WeightSetter.run_step: {ex}")
            # final_weights_for_chain should already be pre-initialized to zeros
            final_weights_for_chain = np.zeros(len(all_uids), dtype=float)

        # set weights on chain
        set_weights(
            final_weights_for_chain,
            step=self.step,
            subtensor=shared_settings.SUBTENSOR,
            metagraph=shared_settings.METAGRAPH,
        )
        # TODO: empty rewards queue only on weight setting success
        self.reward_events = []  # empty reward events queue; ensure it's list
        await asyncio.sleep(0.01)
        return final_weights_for_chain


weight_setter = WeightSetter()
