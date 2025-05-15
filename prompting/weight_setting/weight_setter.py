import asyncio
import os

import bittensor as bt
import numpy as np
import pandas as pd
from loguru import logger

from prompting import __spec_version__
from prompting.llms.model_zoo import ModelZoo
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.inference import InferenceTask
from prompting.tasks.task_registry import TaskConfig, TaskRegistry
from prompting.weight_setting.weight_synchronizer import WeightSynchronizer
from shared import settings
from shared.loop_runner import AsyncLoopRunner
from shared.misc import ttl_get_block

shared_settings = settings.shared_settings

FILENAME = "validator_weights.npz"
WEIGHTS_HISTORY_LENGTH = 24
PAST_WEIGHTS: list[np.ndarray] = []

weight_synchronizer = WeightSynchronizer(metagraph=shared_settings.METAGRAPH, wallet=shared_settings.WALLET)


def apply_reward_func(raw_rewards: np.ndarray, p=0.5):
    """Apply the reward function to the raw rewards. P adjusts the steepness of the function - p = 0.5 leaves
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


async def set_weights(
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

        # Replace any NaN values with 0
        weights = np.nan_to_num(weights, nan=0.0)
        # Calculate the average reward for each uid across non-zero values.
        PAST_WEIGHTS.append(weights)
        if len(PAST_WEIGHTS) > WEIGHTS_HISTORY_LENGTH:
            PAST_WEIGHTS.pop(0)
        averaged_weights = np.average(np.array(PAST_WEIGHTS), axis=0)
        save_weights(PAST_WEIGHTS)
        try:
            if shared_settings.NEURON_DISABLE_SET_WEIGHTS: # If weights will not be set on chain, we should not synchronize
                augmented_weights = averaged_weights
            else:
                augmented_weights = await weight_synchronizer.get_augmented_weights(
                    weights=averaged_weights, uid=shared_settings.UID
                )
        except Exception as ex:
            logger.exception(f"Issue with setting weights: {ex}")
            augmented_weights = averaged_weights
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=shared_settings.METAGRAPH.uids,
            weights=augmented_weights,
            netuid=shared_settings.NETUID,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
    except Exception as ex:
        logger.exception(f"Issue with setting weights: {ex}")

    # Create a dataframe from weights and uids and save it as a csv file, with the current step as the filename.
    if shared_settings.LOG_WEIGHTS:
        try:
            weights_df = pd.DataFrame(
                {
                    "step": step,
                    "uids": uint_uids,
                    "weights": processed_weights.flatten(),
                    "raw_weights": str(list(weights.flatten())),
                    "averaged_weights": str(list(averaged_weights.flatten())),
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
    # interval: int = 60

    class Config:
        arbitrary_types_allowed = True

    async def start(self, reward_events, name: str | None = None, **kwargs):
        self.reward_events = reward_events
        global PAST_WEIGHTS

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
        try:
            if len(self.reward_events) == 0:
                logger.warning("No reward events in queue, skipping weight setting...")
                return
            # reward_events is a list of lists of WeightedRewardEvents - the 'sublists' each contain the multiple reward events for a single task
            self.reward_events: list[list[WeightedRewardEvent]] = self.reward_events  # to get correct typehinting

            # reward_dict = {uid: 0 for uid in get_uids(sampling_mode="all")}
            all_uids = range(shared_settings.METAGRAPH.n.item())
            reward_dict = {uid: 0 for uid in all_uids}
            logger.info(f"Setting weights for {len(reward_dict)} uids")
            # miner_rewards is a dictionary that separates each task config into a dictionary of uids with their rewards
            miner_rewards: dict[TaskConfig, dict[int, float]] = {
                config: {uid: {"reward": 0, "count": 0} for uid in all_uids} for config in TaskRegistry.task_configs
            }

            inference_events: list[WeightedRewardEvent] = []
            for reward_events in self.reward_events:
                await asyncio.sleep(0.01)
                for reward_event in reward_events:
                    if np.sum(reward_event.rewards) > 0:
                        logger.debug("Identified positive reward event")
                    task_config = TaskRegistry.get_task_config(reward_event.task)

                    # inference task uses a different reward model
                    if task_config.task == InferenceTask:
                        inference_events.append(reward_event)
                        continue

                    # give each uid the reward they received
                    for uid, reward in zip(reward_event.uids, reward_event.rewards):
                        miner_rewards[task_config][uid]["reward"] += (
                            reward * reward_event.weight
                        )  # TODO: Double check I actually average at the end
                        miner_rewards[task_config][uid]["count"] += (
                            1 * reward_event.weight
                        )  # TODO: Double check I actually average at the end

            for inference_event in inference_events:
                for uid, reward in zip(inference_event.uids, inference_event.rewards):
                    llm_model = inference_event.task.llm_model_id

                    model_specific_reward = ModelZoo.get_model_by_id(llm_model).reward if llm_model else 1
                    miner_rewards[TaskRegistry.get_task_config(InferenceTask)][uid]["reward"] += (
                        reward * model_specific_reward
                    )  # for inference 2x responses should mean 2x the reward

            for task_config, rewards in miner_rewards.items():
                r = np.array([x["reward"] / max(1, x["count"]) for x in list(rewards.values())])
                u = np.array(list(rewards.keys()))
                if task_config.task == InferenceTask:
                    processed_rewards = r / max(1, (np.sum(r[r > 0]) + 1e-10))
                else:
                    processed_rewards = apply_reward_func(raw_rewards=r, p=shared_settings.REWARD_STEEPNESS)
                processed_rewards *= task_config.probability
                # update reward dict
                for uid, reward in zip(u, processed_rewards):
                    reward_dict[uid] += reward
            final_rewards = np.array(list(reward_dict.values())).astype(float)
            final_rewards[final_rewards < 0] = 0
            final_rewards /= np.sum(final_rewards) + 1e-10
        except Exception as ex:
            logger.exception(f"{ex}")

        # set weights on chain
        await set_weights(
            final_rewards, step=self.step, subtensor=shared_settings.SUBTENSOR, metagraph=shared_settings.METAGRAPH
        )
        # TODO: empty rewards queue only on weight setting success
        self.reward_events[:] = []  # empty reward events queue
        await asyncio.sleep(0.01)
        return final_rewards


weight_setter = WeightSetter()
