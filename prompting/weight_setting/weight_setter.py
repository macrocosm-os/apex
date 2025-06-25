import asyncio
from collections import deque
import datetime
import json
from pathlib import Path
from typing import Any

import bittensor as bt
import numpy as np
import numpy.typing as npt
from loguru import logger

from prompting import __spec_version__
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.inference import InferenceTask
from prompting.tasks.msrv2_task import MSRv2Task
from prompting.tasks.task_registry import TaskConfig, TaskRegistry
from prompting.weight_setting.weight_synchronizer import WeightSynchronizer
from shared import settings
from shared.loop_runner import AsyncLoopRunner

shared_settings = settings.shared_settings


async def set_weights(
    weights: np.ndarray,
    subtensor: bt.Subtensor | None = None,
    metagraph: bt.Metagraph | None = None,
    weight_syncer: WeightSynchronizer | None = None,
):
    """Set the validator weights to the metagraph hotkeys based on the scores it has received from the miners.

    The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
    """
    try:
        if any(np.isnan(weights).flatten()):
            logger.warning(f"Scores used for weight setting contain NaN values: {weights}")
        weights = np.nan_to_num(weights, nan=0.0)

        try:
            if shared_settings.NEURON_DISABLE_SET_WEIGHTS:
                # If weights will not be set on chain, we should not synchronize.
                augmented_weights = weights
            else:
                augmented_weights = await weight_syncer.get_augmented_weights(
                    weights=weights, uid=shared_settings.UID
                )
        except BaseException as ex:
            logger.exception(f"Issue with setting weights: {ex}")
            augmented_weights = weights

        # Process the raw weights to final_weights via subtensor limitations.
        processed_weight_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids=shared_settings.METAGRAPH.uids,
            weights=augmented_weights,
            netuid=shared_settings.NETUID,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        # Convert to uint16 weights and uids.
        uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids,
            weights=processed_weights
        )
    except Exception as ex:
        logger.exception(f"Issue with setting weights: {ex}")

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
    # interval: int = 60 * 21
    interval: int = 60 * 3
    reward_events: list[list[WeightedRewardEvent]] | None = None
    weight_dict: dict[int, list[float]] | None = None
    weight_syncer: WeightSynchronizer | None = None

    reward_history_path: Path = Path("validator_rewards.jsonl")
    reward_history_len: int = 24
    # List of uids info per each epoch, e.g.: [{1: {"reward": 1.0}, 2: {"reward": 3.0}}].
    reward_history: deque[dict[int, dict[str, Any]]] | None = None

    class Config:
        arbitrary_types_allowed = True

    async def start(
        self,
        reward_events: list[list[WeightedRewardEvent]] | None,
        weight_dict: dict[int, list[float]],
        name: str | None = None,
    ):
        self.reward_events = reward_events
        self.weight_syncer = WeightSynchronizer(
            metagraph=shared_settings.METAGRAPH, wallet=shared_settings.WALLET, weight_dict=weight_dict
        )
        await self._load_rewards()
        return await super().start(name=name)
    
    async def _save_rewards(self, rewards: npt.NDArray[np.float32]):
        """Persist the latest epoch rewards.

        The snapshot is appended to `reward_history` (bounded by `reward_average_len`) and the JSONL file at
        `reward_average_path` is rewritten with the current buffer.

        Args:
            rewards: A one-dimensional array where the index is the uid and the value is its reward.
        """
        epoch_rewards = {int(uid): {"reward": float(r)} for uid, r in enumerate(rewards)}

        if not isinstance(self.reward_history, deque):
            self.reward_history = deque(maxlen=self.reward_history_len)
        self.reward_history.append(epoch_rewards)

        try:
            block = shared_settings.block
            with self.reward_history_path.open("w", encoding="utf-8") as file:
                for snapshot in self.reward_history:
                    row: dict[str, Any] = {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds") + "Z",
                        "block": block,
                        "rewards": {str(uid): v["reward"] for uid, v in snapshot.items()},
                    }
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
        except BaseException as exc:
            logger.error(f"Couldn't write rewards history: {exc}")

    async def _load_rewards(self):
        """Load reward snapshots from disk into `reward_history`.

        Only the newest `reward_average_len` rows are retained.
        """
        self.reward_history: deque[dict[int, dict[str, Any]]] | None = deque(maxlen=self.reward_history_len)
        if not self.reward_history_path.exists():
            logger.info("No rewards file found - starting with empty history.")
            return

        try:
            with self.reward_history_path.open("r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    payload = data.get("rewards")
                    if payload is None:
                        raise ValueError(f"Malformed weight history file: {data}")

                    self.reward_history.append(
                        {int(uid): {"reward": float(reward)} for uid, reward in payload.items()}
                    )
        except BaseException as exc:
            self.reward_history: deque[dict[int, dict[str, Any]]] | None = deque(maxlen=self.reward_history_len)
            logger.error(f"Couldn't load rewards from file, resetting weight history: {exc}")

    @classmethod
    async def merge_task_rewards(cls, reward_events: list[list[WeightedRewardEvent]]) -> npt.NDArray[np.float32] | None:
        if len(reward_events) == 0:
            logger.warning("No reward events in queue, skipping weight setting...")
            return

        all_uids = range(shared_settings.METAGRAPH.n.item())
        reward_dict = {uid: 0 for uid in all_uids}
        logger.info(f"Setting weights for {len(reward_dict)} uids")

        # miner_rewards is a dictionary that separates each task config into a dictionary of uids with their rewards.
        miner_rewards: dict[TaskConfig, dict[int, dict[str, int]]] = {
            config: {uid: {"reward": 0, "count": 0} for uid in all_uids} for config in TaskRegistry.task_configs
        }

        linear_reward_tasks = set([InferenceTask, MSRv2Task])
        linear_events: list[WeightedRewardEvent] = []
        for reward_sub_events in reward_events:
            await asyncio.sleep(0.01)
            for reward_event in reward_sub_events:
                task_config = TaskRegistry.get_task_config(reward_event.task)

                # Inference task uses a different reward model.
                if task_config.task in linear_reward_tasks:
                    linear_events.append(reward_event)
                    continue

                # Give each uid the reward they received.
                for uid, reward in zip(reward_event.uids, reward_event.rewards):
                    miner_rewards[task_config][uid]["reward"] += reward * reward_event.weight
                    miner_rewards[task_config][uid]["count"] += reward_event.weight

        for linear_event in linear_events:
            task_config = TaskRegistry.get_task_config(linear_event.task)
            for uid, reward in zip(linear_event.uids, linear_event.rewards):
                miner_rewards[task_config][uid]["reward"] += reward

        for task_config, rewards in miner_rewards.items():
            task_rewards = np.array([x["reward"] / max(1, x["count"]) for x in list(rewards.values())])
            task_uids = np.array(list(rewards.keys()))
            if task_config.task in linear_reward_tasks:
                processed_rewards = task_rewards / max(1, (np.sum(task_rewards[task_rewards > 0]) + 1e-10))
            else:
                processed_rewards = cls.apply_steepness(
                    raw_rewards=task_rewards,
                    steepness=shared_settings.REWARD_STEEPNESS
                )
            processed_rewards *= task_config.probability

            for uid, reward in zip(task_uids, processed_rewards):
                reward_dict[uid] += reward

        final_rewards = np.array(list(reward_dict.values())).astype(np.float32)
        return final_rewards

    @classmethod
    def apply_steepness(cls, raw_rewards: npt.NDArray[np.float32], steepness: float = 0.5) -> npt.NDArray[np.float32]:
        """Apply steepness function to the raw rewards.

        Args:
            steepness: Adjusts the steepness of the function - p = 0.5 leaves the rewards unchanged,
            p < 0.5 makes the function more linear (at p=0 all miners with positives reward values get the same reward),
            p > 0.5 makes the function more exponential (winner takes all).
        """
        # 6.64385619 = ln(100)/ln(2) -> this way if p = 0.5, the exponent is exactly 1.
        exponent = (steepness ** 6.64385619) * 100
        raw_rewards = np.array(raw_rewards) / max(1, (np.sum(raw_rewards[raw_rewards > 0]) + 1e-10))
        positive_rewards = np.clip(raw_rewards, 1e-10, np.inf)
        normalised_rewards = positive_rewards / np.max(positive_rewards)
        post_func_rewards = normalised_rewards ** exponent
        all_rewards = post_func_rewards / (np.sum(post_func_rewards) + 1e-10)
        all_rewards[raw_rewards <= 0] = raw_rewards[raw_rewards <= 0]
        return all_rewards

    async def run_step(self):
        await asyncio.sleep(0.01)
        try:
            if self.reward_events is None:
                logger.error(f"No rewards evants were found, skipping weight setting")
                return

            final_rewards = await self.merge_task_rewards(self.reward_events)

            if final_rewards is None:
                logger.error(f"No rewards were found, skipping weight setting")
                return

            await self._save_rewards(final_rewards)
            final_rewards[final_rewards < 0] = 0
            final_rewards /= np.sum(final_rewards) + 1e-10
        except BaseException as ex:
            logger.exception(f"{ex}")

        # Set weights on chain.
        await set_weights(
            final_rewards,
            subtensor=shared_settings.SUBTENSOR,
            metagraph=shared_settings.metagraph_force_sync(),
            weight_syncer=self.weight_syncer,
        )
        # TODO: Empty rewards queue only on weight setting success.
        self.reward_events[:] = []
        await asyncio.sleep(0.01)
        return final_rewards


weight_setter = WeightSetter()
