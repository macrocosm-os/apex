import asyncio
import random
from typing import Dict

import numpy as np
from loguru import logger

from prompting.tasks.base_task import BaseTask
from prompting.tasks.task_registry import TaskRegistry
from shared import settings
from shared.epistula import query_availabilities
from shared.loop_runner import AsyncLoopRunner
from shared.uids import get_uids

shared_settings = settings.shared_settings

task_config: dict[str, bool] = {str(task_config.task.__name__): True for task_config in TaskRegistry.task_configs}
model_config: dict[str, bool] = {model: False for model in shared_settings.LLM_MODEL}


class MinerAvailabilities:
    """Static class that provides methods to query miner availabilities from a miners dictionary"""

    @staticmethod
    def get_available_miners(
        miners: dict[int, dict], task: BaseTask | None = None, model: str | None = None, k: int | None = None
    ) -> list[int]:
        available = list(miners.keys())
        if task:
            task_name = task.__class__.__name__ if isinstance(task, BaseTask) else task.__name__
            available = [uid for uid in available if miners[uid]["task_availabilities"].get(task_name, False)]
        if model:
            available = [uid for uid in available if miners[uid]["llm_model_availabilities"].get(model, False)]
        if k:
            available = random.sample(available, min(len(available), k))
        return list(map(int, available))


class CheckMinerAvailability(AsyncLoopRunner):
    interval: int = 30  # Miners will be queried approximately once every hour
    uids: np.ndarray = shared_settings.TEST_MINER_IDS or get_uids(sampling_mode="all")
    current_index: int = 0
    uids_per_step: int = 10
    miners_dict: dict[int, dict] = {}

    class Config:
        arbitrary_types_allowed = True

    async def start(self, miners_dict: dict[int, dict], **kwargs):
        self.miners_dict = miners_dict
        logger.debug("Starting availability checking loop...")
        return await super().start(**kwargs)

    async def run_step(self):
        start_index = self.current_index
        end_index = min(start_index + self.uids_per_step, len(self.uids))
        uids_to_query = self.uids[start_index:end_index]
        if self.step == 0:
            uids_to_query = self.uids

        if any(uid >= len(shared_settings.METAGRAPH.axons) for uid in uids_to_query):
            raise ValueError("Some UIDs are out of bounds. Make sure all the TEST_MINER_IDS are valid.")
        responses: list[Dict[str, bool]] = await query_availabilities(uids_to_query, task_config, model_config)

        for response, uid in zip(responses, uids_to_query):
            try:
                self.miners_dict[uid] = {
                    "task_availabilities": response.get("task_availabilities", {task: True for task in task_config}),
                    "llm_model_availabilities": response.get(
                        "llm_model_availabilities", {model: False for model in model_config}
                    ),
                }
            except BaseException as e:
                logger.debug(f"Availability Response Invalid for miner {uid}: {e}")

        self.current_index = end_index

        if self.current_index >= len(self.uids):
            self.current_index = 0

        tracked_miners = [m for m in self.miners_dict.values() if m is not None]
        logger.debug(
            f"TRACKED MINERS: {len(tracked_miners)} --- UNTRACKED MINERS: {len(self.uids) - len(tracked_miners)}"
        )
        if tracked_miners:
            logger.debug(f"SAMPLE MINER: {tracked_miners[0]}")
        await asyncio.sleep(0.1)


# Initialize global miners dictionary
# miners_dict: dict[int, dict] = {}
availability_checking_loop = CheckMinerAvailability()
