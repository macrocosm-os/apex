import asyncio
import threading

from loguru import logger
from pydantic import ConfigDict

from prompting.miner_availability.miner_availability import MinerAvailabilities
from prompting.tasks.task_registry import TaskRegistry
from shared import settings
from shared.loop_runner import AsyncLoopRunner
from shared.timer import Timer

shared_settings = settings.shared_settings

RETRIES = 3


class TaskLoop(AsyncLoopRunner):
    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 2
    task_queue: list | None = []
    scoring_queue: list | None = []
    miners_dict: dict | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(self, task_queue, scoring_queue, miners_dict, **kwargs):
        self.task_queue = task_queue
        self.scoring_queue = scoring_queue
        self.miners_dict = miners_dict
        await super().start(**kwargs)

    async def run_step(self):
        if len(self.task_queue) > shared_settings.TASK_QUEUE_LENGTH_THRESHOLD:
            await asyncio.sleep(10)
            return None
        if len(self.scoring_queue) > shared_settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            await asyncio.sleep(10)
            return None
        try:
            task = None
            for i in range(RETRIES):
                try:
                    task = TaskRegistry.create_random_task_with_dataset()
                    break
                except Exception as ex:
                    logger.error(f"Failed to get task or dataset entry: {ex}")
                await asyncio.sleep(0.1)

            if (
                len(
                    MinerAvailabilities.get_available_miners(
                        miners=self.miners_dict, task=task, model=task.llm_model_id
                    )
                )
                == 0
            ):
                logger.debug(
                    f"No available miners for Task: {task.__class__.__name__} and Model ID: {task.llm_model_id}. Skipping step."
                )
                return None

            if not (dataset_entry := task.dataset_entry):
                logger.warning(f"Dataset for task {task.__class__.__name__} returned None. Skipping step.")
                return None

            with Timer(label=f"Generating query for task: {task.__class__.__name__}"):
                if not task.query:
                    logger.debug(f"Generating query for task: {task.__class__.__name__}.")
                    await task.make_query(dataset_entry=dataset_entry)
                logger.debug(f"Generated Messages: {task.task_messages}")

            logger.debug(f"Appending task: {task.__class__.__name__} to task queue.")
            self.task_queue.append(task)
        except Exception as ex:
            logger.exception(ex)
            return None


task_loop = TaskLoop()
