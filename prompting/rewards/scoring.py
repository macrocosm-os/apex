import asyncio
import copy
import threading
from multiprocessing.managers import AcquirerProxy

import wandb
from loguru import logger
from pydantic import ConfigDict

from prompting.llms.model_manager import AsyncModelScheduler
from prompting.rewards.scoring_config import ScoringConfig
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.MSRv2_task import MSRv2Task
from prompting.tasks.task_registry import TaskRegistry
from shared import settings
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent
from shared.logging import RewardLoggingEvent, log_event
from shared.logging.logging import reinit_wandb, should_reinit_wandb
from shared.loop_runner import AsyncLoopRunner
from shared.timer import Timer


class TaskScorer(AsyncLoopRunner):
    """Maintains a queue of tasks and responses to score and then runs a scoring loop in a background thread.

    This scoring loop will score the responses once the LLM needed is loaded in the model_manager and log the rewards.
    """

    mp_lock: AcquirerProxy | None = None
    is_running: bool = False
    model_scheduler: AsyncModelScheduler | None = None
    thread: threading.Thread = None
    interval: int = 1
    scoring_queue: list | None = None
    reward_events: list | None = None
    task_queue: list | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(
        self,
        model_scheduler: AsyncModelScheduler,
        scoring_queue,
        reward_events,
        mp_lock: AcquirerProxy,
        name: str | None = None,
        task_queue: list | None = None,
        **kwargs,
    ):
        self.scoring_queue = scoring_queue
        self.reward_events = reward_events
        self.model_scheduler = model_scheduler
        self.mp_lock = mp_lock
        self.task_queue = task_queue
        return await super().start(name=name, **kwargs)

    def add_to_queue(
        self,
        task: BaseTextTask,
        response: DendriteResponseEvent,
        dataset_entry: DatasetEntry,
        block: int,
        step: int,
        task_id: str,
    ) -> None:
        self.scoring_queue.append(
            ScoringConfig(
                task=task,
                response=response,
                dataset_entry=dataset_entry,
                block=block,
                step=step,
                task_id=task_id,
            )
        )

    async def run_step(self) -> RewardLoggingEvent:
        await asyncio.sleep(0.1)
        # Only score responses for which the model is loaded
        await self.model_scheduler.llm_model_manager.lock.acquire()
        with self.mp_lock:
            scorable = [
                scoring_config
                for scoring_config in self.scoring_queue
                if (scoring_config.task.llm_model in self.model_scheduler.llm_model_manager.active_models.keys())
                or (scoring_config.task.llm_model is None)
            ]
            if len(scorable) == 0:
                return
            self.scoring_queue.remove(scorable[0])
        scoring_config: ScoringConfig = scorable.pop(0)

        # here we generate the actual reference
        with Timer(label=f"Generating reference for {scoring_config.task.__class__.__name__}"):
            await scoring_config.task.make_reference(
                dataset_entry=scoring_config.dataset_entry,
                model_manager=self.model_scheduler.llm_model_manager,
            )

        # and there we then calculate the reward
        reward_pipeline = TaskRegistry.get_task_reward(scoring_config.task)
        with Timer(label=f"Scoring {scoring_config.task.__class__.__name__}") as scoring_timer:
            if self.task_queue is None:
                raise ValueError("Task queue must be provided to TaskScorer.run_step()")
            reward_events = await reward_pipeline.apply(
                response_event=scoring_config.response,
                challenge=scoring_config.task.query,
                reference=scoring_config.task.reference,
                model_id=scoring_config.task.llm_model,
                task=scoring_config.task,
                model_manager=self.model_scheduler.llm_model_manager,
                task_queue=self.task_queue,
            )
            if scoring_config.task.organic:
                logger.debug(f"Reward events size: {len(reward_events)}")
                organic_reward_value = None
                # Ensure reward_events and rewards list are not empty
                if reward_events and reward_events[0].rewards:
                    organic_reward_value = reward_events[0].rewards[0]

                final_scoring_time = scoring_timer.final_time

                logger.info(
                    f"Organic task {scoring_config.task.task_id} scored. "
                    f"Reward: {organic_reward_value}, Duration: {final_scoring_time:.2f}s"
                )

                # Log only specific fields to wandb for organic tasks
                if settings.shared_settings.WANDB_ON:
                    if should_reinit_wandb():
                        reinit_wandb()

                    organic_log_data = {
                        "organic_reward": organic_reward_value,
                        "organic_chunk_timings": scoring_config.response.stream_results_all_chunks_timings,
                        "organic_task_name": scoring_config.task.name,
                        "organic_task_id": scoring_config.task.task_id,
                    }
                    try:
                        wandb.log(organic_log_data, step=scoring_config.step)
                    except Exception as e:
                        logger.error(
                            f"Error during wandb log for organic task {scoring_config.task_id}: {e} - data: {organic_log_data}"
                        )
        self.reward_events.append(reward_events)

        logger.debug(
            f"Scored {scoring_config.task.__class__.__name__} {scoring_config.task.task_id} with model "
            f"{scoring_config.task.llm_model_id}"
        )
        if not scoring_config.task.organic:
            # Reduce log size for raw chunks, wandb fails to log any data when overloaded.
            response = copy.deepcopy(scoring_config.response)
            response.stream_results_all_chunk_dicts_raw = []
            for idx in range(len(response.stream_results)):
                response.stream_results[idx].accumulated_chunk_dicts_raw = []

            if isinstance(scoring_config.task, MSRv2Task):
                if scoring_config.task.ground_truth is not None:
                    reference_value = str(scoring_config.task.ground_truth)  # "0" or "1"
                else:
                    reference_value = None
            else:
                reference_value = scoring_config.task.reference

            log_event(
                RewardLoggingEvent(
                    response_event=response,
                    reward_events=reward_events,
                    reference=reference_value,
                    challenge=scoring_config.task.query,
                    task=scoring_config.task.name,
                    block=scoring_config.block,
                    step=scoring_config.step,
                    task_id=scoring_config.task_id,
                    task_dict=scoring_config.task.model_dump(),
                    source=scoring_config.dataset_entry.source,
                )
            )

        self.model_scheduler.llm_model_manager.lock.release()
        await asyncio.sleep(0.01)


task_scorer = TaskScorer()
