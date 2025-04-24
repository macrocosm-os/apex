import asyncio
import threading
from multiprocessing.managers import AcquirerProxy

from loguru import logger
from pydantic import ConfigDict

from prompting.llms.model_manager import AsyncModelScheduler
from prompting.rewards.scoring_config import ScoringConfig
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.task_registry import TaskRegistry
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent
from shared.logging import RewardLoggingEvent, log_event
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
        # with self.mp_lock:
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
        with Timer(label=f"Scoring {scoring_config.task.__class__.__name__}"):
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

        # with self.mp_lock:
        self.reward_events.append(reward_events)

        # TODO: Remove this once we have a better way to handle organic tasks
        # if scoring_config.task.organic:
        #     self.reward_events.append(
        #         reward_events
        #     )  # Add the organic a second time, doubling the weight of the organic
        logger.debug(
            f"Scored {scoring_config.task.__class__.__name__} {scoring_config.task.task_id} with model "
            f"{scoring_config.task.llm_model_id}"
        )
        if not scoring_config.task.organic:
            log_event(
                RewardLoggingEvent(
                    response_event=scoring_config.response,
                    reward_events=reward_events,
                    reference=scoring_config.task.reference,
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
