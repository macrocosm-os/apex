import asyncio
import copy
import threading
import time
from multiprocessing.managers import AcquirerProxy

from loguru import logger
from pydantic import ConfigDict

from prompting.rewards.scoring_config import ScoringConfig
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.msrv2_task import MSRv2Task
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
    thread: threading.Thread = None
    interval: int = 1
    scoring_queue: list | None = None
    reward_events: list | None = None
    task_queue: list | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(
        self,
        scoring_queue,
        reward_events,
        mp_lock: AcquirerProxy,
        name: str | None = None,
        task_queue: list | None = None,
        **kwargs,
    ):
        self.scoring_queue = scoring_queue
        self.reward_events = reward_events
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

        scoring_config: ScoringConfig | None = None
        while self.scoring_queue:
            # Pop the oldest item from the queue.
            config = self.scoring_queue.pop(0)
            # Check if the config is recent enough to be processed.
            if config.created_at >= time.time() - 60 * 60 * 20:
                scoring_config = config
                break
            # Otherwise, the old config is discarded and we continue to the next one.

        if not scoring_config:
            return

        # here we generate the actual reference
        with Timer(label=f"Generating reference for {scoring_config.task.__class__.__name__}"):
            await scoring_config.task.make_reference(
                dataset_entry=scoring_config.dataset_entry,
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
                model_id=scoring_config.task.llm_model_id,
                task=scoring_config.task,
                task_queue=self.task_queue,
            )

        self.reward_events.append(reward_events)

        logger.debug(
            f"Scored {scoring_config.task.__class__.__name__} {scoring_config.task.task_id} with model "
            f"{scoring_config.task.llm_model_id}"
        )

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

        if scoring_config.task.organic:
            response.stream_results = []
            response.axons = []
            response.completions = []
            response.stream_results_all_chunks = []
            response.stream_results_all_tokens_per_chunk = []
            reward_events = copy.deepcopy(reward_events)
            for event in reward_events:
                event.task = event.task.__class__()

            reference = None
            challenge = ""
            task_dict = {}
            source = "organic"
        else:
            reference = reference_value
            challenge = scoring_config.task.query
            task_dict = scoring_config.task.model_dump()
            source = scoring_config.dataset_entry.source

        log_event(
            RewardLoggingEvent(
                response_event=response,
                reward_events=reward_events,
                reference=reference,
                challenge=challenge,
                task=scoring_config.task.name,
                block=scoring_config.block,
                step=scoring_config.step,
                task_id=scoring_config.task_id,
                task_dict=task_dict,
                source=source,
            )
        )
        await asyncio.sleep(0.01)


task_scorer = TaskScorer()
