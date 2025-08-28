import asyncio
import random
import uuid
from collections.abc import Sequence
from typing import Any

from loguru import logger

from apex.common.models import QueryTask
from apex.services.deep_research.deep_research_base import DeepResearchBase
from apex.services.llm.llm_base import LLMBase
from apex.services.websearch.websearch_base import WebSearchBase
from apex.validator import generate_query, generate_reference
from apex.validator.logger_local import LoggerLocal
from apex.validator.logger_wandb import LoggerWandb
from apex.validator.miner_sampler import MinerSampler


class Pipeline:
    def __init__(
        self,
        websearch: WebSearchBase,
        miner_sampler: MinerSampler,
        llm: LLMBase,
        deep_research: DeepResearchBase,
        logger_wandb: LoggerWandb | None = None,
        num_consumers: int = 5,
        timeout_consumer: float = 1200,
        timeout_producer: float = 240,
        queue_size: int = 10_000,
        redundancy_rate: float = 0.05,  # The rate that references are generated in addition to generator steps
        reference_rate: float = 0.5,  # The rate that references are generated as opposed to generator steps
        debug: bool = False,
    ):
        self.websearch = websearch
        self.miner_registry = miner_sampler
        self.llm = llm
        self.deep_research = deep_research
        self.logger_wandb = logger_wandb
        self.num_consumers = num_consumers
        self.timeout_consumer = timeout_consumer
        self.timeout_producer = timeout_producer
        self.queue_size = queue_size
        self.q_in: asyncio.Queue[QueryTask] = asyncio.Queue(maxsize=self.queue_size)
        self.q_out: asyncio.Queue[str] = asyncio.Queue()
        self.redundancy_rate = redundancy_rate
        self.reference_rate = reference_rate
        self._debug = debug
        self._logger_local = LoggerLocal()

    async def start_loop(self, initial_queries: Sequence[str] | None = None) -> None:
        """Kick off producer -> consumer workers. Runs in perpetuity, generating unique IDs for each task."""
        self.q_in = asyncio.Queue(maxsize=self.queue_size)
        self.q_out = asyncio.Queue()

        # If initial queries provided, enqueue them before starting workers.
        if initial_queries:
            for query in initial_queries:
                await self.q_in.put(QueryTask(query_id=str(uuid.uuid4()), query=query))

        producer_task: asyncio.Task[Any] = asyncio.create_task(self._periodic_producer())

        consumer_tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(self._periodic_consumer()) for _ in range(self.num_consumers)
        ]

        try:
            while True:
                await asyncio.sleep(60)
        finally:
            producer_task.cancel()
            for w in consumer_tasks:
                w.cancel()

    async def run_single(self, task: QueryTask) -> str:
        """End-to-end scoring for ONE logical query.

        If `task.query_text` is provided, it is used directly; otherwise `generate_query` is called.
        """
        query = task.query
        if query is None:
            logger.debug("Generating task query")
            query = await generate_query(llm=self.llm, websearch=self.websearch)

        reference = None
        tool_history: list[dict[str, str]] = []
        if random.random() < self.reference_rate:
            try:
                generator_results = None
                ground_truth = 0
                logger.debug(f"Generating task reference for query: {query[:20]}..")
                reference, tool_history = await generate_reference(llm=self.deep_research, query=query)
            except BaseException as exc:
                logger.exception(f"Failed to generate reference: {exc}")

        if reference is None:
            ground_truth = 1
            logger.debug(f"Querying generators with query: {query[:20]}..")
            generator_results = await self.miner_registry.query_generators(query=query)
            if random.random() < self.redundancy_rate:
                try:
                    logger.debug(f"Generating redundant task reference for query: {query[:20]}..")
                    reference, tool_history = await generate_reference(llm=self.deep_research, query=query)
                except BaseException as exc:
                    logger.warning(f"Failed to generate redundant reference: {exc}")

        logger.debug(f"Querying discriminators with reference GT: {ground_truth}, query: {query[:20]}..")
        discriminator_results = await self.miner_registry.query_discriminators(
            query=query, generator_results=generator_results, reference=reference, ground_truth=ground_truth
        )

        if self.logger_wandb:
            await self.logger_wandb.log(
                reference=reference, discriminator_results=discriminator_results, tool_history=tool_history
            )

        if self._debug:
            await self._logger_local.log(
                query=query,
                ground_truth=ground_truth,
                reference=reference,
                generator_results=generator_results,
                discriminator_results=discriminator_results,
            )

        return task.query_id

    async def _periodic_consumer(self) -> None:
        while True:
            task = await self.q_in.get()
            try:
                completed_id = await self.run_single(task=task)
                await self.q_out.put(completed_id)
                await asyncio.sleep(self.timeout_consumer)
            except Exception:
                logger.exception(f"Failed pipeline for {task.query_id}")
            finally:
                self.q_in.task_done()

    async def _periodic_producer(self) -> None:
        """Continuously enqueue tasks with unique IDs and no query text."""
        while True:
            await self.q_in.put(QueryTask(query_id=str(uuid.uuid4()), query=None))
            await asyncio.sleep(self.timeout_producer)
