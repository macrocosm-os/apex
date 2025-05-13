import asyncio
import datetime
import json
from collections import deque
from typing import Any

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from shared import settings
from shared.epistula import create_header_hook
from shared.loop_runner import AsyncLoopRunner
from validator_api.validator_forwarding import Validator, ValidatorRegistry

validator_registry = ValidatorRegistry()

shared_settings = settings.shared_settings


class ScoringPayload(BaseModel):
    payload: dict[str, Any]
    retries: int = 0
    date: datetime.datetime | None = None


class ScoringQueue(AsyncLoopRunner):
    """Performs organic scoring every `interval` seconds."""

    interval: float = shared_settings.SCORING_RATE_LIMIT_SEC
    scoring_queue_threshold: int = shared_settings.SCORING_QUEUE_API_THRESHOLD
    max_scoring_retries: int = 2
    _scoring_lock = asyncio.Lock()
    _scoring_queue: deque[ScoringPayload] = deque()
    _queue_maxlen: int = 50
    _min_wait_time: float = 1

    async def wait_for_next_execution(self, last_run_time) -> datetime.datetime:
        """If scoring queue is small, execute immediately, otherwise wait until the next execution time."""
        async with self._scoring_lock:
            if self.scoring_queue_threshold < self.size > 0:
                # If scoring queue is small and non-empty, score more frequently.
                await asyncio.sleep(self._min_wait_time)
                return datetime.datetime.now()

        return await super().wait_for_next_execution(last_run_time)

    async def run_step(self):
        """Perform organic scoring: pop queued payload, forward to the validator API."""
        async with self._scoring_lock:
            if not self._scoring_queue:
                return

            scoring_payload = self._scoring_queue.popleft()
            payload = scoring_payload.payload
            uids = payload["uids"]
            logger.info(
                f"Trying to score organic from {scoring_payload.date}, uids: {uids}. "
                f"Queue size: {len(self._scoring_queue)}"
            )
        validators: list[Validator] = []
        try:
            if shared_settings.OVERRIDE_AVAILABLE_AXONS:
                for idx, vali_axon in enumerate(shared_settings.OVERRIDE_AVAILABLE_AXONS):
                    validators.append(Validator(uid=-idx, axon=vali_axon, hotkey=shared_settings.API_HOTKEY, stake=1e6))
            else:
                validators = await validator_registry.get_available_axons(balance=shared_settings.API_ENABLE_BALANCE)
        except Exception as e:
            logger.exception(f"Could not find available validator scoring endpoint: {e}")

        try:
            if hasattr(payload, "to_dict"):
                payload = payload.to_dict()
            elif isinstance(payload, BaseModel):
                payload = payload.model_dump()
            payload_bytes = json.dumps(payload).encode()
        except BaseException as e:
            logger.exception(f"Error when encoding payload: {e}")
            await asyncio.sleep(0.1)
            return

        tasks = []
        for _, validator in validators.items():
            task = asyncio.create_task(self._send_result(payload_bytes, scoring_payload, validator, uids))
            tasks.append(task)
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)

    async def append_response(
        self,
        uids: list[int],
        body: dict[str, Any],
        chunks: list[list[str]],
        chunk_dicts_raw: list[ChatCompletionChunk | None] | None = None,
        timings: list[list[float]] | None = None,
    ):
        if not shared_settings.SCORE_ORGANICS:
            return

        if body.get("task") != "InferenceTask" and body.get("task") != "WebRetrievalTask":
            # logger.debug(f"Skipping forwarding for non-inference/web retrieval task: {body.get('task')}")
            return

        uids = list(map(int, uids))
        chunk_dict = {str(u): c for u, c in zip(uids, chunks)}  # TODO: Remove chunk_dict if we have chunk_dicts_raw
        if chunk_dicts_raw:
            # Iterate over the chunk_dicts_raw and convert each chunk to a dictionary
            chunk_dict_raw = {}
            for u, c in zip(uids, chunk_dicts_raw):
                chunk_dict_raw[str(u)] = [chunk.model_dump() for chunk in c]
        else:
            chunk_dict_raw = {}
        if timings:
            timing_dict = {str(u): t for u, t in zip(uids, timings)}
        else:
            timing_dict = {}
        payload = {
            "body": body,
            "chunks": chunk_dict,
            "uids": uids,
            "timings": timing_dict,
            "chunk_dicts_raw": chunk_dict_raw,
        }
        scoring_item = ScoringPayload(payload=payload, date=datetime.datetime.now().replace(microsecond=0))
        # logger.info(f"Appending organic to scoring queue: {scoring_item}")
        async with self._scoring_lock:
            if len(self._scoring_queue) >= self._queue_maxlen:
                scoring_payload = self._scoring_queue.popleft()
                logger.info(f"Dropping oldest organic from {scoring_payload.date} for uids {uids}")
            self._scoring_queue.append(scoring_item)

    async def _send_result(self, payload_bytes, scoring_payload, validator: Validator, uids):
        try:
            vali_url = f"http://{validator.axon}/scoring"
            timeout = httpx.Timeout(timeout=120.0)
            async with httpx.AsyncClient(
                timeout=timeout,
                event_hooks={"request": [create_header_hook(shared_settings.WALLET.hotkey, validator.hotkey)]},
            ) as client:
                response = await client.post(
                    url=vali_url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                )
                validator_registry.update_validators(uid=validator.uid, response_code=response.status_code)
                if response.status_code != 200:
                    raise Exception(
                        f"Status code {response.status_code} response for validator {validator.uid} - {vali_url}: "
                        f"{response.status_code} for uids {len(uids)}"
                    )
                logger.debug(f"Successfully forwarded response to uid {validator.uid} - {vali_url}")
        except httpx.ConnectError as e:
            logger.warning(
                f"Couldn't connect to validator {validator.uid} {vali_url} for scoring {len(uids)}. Exception: {e}"
            )
        except Exception as e:
            if shared_settings.API_ENABLE_BALANCE and scoring_payload.retries < self.max_scoring_retries:
                scoring_payload.retries += 1
                async with self._scoring_lock:
                    self._scoring_queue.appendleft(scoring_payload)
                logger.warning(
                    f"Tried to forward response from {scoring_payload.date} "
                    f"to validator {validator.uid} {vali_url} for  {len(uids)} uids. "
                    f"Queue size: {len(self._scoring_queue)}. Exception: {e}"
                )
            else:
                logger.warning(
                    f"Error while forwarding response from {scoring_payload.date} "
                    f"to validator {validator.uid} {vali_url} for {len(uids)} uids "
                    f"retries. Queue size: {len(self._scoring_queue)}. Exception: {e}"
                )

    @property
    def size(self) -> int:
        return len(self._scoring_queue)

    def __len__(self) -> int:
        return self.size


# TODO: Leaving it as a global var to make less architecture changes, refactor as DI.
scoring_queue = ScoringQueue()
