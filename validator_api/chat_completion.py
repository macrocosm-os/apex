import asyncio
import json
import random
import time
from typing import Any, AsyncGenerator, Optional

import numpy as np
from numpy import ndarray

import openai
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared import settings
from shared.uids import get_uids
from validator_api.chain.uid_tracker import SUCCESS_RATE_MIN, CompletionFormat, TaskType, UidTracker

shared_settings = settings.shared_settings

from shared.epistula import make_openai_query
from validator_api import scoring_queue
from validator_api.utils import filter_available_uids

TOKENS_MIN_STREAM = 21


async def collect_streams(  # noqa: C901
    primary_responses: list[asyncio.Task[Any]],
    primary_uids: list[int],
    responses: list[asyncio.Task[Any]],
    stream_finished: list[bool],
    collected_chunks_list: list[list[str]],
    collected_chunks_raw_list: list[list[Any]],
    body: dict[str, Any],
    uids: list[int],
    timings_list: list[list[float]],
    uid_tracker: UidTracker,
    timeout: float = shared_settings.INFERENCE_TIMEOUT,
) -> AsyncGenerator[str, None]:
    """Start streaming as soon as any miner produces the first non-empty chunk.

    While streaming, collect primary miner and all other miners chunks for scoring.

    A chunk is considered non-empty if it has:
      - chunk.choices[0].delta
      - chunk.choices[0].delta.content
      - chunk.choices[0].logprobs
      - chunk.choices[0].logprobs.content
    """
    response_start_time = time.monotonic()

    def _is_valid(chunk: Any) -> bool:
        """Return True for the first chunk we care about (delta + logprobs.content)."""
        try:
            choice = chunk.choices[0]
            return (
                choice.delta is not None
                and getattr(choice.delta, "content", None) is not None
                and choice.logprobs is not None
                and getattr(choice.logprobs, "content", None) is not None
            )
        except (AttributeError, IndexError):
            return False

    # Guards first stream.
    first_found_evt = asyncio.Event()
    first_queue: asyncio.Queue[tuple[int, Any, AsyncGenerator]] = asyncio.Queue()

    async def _collector(idx: int, resp_task: asyncio.Task, reliable_uid: bool) -> None:
        """Miner stream collector with enhanced N-token buffering for reliable miners."""
        try:
            resp_gen = await resp_task
            if not resp_gen or isinstance(resp_gen, Exception):
                return

            # Buffer for reliable miners competing to be primary.
            token_buffer: list[str] = []

            async for chunk in resp_gen:
                if not _is_valid(chunk):
                    continue

                # If this is a reliable miner and no primary chosen yet.
                if reliable_uid and not first_found_evt.is_set():
                    # Add to buffer.
                    token_buffer.append(chunk)
                    collected_chunks_raw_list[idx].append(chunk)
                    collected_chunks_list[idx].append(chunk.choices[0].delta.content)
                    timings_list[idx].append(time.monotonic() - response_start_time)

                    # If we've collected N tokens or stream ended, compete to be primary.
                    if len(token_buffer) >= TOKENS_MIN_STREAM:
                        first_found_evt.set()
                        await first_queue.put((idx, token_buffer, resp_gen))
                        return

                    # Continue buffering (don't break the loop).
                    continue

                # We're NOT competing for primary – just collect for scoring.
                collected_chunks_raw_list[idx].append(chunk)
                collected_chunks_list[idx].append(chunk.choices[0].delta.content)
                timings_list[idx].append(time.monotonic() - response_start_time)

            stream_finished[idx] = True
            # Stream ended before N tokens, and all miners finished the stream - submit what we have.
            if reliable_uid and not first_found_evt.is_set() and token_buffer and sum(stream_finished) == len(stream_finished):
                first_found_evt.set()
                # None = stream ended.
                await first_queue.put((idx, token_buffer, None))

        except (openai.APIConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.exception(f"Collector error for miner index {idx}: {e}")

    # Spawn collectors for every miner.
    reliable_uids: list[int] = []
    for uid in uids:
        success_rate = await uid_tracker.uids[uid].success_rate(TaskType.Inference)
        if success_rate > SUCCESS_RATE_MIN:
            reliable_uids.append(uid)

    reliable_uids_exist = bool(reliable_uids)
    logger.debug(f"Reliable uids {reliable_uids}")
    # Trigger primary (client stream) candidates first to reduce latency.
    collectors: list[asyncio.Task] = []
    for idx, (stream, uid) in enumerate(zip(primary_responses, primary_uids)):
        try:
            if reliable_uids_exist:
                # Reliable uids are preffered for primary client stream.
                collectors.append(asyncio.create_task(_collector(idx, stream, uid in reliable_uids)))
            else:
                # If no reliable uids, we just collect everything.
                collectors.append(asyncio.create_task(_collector(idx, stream, True)))
        except BaseException as exc:
            logger.exception(f"Error during primary stream, uids: {primary_uids}: {exc}")

    # Collect remaining streams.
    for idx, (stream, uid) in enumerate(zip(responses, uids)):
        try:
            collectors.append(asyncio.create_task(_collector(idx, stream, True)))
        except BaseException as exc:
            logger.exception(f"Error during extra stream, uids: {uids}: {exc}")

    # Wait for the first valid chunk.
    try:
        try:
            primary_idx, token_buffer, primary_gen = await asyncio.wait_for(first_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("No miner produced a valid chunk")
            yield 'data: {"error": "502 - No valid response received"}\n\n'
            return

        # Stream all buffered tokens first.
        for chunk in token_buffer:
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

        # Continue with remaining stream (if any).
        if primary_gen is not None:
            try:
                async for chunk in primary_gen:
                    if not _is_valid(chunk):
                        continue
                    collected_chunks_raw_list[primary_idx].append(chunk)
                    collected_chunks_list[primary_idx].append(chunk.choices[0].delta.content)
                    timings_list[primary_idx].append(time.monotonic() - response_start_time)
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            except (openai.APIConnectionError, asyncio.CancelledError):
                logger.warning(f"Primary miner stream cancelled: {uids[primary_idx]}")

        yield "data: [DONE]\n\n"

        # Wait for background collectors to finish.
        await asyncio.gather(*collectors, return_exceptions=True)
        # Push everything to the scoring queue.
        asyncio.create_task(
            scoring_queue.scoring_queue.append_response(
                uids=uids,
                body=body,
                chunks=collected_chunks_list,
                chunk_dicts_raw=collected_chunks_raw_list,
                timings=timings_list,
            )
        )
        # await uid_tracker.score_uid_chunks(uids, collected_chunks_list, TaskType.Inference, format)
    except (openai.APIConnectionError, asyncio.CancelledError):
        logger.debug("Client disconnected, stream cancelled")
        raise
    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        yield 'data: {"error": "Internal server Error"}\n\n'


async def get_response_from_miner(body: dict[str, Any], uid: int, timeout_seconds: int) -> tuple:
    """Get response from a single miner."""
    try:
        return await make_openai_query(
            metagraph=shared_settings.METAGRAPH,
            wallet=shared_settings.WALLET,
            body=body,
            uid=uid,
            stream=False,
            timeout_seconds=timeout_seconds,
        )
    except BaseException as e:
        logger.warning(f"Error getting response from miner {uid}: {e}")
        return None



# Number of uids to choose from to so stream back to the client.
# settings.API_STREAM_UIDS
# Number of uids to sample on each API query (used to normalize scoring).
# settings.API_MIN_QUERY_UIDS
# 
async def chat_completion(
    body: dict[str, Any],
    # uids parameter is deprecated.
    uids: Optional[list[int]] = None,
    num_miners: int = 5,
    uid_tracker: UidTracker | None = None,
    add_reliable_miners: int = 2,
) -> tuple | StreamingResponse:
    # TODO: Add docstring.
    """Handle chat completion with multiple miners in parallel."""
    body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
    logger.debug(
        "Finding miners for task: {} model: {} test: {} n_miners: {}",
        body.get("task"),
        body.get("model"),
        shared_settings.API_TEST_MODE,
        num_miners,
    )
    primary_uids = filter_available_uids(
        task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=num_miners
    )
    if not primary_uids:
        raise HTTPException(status_code=500, detail="No available miners")
    primary_uids: list[int] = random.sample(primary_uids, min(len(primary_uids), num_miners))

    extra_uids: np.ndarray = get_uids(sampling_mode="random", k=settings.API_MIN_UIDS_QUERY, exclude=primary_uids)
    # TODO: Revisit why returned uids types are not unified for different modes.
    if isinstance(extra_uids, np.ndarray):
        extra_uids = extra_uids.tolist()
    total_amount = len(primary_uids) + len(extra_uids)

    STREAM = body.get("stream", False)
    timeout_seconds = float(body.get("timeout", shared_settings.INFERENCE_TIMEOUT * 2))
    timeout_seconds = max(timeout_seconds, shared_settings.INFERENCE_TIMEOUT * 2)
    timeout_seconds = min(timeout_seconds, shared_settings.MAX_TIMEOUT)
    logger.debug(f"Changing timeout from {body.get('timeout')} to {timeout_seconds}. Messages: {body.get('messages')}")

    # Initialize chunks collection for each miner
    stream_finished = [False for _ in range(total_amount)]
    collected_chunks_list = [[] for _ in range(total_amount)]
    collected_chunks_raw_list = [[] for _ in range(total_amount)]
    timings_list = [[] for _ in range(total_amount)]

    if STREAM:
        # Query primary first.
        primary_streams: list[asyncio.Task] = []
        for uid in primary_uids:
            try:
                primary_streams.append(
                    asyncio.create_task(
                        make_openai_query(
                            shared_settings.METAGRAPH, shared_settings.WALLET, timeout_seconds, body, uid, stream=True
                        )
                    )
                )
            except BaseException as e:
                logger.error(f"Error creating task for miner {uid}: {e}")
                continue

        responses: list[asyncio.Task] = []
        # Query remaining uids.
        for uid in extra_uids:
            try:
                responses.append(
                    asyncio.create_task(
                        make_openai_query(
                            shared_settings.METAGRAPH, shared_settings.WALLET, timeout_seconds, body, uid, stream=True
                        )
                    )
                )
            except BaseException as e:
                logger.error(f"Error creating task for miner {uid}: {e}")
                continue

        assert len(responses) == total_amount, f"Requested streams {len(responses)} != {total_amount} uids"

        logger.debug(f"Created {len(primary_streams)}/{len(primary_uids)} response tasks for streaming.")

        return StreamingResponse(
            collect_streams(
                primary_responses=primary_responses,
                primary_uids=primary_uids,
                responses=responses,
                uids=extra_uids,
                stream_finished=stream_finished,
                collected_chunks_list=collected_chunks_list,
                collected_chunks_raw_list=collected_chunks_raw_list,
                body=body,
                timings_list=timings_list,
                uid_tracker=uid_tracker,
                timeout=timeout_seconds,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # For non-streaming requests, wait for first valid response
        responses = [
            asyncio.create_task(get_response_from_miner(body=body, uid=uid, timeout_seconds=timeout_seconds))
            for uid in primary_uids
        ]

        first_valid_response = None
        collected_responses = []

        while responses and first_valid_response is None:
            done, pending = await asyncio.wait(responses, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    response = await task
                    if response and isinstance(response, tuple) and response[0].choices and response[0].choices[0]:
                        if first_valid_response is None:
                            first_valid_response = response
                        collected_responses.append(response)
                except Exception as e:
                    logger.error(f"Error in miner response: {e}")
                responses.remove(task)

        if first_valid_response is None:
            raise HTTPException(status_code=502, detail="No valid response received")

        # TODO: Non-stream scoring is not supported right now.
        # asyncio.create_task(
        #     collect_remaining_nonstream_responses(
        #         pending=pending,
        #         collected_responses=collected_responses,
        #         body=body,
        #         uids=uids,
        #         timings_list=timings_list,
        #     )
        # )
        # Return only the response object, not the chunks.
        return first_valid_response[0]


async def collect_remaining_nonstream_responses(
    pending: set[asyncio.Task],
    collected_responses: list,
    body: dict,
    uids: list,
    timings_list: list,
):
    """Wait for all pending miner tasks to complete and append their responses to the scoring queue."""

    try:
        # Wait for all remaining tasks; allow exceptions to be returned.
        remaining_responses = await asyncio.gather(*pending, return_exceptions=True)
        for response in remaining_responses:
            if not isinstance(response, Exception) and response and isinstance(response, tuple):
                collected_responses.append(response)
    except Exception as e:
        logger.error(f"Error gathering pending non-stream responses for scoring: {e}")

    try:
        chunks = [response[1] if response else [] for response in collected_responses]
        # TODO: Add timings.
        # Append all collected responses to the scoring queue for later processing.
        await scoring_queue.scoring_queue.append_response(
            uids=uids,
            body=body,
            chunks=chunks,
            chunk_dicts_raw=None,
            timings=None,  # We do not need chunk_dicts_raw or timings for non-stream responses.
        )
    except Exception as e:
        logger.error(f"Error appending non-stream responses to scoring queue: {e}")
