import asyncio
import contextlib
import itertools
import json
import random
import time
from typing import Any, AsyncGenerator, Optional

import numpy as np
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
TIMEOUT_ALL_UIDS_FALLBACK = 7
_END_OF_STREAM: object = object()


async def _prepare_chunk(chunk, body: dict[str, Any]):
    chunk_dict = chunk.model_dump()
    if not body.get("logprobs"):
        chunk_dict["choices"][0].pop("logprobs", None)
    return f"data: {json.dumps(chunk_dict)}\n\n"


async def collect_streams(  # noqa: C901
    primary_streams: list[asyncio.Task[Any]],
    primary_uids: list[int],
    extra_streams: list[asyncio.Task[Any]],
    stream_finished: list[bool],
    collected_chunks_list: list[list[str]],
    collected_chunks_raw_list: list[list[Any]],
    body: dict[str, Any],
    extra_uids: list[int],
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
    producer_found = asyncio.Event()
    producer_idx: int | None = None
    producer_chunks: asyncio.Queue[str | object] = asyncio.Queue()

    async def _collector(idx: int, resp_task: asyncio.Task, reliable_uid: bool, top_incentive: bool) -> None:
        """Miner stream collector with enhanced N-token buffering for reliable miners."""
        nonlocal producer_idx
        nonlocal response_start_time
        try:
            resp_gen = await resp_task
            if not resp_gen or isinstance(resp_gen, Exception):
                return

            async for chunk in resp_gen:
                if not _is_valid(chunk):
                    continue

                if not producer_found.is_set():
                    # if reliable_uid and top_incentive and len(collected_chunks_raw_list[idx]) >= TOKENS_MIN_STREAM:
                    if reliable_uid and len(collected_chunks_raw_list[idx]) >= TOKENS_MIN_STREAM:
                        # Set UID as a primary stream if it's in reliable list and top incentive.
                        producer_idx = idx
                        producer_found.set()
                        # Flush buffered chunks collected so far.
                        for cached_chunk in collected_chunks_raw_list[idx]:
                            await producer_chunks.put(cached_chunk)
                    elif (
                        time.monotonic() - response_start_time
                    ) > TIMEOUT_ALL_UIDS_FALLBACK and collected_chunks_raw_list:
                        # If no reliable UID has declared the primary stream, fallback to any UID for primary stream
                        # with the longest response.
                        producer_idx, _ = max(
                            enumerate(collected_chunks_raw_list), key=lambda response: len(response[1])
                        )
                        producer_found.set()
                        for cached_chunk in collected_chunks_raw_list[idx]:
                            await producer_chunks.put(cached_chunk)
                        if stream_finished[idx]:
                            # Fallback stream might be already finished.
                            await producer_chunks.put(_END_OF_STREAM)

                if idx == producer_idx:
                    await producer_chunks.put(chunk)

                # We're NOT competing for primary – just collect for scoring.
                collected_chunks_raw_list[idx].append(chunk)
                collected_chunks_list[idx].append(chunk.choices[0].delta.content)
                timings_list[idx].append(time.monotonic() - response_start_time)

        except (openai.APIConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.exception(f"Collector error for miner index {idx}: {e}")
        finally:
            stream_finished[idx] = True

            if idx == producer_idx:
                await producer_chunks.put(_END_OF_STREAM)
            elif all(stream_finished) and not producer_found.is_set():
                # All streams ended before finding primary uid and streaming back to client, select longest response.
                producer_idx, _ = max(enumerate(collected_chunks_raw_list), key=lambda response: len(response[1]))
                producer_found.set()
                for cached_chunk in collected_chunks_raw_list[idx]:
                    await producer_chunks.put(cached_chunk)
                await producer_chunks.put(_END_OF_STREAM)

    # Trigger primary (client stream) candidates first to reduce latency.
    streams = itertools.chain(
        zip(primary_streams, primary_uids),
        zip(extra_streams, extra_uids),
    )

    reliable_top = {}
    with contextlib.suppress():
        reliable_all = {
            uid: (await uid_tracker.uids[uid].success_rate(TaskType.Inference))
            for uid in itertools.chain(primary_uids, extra_uids)
        }
        top_amount = 3
        reliable_top = dict(sorted(reliable_all.items(), key=lambda kv: kv[1], reverse=True)[:top_amount])
        logger.debug(f"Primary stream candidates (success rate): {reliable_top}")

    if not reliable_top:
        reliable_top = {uid: 1.0 for uid in primary_uids}
        logger.debug(f"No reliable found, fallback to all uids: {reliable_top}")

    collectors: list[asyncio.Task] = []
    all_uids: list[int] = []
    for idx, (stream, uid) in enumerate(streams):
        try:
            top_incentive = uid in primary_uids
            reliable = uid in reliable_top
            collectors.append(asyncio.create_task(_collector(idx, stream, reliable, top_incentive)))
            all_uids.append(uid)
        except BaseException as exc:
            logger.exception(f"Error during primary stream, uids: {primary_uids}: {exc}")

    # Primary stream.
    try:
        chunk = await asyncio.wait_for(producer_chunks.get(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"No miner produced a valid chunk within {timeout:.2f}")
        yield 'data: {"error": "502 - No valid response received"}\n\n'
        return

    if chunk is not _END_OF_STREAM:
        yield await _prepare_chunk(chunk=chunk, body=body)

    # Drain the queue until end‑of‑stream sentinel.
    while True:
        chunk = await producer_chunks.get()
        if chunk is _END_OF_STREAM:
            break
        yield await _prepare_chunk(chunk=chunk, body=body)
    yield "data: [DONE]\n\n"

    # Wait for background collectors to finish.
    await asyncio.gather(*collectors)

    # Update UID reliability tracker.
    await uid_tracker.score_uid_chunks(
        uids=all_uids,
        chunks=collected_chunks_list,
        task_name=TaskType.Inference,
        format=CompletionFormat.STR,
        # UIDs response is lower than average it's probably not complete.
        # Might have FP, althrough reliability tracker used only to choose primary stream.
        min_chunks=int(np.mean([len(chunks) for chunks in collected_chunks_list]) * 0.7),
    )

    # Push everything to the scoring queue.
    asyncio.create_task(
        scoring_queue.scoring_queue.append_response(
            uids=all_uids,
            body=body,
            chunks=collected_chunks_list,
            chunk_dicts_raw=collected_chunks_raw_list,
            timings=timings_list,
        )
    )


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


async def chat_completion(
    body: dict[str, Any],
    # uids parameter is deprecated.
    uids: Optional[list[int]] = None,
    num_miners: int = 5,
    uid_tracker: UidTracker | None = None,
    add_reliable_miners: int = 1,
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
    if uid_tracker is not None:
        # Add reliable uids, or ones with highest success rate to guarantee completed stream.
        reliable_uids = await uid_tracker.sample_reliable(
            task=TaskType.Inference, amount=add_reliable_miners, success_rate=0.99, add_random_extra=False
        )
        primary_uids.extend(list(reliable_uids.keys()))
        primary_uids = list(set(primary_uids))
        logger.debug(
            f"Added reliable miners: {list(reliable_uids.keys())} to the request, "
            f"total primary uids: {len(primary_uids)}"
        )

    if not primary_uids:
        raise HTTPException(status_code=500, detail="No available miners")

    primary_uids: list[int] = random.sample(primary_uids, min(len(primary_uids), num_miners))

    # TODO: Revisit why returned uids types are not unified for different modes.
    extra_uids: np.ndarray = get_uids(
        sampling_mode="random", k=shared_settings.API_EXTRA_UIDS_QUERY, exclude=primary_uids
    )
    if isinstance(extra_uids, np.ndarray):
        extra_uids = extra_uids.tolist()

    total_amount = len(primary_uids) + len(extra_uids)

    STREAM = body.get("stream", False)
    timeout_seconds = float(body.get("timeout", shared_settings.INFERENCE_TIMEOUT * 2))
    timeout_seconds = max(timeout_seconds, shared_settings.INFERENCE_TIMEOUT * 2)
    timeout_seconds = min(timeout_seconds, shared_settings.MAX_TIMEOUT)
    logger.debug(f"Changing timeout from {body.get('timeout')} to {timeout_seconds}. Messages: {body.get('messages')}")

    # Initialize chunks collection for each miner.
    stream_finished: list[bool] = [False for _ in range(total_amount)]
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

        extra_streams: list[asyncio.Task] = []
        # Query remaining uids.
        for uid in extra_uids:
            try:
                extra_streams.append(
                    asyncio.create_task(
                        make_openai_query(
                            shared_settings.METAGRAPH, shared_settings.WALLET, timeout_seconds, body, uid, stream=True
                        )
                    )
                )
            except BaseException as e:
                logger.error(f"Error creating task for miner {uid}: {e}")
                continue

        total_streams = len(primary_streams) + len(extra_streams)
        assert total_streams == total_amount, f"Requested streams {total_streams} != {total_amount} uids"

        logger.debug(f"Created {total_streams} response tasks for streaming")

        return StreamingResponse(
            collect_streams(
                primary_streams=primary_streams,
                primary_uids=primary_uids,
                extra_streams=extra_streams,
                extra_uids=extra_uids,
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
        extra_streams = [
            asyncio.create_task(get_response_from_miner(body=body, uid=uid, timeout_seconds=timeout_seconds))
            for uid in primary_uids
        ]

        first_valid_response = None
        collected_responses = []

        while extra_streams and first_valid_response is None:
            done, pending = await asyncio.wait(extra_streams, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    response = await task
                    if response and isinstance(response, tuple) and response[0].choices and response[0].choices[0]:
                        if first_valid_response is None:
                            first_valid_response = response
                        collected_responses.append(response)
                except Exception as e:
                    logger.error(f"Error in miner response: {e}")
                extra_streams.remove(task)

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
