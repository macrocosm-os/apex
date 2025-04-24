import asyncio
import json
import math
import random
import time
from typing import Any, AsyncGenerator, Callable, List, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared import settings

shared_settings = settings.shared_settings

from shared.epistula import make_openai_query
from validator_api import scoring_queue
from validator_api.utils import filter_available_uids


async def peek_until_valid_chunk(
    response: AsyncGenerator, is_valid_chunk: Callable[[Any], bool]
) -> tuple[Optional[Any], Optional[AsyncGenerator]]:
    """
    Keep reading chunks until we find a 'valid' one or run out of chunks.
    Return (first_valid_chunk, a_generator_of_all_chunks_including_this_one).
    If no chunks or no valid chunks, return (None, None).
    """
    consumed = []
    valid_chunk = None

    try:
        async for chunk in response:
            consumed.append(chunk)
            if is_valid_chunk(chunk):
                valid_chunk = chunk
                break  # we found our valid chunk
    except StopAsyncIteration:
        # no more chunks
        pass

    if not consumed or valid_chunk is None:
        # Either the generator is empty or we never found a valid chunk
        return None, None

    # Rebuild a generator from the chunks we already consumed
    # plus any remaining chunks that weren't pulled yet.
    async def rebuilt_generator() -> AsyncGenerator:
        # yield everything we consumed
        for c in consumed:
            yield c
        # yield anything else still left in 'response'
        async for c in response:
            yield c

    return valid_chunk, rebuilt_generator()


def is_valid_chunk(chunk: Any) -> bool:
    if chunk:
        return (
            hasattr(chunk, "choices")
            and len(chunk.choices) > 0
            and getattr(chunk.choices[0].delta, "content", None) is not None
        )


async def peek_first_chunk(
    response: AsyncGenerator,
) -> tuple[Optional[any], Optional[AsyncGenerator]]:
    """
    Pull one chunk from the async generator and return:
      (the_chunk, a_new_generator_that_includes_this_chunk)
    If the generator is empty, return (None, None).
    """
    try:
        first_chunk = await anext(response)  # or: await anext(response, default=None) in Python 3.10+
    except StopAsyncIteration:
        # Generator is empty
        return None, None

    # At this point, we have the first chunk. We need to rebuild a generator
    # that yields this chunk first, then yields the rest of the original response.
    async def reconstructed_response() -> AsyncGenerator:
        yield first_chunk
        async for c in response:
            yield c

    return first_chunk, reconstructed_response()


async def stream_chunks(
    first_valid_response: AsyncGenerator,
    collected_chunks_list: List[List[str]],
    timings_list: List[List[float]],
    response_start_time: float,
) -> AsyncGenerator[str, None]:
    """Stream chunks from a valid response and collect timing data.

    Args:
        first_valid_response: The async generator containing response chunks
        collected_chunks_list: List to collect response chunks
        timings_list: List to collect timing data
        response_start_time: Start time of the response for timing calculations
    """
    chunks_received = False
    async for chunk in first_valid_response:
        # Safely handle the chunk
        if not chunk.choices or not chunk.choices[0].delta:
            continue

        content = getattr(chunk.choices[0].delta, "content", None)
        if content is None:
            continue

        chunks_received = True
        timings_list[0].append(time.monotonic() - response_start_time)
        collected_chunks_list[0].append(content)
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    if not chunks_received:
        logger.error("Stream is empty: No chunks were received")
        yield 'data: {"error": "502 - Response is empty"}\n\n'

    yield "data: [DONE]\n\n"

    if timings_list and timings_list[0]:
        logger.info(f"Response completion time: {timings_list[0][-1]:.2f}s")


async def stream_from_first_response(
    responses: List[asyncio.Task],
    collected_chunks_list: List[List[str]],
    collected_chunks_raw_list: List,
    body: dict[str, any],
    uids: List[int],
    timings_list: List[List[float]],
) -> AsyncGenerator[str, None]:
    first_valid_response = None
    response_start_time = time.monotonic()

    try:
        # Keep looping until we find a valid response or run out of tasks
        while responses and first_valid_response is None:
            done, pending = await asyncio.wait(responses, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                responses.remove(task)
                try:
                    response = await task  # This is (presumably) an async generator

                    if not response or isinstance(response, Exception):
                        continue
                    # Peak at the first chunk
                    first_chunk, rebuilt_generator = await peek_until_valid_chunk(response, is_valid_chunk)
                    if first_chunk is None:
                        continue

                    first_valid_response = rebuilt_generator
                    break

                except Exception as e:
                    logger.exception(f"Error in miner response: {e}")
                    # just skip and continue to the next task

        if first_valid_response is None:
            logger.error("No valid response received from any miner")
            yield 'data: {"error": "502 - No valid response received"}\n\n'
            return

        # Stream the first valid response
        async for chunk_data in stream_chunks(
            first_valid_response, collected_chunks_list, timings_list, response_start_time
        ):
            yield chunk_data

        # Continue collecting remaining responses in background for scoring
        remaining = asyncio.gather(*pending, return_exceptions=True)
        remaining_tasks = asyncio.create_task(
            collect_remaining_responses(
                remaining=remaining,
                collected_chunks_list=collected_chunks_list,
                collected_chunks_raw_list=collected_chunks_raw_list,
                body=body,
                uids=uids,
                timings_list=timings_list,
                response_start_time=response_start_time,
            )
        )
        await remaining_tasks
        asyncio.create_task(
            scoring_queue.scoring_queue.append_response(
                uids=uids,
                body=body,
                chunks=collected_chunks_list,
                chunk_dicts_raw=collected_chunks_raw_list,
                timings=timings_list,
            )
        )

    except asyncio.CancelledError:
        logger.info("Client disconnected, streaming cancelled")
        for task in responses:
            task.cancel()
        raise
    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        yield 'data: {"error": "Internal server Error"}\n\n'


async def collect_remaining_responses(
    remaining: asyncio.Task,
    collected_chunks_list: List[List[str]],
    collected_chunks_raw_list: List,
    body: dict[str, any],
    uids: List[int],
    timings_list: List[List[float]],
    response_start_time: float,
):
    """Collect remaining responses for scoring without blocking the main response."""
    try:
        responses = await remaining
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error collecting response from uid {uids[i+1]}: {response}")
                continue

            async for chunk in response:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content is None:
                    continue
                timings_list[i + 1].append(time.monotonic() - response_start_time)
                collected_chunks_list[i + 1].append(content)
                collected_chunks_raw_list[i + 1].append(chunk)

    except Exception as e:
        logger.exception(f"Error collecting remaining responses: {e}")


async def get_response_from_miner(body: dict[str, any], uid: int, timeout_seconds: int) -> tuple:
    """Get response from a single miner."""
    return await make_openai_query(
        metagraph=shared_settings.METAGRAPH,
        wallet=shared_settings.WALLET,
        body=body,
        uid=uid,
        stream=False,
        timeout_seconds=timeout_seconds,
    )


async def chat_completion(
    body: dict[str, any], uids: Optional[list[int]] = None, num_miners: int = 10
) -> tuple | StreamingResponse:
    """Handle chat completion with multiple miners in parallel."""
    body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
    if not uids:
        logger.debug(
            "Finding miners for task: {} model: {} test: {} n_miners: {}",
            body.get("task"),
            body.get("model"),
            shared_settings.API_TEST_MODE,
            num_miners,
        )
        uids = body.get("uids") or filter_available_uids(
            task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=num_miners
        )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")
        uids = random.sample(uids, min(len(uids), num_miners))

    STREAM = body.get("stream", False)

    # Initialize chunks collection for each miner
    collected_chunks_list = [[] for _ in uids]
    collected_chunks_raw_list = [[] for _ in uids]
    timings_list = [[] for _ in uids]

    timeout_seconds = max(
        30,
        max(
            0,
            math.floor(
                math.log2(
                    body.get("sampling_parameters", shared_settings.SAMPLING_PARAMS).get("max_new_tokens", 256) / 256
                )
            ),
        )
        * 10
        + 30,
    )

    if STREAM:
        # Create tasks for all miners
        response_tasks = [
            asyncio.create_task(
                make_openai_query(
                    shared_settings.METAGRAPH, shared_settings.WALLET, timeout_seconds, body, uid, stream=True
                )
            )
            for uid in uids
        ]

        return StreamingResponse(
            stream_from_first_response(
                response_tasks, collected_chunks_list, collected_chunks_raw_list, body, uids, timings_list
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # For non-streaming requests, wait for first valid response
        response_tasks = [
            asyncio.create_task(get_response_from_miner(body=body, uid=uid, timeout_seconds=timeout_seconds))
            for uid in uids
        ]

        first_valid_response = None
        collected_responses = []

        while response_tasks and first_valid_response is None:
            done, pending = await asyncio.wait(response_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    response = await task
                    if response and isinstance(response, tuple) and response[0].choices and response[0].choices[0]:
                        if first_valid_response is None:
                            first_valid_response = response
                        collected_responses.append(response)
                except Exception as e:
                    logger.error(f"Error in miner response: {e}")
                response_tasks.remove(task)

        if first_valid_response is None:
            raise HTTPException(status_code=502, detail="No valid response received")

        asyncio.create_task(
            collect_remaining_nonstream_responses(
                pending=pending,
                collected_responses=collected_responses,
                body=body,
                uids=uids,
                timings_list=timings_list,
            )
        )
        return first_valid_response[0]  # Return only the response object, not the chunks


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
