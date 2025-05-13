import asyncio
import json
import math
import random
import time
from typing import Any, AsyncGenerator, Optional

import openai
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared import settings

shared_settings = settings.shared_settings

from shared.epistula import make_openai_query
from validator_api import scoring_queue
from validator_api.utils import filter_available_uids


async def stream_from_first_response(  # noqa: C901
    responses: list[asyncio.Task],
    collected_chunks_list: list[list[str]],
    collected_chunks_raw_list: list[list[Any]],
    body: dict[str, Any],
    uids: list[int],
    timings_list: list[list[float]],
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

    async def _collector(idx: int, resp_task: asyncio.Task) -> None:
        """Miner stream collector.

        1. Wait for the miner's async-generator.
        2. On first valid chunk:
          - if we’re FIRST → notify main via queue and exit
          - else (someone already started) → keep collecting for scoring
        """
        try:
            resp_gen = await resp_task
            if not resp_gen or isinstance(resp_gen, Exception):
                return

            async for chunk in resp_gen:
                if not _is_valid(chunk):
                    continue

                # Someone already claimed the stream?
                if not first_found_evt.is_set():
                    first_found_evt.set()
                    await first_queue.put((idx, chunk, resp_gen))
                    return

                # We’re NOT the first – just collect for scoring.
                collected_chunks_raw_list[idx].append(chunk)
                collected_chunks_list[idx].append(chunk.choices[0].delta.content)
                timings_list[idx].append(time.monotonic() - response_start_time)

        except (openai.APIConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.exception(f"Collector error for miner index {idx}: {e}")

    # Spawn collectors for every miner.
    collectors = [asyncio.create_task(_collector(idx, stream)) for idx, stream in enumerate(responses)]

    # Wait for the first valid chunk.
    try:
        try:
            primary_idx, first_chunk, primary_gen = await asyncio.wait_for(first_queue.get(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("No miner produced a valid chunk within 30 s")
            yield 'data: {"error": "502 - No valid response received"}\n\n'
            return

        # Stream the very first chunk immediately.
        collected_chunks_raw_list[primary_idx].append(first_chunk)
        collected_chunks_list[primary_idx].append(first_chunk.choices[0].delta.content)
        timings_list[primary_idx].append(time.monotonic() - response_start_time)
        yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"

        # Continue streaming the primary miner.
        async for chunk in primary_gen:
            if not _is_valid(chunk):
                continue
            collected_chunks_raw_list[primary_idx].append(chunk)
            collected_chunks_list[primary_idx].append(chunk.choices[0].delta.content)
            timings_list[primary_idx].append(time.monotonic() - response_start_time)
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

        # End of stream.
        yield "data: [DONE]\n\n"
        if timings_list[primary_idx]:
            logger.info(f"Response completion time: {timings_list[primary_idx][-1]:.2f}s")

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

    except (openai.APIConnectionError, asyncio.CancelledError):
        logger.info("Client disconnected, streaming cancelled")
        for c in collectors:
            c.cancel()
        raise
    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        yield 'data: {"error": "Internal server Error"}\n\n'


async def get_response_from_miner(body: dict[str, any], uid: int, timeout_seconds: int) -> tuple:
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
