from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from shared import settings

shared_settings = settings.shared_settings
import asyncio
import json
import random

from loguru import logger

from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import validate_api_key
from validator_api.serializers import WebRetrievalRequest, WebRetrievalResponse, WebSearchResult
from validator_api.utils import filter_available_uids

router = APIRouter()


@router.post(
    "/web_retrieval",
    response_model=WebRetrievalResponse,
    summary="Web retrieval endpoint",
    description="Retrieves information from the web based on a search query using multiple miners.",
    response_description="List of unique web search results",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successful response with web search results",
            "model": WebRetrievalResponse,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error, no available miners, or no successful miner responses"
        },
    },
)
async def web_retrieval(  # noqa: C901
    request: WebRetrievalRequest,
    api_key: str = Depends(validate_api_key),
):
    """Launch *all* requested miners in parallel, return immediately when the first miner delivers a valid result."""
    # Choose miners.
    available = filter_available_uids(
        task="WebRetrievalTask",
        test=shared_settings.API_TEST_MODE,
        n_miners=request.n_miners,
        explore=shared_settings.API_UIDS_EXPLORE,
    )
    uids = random.sample(available, min(len(available), request.n_miners))

    if not uids:
        raise HTTPException(status_code=500, detail="No available miners")

    # Shared miner request body.
    body: dict[str, Any] = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "target_results": request.n_results,
        "timeout": request.max_response_time,
        "messages": [{"role": "user", "content": request.search_query}],
    }
    timeout_seconds = body["timeout"] or 15

    async def _call_miner(idx: int, uid: int) -> tuple[int, list[str], list[str]]:
        """Fire a single miner and return (index, accumulated_chunks, raw_chunks).

        The result is per-miner; we don't wait for the others here.
        """
        stream_results = await query_miners([uid], body, timeout_seconds)
        if not stream_results:
            return idx, [], []

        res: SynapseStreamResult = stream_results[0]
        return idx, res.accumulated_chunks or [], getattr(res, "raw_chunks", [])

    def _parse_chunks(chunks: list[str]) -> list[dict[str, Any]] | None:
        """Load JSON, filter dicts with required keys, None on failure/empty."""
        if not chunks:
            return None
        try:
            payload: Any = json.loads("".join(chunks))
            # Handle double-encoded JSON.
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                return None
            required = ("url", "content", "relevant")
            filtered = [d for d in payload if (isinstance(d, dict) and all(k in d and d[k] for k in required))]
            return filtered or None
        except Exception:
            return None

    # Fire miners concurrently.
    logger.debug(f"🔍 Querying miners for web retrieval: {uids}")
    miner_tasks = [asyncio.create_task(_call_miner(i, uid)) for i, uid in enumerate(uids)]

    # Pre-allocate structures (same order as `uids`) for later scoring.
    collected_chunks_list: list[list[str]] = [[] for _ in uids]
    collected_chunks_raw_list: list[list[Any]] = [[] for _ in uids]

    try:
        first_valid: list[dict[str, Any]] | None = None
        primary_idx: int | None = None

        # as_completed yields tasks exactly when each finishes.
        for fut in asyncio.as_completed(miner_tasks):
            idx, chunks, raw_chunks = await fut
            collected_chunks_list[idx] = chunks
            collected_chunks_raw_list[idx] = raw_chunks

            parsed = _parse_chunks(chunks)
            if parsed:
                first_valid = parsed[: request.n_results]
                primary_idx = idx
                # Stop iterating; others handled in background
                break

        if first_valid is None:
            logger.warning("No miner produced a valid (non-empty) result list")
            raise HTTPException(status_code=500, detail="No miner responded successfully")

        # Build client response from the winner.
        unique, seen = [], set()
        for item in first_valid:
            if item["url"] not in seen:
                seen.add(item["url"])
                unique.append(WebSearchResult(**item))

        # Collect all remaining miners *quietly* then push to scoring.
        async def _collect_remaining(pending: list[asyncio.Task]) -> None:
            try:
                for fut in asyncio.as_completed(pending):
                    idx, chunks, raw_chunks = await fut
                    collected_chunks_list[idx] = chunks
                    collected_chunks_raw_list[idx] = raw_chunks
            except Exception as exc:
                logger.debug(f"Error collecting remaining miners: {exc}")

            await scoring_queue.scoring_queue.append_response(
                uids=uids,
                body=body,
                chunks=collected_chunks_list,
                chunk_dicts_raw=collected_chunks_raw_list,
                timings=None,
            )

        # Pending tasks still not finished.
        pending_tasks = [t for t in miner_tasks if not t.done()]
        asyncio.create_task(_collect_remaining(pending_tasks))

        logger.info(f"✅ Returning {len(unique)} results from miner idx={primary_idx}, uid={uids[primary_idx]}")
        return WebRetrievalResponse(results=unique)

    # Cleanup and error handling.
    except asyncio.CancelledError:
        logger.warning("Client disconnected – cancelling miner tasks")
        for t in miner_tasks:
            t.cancel()
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unhandled error in web_retrieval: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
