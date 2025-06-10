from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from shared import settings
from shared.uids import get_uids

shared_settings = settings.shared_settings
import asyncio
import json
import random

import numpy as np
from loguru import logger

from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import validate_api_key
from validator_api.serializers import WebRetrievalRequest, WebRetrievalResponse, WebSearchResult

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
    # Requesting UIDs is deprecated.
    uids: np.ndarray = get_uids(sampling_mode="random", k=shared_settings.API_EXTRA_UIDS_QUERY)

    if isinstance(uids, np.ndarray):
        uids = uids.tolist()

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
        done, pending = await asyncio.wait(miner_tasks, timeout=timeout_seconds, return_when=asyncio.ALL_COMPLETED)
        for task in pending:
            task.cancel()

        # Gather results that finished on time.
        per_miner_items: list[list[dict[str, Any]]] = [[] for _ in uids]
        for fut in done:
            idx, chunks, raw_chunks = fut.result()
            collected_chunks_list[idx] = chunks
            collected_chunks_raw_list[idx] = raw_chunks
            parsed = _parse_chunks(chunks)
            if parsed:
                per_miner_items[idx] = parsed  # keep association with this miner

        # nothing parsed at all?
        if not any(per_miner_items):
            logger.warning("No miner produced a valid (non-empty) result list")
            # raise HTTPException(status_code=500, detail="No miner responded successfully")
            return WebRetrievalResponse(results=[WebSearchResult(url="", content="", relevant="")])

        # Deduplicate then sample.
        max_needed = request.n_results
        unique_urls: set[str] = set()
        selected_items: list[WebSearchResult] = []

        round_idx = 0
        while len(selected_items) < max_needed:
            made_progress = False
            for miner_items in per_miner_items:
                if round_idx < len(miner_items):
                    item = miner_items[round_idx]
                    url = item["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        selected_items.append(WebSearchResult(**item))
                    # This miner had >= round_idx items.
                    made_progress = True
                    if len(selected_items) == max_needed:
                        # Early exit.
                        break
            if not made_progress:
                # No miner had an item at this depth - we’re out.
                break
            round_idx += 1
        logger.info(
            f"Returning {len(selected_items)} unique results across {sum(bool(m) for m in per_miner_items)} miners"
        )

        # Launch background for scoring.
        asyncio.create_task(
            scoring_queue.scoring_queue.append_response(
                uids=uids,
                body=body,
                chunks=collected_chunks_list,
                chunk_dicts_raw=collected_chunks_raw_list,
                timings=None,
            )
        )

        return WebRetrievalResponse(results=selected_items)

    # Cleanup and error handling.
    except asyncio.CancelledError:
        logger.warning("Client disconnected - cancelling miner tasks")
        for t in miner_tasks:
            t.cancel()
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unhandled error in web_retrieval: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
