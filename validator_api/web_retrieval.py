from fastapi import APIRouter, Depends, HTTPException, status

from shared import settings

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
async def web_retrieval(
    request: WebRetrievalRequest,
    api_key: str = Depends(validate_api_key),
):
    """
    Web retrieval endpoint that queries multiple miners to search the web.

    This endpoint distributes a search query to multiple miners, which perform web searches
    and return relevant results. The results are deduplicated based on URLs before being returned.

    ## Request Parameters:
    - **search_query** (str): The query to search for on the web. Required.
    - **n_miners** (int, default=10): Number of miners to query for results.
    - **n_results** (int, default=5): Maximum number of results to return in the response.
    - **max_response_time** (int, default=10): Maximum time to wait for responses in seconds.
    - **uids** (List[int], optional): Optional list of specific miner UIDs to query.

    ## Response:
    Returns a list of unique web search results, each containing:
    - **url** (str): The URL of the web page
    - **content** (str, optional): The relevant content from the page
    - **relevant** (str, optional): Information about why this result is relevant

    Example request:
    ```json
    {
      "search_query": "latest advancements in quantum computing",
      "n_miners": 15,
      "n_results": 10
    }
    ```
    """
    if request.uids:
        uids = request.uids
        try:
            uids = list(map(int, uids))
        except Exception:
            logger.error(f"Error in uids: {uids}")
    else:
        uids = filter_available_uids(
            task="WebRetrievalTask", test=shared_settings.API_TEST_MODE, n_miners=request.n_miners
        )
        uids = random.sample(uids, min(len(uids), request.n_miners))

    if len(uids) == 0:
        raise HTTPException(status_code=500, detail="No available miners")

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "target_results": request.n_results,
        "timeout": request.max_response_time,
        "messages": [
            {"role": "user", "content": request.search_query},
        ],
    }

    timeout_seconds = 30  # TODO: We need to scale down this timeout
    logger.debug(f"🔍 Querying miners: {uids} for web retrieval")
    stream_results = await query_miners(uids, body, timeout_seconds)
    results = [
        "".join(res.accumulated_chunks)
        for res in stream_results
        if isinstance(res, SynapseStreamResult) and res.accumulated_chunks
    ]
    distinct_results = list(np.unique(results))
    loaded_results = []
    for result in distinct_results:
        try:
            loaded_results.append(json.loads(result))
            logger.info(f"🔍 Result: {result}")
        except Exception:
            logger.error(f"🔍 Result: {result}")
    if len(loaded_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    collected_chunks_list = [res.accumulated_chunks if res and res.accumulated_chunks else [] for res in stream_results]
    asyncio.create_task(scoring_queue.scoring_queue.append_response(uids=uids, body=body, chunks=collected_chunks_list, chunk_dicts_raw=None, timings=None))
    loaded_results = [json.loads(r) if isinstance(r, str) else r for r in loaded_results]
    flat_results = [item for sublist in loaded_results for item in sublist]
    unique_results = []
    seen_urls = set()

    for result in flat_results:
        if isinstance(result, dict) and "url" in result:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                # Convert dict to WebSearchResult
                unique_results.append(WebSearchResult(**result))

    return WebRetrievalResponse(results=unique_results)
