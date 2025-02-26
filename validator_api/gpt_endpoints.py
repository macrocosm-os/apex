import asyncio
import json
import random
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from starlette.responses import StreamingResponse

from shared import settings

shared_settings = settings.shared_settings
from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import validate_api_key
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.serializers import (
    CompletionsRequest,
    TestTimeInferenceRequest,
    WebRetrievalRequest,
    WebRetrievalResponse,
    WebSearchResult,
)
from validator_api.test_time_inference import generate_response
from validator_api.utils import filter_available_uids

router = APIRouter()
N_MINERS = 5


@router.post("/v1/chat/completions")
async def completions(request: CompletionsRequest, api_key: str = Depends(validate_api_key)):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = request.model_dump()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        if body.get("uids"):
            try:
                uids = list(map(int, body.get("uids")))
            except Exception:
                logger.error(f"Error in uids: {body.get('uids')}")
        else:
            uids = filter_available_uids(
                task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=N_MINERS
            )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")

        # Choose between regular inference, test time inference, and mixture of miners.
        if body.get("test_time_inference", False):
            return await test_time_inference(request)
        elif body.get("mixture", False):
            return await mixture_of_miners(body, uids=uids)
        else:
            return await chat_completion(body, uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post("/web_retrieval", response_model=WebRetrievalResponse)
async def web_retrieval(
    request: WebRetrievalRequest,
    api_key: str = Depends(validate_api_key),
):
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

    timeout_seconds = 30
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
    asyncio.create_task(scoring_queue.scoring_queue.append_response(uids=uids, body=body, chunks=collected_chunks_list))
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


@router.post("/test_time_inference")
async def test_time_inference(request: TestTimeInferenceRequest):
    
    async def create_response_stream(request):
        async for steps, total_thinking_time in generate_response(request.messages, model=request.model, uids=request.uids):
            if total_thinking_time is not None:
                logger.debug(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            yield steps, total_thinking_time

    # Create a streaming response that yields each step
    async def stream_steps():
        try:
            i = 0
            async for steps, thinking_time in create_response_stream(request):
                i += 1
                yield "data: " + ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=request.model or "None",
                    object="chat.completion.chunk",
                    choices=[
                        Choice(index=i, delta=ChoiceDelta(content=f"## {steps[-1][0]}\n\n{steps[-1][1]}" + "\n\n"))
                    ],
                ).model_dump_json() + "\n\n"
        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            yield f'data: {{"error": "Internal Server Error: {str(e)}"}}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_steps(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
