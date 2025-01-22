import asyncio
import json
import random

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from loguru import logger
from starlette.responses import StreamingResponse

from shared.epistula import SynapseStreamResult, query_miners
from shared.settings import shared_settings
from shared.uids import get_uids
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.utils import forward_response
from validator_api.test_time_inference import generate_response

router = APIRouter()

# load api keys from api_keys.json
with open("api_keys.json", "r") as f:
    _keys = json.load(f)


def validate_api_key(api_key: str = Header(...)):
    if api_key not in _keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return _keys[api_key]


@router.post("/v1/chat/completions")
async def completions(request: Request, api_key: str = Depends(validate_api_key)):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))

        # Choose between regular completion and mixture of miners.
        # if body.get("test_time_inference", False):
        #     return await test_time_inference(body)
        if body.get("mixture", False):
            return await mixture_of_miners(body)
        else:
            return await chat_completion(body)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post("/web_retrieval")
async def web_retrieval(search_query: str, n_miners: int = 10, uids: list[int] = None):
    uids = list(get_uids(sampling_mode="random", k=n_miners))
    logger.debug(f"🔍 Querying uids: {uids}")
    if len(uids) == 0:
        logger.warning("No available miners. This should already have been caught earlier.")
        return

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "messages": [
            {"role": "user", "content": search_query},
        ],
    }

    timeout_seconds = 30
    stream_results = await query_miners(uids, body, timeout_seconds)
    results = [
        "".join(res.accumulated_chunks)
        for res in stream_results
        if isinstance(res, SynapseStreamResult) and res.accumulated_chunks
    ]
    distinct_results = list(np.unique(results))
    logger.info(
        f"🔍 Collected responses from {len(stream_results)} miners. {len(results)} responded successfully with a total of {len(distinct_results)} distinct results"
    )
    loaded_results = []
    for result in distinct_results:
        try:
            loaded_results.append(json.loads(result))
            logger.info(f"🔍 Result: {result}")
        except Exception:
            logger.error(f"🔍 Result: {result}")
    if len(loaded_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    for uid, res in zip(uids, stream_results):
        asyncio.create_task(
            forward_response(
                uid=uid, body=body, chunks=res.accumulated_chunks if res and res.accumulated_chunks else []
            )
        )
    return loaded_results


@router.post("/test_time_inference")
async def test_time_inference(messages: list[dict]):
    async def create_response_stream(user_query):
        async for steps, total_thinking_time in generate_response(user_query):
            if total_thinking_time is not None:
                logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            yield steps, total_thinking_time

    # Create a streaming response that yields each step
    async def stream_steps():
        try:
            query = messages[-1]["content"]
            logger.info(f"Query: {query}")
            async for steps, thinking_time in create_response_stream(query):
                step_data = {
                    "steps": [{"title": step[0], "content": step[1], "thinking_time": step[2]} for step in steps][-1],
                    "total_thinking_time": thinking_time,
                }
                yield f"data: {json.dumps(step_data)}\n\n"
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
