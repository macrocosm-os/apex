import asyncio
import json
import random
import time
from hashlib import sha256
from math import ceil
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import bittensor as bt
import httpx
import openai
from httpx import Timeout
from loguru import logger
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from substrateinterface import Keypair

from prompting.llms.utils import model_factory
from shared import settings
from shared.dendrite import SynapseStreamResult

shared_settings = settings.shared_settings

# from openai.types import Com


def verify_signature(
    signature, body: bytes, timestamp, uuid, signed_for, signed_by, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 8000
    keypair = Keypair(ss58_address=signed_by)
    if timestamp + ALLOWED_DELTA_MS < now:
        return "Request is too stale"
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
    verified = keypair.verify(message, signature)
    if not verified:
        return "Signature Mismatch"
    return None


def generate_header(
    hotkey: Keypair,
    body_bytes: Dict[str, Any],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    headers = {
        "Epistula-Version": "2",
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{sha256(body_bytes).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        headers["Epistula-Secret-Signature-1"] = "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        headers["Epistula-Secret-Signature-2"] = "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        headers.update(json.loads(body_bytes))
    return headers


def create_header_hook(hotkey, axon_hotkey, timeout_seconds=20):
    async def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            if key not in ["messages", "model", "stream"]:
                request.headers[key] = str(header)
        request.headers["X-Client-Timeout"] = str(timeout_seconds)
        return request

    return add_headers


async def merged_stream(responses: list[AsyncGenerator]):
    streams = [response.__aiter__() for response in responses if not isinstance(response, Exception)]
    pending = {}
    for stream in streams:
        try:
            task = asyncio.create_task(stream.__anext__())
            pending[task] = stream
        except StopAsyncIteration:
            continue  # Skip empty streams

    while pending:
        done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            stream = pending.pop(task)
            try:
                result = task.result()
                yield result
                # Schedule the next item from the same stream
                next_task = asyncio.create_task(stream.__anext__())
                pending[next_task] = stream
            except StopAsyncIteration:
                # Stream is exhausted
                pass
            except Exception as e:
                logger.error(f"Error while streaming: {e}")


async def query_miners(
    uids, body: dict[str, Any], timeout_seconds: int = shared_settings.NEURON_TIMEOUT
) -> list[SynapseStreamResult]:
    try:
        tasks = []
        for uid in uids:
            try:
                timeout_connect = 10
                timeout_postprocess = 1
                response = asyncio.wait_for(
                    asyncio.create_task(
                        make_openai_query(
                            shared_settings.METAGRAPH,
                            shared_settings.WALLET,
                            timeout_seconds,
                            body,
                            uid,
                            timeout_connect=timeout_connect,
                        )
                    ),
                    # Give additional time for result post-processings.
                    timeout=timeout_seconds + timeout_postprocess,
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout exceeded while querying miner {uid}")
                response = Exception(f"Timeout exceeded while querying miner {uid}")
            tasks.append(response)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        responses_valid = 0
        responses_error = 0
        responses_exception = 0
        results: list[SynapseStreamResult] = []
        for response, uid in zip(responses, uids):
            if isinstance(response, Exception):
                responses_exception += 1
                results.append(SynapseStreamResult(exception=str(response)))
            elif isinstance(response, tuple) and isinstance(response[0], ChatCompletion):
                if response and response[1]:
                    responses_valid += 1
                else:
                    responses_error += 1
                results.append(
                    SynapseStreamResult(
                        uid=uid,
                        accumulated_chunks=response[1],
                        accumulated_chunks_timings=response[2],
                        accumulated_chunk_dicts_raw=response[3],
                    )
                )
            else:
                responses_error += 1
                # logger.error(f"Unknown response type: {response}")
                results.append(SynapseStreamResult(uid=uid, exception=f"Unknown response type: {response}"))

        logger.info(
            f"Responses success: {responses_valid}/{len(uids)}. "
            f"Responses exception: {responses_exception}/{len(uids)}. "
            f"Responses invalid: {responses_error}/{len(uids)}"
        )
        return results
    except Exception as e:
        logger.exception(f"Error in query_miners: {e}")
        return []


async def query_availabilities(uids, task_config, model_config):
    """Query the availability of the miners"""
    availability_dict = {"task_availabilities": task_config, "llm_model_availabilities": model_config}
    # Query the availability of the miners
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_availability(
                        shared_settings.METAGRAPH,
                        availability_dict,
                        uid,
                    )
                )
            )
        responses: List[SynapseStreamResult] = await asyncio.gather(*tasks)
        return responses

    except Exception as e:
        logger.error(f"Error in availability call: {e}")
        return []


async def handle_availability(
    metagraph: "bt.NonTorchMetagraph",
    request: Dict[str, Any],
    uid: int,
) -> Dict[str, bool]:
    try:
        axon_info = metagraph.axons[uid]
        url = f"http://{axon_info.ip}:{axon_info.port}/availability"
        # logger.debug(f"Querying availability from {url}")
        timeout = httpx.Timeout(shared_settings.NEURON_TIMEOUT, connect=5, read=5)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=request)

        response.raise_for_status()
        return response.json()

    except Exception:
        return {}


async def make_openai_query(
    metagraph: "bt.NonTorchMetagraph",
    wallet: "bt.wallet",
    timeout_seconds: int,
    body: dict[str, Any],
    uid: int,
    stream: bool = False,
    timeout_connect: int = 10,
) -> tuple[ChatCompletion, list, list] | AsyncGenerator:
    body["seed"] = body.get("seed", random.randint(0, 1000000))
    axon_info = metagraph.axons[uid]
    miner = openai.AsyncOpenAI(
        base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
        api_key="Apex",
        max_retries=0,
        timeout=Timeout(timeout_seconds, connect=timeout_connect, read=timeout_seconds),
        http_client=openai.DefaultAsyncHttpxClient(
            event_hooks={
                "request": [create_header_hook(wallet.hotkey, axon_info.hotkey, timeout_seconds=timeout_seconds)]
            }
        ),
    )
    extra_body = {k: v for k, v in body.items() if k not in ["messages", "model"]}
    if extra_body.get("task") == "InferenceTask":
        body["messages"] = model_factory(body.get("model")).format_messages(body["messages"])

    start_time = time.perf_counter()
    try:
        chat = await miner.chat.completions.create(
            model=body.get("model", None),
            messages=body["messages"],
            stream=True,
            extra_body=extra_body,
        )
    except BaseException:
        # logger.warning(f"Error while querying UID {uid}: {e}")
        return

    if stream:
        return chat
    else:
        choices = []
        chunks = []
        chunk_timings = []
        last_finish_reason = None  # Only track the finish reason of the last chunk

        chunk_dicts_raw = []
        async for chunk in chat:
            if not chunk.choices:
                continue
            for i, choice in enumerate(chunk.choices):
                if i >= len(choices):
                    choices.append("")
                if choice.delta.content:
                    choices[i] += choice.delta.content
                # Save finish reason from the last chunk, safely handling the attribute
                if hasattr(choice, "finish_reason") and choice.finish_reason is not None:
                    last_finish_reason = choice.finish_reason
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
                chunk_timings.append(time.perf_counter() - start_time)
                chunk_dicts_raw.append(chunk)
        choices = [
            Choice(
                index=i,
                message=ChatCompletionMessage(content=choice, role="assistant"),
                finish_reason=last_finish_reason or "stop",  # Use the captured finish_reason or fallback to "stop"
            )
            for i, choice in enumerate(choices)
        ]
        # TODO: We need to find a better way to do this instead of sometimes returning a tuple and sometimes not, but for now this has to do
        return (
            ChatCompletion(
                id=str(uuid4()),
                choices=choices,
                created=int(time.time()),
                model=body.get("model") or "",
                object="chat.completion",
                service_tier=None,
                system_fingerprint=None,
                usage=None,
            ),
            chunks,
            chunk_timings,
            chunk_dicts_raw,
        )
