import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.llms.model_zoo import ModelZoo
from prompting.tasks.inference import InferenceTask
from prompting.tasks.web_retrieval import WebRetrievalTask
from shared import settings
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent
from shared.epistula import SynapseStreamResult, verify_signature
from shared.settings import shared_settings

router = APIRouter()


async def verify_scoring_signature(request: Request):
    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")
    if signed_for != shared_settings.WALLET.hotkey.ss58_address:
        raise HTTPException(status_code=400, detail="Bad Request, message is not intended for self")
    if signed_by != shared_settings.API_HOTKEY:
        raise HTTPException(status_code=401, detail="Signer not the expected ss58 address")

    body = await request.body()
    now = time.time()
    err = verify_signature(
        request.headers.get("Epistula-Request-Signature"),
        body,
        request.headers.get("Epistula-Timestamp"),
        request.headers.get("Epistula-Uuid"),
        signed_for,
        signed_by,
        now,
    )
    if err:
        logger.error(err)
        raise HTTPException(status_code=400, detail=err)


def validate_scoring_key(request: Request):
    if request.headers.api_key != settings.shared_settings.SCORING_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def get_task_scorer(request: Request):
    return request.app.state.task_scorer


@router.post("/scoring")
async def score_response(
    request: Request, api_key_data: dict = Depends(verify_scoring_signature), task_scorer=Depends(get_task_scorer)
):
    model = None
    payload: dict[str, Any] = await request.json()
    logger.debug(f"Awaited body: {payload}")
    body = payload.get("body")
    timeout = payload.get("timeout", shared_settings.NEURON_TIMEOUT)
    uids = payload.get("uids", [])
    chunks = payload.get("chunks", {})
    timings = payload.get("timings", {})
    logger.debug("About to check chunks and uids")
    if not uids or not chunks:
        logger.error(f"Either uids: {uids} or chunks: {chunks} is not valid, skipping scoring")
        return
    uids = [int(uid) for uid in uids]
    model = body.get("model")
    logger.debug("About to check model")
    if model and model != shared_settings.LLM_MODEL:
        logger.error(f"Model {model} not available for scoring on this validator.")
        return
    logger.debug("Model has been checked")
    llm_model = ModelZoo.get_model_by_id(model)
    logger.debug("Got LLM Model from ModelZoo")
    task_name = body.get("task")
    logger.debug(f"Task name set: {task_name}")
    logger.debug(f"Length pre-insertion: {len(task_scorer.scoring_queue)}")
    if task_name == "InferenceTask":
        organic_task = InferenceTask(
            messages=body.get("messages"),
            llm_model=llm_model,
            llm_model_id=llm_model,
            seed=int(body.get("seed", 0)),
            sampling_params=body.get("sampling_parameters", shared_settings.SAMPLING_PARAMS),
            query=body.get("messages"),
            organic=True,
        )
        task_scorer.add_to_queue(
            task=organic_task,
            response=DendriteResponseEvent(
                uids=uids,
                stream_results=[SynapseStreamResult(accumulated_chunks=chunks.get(str(uid), None)) for uid in uids],
                timeout=timeout,
                stream_results_all_chunks_timings=[timings.get(str(uid), None) for uid in uids],
            ),
            dataset_entry=DatasetEntry(),
            block=shared_settings.METAGRAPH.block,
            step=-1,
            task_id=str(uuid.uuid4()),
        )

    elif task_name == "WebRetrievalTask":
        try:
            search_term = body.get("messages")[0].get("content")
        except Exception as ex:
            logger.error(f"Failed to get search term from messages: {ex}, can't score WebRetrievalTask")
            return

        task_scorer.add_to_queue(
            task=WebRetrievalTask(
                messages=[msg["content"] for msg in body.get("messages")],
                seed=int(body.get("seed", 0)),
                sampling_params=body.get("sampling_params", {}),
                query=search_term,
            ),
            response=DendriteResponseEvent(
                uids=uids,
                stream_results=[SynapseStreamResult(accumulated_chunks=chunks.get(str(uid), [])) for uid in uids],
                timeout=body.get("timeout", shared_settings.NEURON_TIMEOUT),
            ),
            dataset_entry=DDGDatasetEntry(search_term=search_term),
            block=shared_settings.METAGRAPH.block,
            step=-1,
            task_id=str(uuid.uuid4()),
        )
    logger.debug(f"Current Queue: {len(task_scorer.scoring_queue)}")
    logger.info("Organic task appended to scoring queue")
