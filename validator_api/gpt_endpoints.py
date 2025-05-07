import random

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from loguru import logger
from starlette.responses import StreamingResponse

from shared import settings

shared_settings = settings.shared_settings
from validator_api.api_management import validate_api_key
from validator_api.chat_completion import chat_completion
from validator_api.deep_research.orchestrator_v2 import OrchestratorV2
from validator_api.job_store import JobStatus, job_store, process_chain_of_thought_job
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.serializers import CompletionsRequest, JobResponse, JobResultResponse, TestTimeInferenceRequest
from validator_api.utils import filter_available_uids

router = APIRouter()
N_MINERS = 10


@router.post(
    "/v1/chat/completions",
    summary="Chat completions endpoint",
    description="Main endpoint that handles both regular, multi step reasoning, test time inference, and mixture of miners chat completion.",
    response_description="Streaming response with generated text",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successful response with streaming text",
            "content": {"text/event-stream": {}},
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error or no available miners"},
    },
)
async def completions(request: CompletionsRequest, api_key: str = Depends(validate_api_key)):
    """
    Chat completions endpoint that supports different inference modes.

    This endpoint processes chat messages and returns generated completions using
    different inference strategies based on the request parameters.

    ## Inference Modes:
    - Regular chat completion
    - Multi Step Reasoning
    - Test time inference
    - Mixture of miners

    ## Request Parameters:
    - **uids** (List[int], optional): Specific miner UIDs to query. If not provided, miners will be selected automatically.
    - **messages** (List[dict]): List of message objects with 'role' and 'content' keys. Required.
    - **seed** (int, optional): Random seed for reproducible results.
    - **task** (str, optional): Task identifier to filter available miners.
    - **model** (str, optional): Model identifier to filter available miners.
    - **test_time_inference** (bool, default=False): Enable step-by-step reasoning mode.
    - **mixture** (bool, default=False): Enable mixture of miners mode.
    - **sampling_parameters** (dict, optional): Parameters to control text generation.

    The endpoint selects miners based on the provided UIDs or filters available miners
    based on task and model requirements.

    Example request:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Tell me about neural networks"}
      ],
      "model": "gpt-4",
      "seed": 42
    }
    ```
    """
    try:
        body = request.model_dump()
        if body.get("inference_mode") == "Reasoning-Fast":
            body["task"] = "MultiStepReasoningTask"
        if body.get("model") == "Default":
            # By setting default, we are allowing a user to use whatever model we define as the standard, could also set to None.
            body["model"] = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
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

        if body.get("test_time_inference", False) or body.get("inference_mode", None) == "Chain-of-Thought":
            test_time_request = TestTimeInferenceRequest(
                messages=request.messages,
                model=request.model,
                uids=uids if uids else None,
                json_format=request.json_format,
            )
            return await test_time_inference(test_time_request)
        elif body.get("mixture", False) or body.get("inference_mode", None) == "Mixture-of-Agents":
            return await mixture_of_miners(body, uids=uids)
        else:
            return await chat_completion(body, uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


async def test_time_inference(request: TestTimeInferenceRequest):
    """
    Test time inference endpoint that provides step-by-step reasoning.

    This endpoint streams the thinking process and reasoning steps during inference,
    allowing visibility into how the model arrives at its conclusions. Each step of
    the reasoning process is streamed as it becomes available.

    ## Request Parameters:
    - **messages** (List[dict]): List of message objects with 'role' and 'content' keys. Required.
    - **model** (str, optional): Optional model identifier to use for inference.
    - **uids** (List[int], optional): Optional list of specific miner UIDs to query.

    ## Response:
    The response is streamed as server-sent events (SSE) with each step of reasoning.
    Each event contains:
    - A step title/heading
    - The content of the reasoning step
    - Timing information (debug only)

    Example request:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Solve the equation: 3x + 5 = 14"}
      ],
      "model": "gpt-4"
    }
    ```
    """
    orchestrator = OrchestratorV2(completions=completions)

    async def create_response_stream(request):
        async for chunk in orchestrator.run(messages=request.messages):
            yield chunk

    return StreamingResponse(
        create_response_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/v1/chat/completions/jobs",
    summary="Asynchronous chat completions endpoint for Chain-of-Thought",
    description="Submit a Chain-of-Thought inference job to be processed in the background and get a job ID immediately.",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Job accepted for processing",
            "model": JobResponse,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error or no available miners"},
    },
)
async def create_chain_of_thought_job(
    request: CompletionsRequest, background_tasks: BackgroundTasks, api_key: str = Depends(validate_api_key)
):
    """
    Submit a Chain-of-Thought inference job to be processed in the background.

    This endpoint accepts the same parameters as the /v1/chat/completions endpoint,
    but instead of streaming the response, it submits the job to the background and
    returns a job ID immediately. The job results can be retrieved using the
    /v1/chat/completions/jobs/{job_id} endpoint.

    ## Request Parameters:
    - **uids** (List[int], optional): Specific miner UIDs to query. If not provided, miners will be selected automatically.
    - **messages** (List[dict]): List of message objects with 'role' and 'content' keys. Required.
    - **model** (str, optional): Model identifier to filter available miners.

    ## Response:
    - **job_id** (str): Unique identifier for the job.
    - **status** (str): Current status of the job (pending, running, completed, failed).
    - **created_at** (str): Timestamp when the job was created.
    - **updated_at** (str): Timestamp when the job was last updated.

    Example request:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Solve the equation: 3x + 5 = 14"}
      ],
      "model": "gpt-4"
    }
    ```
    """
    try:
        body = request.model_dump()

        # Check if inference mode is Chain-of-Thought, if not return error
        if body.get("inference_mode") != "Chain-of-Thought":
            raise HTTPException(status_code=400, detail="This endpoint only accepts Chain-of-Thought inference mode")

        body["model"] = (
            "mrfakename/mistral-small-3.1-24b-instruct-2503-hf" if body.get("model") == "Default" else body.get("model")
        )

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

        # Create a new job
        job_id = job_store.create_job()

        # Create the test time inference request
        test_time_request = TestTimeInferenceRequest(
            messages=request.messages,
            model=request.model,
            uids=uids if uids else None,
            json_format=request.json_format,
        )

        # Create the orchestrator
        orchestrator = OrchestratorV2(completions=completions)

        # Add the background task
        background_tasks.add_task(
            process_chain_of_thought_job,
            job_id=job_id,
            orchestrator=orchestrator,
            messages=test_time_request.messages,
        )

        # Get the job
        job = job_store.get_job(job_id)

        # Return the job response
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
        )

    except Exception as e:
        logger.exception(f"Error in creating chain of thought job: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.get(
    "/v1/chat/completions/jobs/{job_id}",
    summary="Get the status and result of a Chain-of-Thought job",
    description="Retrieve the status and result of a Chain-of-Thought job by its ID.",
    response_model=JobResultResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Job status and result",
            "model": JobResultResponse,
        },
        status.HTTP_404_NOT_FOUND: {"description": "Job not found"},
    },
)
async def get_chain_of_thought_job(job_id: str, api_key: str = Depends(validate_api_key)):
    """
    Get the status and result of a Chain-of-Thought job.

    This endpoint retrieves the status and result of a Chain-of-Thought job by its ID.
    If the job is completed, the result will be included in the response.
    If the job failed, the error message will be included in the response.

    ## Path Parameters:
    - **job_id** (str): The ID of the job to retrieve.

    ## Response:
    - **job_id** (str): Unique identifier for the job.
    - **status** (str): Current status of the job (pending, running, completed, failed).
    - **created_at** (str): Timestamp when the job was created.
    - **updated_at** (str): Timestamp when the job was last updated.
    - **result** (List[str], optional): Result of the job if completed.
    - **error** (str, optional): Error message if the job failed.
    """
    job = job_store.get_job(job_id)

    if job.status == JobStatus.COMPLETED:  # todo check if job is deleted
        job_store.delete_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        result=job.result,
        error=job.error,
    )
