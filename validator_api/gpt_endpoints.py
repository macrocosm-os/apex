import random

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from starlette.responses import StreamingResponse

from shared import settings

shared_settings = settings.shared_settings
from validator_api.api_management import validate_api_key
from validator_api.chat_completion import chat_completion
from validator_api.deep_research.orchestrator_v2 import OrchestratorV2
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.serializers import CompletionsRequest, TestTimeInferenceRequest
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
