from fastapi import APIRouter, Request

from gpu_container.decorators import require_resource

router = APIRouter()


@router.post("/v1/chat/generate_logits")
@require_resource()
async def generate_logits(request: Request):
    json_request = await request.json()
    return await request.app.state.vllm_engine.generate_logits(
        messages=json_request["messages"],
        sampling_params=json_request["sampling_params"],
        seed=json_request["seed"],
        continue_last_message=json_request["continue_last_message"],
        top_logprobs=json_request["top_logprobs"],
    )


@router.post("/v1/chat/generate")
@require_resource()
async def generate(request: Request):
    json_request = await request.json()
    return await request.app.state.vllm_engine.generate(
        messages=json_request["messages"],
        sampling_params=json_request["sampling_params"],
        seed=json_request["seed"],
        continue_last_message=json_request.get("continue_last_message", False),
    )
