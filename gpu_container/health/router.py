from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check that always returns 'ok'."""
    return {"status": "ok"}


@router.get("/health/model")
async def model_health_check(request: Request):
    """
    Model health check that returns 200 only when both the engine and embeddings model are loaded.
    Returns 503 Service Unavailable if either the engine or embeddings model is not loaded yet.
    """
    vllm_ready = hasattr(request.app.state, "vllm_engine") and request.app.state.vllm_engine is not None
    embeddings_ready = hasattr(request.app.state, "embeddings_model") and request.app.state.embeddings_model is not None

    if vllm_ready and embeddings_ready:
        return {"status": "ok", "model": request.app.state.vllm_model_id}
    else:
        message = "Models are still loading: "
        if not vllm_ready:
            message += "LLM engine not ready. "
        if not embeddings_ready:
            message += "Embeddings model not ready."

        return JSONResponse(
            content={"status": "not ready", "message": message.strip()}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
