from functools import wraps

from fastapi import Request, status
from fastapi.responses import JSONResponse


def require_resource():
    """
    Decorator that checks if all resources are loaded before executing the endpoint.
    Returns a 503 Service Unavailable response if any resource is not loaded.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check if engine is loaded
            if not hasattr(request.app.state, "vllm_engine") or request.app.state.vllm_engine is None:
                return JSONResponse(
                    content={"error": "LLM engine is still loading"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            # Check if embeddings model is loaded
            if not hasattr(request.app.state, "embeddings_model") or request.app.state.embeddings_model is None:
                return JSONResponse(
                    content={"error": "Embeddings model is still loading"},
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            # If all required resources are loaded, execute the endpoint
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
