import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI
from loguru import logger

from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api import scoring_queue
from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api
from validator_api.web_retrieval import router as web_retrieval_router

# import pydevd_pycharm

# pydevd_pycharm.settrace(
#     host='46.199.201.61',
#     port=12345,
#     stdoutToServer=True,
#     stderrToServer=True,
#     suspend=False  # Set to True if you want to pause execution here
# )

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if shared_settings.SCORE_ORGANICS:
        scoring_task = asyncio.create_task(scoring_queue.scoring_queue.start())
    miner_task = asyncio.create_task(update_miner_availabilities_for_api.start())
    yield
    miner_task.cancel()
    if shared_settings.SCORE_ORGANICS:
        scoring_task.cancel()
    try:
        await miner_task
        if shared_settings.SCORE_ORGANICS:
            await scoring_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Validator API",
    description="API for interacting with the validator network and miners",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "GPT Endpoints",
            "description": "Endpoints for chat completions, web retrieval, and test time inference",
        },
        {
            "name": "API Management",
            "description": "Endpoints for API key management and validation",
        },
        {
            "name": "Web Retrieval",
            "description": "Endpoints for retrieving information from the web using miners",
        },
    ],
    lifespan=lifespan,
)
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])
app.include_router(web_retrieval_router, tags=["Web Retrieval"])


@app.get(
    "/health",
    summary="Health check endpoint",
    description="Simple endpoint to check if the API is running",
    tags=["Health"],
    response_description="Status of the API",
)
async def health():
    """
    Health check endpoint to verify the API is operational.

    Returns a simple JSON object with status "ok" if the API is running.
    """
    return {"status": "ok"}


async def main():
    logger.info(f"Starting API with {shared_settings.WORKERS} worker(s).")
    config = uvicorn.Config(
        "validator_api.api:app",
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        # Note: The `workers` parameter is typically only supported via the CLI.
        # When running programmatically with `server.serve()`, only a single worker will run.
        workers=shared_settings.WORKERS,
        reload=False,
    )
    server = uvicorn.Server(config)

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
