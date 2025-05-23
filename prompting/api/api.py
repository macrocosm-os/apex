import uvicorn
from fastapi import FastAPI
from loguru import logger

from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from prompting.api.scoring.api import router as scoring_router
from prompting.api.weight_syncing.api import router as weight_syncing_router

# from prompting.rewards.scoring import task_scorer
from shared import settings

app = FastAPI()
app.include_router(miner_availabilities_router, prefix="/miner_availabilities", tags=["miner_availabilities"])
app.include_router(scoring_router, tags=["scoring"])
app.include_router(weight_syncing_router, tags=["weight_syncing"])


@app.get("/health")
def health():
    return {"status": "healthy"}


async def start_scoring_api(task_scorer, scoring_queue, reward_events, miners_dict, weight_dict):
    logger.info("Starting Scoring/Weight Setting API")
    app.state.task_scorer = task_scorer
    app.state.task_scorer.scoring_queue = scoring_queue
    app.state.task_scorer.reward_events = reward_events
    app.state.miners_dict = miners_dict
    app.state.weight_dict = weight_dict

    logger.info(f"Starting Scoring/Weight Setting API on https://0.0.0.0:{settings.shared_settings.SCORING_API_PORT}")
    config = uvicorn.Config(
        "prompting.api.api:app",
        host="0.0.0.0",
        port=settings.shared_settings.SCORING_API_PORT,
        loop="asyncio",
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
