from fastapi import FastAPI
from contextlib import asynccontextmanager
from gpu_container.embeddings.router import router as embeddings_router
from gpu_container.embeddings.lifespan import lifespan as embeddings_lifespan
from gpu_container.vllm.router import router as vllm_router
from gpu_container.vllm.lifespan import lifespan as vllm_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A top-level lifespan handler that calls the lifespan handlers
    for different parts of the application.
    """
    async with embeddings_lifespan(app):
        async with vllm_lifespan(app):
            yield

app = FastAPI(lifespan=lifespan)

app.include_router(embeddings_router)
app.include_router(vllm_router)
