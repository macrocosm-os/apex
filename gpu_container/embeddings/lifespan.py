import asyncio
import os
from contextlib import asynccontextmanager

import torch
from angle_emb import AnglE
from fastapi import FastAPI

# Share the executor with vLLM lifespan
from gpu_container.vllm.lifespan import _executor


def load_config_from_env():
    """Loads configuration from environment variables."""
    model_id = os.getenv("MODEL_ID", "WhereIsAI/UAE-Large-V1")
    device = os.getenv("DEVICE", "cpu")
    # Whether to block app startup until the model is loaded
    block_startup = os.getenv("BLOCK_STARTUP", "true").lower() == "true"

    return {"model_id": model_id, "device": device, "block_startup": block_startup}


def initialize_model(model_id, device):
    """Initialize the embeddings model in a separate thread."""
    print(f"Loading model: {model_id} on device: {device}")
    model = AnglE.from_pretrained(model_id, pooling_strategy="cls")

    if device == "cuda" and torch.cuda.is_available():
        model.to(torch.device("cuda"))
        print("Embeddings model moved to CUDA.")
    else:
        model.to(torch.device("cpu"))
        print("Embeddings model moved to CPU.")

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle embedding model startup and shutdown."""
    print("Loading embeddings model...")
    config = load_config_from_env()

    # Set initial model state to None
    app.state.embeddings_model = None
    app.state.embeddings_model_id = config["model_id"]
    app.state.embeddings_model_loading = True

    # Submit the task to the global executor
    model_future = _executor.submit(initialize_model, config["model_id"], config["device"])

    if config["block_startup"]:
        # Block until model is loaded (original behavior)
        print("Blocking app startup until embeddings model is loaded...")
        model = await asyncio.wrap_future(model_future)
        app.state.embeddings_model = model
        app.state.embeddings_model_loading = False
        print("Embeddings model loaded.")
    else:
        # Don't block, start app immediately and load model in background
        print("Starting app immediately, embeddings model will load in background...")

        async def set_model_when_ready():
            try:
                model = await asyncio.wrap_future(model_future)
                app.state.embeddings_model = model
                app.state.embeddings_model_loading = False
                print("Embeddings model loaded.")
            except Exception as e:
                print(f"Error loading embeddings model: {e}")
                app.state.embeddings_model_loading = False

        # Start the background task to set the model when ready
        asyncio.create_task(set_model_when_ready())

    yield

    print("Shutting down embeddings model...")
    app.state.embeddings_model = None
    app.state.embeddings_model_id = None
    print("Embeddings model shut down.")
