import os
from contextlib import asynccontextmanager

import torch
from angle_emb import AnglE
from fastapi import FastAPI


def load_config_from_env():
    """Loads configuration from environment variables."""
    model_id = os.getenv("MODEL_ID", "WhereIsAI/UAE-Large-V1")
    device = os.getenv("DEVICE", "cpu")

    return {"model_id": model_id, "device": device}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle embedding model startup and shutdown."""
    print("Loading embeddings model...")
    config = load_config_from_env()
    print(f"Loading model: {config['model_id']} on device: {config['device']}")

    model = AnglE.from_pretrained(config["model_id"], pooling_strategy="cls")

    if config["device"] == "cuda" and torch.cuda.is_available():
        model.to(torch.device("cuda"))
        print("Embeddings model moved to CUDA.")
    else:
        model.to(torch.device("cpu"))
        print("Embeddings model moved to CPU.")

    app.state.embeddings_model = model
    app.state.embeddings_model_id = config["model_id"]
    print("Embeddings model loaded.")

    yield

    print("Shutting down embeddings model...")
    app.state.model = None
    app.state.model_id = None
    print("Embeddings model shut down.")
