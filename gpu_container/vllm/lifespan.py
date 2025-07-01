import asyncio
import concurrent.futures
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gpu_container.vllm.reproducible_vllm import ReproducibleVLLM

# Global executor to avoid creating/destroying it with each lifespan
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def load_config_from_env():
    """Loads vLLM configuration from environment variables."""
    vllm_model_id = os.getenv("VLLM_MODEL_ID", "default_model_id")
    hf_model_path = os.getenv("HF_MODEL_PATH", None)  # Use pre-downloaded model
    device = os.getenv("DEVICE", "cuda")
    # Whether to block app startup until the engine is loaded
    block_startup = os.getenv("BLOCK_STARTUP", "true").lower() == "true"
    # Add any other vLLM-specific environment variables here
    return {
        "vllm_model_id": vllm_model_id,
        "hf_model_path": hf_model_path,
        "device": device,
        "block_startup": block_startup,
    }


def initialize_engine(config):
    """Initialize the vLLM engine in a separate thread."""
    model_id = config["vllm_model_id"]
    device = config["device"]
    print(f"Initializing vLLM engine with model {model_id} on {device}...")

    # Use pre-downloaded model path or vllm_model_id
    vllm_model_to_load = config["hf_model_path"] or config["vllm_model_id"]

    return ReproducibleVLLM(model_id=vllm_model_to_load, device=device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle vLLM engine startup and shutdown."""
    print("Loading vLLM engine...")
    config = load_config_from_env()

    # Set initial engine state to None
    app.state.vllm_engine = None
    app.state.vllm_model_id = config["vllm_model_id"]
    app.state.vllm_engine_loading = True

    # Submit the task to the global executor
    engine_future = _executor.submit(initialize_engine, config)

    if config["block_startup"]:
        # Block until engine is loaded (original behavior)
        print("Blocking app startup until engine is loaded...")
        engine = await asyncio.wrap_future(engine_future)
        app.state.vllm_engine = engine
        app.state.vllm_engine_loading = False
        print("vLLM engine loaded.")
    else:
        # Don't block, start app immediately and load engine in background
        print("Starting app immediately, engine will load in background...")

        async def set_engine_when_ready():
            try:
                engine = await asyncio.wrap_future(engine_future)
                app.state.vllm_engine = engine
                app.state.vllm_engine_loading = False
                print("vLLM engine loaded.")
            except Exception as e:
                print(f"Error loading vLLM engine: {e}")
                app.state.vllm_engine_loading = False

        # Start the background task to set the engine when ready
        asyncio.create_task(set_engine_when_ready())

    yield

    print("Shutting down vLLM engine...")
    app.state.vllm_engine = None
    app.state.vllm_model_id = None
    print("vLLM engine shut down.")
