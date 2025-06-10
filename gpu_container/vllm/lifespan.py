import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

def load_config_from_env():
    """Loads vLLM configuration from environment variables."""
    vllm_model_id = os.getenv("VLLM_MODEL_ID", "default_model_id")
    # Add any other vLLM-specific environment variables here
    return {
        "vllm_model_id": vllm_model_id
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle vLLM engine startup and shutdown."""
    print("Loading vLLM engine...")
    config = load_config_from_env()
    
    engine_args = AsyncEngineArgs(model=config['vllm_model_id'])
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    app.state.vllm_engine = engine
    app.state.vllm_model_id = config['vllm_model_id']
    print("vLLM engine loaded.")
    
    yield
    
    print("Shutting down vLLM engine...")
    app.state.vllm_engine = None
    app.state.vllm_model_id = None
    print("vLLM engine shut down.") 