import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from schema import ChatRequest, LogitsRequest
from vllm_llm import ReproducibleVLLM

MODEL_PATH = os.getenv("MODEL_PATH")


class ReproducibleVllmApp:
    def __init__(self):
        self.llm = ReproducibleVLLM(model_id=MODEL_PATH)
        self.app = FastAPI()
        self.app.post("/generate")(self.generate)
        self.app.post("/generate_logits")(self.generate_logits)

    async def generate(self, request: ChatRequest):
        try:
            result = await self.llm.generate(
                messages=[m.dict() for m in request.messages],
                sampling_params=request.sampling_parameters.dict(),
                seed=request.seed,
                continue_last_message=request.continue_last_message,
            )
            return {"result": result}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def generate_logits(self, request: LogitsRequest):
        try:
            logits, prompt = await self.llm.generate_logits(
                messages=[m.dict() for m in request.messages],
                top_logprobs=request.top_logprobs,
                sampling_params=request.sampling_parameters.dict(),
                seed=request.seed,
                continue_last_message=request.continue_last_message,
            )
            return {"logits": logits, "prompt": prompt}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    server = ReproducibleVllmApp()
    server.run()
