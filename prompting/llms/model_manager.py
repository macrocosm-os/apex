import asyncio
from multiprocessing.managers import AcquirerProxy
from typing import ClassVar

import requests
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.vllm_llm import ReproducibleVLLM
from shared import constants, settings
from shared.loop_runner import AsyncLoopRunner


class AsyncRLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def acquire(self):
        current_task = asyncio.current_task()
        if self._owner == current_task:
            self._count += 1
            return True
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return True

    def release(self):
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError("Lock can only be released by the owner")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()


class ModelManager(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleVLLM] = {}
    loading_tasks: dict[ModelConfig, asyncio.Future] = {}
    used_ram: float = 0.0
    lock: ClassVar[AsyncRLock] = AsyncRLock()

    async def generate(
        self,
        messages: list[str] | list[dict],
        roles: list[str] | None = None,
        model: ModelConfig | str | None = None,
        seed: int = None,
        sampling_params: dict[str, float] = None,
    ) -> str:
        if messages and isinstance(messages[0], dict):
            dict_messages = messages
        else:
            dict_messages = [
                {"content": message, "role": role} for message, role in zip(messages, roles or ["user"] * len(messages))
            ]
        url = f"{constants.DOCKER_BASE_URL}/v1/chat/generate"
        headers = {"Content-Type": "application/json"}
        payload = {"messages": dict_messages, "seed": seed, "sampling_params": sampling_params}
        response = requests.post(url, headers=headers, json=payload)
        try:
            json_response = response.json()
            logger.info(f"Response: {json_response}")
            return json_response["choices"][0]["message"]["content"]
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Error generating response. Status: {response.status_code}, Body: {response.text}")
            return ""

        # async with self.lock:
        #     if isinstance(model, str):
        #         model = ModelZoo.get_model_by_id(model)
        #     if not model:
        #         model = ModelZoo.get_random(max_ram=self.total_ram)

        # model_instance: ReproducibleVLLM = await self.get_model(model)

        # async with self.lock:
        #     if model_instance is None:
        #         raise ValueError("Model is None, which may indicate the model is still loading.")
        #     responses = await model_instance.generate(
        #         messages=dict_messages, sampling_params=sampling_params, seed=seed
        #     )
        #     return responses


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    mp_lock: AcquirerProxy
    interval: int = 3600
    scoring_queue: list | None = None
    memory_error: MemoryError | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(self, scoring_queue: list, name: str | None = None, **kwargs):
        self.scoring_queue = scoring_queue
        await super().start(name=name, **kwargs)
        # Load the model immediately.
        await self.run_step()

    async def run_step(self):
        return
        """This method is called periodically according to the interval."""
        # try to load the model belonging to the oldest task in the queue
        with self.mp_lock:
            selected_model = self.scoring_queue[0].task.llm_model if self.scoring_queue else None
        if not selected_model:
            selected_model = ModelZoo.get_random(max_ram=self.llm_model_manager.total_ram)
        logger.info(f"Loading model {selected_model.llm_model_id} for {self.interval} seconds.")

        if selected_model in self.llm_model_manager.active_models:
            logger.info(f"Model {selected_model.llm_model_id} is already loaded.")
            return

        try:
            await self.llm_model_manager.load_model(selected_model)
        except MemoryError as e:
            self.memory_error = e
        logger.debug(f"Active models: {list(self.llm_model_manager.active_models.keys())}")
        await asyncio.sleep(0.01)
