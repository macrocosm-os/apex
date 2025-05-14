import asyncio
import gc
from multiprocessing.managers import AcquirerProxy
from typing import ClassVar

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from prompting.llms.vllm_llm import ReproducibleVLLM
from shared import settings
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
    # lock: ClassVar[AsyncRLock] = asyncio.Lock()

    async def load_model(self, model_config: ModelConfig, force: bool = True) -> ReproducibleVLLM:
        """Load model into GPU.

        Warning: This operation will block execution until the model is successfully loaded into VRAM.

        Args:
            model_config: Model config to load.
            force: If enabled, will unload all other models.
        """
        async with self.lock:
            # Copy active models, since they will be modified in the loop.
            active_models = set(self.active_models.keys())

            if model_config in active_models:
                logger.debug(f"Model {model_config.llm_model_id} is already loaded.")
                return self.active_models[model_config]

            if force:
                logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
                for active_model in active_models:
                    logger.debug(f"Unloading {active_model.llm_model_id} to make room for {model_config.llm_model_id}")
                    await self._unload_model(active_model)
                await self.cleanup()

            try:
                GPUInfo.log_gpu_info()
                model_class = model_factory(model_config.llm_model_id)
                model = model_class(
                    model_id=model_config.llm_model_id,
                    device=settings.shared_settings.NEURON_DEVICE,
                    sampling_params=settings.shared_settings.SAMPLING_PARAMS,
                )
                self.active_models[model_config] = model
                self.used_ram += model_config.min_ram
                logger.info(
                    f"Model {model_config.llm_model_id} has been successfully loaded. "
                    f"Approx. used VRAM: {self.used_ram:.0f}GB"
                )
                await asyncio.sleep(1.0)
                return model
            except BaseException as e:
                await self.cleanup()
                # In case of VRAM leak, raise an exception to terminate the process.
                raise MemoryError(f"Failed to load model {model_config.llm_model_id}: {e}")

    async def _cleanup_model(self, model_instance: ReproducibleVLLM, cpu_offload: bool = False):
        """Free VRAM from given model."""
        if cpu_offload:
            try:
                model_instance.model = model_instance.model.to("cpu")
            except NotImplementedError as e:
                logger.exception(f"Standard move to CPU failed: {str(e)}")
                try:
                    # Fallback for meta tensors.
                    model_instance.model = model_instance.model.to_empty("cpu")
                except Exception as fallback_e:
                    logger.exception(f"Could not move meta model to CPU, proceeding with generic GC: {str(fallback_e)}")
            except Exception as e:
                logger.exception(f"Unexpected error when moving model to CPU: {str(e)}")

        model_instance.unload_model()
        del model_instance

    async def _unload_model(self, model_config: ModelConfig):
        if model_config not in self.active_models:
            logger.warning(f"Couldn't find given model to unload: {model_config}")
            return

        try:
            initial_free_memory = GPUInfo.free_memory
            logger.debug(f"Initial free GPU memory before unloading: {initial_free_memory} GB")
            # async with self.rlock:
            model_instance = self.active_models.pop(model_config)
            await self._cleanup_model(model_instance, cpu_offload=False)
            await self.cleanup()

            memory_freed = GPUInfo.free_memory - initial_free_memory
            logger.info(f"Successfully unloaded model {model_config.llm_model_id}. Memory freed: {memory_freed:.2f} GB")

        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")

        # Update used RAM tracking
        self.used_ram -= model_config.min_ram

        GPUInfo.log_gpu_info()

    async def get_model(self, llm_model: ModelConfig | str) -> ReproducibleVLLM:
        async with self.lock:
            if not llm_model:
                llm_model = next(iter(self.active_models.keys())) if self.active_models else ModelZoo.get_random()
            if isinstance(llm_model, str):
                llm_model = ModelZoo.get_model_by_id(llm_model)
            if llm_model in self.active_models:
                return self.active_models[llm_model]

        return await self.load_model(llm_model)

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
            dict_messages = [{"content": message, "role": role} for message, role in zip(messages, roles)]

        async with self.lock:
            if isinstance(model, str):
                model = ModelZoo.get_model_by_id(model)
            if not model:
                model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleVLLM = await self.get_model(model)

        async with self.lock:
            if model_instance is None:
                raise ValueError("Model is None, which may indicate the model is still loading.")
            responses = await model_instance.generate(
                messages=dict_messages, sampling_params=sampling_params, seed=seed
            )
            return responses

    async def generate_logits(
        self,
        messages: list[str],
        model: ModelConfig | str | None = None,
        sampling_params: dict[str, float] = None,
        seed: int = None,
        continue_last_message: bool = False,
        top_logprobs: int = 10,
    ):
        model_instance: ReproducibleVLLM = await self.get_model(model)
        return await model_instance.generate_logits(
            messages=messages,
            sampling_params=sampling_params,
            seed=seed,
            continue_last_message=continue_last_message,
            top_logprobs=top_logprobs,
        )

    async def cleanup(self):
        """Perform VRAM clean-up."""
        for _, model in self.active_models.items():
            del model.model
            del model

        self.active_models = {}
        self.used_ram = 0.0

        if torch.cuda.is_available():
            # Reset all CUDA cached memory.
            try:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                await asyncio.sleep(1.0)
            except BaseException as e:
                logger.warning(f"Error during CUDA empty cache: {e}")
        else:
            logger.warning("CUDA is not available")

        gc.collect()
        gc.collect(generation=2)
        await asyncio.sleep(1.0)

        logger.info(f"VRAM clean-up completed. Current GPU usage: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


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
