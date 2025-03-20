import asyncio
import gc
import time
from typing import Dict

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.llms.hf_llm import ReproducibleHF
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from shared import settings
from shared.loop_runner import AsyncLoopRunner

# This maintains a list of tasks for which we need to generate references. Since
# we can only generate the references, when the correct model is loaded, we work
# through the tasks based on the currently loaded model.
open_tasks = []


class ModelManager(BaseModel):
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleHF] = {}
    used_ram: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_always_active_models(self):
        for model_config in self.always_active_models:
            self.load_model(model_config)

    def load_model(self, model_config: ModelConfig, force: bool = True):
        torch.cuda.empty_cache()
        if model_config in self.active_models.keys():
            print(f"Model {model_config.llm_model_id} is already loaded.")
            return

        # if force loading is enabled, unload models until there is enough RAM
        if force:
            logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
            for active_model in list(self.active_models.keys()):
                if active_model in self.always_active_models:
                    logger.debug(f"Model {active_model.llm_model_id} is always active. Skipping.")
                    continue
                if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
                    logger.debug(f"Unloading {active_model.llm_model_id} to make room for {model_config.llm_model_id}")
                    self.unload_model(active_model)
                else:
                    logger.debug(f"Enough RAM for model {model_config.llm_model_id} free")
                    GPUInfo.log_gpu_info()
                    break

            # If no active models remain but GPU is still showing significant usage,
            # perform emergency cleanup
            if len(self.active_models) == 0 and GPUInfo.gpu_utilization > 0.25:  # More than 25% still in use
                logger.warning(
                    "GPU still showing high utilization after unloading all models. Performing emergency cleanup."
                )
                self._emergency_gpu_cleanup()

        if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
            if not force:
                logger.warning(f"Not enough RAM to load model {model_config.llm_model_id}.")
                GPUInfo.log_gpu_info()
            raise MemoryError(
                f"""Not enough RAM to load model {model_config.llm_model_id}.
                    Required: {model_config.min_ram} GB
                    Available in Model Manager: {self.total_ram - self.used_ram} GB
                    Available in GPU: {GPUInfo.free_memory} GB"""
            )

        try:
            GPUInfo.log_gpu_info()

            model = model_factory(model_config.llm_model_id)(
                model_id=model_config.llm_model_id,
                device=settings.shared_settings.NEURON_DEVICE,
                sampling_params=settings.shared_settings.SAMPLING_PARAMS,
            )

            self.active_models[model_config] = model
            self.used_ram += model_config.min_ram
            logger.info(f"Model {model_config.llm_model_id} loaded. Current used RAM: {self.used_ram} GB")
            return model
        except Exception as e:
            logger.exception(f"Failed to load model {model_config.llm_model_id}. Error: {str(e)}")

    def _cleanup_pytorch_model(self, model_instance, model_config: ModelConfig):
        """Handle cleanup specifically for PyTorch-based models."""
        if hasattr(model_instance.llm, "model"):
            try:
                # Check if it's a PyTorch model with a 'to' method
                if hasattr(model_instance.llm.model, "to"):
                    logger.debug(f"Moving model {model_config.llm_model_id} to CPU before deletion")
                    model_instance.llm.model.to("cpu")
                    time.sleep(0.1)

                    # Explicitly set requires_grad to False for all parameters if possible
                    if hasattr(model_instance.llm.model, "parameters"):
                        for param in model_instance.llm.model.parameters():
                            if hasattr(param, "requires_grad"):
                                param.requires_grad = False

            except Exception as e:
                logger.debug(f"Could not move model to CPU: {str(e)}, proceeding with direct deletion")

            # Delete the model reference and any cached states
            if hasattr(model_instance.llm.model, "_clear_cache"):
                model_instance.llm.model._clear_cache()

            # Explicitly delete model components if available
            if hasattr(model_instance.llm.model, "modules"):
                for module in list(model_instance.llm.model.modules()):
                    del module

            # Final deletion of model
            del model_instance.llm.model

    def unload_model(self, model_config: ModelConfig):
        if model_config not in self.active_models:
            logger.warning("Couldn't find model to unload.")
            return

        try:
            # Get the model instance
            model_instance = self.active_models[model_config]

            # Record initial memory state for debugging
            initial_free_memory = GPUInfo.free_memory
            logger.debug(f"Initial free GPU memory before unloading: {initial_free_memory} GB")

            # Different model implementations have different structures
            # Handle vLLM-based models
            if hasattr(model_instance, "llm") and hasattr(model_instance.llm, "llm_engine"):
                if hasattr(model_instance.llm.llm_engine, "model_executor") and hasattr(
                    model_instance.llm.llm_engine.model_executor, "driver_worker"
                ):
                    del model_instance.llm.llm_engine.model_executor.driver_worker

            # Handle pipeline-based models with a hybrid approach
            if hasattr(model_instance, "llm"):
                # Try to move model to CPU first if it's a PyTorch model
                self._cleanup_pytorch_model(model_instance, model_config)

                # Handle tokenizer
                if hasattr(model_instance.llm, "tokenizer"):
                    del model_instance.llm.tokenizer

                # Delete the llm object itself
                del model_instance.llm

            # Remove the model from active models dictionary
            del self.active_models[model_config]

            # Force Python garbage collection multiple times to ensure cleanup
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            # Additional memory cleanup for PyTorch
            if torch.cuda.is_available():
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                # Synchronize CUDA to ensure operations are complete
                torch.cuda.synchronize()

                # Force additional cleanup with multiple empty_cache calls
                torch.cuda.empty_cache()

                # Wait a bit longer to ensure memory is released back to the system
                import time

                time.sleep(0.5)
                torch.cuda.empty_cache()

                # Additional synchronization point
                torch.cuda.synchronize()

            # One final garbage collection
            gc.collect()

            # Report memory change
            memory_freed = GPUInfo.free_memory - initial_free_memory
            logger.info(f"Successfully unloaded model {model_config.llm_model_id}. Memory freed: {memory_freed:.2f} GB")

        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")

        # Update used RAM tracking
        self.used_ram -= model_config.min_ram

        # Log current memory state
        GPUInfo.log_gpu_info()

    def get_or_load_model(self, llm_model_id: str) -> ReproducibleHF:
        model_config = ModelZoo.get_model_by_id(llm_model_id)
        if model_config not in self.active_models:
            self.load_model(model_config)
        return self.active_models[model_config]

    def get_model(self, llm_model: ModelConfig | str) -> ReproducibleHF:
        if not llm_model:
            llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
        if isinstance(llm_model, str):
            llm_model = ModelZoo.get_model_by_id(llm_model)

        if llm_model in self.active_models:
            return self.active_models.get(llm_model)
        else:
            return self.load_model(llm_model, force=True)

    def _make_prompt(self, messages: list[dict[str, str]]) -> str:
        role_template = {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>",
        }

        composed_prompt: list[str] = []

        for message in messages:
            role = message["role"]
            if role not in role_template:
                continue
            content = message["content"]
            composed_prompt.append(role_template[role].format(content))

        # Adds final tag indicating the assistant's turn
        composed_prompt.append(role_template["end"])
        return "".join(composed_prompt)

    def generate(
        self,
        messages: list[str],
        roles: list[str] | None = None,
        model: ModelConfig | str | None = None,
        seed: int = None,
        sampling_params: Dict[str, float] = None,
    ) -> str:
        if messages and isinstance(messages[0], dict):
            dict_messages = messages
        else:
            dict_messages = [{"content": message, "role": role} for message, role in zip(messages, roles)]

        if isinstance(model, str):
            model = ModelZoo.get_model_by_id(model)
        if not model:
            model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleHF = self.get_model(model)
        responses = model_instance.generate(messages=[dict_messages], sampling_params=sampling_params, seed=seed)

        return responses

    def _emergency_gpu_cleanup(self):
        """
        Perform an emergency cleanup of GPU memory when standard unloading
        doesn't free up expected memory.
        """
        logger.info("Performing emergency GPU cleanup")

        # Reset model tracking state
        self.active_models = {}
        self.used_ram = 0.0

        # Run aggressive cleanup sequence
        import time

        # Multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.1)

        # Force CUDA synchronization
        torch.cuda.synchronize()

        # Reset all CUDA cached memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Try to release all unreachable objects
        gc.collect(generation=2)

        # Delay to allow OS to reclaim memory
        time.sleep(1.0)

        # Final cache clear
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info(f"Emergency cleanup complete. Current GPU utilization: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    interval: int = 14400
    scoring_queue: list | None = None

    async def start(self, scoring_queue: list, name: str | None = None):
        self.scoring_queue = scoring_queue
        return await super().start(name=name)

    async def initialise_loop(self):
        model_manager.load_always_active_models()

    async def run_step(self):
        """This method is called periodically according to the interval."""
        # try to load the model belonging to the oldest task in the queue
        selected_model = self.scoring_queue[0].task.llm_model if self.scoring_queue else None
        if not selected_model:
            selected_model = ModelZoo.get_random(max_ram=self.llm_model_manager.total_ram)
        logger.info(f"Loading model {selected_model.llm_model_id} for {self.interval} seconds.")

        if selected_model in self.llm_model_manager.active_models:
            logger.info(f"Model {selected_model.llm_model_id} is already loaded.")
            return

        logger.debug(f"Active models: {self.llm_model_manager.active_models.keys()}")
        # Load the selected model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.llm_model_manager.load_model, selected_model)
        await asyncio.sleep(0.01)


model_manager = ModelManager()
model_scheduler = AsyncModelScheduler(llm_model_manager=model_manager, sync=True)
