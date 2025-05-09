from typing import ClassVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from shared import settings


class ModelConfig(BaseModel):
    llm_model_id: str
    reward: float
    min_ram: float
    model_config = ConfigDict(frozen=True)

    def __hash__(self):
        return hash((self.llm_model_id, self.reward, self.min_ram))


class ModelZoo:
    # Dynamically create model configs from the list of models in settings
    models_configs: ClassVar[list[ModelConfig]] = []

    # Initialize models directly in the class
    # Handle both string and list configurations
    models = settings.shared_settings.LLM_MODEL
    if isinstance(models, str):
        models = [models]

    # Add each model from settings to the configs
    for model in models:
        models_configs.append(
            ModelConfig(
                llm_model_id=model,
                reward=1 / len(models),
                min_ram=settings.shared_settings.MAX_ALLOWED_VRAM_GB,
            )
        )

    @classmethod
    def get_all_models(cls) -> list[str]:
        return [model.llm_model_id for model in cls.models_configs]

    @classmethod
    def get_random(cls, max_ram: float = np.inf) -> ModelConfig:
        # models = [model for model in cls.models_configs if model.min_ram <= max_ram]
        # if len(models) == 0:
        #     raise Exception(f"No model with < {max_ram}GB memory requirements found")
        return np.random.choice(cls.models_configs)

    @classmethod
    def get_model_by_id(cls, model_id: str) -> ModelConfig:
        if not model_id:
            logger.error("model_id cannot be None or empty. Returning None...")
            return None
        try:
            return [model for model in cls.models_configs if model.llm_model_id == model_id][0]
        except Exception as ex:
            logger.error(f"Model {model_id} not found in ModelZoo: {ex}")
