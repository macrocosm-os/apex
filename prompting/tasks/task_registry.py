import random
from typing import ClassVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.datasets.random_website import DDGDataset
from prompting.datasets.sn13 import SN13Dataset
from prompting.rewards.reward import BaseRewardConfig
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.inference import InferenceRewardConfig, InferenceTask
from prompting.tasks.MSRv2_task import MSRv2RewardConfig, MSRv2Task
from prompting.tasks.web_retrieval import WebRetrievalRewardConfig, WebRetrievalTask
from shared.base import BaseDataset


class TaskConfig(BaseModel):
    task: type[BaseTextTask]
    probability: float
    datasets: list[type[BaseDataset]]
    reward_model: type[BaseRewardConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __hash__(self):
        return hash(self.task)


class TaskRegistry(BaseModel):
    task_configs: ClassVar[list[TaskConfig]] = [
        TaskConfig(task=MSRv2Task, probability=0.10, datasets=[DDGDataset], reward_model=MSRv2RewardConfig),
        TaskConfig(
            task=InferenceTask,
            probability=0.55,
            datasets=[SN13Dataset],
            reward_model=InferenceRewardConfig,
        ),
        TaskConfig(
            task=WebRetrievalTask,
            probability=0.35,
            datasets=[DDGDataset],
            reward_model=WebRetrievalRewardConfig,
        ),
    ]

    @classmethod
    def get_task_by_name(cls, task_name: str) -> BaseTextTask.__class__:
        if matching_tasks := [t.task for t in cls.task_configs if t.task.__name__ == task_name]:
            return matching_tasks[0]
        return None

    @classmethod
    def get_task_config(cls, task: BaseTextTask.__class__ | BaseTextTask) -> TaskConfig:
        task = task.__class__ if isinstance(task, BaseTextTask) else task
        try:
            return [t for t in cls.task_configs if task is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return

    @classmethod
    def random(cls) -> TaskConfig:
        probabilities = [task.probability for task in cls.task_configs]
        selected_task = random.choices(cls.task_configs, probabilities)[0]
        return selected_task

    @classmethod
    def get_task_datasets(cls, task: type[BaseTextTask] | BaseTextTask) -> list[type[BaseDataset]]:
        task_class = task.__class__ if isinstance(task, BaseTextTask) else task
        try:
            return [t.datasets for t in cls.task_configs if task_class is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return []

    @classmethod
    def get_random_task(cls) -> BaseTextTask:
        return cls.random().task()

    @classmethod
    def get_random_task_dataset(cls, task: type[BaseTextTask] | BaseTextTask) -> type[BaseDataset]:
        return random.choice(cls.get_task_datasets(task))

    @classmethod
    def get_task_reward(cls, task: BaseTextTask | type[BaseTextTask]) -> type[BaseRewardConfig]:
        task_class = task.__class__ if isinstance(task, BaseTextTask) else task
        try:
            return [t.reward_model for t in cls.task_configs if task_class is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return []

    @classmethod
    def create_random_task_with_dataset(cls) -> BaseTextTask:
        task_config = cls.random()
        dataset = cls.get_random_task_dataset(task_config.task)
        return task_config.task(dataset_entry=dataset().next())


assert (
    np.around(np.sum([conf.probability for conf in TaskRegistry.task_configs]), 5) == 1
), f"Task probabilities must sum to 1 but sum to {np.sum([conf.probability for conf in TaskRegistry.task_configs])}"
