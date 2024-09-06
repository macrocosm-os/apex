from prompting.tasks.base_task import BaseTextTask
from prompting.rewards.reward import BaseRewardConfig
from prompting.datasets.lmsys import LMSysDataset
from prompting.datasets.base import BaseDataset
from prompting.tasks.inference import SyntheticInferenceTask, InferenceRewardConfig
from pydantic import BaseModel, ConfigDict
import random
from typing import ClassVar
from loguru import logger
import numpy as np

from prompting.tasks.programming_task import ProgrammingTask, ProgrammingRewardConfig
from prompting.datasets.huggingface_github import HuggingFaceGithubDataset


class TaskConfig(BaseModel):
    task: BaseTextTask.__class__
    probability: float
    datasets: list[BaseDataset.__class__]
    reward_model: BaseRewardConfig.__class__

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskRegistry(BaseModel):
    task_configs: ClassVar[list[TaskConfig]] = [
        TaskConfig(task=QuestionAnsweringTask, probability=0.1, datasets=[WikiDataset], reward_model=QARewardConfig),
        TaskConfig(
            task=SummarizationTask, probability=0.1, datasets=[WikiDataset], reward_model=SummarizationRewardConfig
        ),
        TaskConfig(
            task=DateQuestionAnsweringTask,
            probability=0.1,
            datasets=[WikiDateDataset],
            reward_model=DateQARewardConfig,
        ),
        TaskConfig(
            task=SyntheticInferenceTask,
            probability=0.2,
            datasets=[LMSysDataset],
            reward_model=InferenceRewardConfig,
        ),
        TaskConfig(
            task=MultiChoiceTask,
            probability=0.2,
            datasets=[WikiDataset],
            reward_model=MultiChoiceRewardConfig,
        ),
        TaskConfig(
            task=ProgrammingTask,
            probability=0.1,
            datasets=[HuggingFaceGithubDataset],
            reward_model=ProgrammingRewardConfig,
        ),
        TaskConfig(task=MultiChoiceTask, probability=0.2, datasets=[WikiDataset], reward_model=MultiChoiceRewardConfig),
    ]

    @classmethod
    def random(cls) -> TaskConfig:
        probabilities = [task.probability for task in cls.task_configs]
        selected_task = random.choices(cls.task_configs, probabilities)[0]
        return selected_task

    @classmethod
    def get_task_datasets(cls, task: BaseTextTask.__class__ | BaseTextTask) -> list[BaseDataset.__class__]:
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
    def get_random_task_dataset(cls, task: BaseTextTask.__class__ | BaseTextTask) -> BaseDataset.__class__:
        return random.choice(cls.get_task_datasets(task))

    @classmethod
    def get_task_reward(cls, task: BaseTextTask | BaseTextTask.__class__) -> BaseRewardConfig.__class__:
        task_class = task.__class__ if isinstance(task, BaseTextTask) else task
        try:
            return [t.reward_model for t in cls.task_configs if task_class is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return []

    @classmethod
    def create_random_task_with_dataset(cls) -> tuple[BaseTextTask, BaseDataset]:
        task_config = cls.random()
        dataset = cls.get_random_task_dataset(task_config.task)
        return task_config.task(), dataset()


assert (
    np.around(np.sum([conf.probability for conf in TaskRegistry.task_configs]), 5) == 1
), f"Task probabilities must sum to 1 but sum to {np.sum([conf.probability for conf in TaskRegistry.task_configs]) }"
