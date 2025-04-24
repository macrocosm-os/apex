import time
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from prompting.llms.model_manager import ModelManager
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent

RewardTypeLiteral = Literal["reward", "penalty"]


class WeightedRewardEvent(BaseModel):
    weight: float
    task: BaseTextTask
    reward_model_name: str
    rewards: list[float]
    rewards_normalized: list[float]
    timings: list[float]
    reward_model_type: RewardTypeLiteral
    batch_time: float
    uids: list[float]

    threshold: float | None = None
    extra_info: dict | None = None
    reward_type: Literal["reward", "penalty"] = "reward"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.reward_model_name}_raw_{self.reward_model_type.value}": self.rewards,
            f"{self.reward_model_name}_{self.reward_model_type.value}": self.rewards_normalized,
            f"{self.reward_model_name}_{self.reward_model_type.value}_timings": self.timings,
            f"{self.reward_model_name}_{self.reward_model_type.value}_batch_time": self.batch_time,
            f"{self.reward_model_name}_{self.reward_model_type.value}_threshold": self.threshold,
            f"{self.reward_model_name}_{self.reward_model_type.value}_extra_info": self.extra_info,
            f"{self.reward_model_name}_{self.reward_model_type.value}_uids": self.uids,
            f"{self.reward_model_name}_{self.reward_model_type.value}_task": self.task,
            f"{self.reward_model_name}_{self.reward_model_type.value}_weight": self.weight,
        }


class BatchRewardOutput(BaseModel):
    rewards: np.ndarray
    timings: np.ndarray
    threshold: float | None = None
    extra_info: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    uids: list[int] | None = None

    @model_validator(mode="after")
    def validate_rewards_and_timings(cls, v):
        if v.rewards.shape != v.timings.shape:
            raise ValueError(f"rewards.shape {v.rewards.shape} != timings.shape {v.timings.shape}")
        return v

    @property
    def rewards_normalized(self) -> np.ndarray:
        if self.rewards.size == 0:
            return np.array([])
        if self.rewards.min() == self.rewards.max():
            return np.array([1 / len(self.rewards)] * len(self.rewards))
        return (self.rewards - self.rewards.min()) / (self.rewards.max() - self.rewards.min())


class BaseRewardModel(ABC, BaseModel):
    weight: float = 1.0

    @abstractmethod
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        model_manager: ModelManager = None,
        task_queue: list[BaseTextTask] | None = None,
        **kwargs,
    ) -> BatchRewardOutput:
        raise NotImplementedError("You must implement the reward method")

    async def apply(
        self,
        response_event: DendriteResponseEvent,
        reference: str | None = None,
        challenge: str | None = None,
        reward_type: Literal["reward", "penalty"] = "reward",
        task: BaseTextTask | None = None,
        model_manager: ModelManager | None = None,
        task_queue: list[BaseTextTask] | None = None,
        **kwargs,
    ) -> WeightedRewardEvent:
        if task_queue is None:
            raise ValueError("Task queue must be provided to BaseRewardModel.apply()")
        t0 = time.time()
        comparator = reference if reward_type == "reward" else challenge
        batch_rewards_output: BatchRewardOutput = await self.reward(
            comparator, response_event, task=task, model_manager=model_manager, task_queue=task_queue, **kwargs
        )
        batch_rewards_time = time.time() - t0
        uids = batch_rewards_output.uids if batch_rewards_output.uids is not None else response_event.uids

        return WeightedRewardEvent(
            weight=self.weight,
            task=task,
            reward_model_name=self.__class__.__name__,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            reward_model_type=reward_type,
            batch_time=batch_rewards_time,
            threshold=batch_rewards_output.threshold,
            timings=batch_rewards_output.timings,
            extra_info=kwargs,
            uids=uids,
        )


class WeightedRewardModel(BaseModel):
    weight: float
    reward_model: BaseRewardModel


class BaseRewardConfig(ABC, BaseModel):
    """This class takes in a dictionary of rewards and penalties that should be applied. On apply(),
    it then applies all the reward models based on query & reference and returns the reward.

    both reward_definition and penalty_definition must be a list of tuples of type:

    weighting: RewardModel, e.g.

    [ (0.2, RougeRewardModel), (0.8, CosineDistanceRewardModel) ]

    Note that for all the rewards, the percentages must sum up to 1 (100%). For penalties,
    this is not the case, e.g. you may want to only apply a single penalty very lightly
    and weight it with <1.
    """

    reward_definitions: ClassVar[list[BaseRewardModel]]
    penalty_definitions: ClassVar[list[BaseRewardModel]] = []

    @classmethod
    def sum_rewards(cls, reward_events: list[WeightedRewardEvent]) -> np.ndarray:
        if not reward_events:
            return 0
        return np.sum([r.rewards * r.weight for r in reward_events], axis=0)

    @classmethod
    def final_rewards(cls, reward_events: list[WeightedRewardEvent]) -> list[float]:
        reward_events = [r for r in reward_events if r.reward_type == "reward"]
        return cls.sum_rewards(reward_events)

    @classmethod
    async def apply(
        cls,
        response_event: DendriteResponseEvent,
        reference: str,
        challenge: str | None = None,
        model_id: str | None = None,
        task: BaseTextTask | None = None,
        model_manager: ModelManager | None = None,
        task_queue: list[BaseTextTask] | None = None,
    ) -> list[WeightedRewardEvent]:
        if task_queue is None:
            raise ValueError("Task queue must be provided to BaseRewardConfig.apply()")
        reward_events = []
        for weighted_reward in cls.reward_definitions:
            reward_events.append(
                await weighted_reward.apply(
                    reference=reference,
                    response_event=response_event,
                    challenge=challenge,
                    reward_type="reward",
                    model_id=model_id,
                    task=task,
                    model_manager=model_manager,
                    task_queue=task_queue,
                ),
            )
        return reward_events
