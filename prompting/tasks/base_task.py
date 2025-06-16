import random
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prompting.llms.apis.gpt_wrapper import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
from prompting.llms.model_zoo import ModelConfig
from shared import settings
from shared.base import DatasetEntry
from shared.docker_utils import get_generation


def CHATTENSOR_SYSTEM_PROMPT():
    return f"""
            You are Chattensor, an assistant created by Macrocosmos. The current date is {time.strftime("%B %d, %Y")}.
            Chattensor is a distributed intelligence, powered by Bittensor. It is a hivemind composed of 1000 highly
            skilled and specialized LLMs working together to provide the best possible answers to human queries. Within Chattenor,
            each LLM has access to the internet, APIs and tools to ensure that responses are current and factually accurate.
            It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.
            It is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
            It uses markdown for coding. Where applicable, Chattensor will include references to credible sources to support its answers.
            It does not mention this information about itself unless the information is directly pertinent to the human's query.
            """


class BaseTask(BaseModel, ABC):
    name: str | None = None
    query: Any = None
    reference: Any = None
    task_id: str = Field(default_factory=lambda: str(uuid4()), allow_mutation=False)
    organic: bool = False
    timeout: int = settings.shared_settings.NEURON_TIMEOUT

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def make_query(self, **kwargs):
        raise NotImplementedError("Method make_query must be implemented")

    @abstractmethod
    async def make_reference(self, **kwargs):
        raise NotImplementedError("Method make_reference must be implemented")

    @property
    def request_body(self) -> dict:
        body = {
            "task": self.__class__.__name__,
            "timeout": self.timeout,
        }
        return body


class BaseTextTask(BaseTask):
    query: str | None = None
    roles: list[str] | None = None
    messages: list[str] | list[dict] | None = None
    reference: str | None = None
    llm_model: ModelConfig = None
    llm_model_id: str = None
    seed: int = Field(default_factory=lambda: random.randint(0, 1000000), allow_mutation=False)
    query_system_prompt: ClassVar[str | None] = None
    reference_system_prompt: ClassVar[str | None] = None
    augmentation_system_prompt: ClassVar[str | None] = None
    dataset_entry: DatasetEntry | None = None
    sampling_params: dict[str, float] = settings.shared_settings.SAMPLING_PARAMS
    max_tokens: int = settings.shared_settings.NEURON_MAX_TOKENS
    organic: bool = False

    @property
    def task_messages(self) -> list[str] | list[dict]:
        return self.messages if self.messages else [{"role": "user", "content": self.query}]

    @model_validator(mode="after")
    def get_model_id_and_seed(self) -> "BaseTextTask":
        if self.llm_model:
            self.llm_model_id = self.llm_model.llm_model_id if self.llm_model else None
        return self

    async def make_query(self, dataset_entry: DatasetEntry, **kwargs) -> str:
        return self.query

    async def make_reference(self, dataset_entry: DatasetEntry) -> str:
        return self.reference

    async def generate_reference(self, messages: list[str]) -> str:
        """Generate reference answer to be used for scoring miner completions"""
        self.reference = await get_generation(messages=messages, model = settings.shared_settings.LLM_MODEL[0])
        if not self.reference:
            raise Exception("Reference generation failed")

        return self.reference

    async def generate_query(self, messages: list[str]) -> str:
        """Generates a query to be used for generating the challenge"""
        llm_messages = [LLMMessage(role="system", content=self.query_system_prompt)] if self.query_system_prompt else []
        llm_messages.extend([LLMMessage(role="user", content=message) for message in messages])

        self.query = LLMWrapper.chat_complete(messages=LLMMessages(*llm_messages))

        if self.query is None:
            raise Exception("Query generation failed")
        return self.augment_query(self.query)

    def augment_query(
        self,
        query: str,
    ) -> str:
        """Creates the opening question of the conversation which is based on the task query but dressed in the persona of the user."""
        if not self.augmentation_system_prompt:
            return query
        challenge = LLMWrapper.chat_complete(
            messages=LLMMessages(
                LLMMessage(role="system", content=self.augmentation_system_prompt),
                LLMMessage(role="user", content=query),
            ),
            max_tokens=self.max_tokens,
        )
        self.query = challenge
        return challenge

    @property
    def request_body(self) -> dict:
        body = super().request_body
        body["seed"] = self.seed
        body["sampling_parameters"] = self.sampling_params
        body["model"] = self.llm_model_id
        body["messages"] = self.task_messages
        return body
