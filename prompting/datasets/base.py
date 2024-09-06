from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel
from typing import ClassVar
from prompting.utils.timer import Timer
import json
from pydantic import model_validator


class DatasetEntry(BaseModel):
    @property
    def hash(self) -> int:
        return hash(json.dumps(self.model_dump(), sort_keys=True))

    def __hash__(self) -> int:
        return self.hash


class ChatEntry(DatasetEntry):
    messages: list[str]
    roles: list[str]
    organic: bool
    source: str
    query: str | None = None

    @model_validator(mode="after")
    def check_query(self) -> "ChatEntry":
        if self.query is None:
            self.query = self.messages[-1]
        return self


class MMLUEntry(DatasetEntry):
    query: str
    subject: str
    choices: list[str]
    answer: str


class Context(DatasetEntry):
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: list[str]
    external_links: list[str]
    source: str
    tags: list[str] | None = None
    extra: dict | None = None  # additional non-essential information
    stats: dict | None = None  # retrieval stats such as fetch time, number of tries, etc.


class BaseDataset(ABC, BaseModel):
    """Base class for datasets."""

    name: ClassVar[str] = "base"
    max_tries: int = 10

    @abstractmethod
    def random(self) -> Context: ...

    @abstractmethod
    def get(self) -> Context: ...

    def next(self, method: Literal["random", "search", "get"] = "random", **kwargs) -> dict:
        tries = 1
        context: Context  # for some reason the ls doesn't understand it's of type Context without this

        with Timer() as timer:
            while True:
                # TODO: Multithread the get method so that we don't have to suffer nonexistent pages
                if method == "random":
                    context = self.random(**kwargs)
                elif method == "search":
                    context = self.search(**kwargs)
                elif method == "get":
                    context = self.get(**kwargs)

                if context:
                    break

        context.source = self.__class__.__name__
        context.stats = {
            "fetch_time": timer.elapsed_time,
            "num_tries": tries,
            "fetch_method": method,
            "next_kwargs": kwargs,
        }
        return context
