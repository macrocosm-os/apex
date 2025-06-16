import random
from typing import ClassVar

import datasets
from pydantic import model_validator

from shared.base import BaseDataset, ChatEntry


class SN13Dataset(BaseDataset):
    _url: ClassVar[str] = "arrmlet/x_dataset_218"
    name: ClassVar[str] = "x_dataset_218"
    exception: Exception | None = None
    dataset: datasets.Dataset | None = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def load_dataset(self) -> "SN13Dataset":
        self.dataset = datasets.load_dataset(self._url)["train"]
        self

    def get(self) -> ChatEntry:
        return self.sample()

    def next(self) -> ChatEntry:
        return self.sample()

    def random(self) -> ChatEntry:
        return self.sample()

    def sample(self) -> ChatEntry:
        """Sample the data, raises an exception if logging into HuggingFace was unsuccessful."""
        if self.exception is not None:
            raise self.exception
        # Randomly select a sample from the dataset.
        messages = []
        for i in range(4):
            sample_idx = random.randint(0, len(self.dataset) - 1)
            if message := self.dataset[sample_idx]["text"]:
                if i % 2 == 0:
                    messages.append({"role": "user", "content": message})
                else:
                    messages.append({"role": "assistant", "content": message})

        return ChatEntry(messages=messages, organic=False, source=self._url)
