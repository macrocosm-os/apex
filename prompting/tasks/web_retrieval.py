import json
import random
import textwrap
from typing import ClassVar, Optional

from pydantic import Field

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.rewards.web_retrieval import WebRetrievalRewardModel
from prompting.tasks.base_task import BaseTextTask

# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = textwrap.dedent(
    """You are a tool used to train users research skills.
You will ask questions about websites in such a way
that users are able to retrieve the content. Your tone should be causal,
in the same way that a human would be asking.
"""
)

MESSAGE_TEMPLATE = """Ask a question about the following text in such a way that it's not obvious
that you're asking about text from this specific website, but keep the context to make sure that the
question can be answered through an internet search.

WEBSITE CONTENT:

{website_content}"""


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        WebRetrievalRewardModel(weight=1.0),
    ]


class WebRetrievalTask(BaseTextTask):
    name: ClassVar[str] = "web_retrieval"
    augmentation_system_prompt: ClassVar[str] = ""
    query_system_prompt: ClassVar[Optional[str]] = QUERY_SYSTEM_PROMPT
    target_results: int = Field(default_factory=lambda: random.randint(1, 10))
    timeout: int = Field(default_factory=lambda: random.randint(5, 15))

    async def make_query(self, dataset_entry: DDGDatasetEntry) -> str:
        self.query = await self.generate_query(
            messages=[MESSAGE_TEMPLATE.format(website_content=dataset_entry.website_content)]
        )
        return self.query

    async def make_reference(self, dataset_entry: DDGDatasetEntry) -> str:
        dataset_entry.query = self.query
        ref_dict = dataset_entry.model_dump_json()
        self.reference = json.dumps(ref_dict)
        return self.reference

    @property
    def request_body(self) -> dict:
        body = super().request_body
        body["target_results"] = self.target_results
        return body
