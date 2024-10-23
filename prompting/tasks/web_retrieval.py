import json
import textwrap
from typing import ClassVar, Optional

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.rewards.web_retrieval import WebRetrievalRewardModel
from prompting.tasks.base_task import BaseTextTask

# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = textwrap.dedent(
"""Ask a question about the following text in such a way that it's not obvious 
that you're asking about text from this specific website, but keep the context to make sure that the 
question can be answered through the internet search.
"""
)


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        WebRetrievalRewardModel(weight=1.0),
    ]


class WebRetrievalTask(BaseTextTask):
    name: ClassVar[str] = "web_retrieval"
    augmentation_system_prompt: ClassVar[str] = ""
    query_system_prompt: ClassVar[Optional[str]] = QUERY_SYSTEM_PROMPT

    def make_query(self, dataset_entry: DDGDatasetEntry) -> str:
        self.query = self.generate_query(messages=dataset_entry.website_content)
        return self.query

    def make_reference(self, dataset_entry: DDGDatasetEntry) -> str:
        self.reference = json.dumps(dataset_entry.model_dump_json())
        return self.reference
