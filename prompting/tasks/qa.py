from typing import ClassVar

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.rewards.rouge import RougeRewardModel
from prompting.tasks.base_task import BaseTextTask
from shared.base import Context  # type: ignore # noqa: F401

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

REFERENCE_SYSTEM_PROMPT = """\
You are an expert question-answering LLM. You will receive context and a question, and you will generate a detailed and accurate answer to the question. Your answer should be based on the context provided.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}

You must ask a question that can be answered by the context.
"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context.

#Context:
{context}

# Question:
{question}
"""


class QARewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        RougeRewardModel(weight=0.5),
        RelevanceRewardModel(weight=0.5),
    ]
    penalty_definition: ClassVar[list[BaseRewardModel]] = [RougeRewardModel(weight=0.5)]


class WebQuestionAnsweringTask(BaseTextTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    name: ClassVar[str] = "web_qa"
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None

    async def make_query(self, dataset_entry: DDGDatasetEntry):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=dataset_entry.website_content)
        self.query = await self.generate_query(messages=[query_prompt])
        return self.query

    async def make_reference(self, dataset_entry: DDGDatasetEntry):
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=dataset_entry.website_content, question=self.query)
        self.reference = await self.generate_reference(messages=[{"role": "user", "content": reference_prompt}])
        return self.reference
