import random
from typing import ClassVar

from loguru import logger

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.tasks.qa import WebQuestionAnsweringTask
from shared.base import Context
from validator_api.test_time_inference import generate_response

MAX_THINKING_STEPS = 10


def execute_multi_step_reasoning(user_query: str):
    for steps, total_thinking_time in generate_response(user_query):
        if total_thinking_time is not None:
            logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
    return steps, total_thinking_time


class MultiStepReasoningRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        RelevanceRewardModel(weight=1),
    ]


# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a master of crafting intellectually stimulating questions that unfold across multiple sentences. Each question you generate should be structured as a brief narrative or scenario, where crucial information is deliberately distributed across multiple sentences. The complete question can only be understood and answered by carefully considering all the information provided across these sentences.

Your questions should:
1. Begin with context or background information
2. Introduce key variables or constraints in subsequent sentences
3. Present the actual question in the final sentence
4. Require analytical reasoning rather than mere fact recall
5. Draw from the provided context when available
6. Incorporate multiple related concepts or data points

EXAMPLE FORMATS:
✓ "The International Space Station orbits at an average height of 400km above Earth. At this height, it completes one orbit every 92 minutes. Assuming constant speed, how many kilometers does the ISS travel in one Earth day?"

✓ "A new streaming service launches with 500,000 subscribers in January. They observe that they lose 5% of their existing subscribers each month, but also gain 50,000 new subscribers in the same period. Their infrastructure costs increase by $100,000 for every 200,000 subscribers. What will their monthly infrastructure costs be after 6 months?"

✓ "The average American household generates 4.5 pounds of trash daily. Local recycling programs typically reduce landfill waste by 30%. Your city has just implemented a new composting initiative that diverts an additional 25% of waste from landfills. Considering there are 50,000 households in your city, how many pounds of waste would still reach landfills each week?"

AVOID:
- Single-sentence questions
- Questions answerable with simple facts
- Questions without context or background
- Obvious or straightforward calculations
- Questions that don't require analysis

Remember: The goal is to create questions where the context and parameters are revealed progressively, requiring the reader to integrate information across multiple sentences to fully understand and solve the problem. Make sure that the question is spread over at least 3 sentences.
"""

QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}

Remember the question must encourage logical thinking and reasoning and must be spread over at least 3 sentences.
"""

SAMPLE_SYSTEM_PROMPTS = [
    """You are an LLM specialising in reasoning and solving complex questions. You will be given a chat interaction with a user and must answer appropriately.""",
    """You are a step-by-step problem solver. When given a complex question, you break it down into clear logical steps, showing your work and explaining your reasoning at each stage. You maintain a methodical approach to ensure accuracy.""",
    """You are an expert at mathematical and analytical reasoning. You excel at carefully parsing multi-part problems, identifying key information, and systematically working through solutions while clearly documenting your thought process.""",
]


class MultiStepReasoningTask(WebQuestionAnsweringTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    name: ClassVar[str] = "multi_step_reasoning"
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    query_system_prompt: str = QUERY_SYSTEM_PROMPT
    reference: str | None = None

    async def make_query(self, dataset_entry: DDGDatasetEntry):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=dataset_entry.website_content)
        question = await self.generate_query(messages=[query_prompt])
        self.messages = [{"role": "system", "content": random.choice(SAMPLE_SYSTEM_PROMPTS)}] + [
            {"role": "user", "content": question}
        ]
        return self.query

    async def _async_generate_reference(self):
        async for steps, total_thinking_time in generate_response(
            self.messages, model=self.llm_model_id, use_miners=False
        ):
            logger.debug(f"Step generated in reference of MSR: {steps}")
            if total_thinking_time is not None:
                logger.debug(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
        return steps[-1][1]

    async def make_reference(self, dataset_entry: Context):
        try:
            logger.debug(f"Generating reference for MSR: {self.messages}")
            # Run the async function in a new event loop
            self.reference = await self._async_generate_reference()
            logger.debug(f"Generated reference for MSR: {self.reference}")
        except Exception as e:
            logger.error(f"Error getting final answer for MSR: {e}")
            self.reference = None
        if self.reference is None:
            logger.error("No reference found for MSR")
            return None
        return self.reference
