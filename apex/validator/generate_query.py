import random

from loguru import logger

from apex.common.constants import get_english_words
from apex.services.llm.llm_base import LLMBase
from apex.services.websearch.websearch_base import WebSearchBase

QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context.

[Context]
{context}

Remember the question must encourage logical thinking and reasoning and must be spread over at least 3 sentences.
Do not mention the context explicitly."""


async def generate_query(llm: LLMBase, websearch: WebSearchBase) -> str:
    random_words = " ".join(random.sample(get_english_words(), 3))
    # Perform a lightweight search and pick a single result as context.
    try:
        search_results = await websearch.search(random_words, max_results=5)
        search_website = random.choice(search_results)
        search_content = search_website.content
    except BaseException as exc:
        logger.error(f"Error during web search: {exc}")
        search_content = ""
    query = QUERY_PROMPT_TEMPLATE.format(context=search_content)
    query_response, _ = await llm.invoke([{"role": "user", "content": query}])
    logger.debug(f"Generated query.\nPrompt: '{query}'\nResponse: '{query_response}'")
    return query_response
