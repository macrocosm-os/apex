from typing import Any

from loguru import logger

from apex.services.deep_research.deep_research_base import DeepResearchBase


async def generate_reference(
    llm: DeepResearchBase, query: str
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    """Generate a reference response for the given prompt.

    Args:
        llm: LLM provider object.
        query: The research question or topic to investigate.

    Returns:
        A tuple containing the model's response and tool history.
    """
    system_message: dict[str, str] = {
        "role": "system",
        "content": (
            "You are Deep Researcher, a meticulous assistant. For each claim you make, provide step-by-step reasoning "
            "and cite exact source numbers from the provided context."
        ),
    }
    user_message: dict[str, str] = {
        "role": "user",
        "content": (
            f"Research Question: {query}\n\n"
            "Please think through the answer carefully, annotate each step with citations like [1], [2], etc., "
            'and conclude with a "Sources:" list mapping each [n] to its source URL or title.'
        ),
    }

    response, tool_history, reasoning_traces = await llm.invoke([system_message, user_message])
    logger.debug(f"Generated reference.\nPrompt: '{user_message}'\nResponse: '{response}'")
    return response, tool_history, reasoning_traces
