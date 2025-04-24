import json
import re
import traceback
from functools import wraps

from loguru import logger


def parse_llm_json(json_str):
    """
    Parse JSON output from LLM that may contain code blocks, newlines and other formatting.
    Extracts JSON from code blocks if present.

    Args:
        json_str (str): The JSON string to parse

    Returns:
        dict: The parsed JSON object
    """
    # logger.info(f"Parsing JSON from: {json_str}")
    # First try to extract JSON from code blocks if they exist
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_block_matches = re.findall(code_block_pattern, json_str)

    if code_block_matches:
        # Use the first code block found
        json_str = code_block_matches[0]
        # logger.info(f"Extracted JSON from code block: {json_str}")
    else:
        raise json.JSONDecodeError("Expecting value", json_str, 0)

    # Replace escaped newlines with actual newlines
    json_str = json_str.replace("\\n", "\n")

    # Remove any redundant newlines/whitespace while preserving content
    json_str = " ".join(line.strip() for line in json_str.splitlines())

    # Parse the cleaned JSON string
    return json.loads(json_str)


def with_retries(max_retries: int = 3):
    """
    A decorator that retries a function on failure and logs attempts using loguru.

    Args:
        max_retries (int): Maximum number of retry attempts before giving up
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get the full stack trace
                    stack_trace = traceback.format_exc()
                    # If this is the last attempt, log as critical with full stack trace
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Function '{func.__name__}' failed on final attempt {attempt + 1}/{max_retries}. "
                            f"Error: {str(e)}\nStack trace:\n{stack_trace}"
                        )
                        raise  # Re-raise the exception after logging
                    # Otherwise log as error without stack trace
                    logger.error(
                        f"Function '{func.__name__}' failed on attempt {attempt + 1}/{max_retries}. "
                        f"Error: {str(e)}. Retrying..."
                    )
            return None  # In case all retries fail

        return wrapper

    return decorator


def convert_to_gemma_messages(messages):
    """Convert a list of messages to a list of gemma messages by alternating roles and adding empty messages."""
    gemma_messages = []
    for message in messages:
        if gemma_messages and gemma_messages[-1]["role"] == message["role"]:
            # Gemma requires alternating roles, so we need to add an empty message with the opposite role
            gemma_messages.append(
                {"type": "text", "content": "", "role": "assistant" if message["role"] == "user" else "user"}
            )
        gemma_messages.append({"type": "text", "role": message["role"], "content": message["content"]})
    return gemma_messages
