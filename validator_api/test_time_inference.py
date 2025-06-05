import asyncio
import json
import random
import re
import time

from loguru import logger

from prompting.llms.apis.llm_messages import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
from shared.prompts.test_time_inference import final_answer_prompt, intro_prompt, system_acceptance_prompt
from shared.timer import Timer
from validator_api.chat_completion import chat_completion

MAX_THINKING_STEPS = 10
ATTEMPTS_PER_STEP = 10


def parse_multiple_json(api_response):
    """
    Parses a string containing multiple JSON objects and returns a list of dictionaries.

    Args:
        api_response (str): The string returned by the API containing JSON objects.

    Returns:
        list: A list of dictionaries parsed from the JSON objects.
    """
    # Regular expression pattern to match individual JSON objects
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

    # Find all JSON object strings in the response
    json_strings = json_pattern.findall(api_response)

    parsed_objects = []
    for json_str in json_strings:
        try:
            # Replace escaped single quotes with actual single quotes
            json_str_clean = json_str.replace("\\'", "'")

            # Remove or replace invalid control characters
            json_str_clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", json_str_clean)

            # Parse the JSON string into a dictionary
            obj = json.loads(json_str_clean)
            parsed_objects.append(obj)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON object: {e}")

            # Try a more aggressive approach if standard cleaning failed
            try:
                clean_str = "".join(c if ord(c) >= 32 or c in ["\n", "\r", "\t"] else " " for c in json_str)
                clean_str = re.sub(r"\s+", " ", clean_str)  # Normalize whitespace

                # Try to parse again
                obj = json.loads(clean_str)
                parsed_objects.append(obj)
                logger.info("Successfully parsed JSON after aggressive cleaning")
            except json.JSONDecodeError:
                # If still failing, log and continue
                continue

    if len(parsed_objects) == 0:
        logger.debug(
            f"No valid JSON objects found in the response - couldn't parse json. The miner response was: {api_response}"
        )
        return None
    if (
        not parsed_objects[0].get("title")
        or not parsed_objects[0].get("content")
        or not parsed_objects[0].get("next_action")
    ):
        logger.debug(
            f"Invalid JSON object found in the response - field missing. The miner response was: {api_response}"
        )
        return None

    return parsed_objects


async def make_api_call(
    messages, model=None, is_final_answer: bool = False, use_miners: bool = True, uids: list[int] | None = None
):
    async def single_attempt():
        try:
            if use_miners:
                response = await chat_completion(
                    body={
                        "messages": messages,
                        "model": model,
                        "task": "InferenceTask",
                        "test_time_inference": True,
                        "mixture": False,
                        "sampling_parameters": {
                            "temperature": 0.5,
                            "max_new_tokens": 500,
                        },
                        "seed": random.randint(0, 1000000),
                    },
                    num_miners=3,
                    top_uids=uids,
                )
                response_str = response.choices[0].message.content
            else:
                logger.debug("Using SN19 API for inference in MSR")
                response_str = LLMWrapper.chat_complete(
                    messages=LLMMessages(*[LLMMessage(role=m["role"], content=m["content"]) for m in messages]),
                    model=model,
                    temperature=0.5,
                    max_tokens=2000,
                )

            response_dict = parse_multiple_json(response_str)[0]
            return response_dict
        except Exception as e:
            logger.warning(f"Failed to get valid response: {e}")
            return None

    # When not using miners, let's try and save tokens
    if not use_miners:
        for _ in range(ATTEMPTS_PER_STEP):
            try:
                result = await single_attempt()
                if result is not None:
                    return result
                else:
                    logger.error("Failed to get valid response")
                    continue
            except Exception as e:
                logger.error(f"Failed to get valid response: {e}")
                continue
    else:
        # when using miners, we try and save time
        tasks = [asyncio.create_task(single_attempt()) for _ in range(ATTEMPTS_PER_STEP)]

        # As each task completes, check if it was successful
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                if result is not None:
                    # Cancel remaining tasks
                    for task in tasks:
                        task.cancel()
                    return result
            except Exception as e:
                logger.error(f"Task failed with error: {e}")
                continue

    # If all tasks failed, return error response
    error_msg = "All concurrent API calls failed"
    logger.error(error_msg)
    if is_final_answer:
        return {
            "title": "Error",
            "content": f"Failed to generate final answer. Error: {error_msg}",
        }
    else:
        return {
            "title": "Error",
            "content": f"Failed to generate step. Error: {error_msg}",
            "next_action": "final_answer",
        }


async def generate_response(
    original_messages: list[dict[str, str]], model: str = None, uids: list[int] | None = None, use_miners: bool = True
):
    messages = [
        {
            "role": "system",
            "content": intro_prompt(),
        }
    ]
    messages += original_messages
    messages += [
        {
            "role": "assistant",
            "content": system_acceptance_prompt(),
        }
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    for _ in range(MAX_THINKING_STEPS):
        with Timer() as timer:
            step_data = await make_api_call(messages, model=model, use_miners=use_miners, uids=uids)
        thinking_time = timer.final_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data["content"], thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" or not step_data.get("next_action"):
            break

        step_count += 1
        yield steps, None

    messages.append(
        {
            "role": "user",
            "content": final_answer_prompt(),
        }
    )

    start_time = time.time()
    final_data = await make_api_call(messages, model=model, is_final_answer=True, use_miners=use_miners, uids=uids)

    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if final_data["title"] == "Error":
        steps.append(("Error", final_data["content"], thinking_time))
        raise ValueError(f"Failed to generate final answer: {final_data['content']}")

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time
