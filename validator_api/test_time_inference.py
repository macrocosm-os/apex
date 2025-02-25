import asyncio
import json
import random
import re
import time

from loguru import logger

from prompting.llms.apis.llm_messages import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
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
            # This regex replaces control characters (ASCII 0-31) except tabs, newlines and carriage returns
            json_str_clean = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str_clean)
            
            # Parse the JSON string into a dictionary
            obj = json.loads(json_str_clean)
            parsed_objects.append(obj)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON object: {e}")
            
            # Try a more aggressive approach if standard cleaning failed
            try:
                # Use a more aggressive approach to clean the JSON string
                # This will remove all non-ASCII characters and normalize whitespace
                clean_str = ''.join(c if ord(c) >= 32 or c in ['\n', '\r', '\t'] else ' ' for c in json_str)
                clean_str = re.sub(r'\s+', ' ', clean_str)  # Normalize whitespace
                
                # Try to parse again
                obj = json.loads(clean_str)
                parsed_objects.append(obj)
                logger.info("Successfully parsed JSON after aggressive cleaning")
            except json.JSONDecodeError:
                # If still failing, log and continue
                continue

    if len(parsed_objects) == 0:
        logger.error(
            f"No valid JSON objects found in the response - couldn't parse json. The miner response was: {api_response}"
        )
        return None
    
    # Check if we have the required fields
    # For final answers, the next_action field might be missing
    first_obj = parsed_objects[0]
    if not first_obj.get("title") or not first_obj.get("content"):
        logger.error(
            f"Invalid JSON object found in the response - required field missing. The miner response was: {api_response}"
        )
        return None
    
    # If this is a final answer (title contains "Final" or "Conclusion"), next_action is optional
    is_final_answer = "final" in first_obj.get("title", "").lower() or "conclusion" in first_obj.get("title", "").lower()
    if not is_final_answer and not first_obj.get("next_action"):
        logger.error(
            f"Invalid JSON object found in the response - next_action field missing for non-final answer. The miner response was: {api_response}"
        )
        return None
    
    # If next_action is missing for a final answer, add it with value "final_answer"
    if is_final_answer and not first_obj.get("next_action"):
        first_obj["next_action"] = "final_answer"
        
    return parsed_objects


async def make_api_call(
    messages, model=None, is_final_answer: bool = False, use_miners: bool = True, uids: list[int] = None
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
                    uids=uids,
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

            logger.debug(f"Making API call with\n\nMESSAGES: {messages}\n\nRESPONSE: {response_str}")
            parsed_json = parse_multiple_json(response_str)
            if parsed_json is None or len(parsed_json) == 0:
                logger.warning("parse_multiple_json returned None or empty list")
                return None
            
            response_dict = parsed_json[0]
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
            "title": "Error: Failed to Generate Final Answer",
            "content": f"Failed to generate final answer. Error: {error_msg}. This could be due to issues with parsing the model response or connectivity problems with the miners.",
            "next_action": "final_answer"  # Ensure this field is always present
        }
    else:
        return {
            "title": "Error: Failed to Generate Step",
            "content": f"Failed to generate step. Error: {error_msg}. This could be due to issues with parsing the model response or connectivity problems with the miners.",
            "next_action": "final_answer",
        }


async def generate_response(
    original_messages: list[dict[str, str]], model: str = None, uids: list[int] = None, use_miners: bool = True
):
    messages = [
        {
            "role": "system",
            "content": """You are a world-class expert in analytical reasoning and problem-solving. Your task is to break down complex problems through rigorous step-by-step analysis, carefully examining each aspect before moving forward. For each reasoning step:

OUTPUT FORMAT:
Return a JSON object with these required fields:
{
    "title": "Brief, descriptive title of current reasoning phase",
    "content": "Detailed explanation of your analysis",
    "next_action": "continue" or "final_answer"
}

REASONING PROCESS:
1. Initial Analysis
   - Break down the problem into core components
   - Identify key constraints and requirements
   - List relevant domain knowledge and principles

2. Multiple Perspectives
   - Examine the problem from at least 3 different angles
   - Consider both conventional and unconventional approaches
   - Identify potential biases in initial assumptions

3. Exploration & Validation
   - Test preliminary conclusions against edge cases
   - Apply domain-specific best practices
   - Quantify confidence levels when possible (e.g., 90% certain)
   - Document key uncertainties or limitations

4. Critical Review
   - Actively seek counterarguments to your reasoning
   - Identify potential failure modes
   - Consider alternative interpretations of the data/requirements
   - Validate assumptions against provided context

5. Synthesis & Refinement
   - Combine insights from multiple approaches
   - Strengthen weak points in the reasoning chain
   - Address identified edge cases and limitations
   - Build towards a comprehensive solution

REQUIREMENTS:
- Each step must focus on ONE specific aspect of reasoning
- Explicitly state confidence levels and uncertainty
- When evaluating options, use concrete criteria
- Include specific examples or scenarios when relevant
- Acknowledge limitations in your knowledge or capabilities
- Maintain logical consistency across steps
- Build on previous steps while avoiding redundancy

CRITICAL THINKING CHECKLIST:
✓ Have I considered non-obvious interpretations?
✓ Are my assumptions clearly stated and justified?
✓ Have I identified potential failure modes?
✓ Is my confidence level appropriate given the evidence?
✓ Have I adequately addressed counterarguments?

Remember: Quality of reasoning is more important than speed. Take the necessary steps to build a solid analytical foundation before moving to conclusions.

Example:

User Query: How many piano tuners are in New York City?

Expected Answer:
{
    "title": "Estimating the Number of Piano Tuners in New York City",
    "content": "To estimate the number of piano tuners in NYC, we need to break down the problem into core components. Key factors include the total population of NYC, the number of households with pianos, the average number of pianos per household, and the frequency of piano tuning. We should also consider the number of professional piano tuners and their workload.",
    "next_action": "continue"
}
""",
        }
    ]
    messages += original_messages
    messages += [
        {
            "role": "assistant",
            "content": "I understand. I will now analyze the problem systematically, following the structured reasoning process while maintaining high standards of analytical rigor and self-criticism.",
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

    final_answer_prompt = """Based on your thorough analysis, please provide your final answer. Your response should:

        1. Clearly state your conclusion
        2. Summarize the key supporting evidence
        3. Acknowledge any remaining uncertainties
        4. Include relevant caveats or limitations
        5. Synthesis & Refinement

        Ensure your response uses the correct json format as follows:
        {
            "title": "Final Answer",
            "content": "Conclusion and detailed explanation of your answer",
        }"""

    messages.append(
        {
            "role": "user",
            "content": final_answer_prompt,
        }
    )

    start_time = time.time()
    final_data = await make_api_call(
        messages, model=model, is_final_answer=True, use_miners=use_miners, uids=uids
    )

    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if final_data["title"] == "Error":
        steps.append(("Error", final_data["content"], thinking_time))
        raise ValueError(f"Failed to generate final answer: {final_data['content']}")

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time
