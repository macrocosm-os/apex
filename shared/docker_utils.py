import requests
from shared import constants
from shared.logging import logger

async def get_generation(
    messages: list[str] | list[dict],
    roles: list[str] | None = None,
    model: str | None = None,
    seed: int = None,
    sampling_params: dict[str, float] = None,
) -> str:
    if messages and isinstance(messages[0], dict):
        dict_messages = messages
    else:
        dict_messages = [
            {"content": message, "role": role} for message, role in zip(messages, roles or ["user"] * len(messages))
        ]
    url = f"{constants.DOCKER_BASE_URL}/v1/chat/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"messages": dict_messages, "seed": seed, "sampling_params": sampling_params}
    response = requests.post(url, headers=headers, json=payload)
    try:
        json_response = response.json()
        logger.info(f"Response: {json_response}")
        return json_response["choices"][0]["message"]["content"]
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Error generating response. Status: {response.status_code}, Body: {response.text}")
        return ""