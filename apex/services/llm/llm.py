from typing import Any

import aiohttp

from apex.common import constants
from apex.services.llm.llm_base import LLMBase


class LLM(LLMBase):
    def __init__(self, base_url: str, model: str, key: str):
        self._base_url = base_url
        self._model = model
        self._key = key

    async def invoke(
        self, messages: list[dict[str, str]], body: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
        headers = {
            "Authorization": "Bearer " + self._key,
            "Content-Type": "application/json",
        }

        if body is None:
            body = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "max_tokens": constants.MAX_TOKENS,
                "temperature": constants.TEMPERATURE,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base_url}/chat/completions", headers=headers, json=body) as response:
                response.raise_for_status()

                data = await response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                # This base LLM does not build multi-step chains; return empty reasoning_traces
                return str(content), [], []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._base_url}, {self._model})"
