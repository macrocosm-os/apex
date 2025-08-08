from typing import Any

from apex.services.llm.llm_base import LLMBase


class DeepResearchBase(LLMBase):
    async def invoke(
        self, messages: list[dict[str, str]], body: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, str]]]:
        raise NotImplementedError
