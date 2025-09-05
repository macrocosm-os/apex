from typing import Any


class LLMBase:
    async def invoke(
        self, messages: list[dict[str, str]], body: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
        raise NotImplementedError
