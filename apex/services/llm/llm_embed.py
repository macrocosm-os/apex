import aiohttp


class LLMEmbed:
    def __init__(self, base_url: str, key: str):
        self._base_url = base_url
        self._key = key

    async def invoke(self, inputs: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": "Bearer " + self._key,
            "Content-Type": "application/json",
        }

        body = {"inputs": inputs}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base_url}", headers=headers, json=body) as response:
                response.raise_for_status()
                embeddings = await response.json()
                return embeddings  # type: ignore[no-any-return]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._base_url})"
