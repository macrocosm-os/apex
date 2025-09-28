from loguru import logger
from tavily import TavilyClient

from apex.services.websearch.websearch_base import WebSearchBase, Website


class WebSearchTavily(WebSearchBase):
    QUERY_LIMIT = 399

    def __init__(self, key: str):
        self.client = TavilyClient(key)

    async def search(self, query: str, max_results: int = 5) -> list[Website]:
        websites: list[Website] = []
        try:
            response = self.client.search(
                query=query[: self.QUERY_LIMIT],
                max_results=max_results,
            )
            response_time = response.get("response_time")
            for result in response.get("results", []):
                website = Website(
                    query=query,
                    url=result.get("url"),
                    content=result.get("content"),
                    title=result.get("title"),
                    score=result.get("score"),
                    response_time=response_time,
                )
                websites.append(website)
        except Exception as exc:
            logger.error(f"Exception during web search: {exc}")
        return websites
