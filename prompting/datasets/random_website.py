import random
from functools import lru_cache
from typing import Optional

import trafilatura
from duckduckgo_search.duckduckgo_search import DDGS
from loguru import logger

from prompting.datasets.utils import ENGLISH_WORDS
from shared import settings
from shared.base import BaseDataset, Context, DatasetEntry

MAX_CHARS = 5000


class DDGDatasetEntry(DatasetEntry):
    search_term: str
    website_url: str = None
    website_content: str = None
    query: str | None = None
    source: str | None = None


class DDGDataset(BaseDataset):
    english_words: list[str] = None

    def search_random_term(self, retries: int = 3) -> tuple[Optional[str], Optional[list[dict[str, str]]]]:
        ddg = DDGS(proxy=settings.shared_settings.PROXY_URL, verify=False)
        exception: BaseException | None = None
        for _ in range(retries):
            random_words = " ".join(random.sample(ENGLISH_WORDS, 3))
            try:
                results = list(ddg.text(random_words))
                if results:
                    return random_words, results
            except BaseException as ex:
                exception = ex
        logger.warning(f"Failed to get search results from DuckDuckGo after {retries} tries: {exception}")
        return None, None

    @staticmethod
    @lru_cache(maxsize=1000)
    def extract_website_content(url: str, retries: int = 3) -> Optional[str]:
        exception: Exception | None = None
        for _ in range(retries):
            try:
                website = trafilatura.fetch_url(url)
                extracted = trafilatura.extract(website)
                return extracted[:MAX_CHARS] if extracted else None
            except BaseException as ex:
                exception = ex
        logger.debug(f"Failed to extract content from website {url} after {retries} retries: {exception}")

    def next(self) -> Optional[DDGDatasetEntry]:
        search_term, results = self.search_random_term(retries=5)
        if not results:
            return None
        website_url = results[0]["href"]
        website_content = self.extract_website_content(website_url)
        if not website_content or len(website_content) == 0:
            logger.debug(f"Failed to extract content from website {website_url}")
            return None

        return DDGDatasetEntry(
            search_term=search_term, website_url=website_url, website_content=website_content, source=website_url
        )

    def get(self) -> Optional[DDGDatasetEntry]:
        return self.next()

    def random(self) -> Context:
        return self.next()
