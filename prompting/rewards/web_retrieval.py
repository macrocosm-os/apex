"""Expected miner's response is a JSON object with the following keys: url, content, relevant for each website."""

import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import whois
from loguru import logger
from pydantic import BaseModel
from scipy import spatial
from thefuzz import fuzz

from prompting.datasets.random_website import DDGDataset, DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent
from shared.misc import async_lru_cache

MIN_RELEVANT_CHARS = 300
MIN_MATCH_THRESHOLD = 98

MIN_SIM_THRESHOLD = 0.44
PENALIZE_SIM_THRESHOLD = 0.36
PENALTY = -0.1

# Define file paths
PAST_WEBSITES_FILE = "past_websites.csv"
TOP_DOMAINS_FILE = "data/top100k_domains.csv"

# Define blacklisted terms
BLACKLISTED_TERMS = {
    "howtogeek",
    "docs.google.com",
    "?q=",
    "/search",
    "sheets.google.com",
    "drive.google.com",
    "pastebin",
    "paste",
    "gist",
    "github",
    "gitlab",
    "bitbucket",
    "hastebin",
    "ghostbin",
    "privatebin",
}

# Maximum number of past URLs to store per user
N_PAST_URLS = 200

# Minimum age of the website.
MIN_AGE_DAYS = 90

# Load the past_websites dictionary and top domains
try:
    # Load top domains
    top_domains_df = pd.read_csv(TOP_DOMAINS_FILE)
    TOP_DOMAINS = set(top_domains_df["Domain"].str.lower().values)

    # Load past websites
    if os.path.exists(PAST_WEBSITES_FILE) and os.path.getsize(PAST_WEBSITES_FILE) > 0:
        past_websites_df = pd.read_csv(PAST_WEBSITES_FILE)
        past_websites = defaultdict(list)
        # Group by uid and take only the last N_PAST_URLS entries
        for uid, group in past_websites_df.groupby("uid"):
            past_websites[uid] = group["domain"].tolist()[-N_PAST_URLS:]
    else:
        logger.warning(f"Past websites file {PAST_WEBSITES_FILE} does not exist or empty, creating new dictionary")
        past_websites = defaultdict(list)
except Exception as e:
    logger.error(f"Failed to load domains data: {e}")
    TOP_DOMAINS = set()
    past_websites = defaultdict(list)


def _append_to_past_websites(uid: str, domain: str):
    """Helper function to append domain to past_websites while maintaining max size."""
    past_websites[uid].append(domain)
    if len(past_websites[uid]) > N_PAST_URLS:
        past_websites[uid] = past_websites[uid][-N_PAST_URLS:]


class WebsiteResult(BaseModel):
    url: str | None
    content: str | None
    relevant: str | None


import tldextract


def extract_main_domain(url):
    # Extract domain components
    extracted = tldextract.extract(url)
    # Return domain + suffix (e.g. "google.com")
    return f"{extracted.domain}.{extracted.suffix}"


class WebRetrievalRewardModel(RelevanceRewardModel):
    def __hash__(self):
        # Use the id of the object as its hash
        return hash(self.model_dump_json)

    @staticmethod
    @async_lru_cache(maxsize=1000)
    async def domain_age_days(domain: str, fallback_age: int = 1_000_000) -> int:
        """Returns the age of a domain in days.

        Args:
            domain: Website url.
            fallback_age: If can't fetch domain age, fallback to `fallback_age` age.

        Returns:
            Domain age in days since creation.
        """
        fallback_age = 1_000_000
        try:
            w = await asyncio.to_thread(whois.whois, domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list) and creation_date:
                creation_date = creation_date[0]

            if creation_date is None:
                return fallback_age
            # Convert everything to naive datetime in UTC or local.
            if hasattr(creation_date, "tzinfo") and creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            delta = datetime.now() - creation_date
            return delta.days
        except BaseException as e:
            logger.debug(f"Error fetching domain age data: {e}")
            return fallback_age

    @async_lru_cache(maxsize=1000)
    async def _cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions."""
        reference_emb_flatten = self.embedding_model.encode(content1, to_numpy=True).flatten()
        response_emb_flatten = self.embedding_model.encode(content2, to_numpy=True).flatten()
        return 1.0 - float(spatial.distance.cosine(reference_emb_flatten, response_emb_flatten))

    async def score_website_result(
        self, dataset_entry: DDGDatasetEntry, response_url: str, response_content: str, response_relevant: str, uid: str
    ) -> float:
        if not response_url or not response_content or not response_relevant:
            return PENALTY

        # Extract domain from URL.
        netloc = extract_main_domain(response_url)
        logger.debug(f"Scoring url: {response_url}")

        if any(term in response_url for term in BLACKLISTED_TERMS):
            logger.debug(f"Domain {response_url} contains blacklisted term, scoring 0")
            return 0

        if (days := await self.domain_age_days(response_url)) < MIN_AGE_DAYS:
            logger.debug(f"Domain {response_url} is too young ({days} days old), scoring 0")
            return 0

        # Penalise a completion where the relevant section is contained in the URL (e.g. miners)
        # trying to use a search box to enter exactly the relevant section they need
        discount_factor = 1 - fuzz.token_sort_ratio(response_url, response_relevant) / 100
        # Check if URL is IP-based or has port
        if not response_url or len(response_url) > 500:
            logger.debug(f"URL {response_url} is too long, setting discount factor to 0")
            return 0

        if not netloc or any(c.isdigit() for c in netloc.split(".")) or ":" in netloc:
            discount_factor = 0
            logger.debug(f"URL {response_url} appears to be IP-based or on specific port, setting discount factor to 0")
            return 0
        else:
            domain = netloc

            domain_count = np.sum(np.array([domain == d for d in past_websites[uid]])) + 1

            # If domain is in top 100k, don't apply penalty
            if domain in TOP_DOMAINS:
                # if the domain is in the top 100k, we allow 10 occurrences in the last 200 URLs before penalising
                discount_factor *= 1.0 / (max(1, domain_count - 10))
            else:
                # Count how many times this domain has been used by this miner
                discount_factor *= 1.0 / max(1, domain_count)

            _append_to_past_websites(uid, domain)

            # Content scraped from the URL provided in the completion.
            reference_website_content = DDGDataset.extract_website_content(response_url)
            if not reference_website_content or len(reference_website_content) == 0:
                logger.debug(f"Failed to extract miner {uid} content from website: {response_url}")
                return 0

            if fuzz.ratio(response_content, reference_website_content) < MIN_MATCH_THRESHOLD:
                logger.debug(f"Miner {uid} returned text that doesn't match the website, scoring 0")
                return 0

            if len(response_relevant) > len(response_content) or len(response_relevant) < MIN_RELEVANT_CHARS:
                logger.debug(
                    f"Miner {uid} relevant section is too short (<{MIN_RELEVANT_CHARS} chars) or longer than the whole "
                    f"website content {len(response_relevant)} > {len(response_content)}"
                )
                return 0

            if response_relevant not in response_content:
                return 0

            similarity = await self._cosine_similarity(content1=dataset_entry.query, content2=response_relevant)
            logger.warning(f"SIM: {similarity}")
            if similarity < PENALIZE_SIM_THRESHOLD:
                # Penalise if similarity is too low.
                logger.debug(f"Miner {uid} returned text that doesn't match the query")
                return PENALTY
            elif similarity < MIN_SIM_THRESHOLD:
                logger.debug(f"Miner {uid} returned text has low similarity")
                return 0
            return similarity * discount_factor

    async def score_miner_response(
        self, dataset_entry: DDGDatasetEntry, completion: str, task: BaseTextTask | None = None, uid: str | None = None
    ) -> float:
        scores = []
        miner_websites: list[WebsiteResult] = self._parse_response(completion)
        unique_websites = np.unique([website.url for website in miner_websites])
        if unique_websites.size != len(miner_websites) or unique_websites.size != task.target_results:
            # logger.warning("Miner returned multiple websites with the same URL")
            return PENALTY

        tasks = [
            self.score_website_result(dataset_entry, website.url, website.content, website.relevant, uid)
            for website in miner_websites
        ]
        scores = await asyncio.gather(*tasks)

        if scores:
            weights = np.arange(len(scores), 0, -1)
            return float(np.average(scores, weights=weights))
        return 0

    # TODO: Change base class reference type to Reference pydantic model, in order to store additional data.
    async def reward(
        self, reference: str, response_event: DendriteResponseEvent, task: BaseTextTask | None = None, **kwargs
    ) -> BatchRewardOutput:
        """Score response website content and URL based on the similarity to the search term and reference content."""
        rewards: list[float] = []
        timings: list[float] = []
        dataset_entry = DDGDatasetEntry.model_validate_json(json.loads(reference))
        if not dataset_entry.query:
            # if the dataset doesn't have a query, we can't score the completions
            return BatchRewardOutput(
                rewards=np.array([0] * len(response_event.completions)),
                timings=np.array([0] * len(response_event.completions)),
            )

        for completion, uid in zip(response_event.completions, response_event.uids):
            rewards.append(await self.score_miner_response(dataset_entry, completion, task=task, uid=uid))
            timings.append(0)

        # Save the past_websites dictionary to CSV
        past_websites_data = []
        for uid, domains in past_websites.items():
            for domain in domains:
                past_websites_data.append({"uid": uid, "domain": domain})
        pd.DataFrame(past_websites_data).to_csv(PAST_WEBSITES_FILE, index=False)

        return BatchRewardOutput(rewards=np.array(rewards), timings=np.array(timings))

    @staticmethod
    def _parse_response(completion: str) -> tuple[str | None, ...]:
        result: list[WebsiteResult] = []
        try:
            data = json.loads(completion)
            if not isinstance(data, list) and isinstance(data, dict):
                data = [data]
            for website in data:
                if not isinstance(website, dict):
                    continue
                response_url = website.get("url")
                response_content = website.get("content")
                response_relevant = website.get("relevant")
                result.append(WebsiteResult(url=response_url, content=response_content, relevant=response_relevant))
            return result
        except BaseException:
            result = []
        return result
