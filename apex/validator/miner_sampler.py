import asyncio
import json
import random
import time
from collections import deque
from collections.abc import Coroutine, Sequence
from typing import Any, Literal

import aiohttp
from loguru import logger
from pydantic import BaseModel

from apex.common.async_chain import AsyncChain
from apex.common.constants import TIMEOUT, VALIDATOR_REFERENCE_LABEL
from apex.common.epistula import generate_header
from apex.common.models import MinerDiscriminatorResults, MinerGeneratorResults
from apex.common.utils import async_cache
from apex.validator.logger_db import LoggerDB

_TTL_UIDS_RESYNC = 300


class MinerInfo(BaseModel):
    hotkey: str
    uid: int
    address: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MinerInfo):
            return NotImplemented
        return (self.hotkey, self.uid, self.address) == (other.hotkey, other.uid, other.address)

    def __hash__(self) -> int:
        return hash((self.hotkey, self.uid, self.address))


class MinerSampler:
    def __init__(
        self,
        chain: AsyncChain,
        sample_mode: Literal["random", "sequential"] = "sequential",
        discriminator_sample_size: int = 50,
        generator_sample_size: int = 1,
        logger_db: LoggerDB | None = None,
        available_uids: Sequence[int] | None = None,
        available_addresses: Sequence[str] | None = None,
        validator_min_stake: float = 24_000,
    ):
        """Samples miner uids from metagraph and performs discriminator or generator queries.

        Args:
            chain: Chain object.
            sample_mode: Sampling mode, available modes:
                - random: Samples random uids.
                - sequential: Samples all uids sequentially.
            discriminator_sample_size: Amount of miners to be sampled for discriminator queries.
            generator_sample_size: Amount of miners to be sampled for generator queries.
            logger_db: Optional logger DB object.
            available_uids: List of available UIDs. If None, use all UIDs.
            available_addresses: List of available addresses for given UIDs. If None, use metagraph addresses.
            validator_min_stake: Validator minimum required alpha stake.
                UIDs with stake higher than specified won't be queried as miners.
        """
        self._chain = chain
        self._sample_mode = sample_mode
        self._discriminator_sample_size = discriminator_sample_size
        self._generator_sample_size = generator_sample_size
        self._logger_db = logger_db
        self._available_uids = available_uids
        self._available_addresses = available_addresses
        self._last_sample_idx: int = 0
        self._validator_min_stake = validator_min_stake
        if self._available_uids and self._available_addresses:
            equal_length = len(self._available_uids) == len(self._available_addresses)
            assert equal_length, "Test UIDs and addresses must be the same length."
        self._epoch_deque: deque[MinerInfo] = deque()
        self._sample_lock = asyncio.Lock()

    @async_cache(_TTL_UIDS_RESYNC)
    async def _get_all_miners(self, sample_size: int) -> list[MinerInfo]:
        meta = await self._chain.metagraph()
        miners: list[MinerInfo] = []
        for idx in range(meta.n.item()):
            if meta.stake[idx] >= self._validator_min_stake:
                # Skip validator hotkey.
                continue
            address = f"http://{meta.axons[idx].ip}:{meta.axons[idx].port}"
            miners.append(MinerInfo(uid=meta.uids[idx], hotkey=meta.hotkeys[idx], address=address))

        if self._available_uids is not None:
            # Test mode with predefined available pool of uids.
            logger.info(f"Using test mode with predefined list of available uids: {self._available_uids}")
            miners_test: list[MinerInfo] = []
            for miner_info in miners:
                try:
                    test_uid_index = self._available_uids.index(miner_info.uid)
                except ValueError:
                    continue

                if self._available_addresses:
                    address = self._available_addresses[test_uid_index]
                    logger.info(f"Replacing miner uid {miner_info.uid} endpoint: {miner_info.address} -> {address}")
                    miner_info.address = address
                miners_test.append(miner_info)
            miners = miners_test

        if sample_size > len(miners):
            logger.warning(
                f"Sample size is larger than amount of miners: {sample_size} > {len(miners)}. "
                f"Setting sample size to {len(miners)}"
            )
            sample_size = len(miners)
        return miners

    async def _sample_miners(self, sample_size: int) -> list[MinerInfo]:
        miners = await self._get_all_miners(sample_size=sample_size)

        miners_sample: list[MinerInfo] = []
        if self._sample_mode == "random":
            miners_sample = random.sample(miners, sample_size)

        elif self._sample_mode == "sequential":
            async with self._sample_lock:
                while len(miners_sample) < (sample_size):
                    if not self._epoch_deque:
                        # Get shuffled deque of miners.
                        self._epoch_deque = deque(random.sample(miners, len(miners)))
                    miners_sample.append(self._epoch_deque.popleft())

        else:
            raise ValueError(f"Unknown sampling mode: {self._sample_mode}")

        logger.debug(f"Sampled uids (sample size = {sample_size}): {sorted([miner.uid for miner in miners_sample])}")
        return miners_sample

    async def query_miners(
        self, body: dict[str, Any], endpoint: str, hotkey: str | None = None, timeout: float = TIMEOUT
    ) -> str:
        """Query the miners for the query."""
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession() as session:
                headers = await generate_header(
                    self._chain.wallet.hotkey, body=json.dumps(body).encode("utf-8"), signed_for=hotkey
                )
                async with session.post(
                    f"{endpoint}/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=client_timeout,
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.text()
        except BaseException:
            # Error during miner query, return empty string.
            return ""
        return str(result)

    async def query_generators(self, query: str) -> MinerGeneratorResults:
        """Query the miners for the query."""
        miner_information = await self._sample_miners(sample_size=self._generator_sample_size)
        body = {"step": "generator", "query": query}

        hotkeys: list[str] = []
        tasks: list[Coroutine[str, str, Any]] = []

        for miner_info in miner_information:
            hotkeys.append(miner_info.hotkey)
            tasks.append(self.query_miners(body=body, endpoint=miner_info.address, hotkey=miner_info.hotkey))
        generator_results = await asyncio.gather(*tasks)
        return MinerGeneratorResults(query=query, generator_hotkeys=hotkeys, generator_results=generator_results)

    async def query_discriminators(
        self,
        query: str,
        generator_results: MinerGeneratorResults | None,
        reference: str | None,
        ground_truth: int,
    ) -> MinerDiscriminatorResults:
        """Query the miners for the query."""
        miner_information = await self._sample_miners(sample_size=self._discriminator_sample_size)
        # Flip the coin for the generator.
        if ground_truth and generator_results:
            selected_generator: tuple[str, str] = random.choice(
                list(
                    zip(
                        generator_results.generator_hotkeys,
                        generator_results.generator_results,
                        strict=False,
                    )
                )
            )
        else:
            if reference is None:
                raise ValueError("Reference cannot be None when not using miner generator results")
            selected_generator = (VALIDATOR_REFERENCE_LABEL, reference)

        body = {
            "step": "discriminator",
            "query": query,
            "generation": selected_generator[1],
        }

        hotkeys: list[str] = []
        tasks: list[Coroutine[str, str, Any]] = []
        for miner_info in miner_information:
            hotkeys.append(miner_info.hotkey)
            tasks.append(self.query_miners(body=body, endpoint=miner_info.address, hotkey=miner_info.hotkey))
        discriminator_results = await asyncio.gather(*tasks)

        # Parse discriminator results and calculate scores.
        discriminator_results_float: list[float] = []
        parsed_discriminator_results: list[str] = []
        score_per_miner = 1.0 / len(miner_information)

        for result in discriminator_results:
            if result:
                # Parse the OpenAI response to extract the discriminator's choice.
                try:
                    parsed_result = json.loads(result)
                    # Extract the content from OpenAI response format.
                    choice_content = parsed_result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
                    # If parsing fails, assume it's a direct string response.
                    if not isinstance(result, str):
                        result = str(result)
                    choice_content = result.strip()
            else:
                choice_content = "None"
            parsed_discriminator_results.append(choice_content)

            # Apply scoring logic based on selected generator type.
            if choice_content == str(ground_truth):
                discriminator_score = score_per_miner
            else:
                discriminator_score = 0.0

            discriminator_results_float.append(discriminator_score)

        # Generator result is 1 minus sum of discriminator results.
        generator_result_float = 1.0 - sum(discriminator_results_float)
        miner_discriminator_results = MinerDiscriminatorResults(
            query=query,
            generator_hotkey=selected_generator[0],
            generator_result=selected_generator[1],
            generator_score=generator_result_float,
            discriminator_hotkeys=hotkeys,
            discriminator_results=parsed_discriminator_results,
            discriminator_scores=discriminator_results_float,
            timestamp=int(time.time()),
        )

        if self._logger_db is not None:
            await self._logger_db.log(miner_discriminator_results)

        return miner_discriminator_results
