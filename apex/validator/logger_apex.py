import asyncio

from macrocosmos import AsyncLoggerClient

from apex import __version__
from apex.common.async_chain import AsyncChain
from apex.common.models import MinerDiscriminatorResults
from apex.common.utils_similarity import (
    compute_character_count_diff,
    compute_rouge_score,
    compute_sentence_length_diff,
    compute_similarity_score,
)


class LoggerApex:
    def __init__(
        self,
        async_chain: AsyncChain,
        project: str = "APEX_GAN_ARENA",
    ):
        self.mcl_client = AsyncLoggerClient()
        self.run = asyncio.create_task(
            self.mcl_client.init(
                project=project,
                config={
                    "hotkey": async_chain.wallet.hotkey.ss58_address,
                    "netuid": async_chain.netuid,
                    "version": __version__,
                },
            )
        )

    async def log(
        self,
        reference: str | None = None,
        discriminator_results: MinerDiscriminatorResults | None = None,
        tool_history: list[dict[str, str]] | None = None,
    ) -> None:
        """Log an event to the Apex logger."""
        if discriminator_results:
            event = discriminator_results.model_dump()
            event["reference"] = reference
            event["tool_history"] = tool_history
            await self.mcl_client.log(event)

    # TODO: Move to seperate etl pipeline
    async def preprocess_event(self, event: dict[str, str]) -> dict[str, str | float]:
        """Preprocess an event before logging it."""
        reference = event.get("reference")
        generation = event.get("generation", "")
        if not reference:
            return dict(event)

        rouge_score = compute_rouge_score(reference, generation)
        similarity_score = await compute_similarity_score(reference, generation)
        character_count_difference = compute_character_count_diff(reference, generation)
        sentence_length_difference = compute_sentence_length_diff(reference, generation)

        # Create a new dict with mixed types to satisfy mypy
        processed_event: dict[str, str | float] = dict(event)
        processed_event["rouge_score"] = rouge_score
        processed_event["similarity_score"] = similarity_score
        processed_event["character_count_difference"] = character_count_difference
        processed_event["sentence_length_difference"] = sentence_length_difference
        return processed_event
