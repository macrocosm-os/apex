import wandb
from typing import Any, Mapping

from apex.common.async_chain import AsyncChain
from apex.common.models import MinerDiscriminatorResults
from apex import __version__


def approximate_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    return len(text) // 4


class LoggerWandb:
    def __init__(
        self,
        async_chain: AsyncChain,
        project: str = "apex-gan-arena",
        api_key: str | None = None,
    ):
        self.run: Any | None = None
        if project and api_key:
            # Authenticate with W&B, then initialize the run
            wandb.login(key=api_key)
            self.run = wandb.init(
                entity="macrocosmos",
                project=project,
                config={
                    "hotkey": async_chain.wallet.hotkey.ss58_address,
                    "netuid": async_chain.netuid,
                    "version": __version__,
                },
            )

    async def log(
        self,
        reference: str | None = None,
        discriminator_results: MinerDiscriminatorResults | None = None,
        tool_history: list[dict[str, str]] | None = None,
    ) -> None:
        """Log an event to wandb."""
        if self.run:
            if discriminator_results:
                processed_event = self.process_event(discriminator_results.model_dump())
                processed_event["reference"] = reference
                processed_event["tool_history"] = tool_history
                self.run.log(processed_event)

    def process_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Preprocess an event before logging it."""
        reference = event.get("reference", "")
        generation = event.get("generation", "")
        generator_tokens = approximate_tokens(generation)
        reference_tokens = approximate_tokens(reference)

        processed_event: dict[str, Any] = dict(event)
        processed_event["generator_tokens"] = generator_tokens
        processed_event["reference_tokens"] = reference_tokens

        return processed_event
