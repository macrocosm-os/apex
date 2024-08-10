import asyncio
import json
import time
from functools import partial
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Sequence, Tuple, Union

import bittensor as bt
import numpy as np
from bittensor.dendrite import dendrite
from loguru import logger
from organic_scoring import OrganicScoringBase
from organic_scoring.organic_queue import OrganicQueueBase
from organic_scoring.synth_dataset import SynthDatasetBase
from starlette.types import Send
from transformers import PreTrainedTokenizerFast
from typing_extensions import override

from neurons.forward import handle_response
from prompting import settings
from prompting.base.dendrite import SynapseStreamResult
from prompting.base.protocol import StreamPromptingSynapse
from prompting.llms.vllm_llm import vLLMPipeline
from prompting.organic.organic_task import OrganicRewardConfig, OrganicTask
from prompting.settings import settings

# from prompting.rewards.reward import RewardResult


class OrganicEntry:
    roles: list[str]
    messages: list[str]
    is_organic: bool
    source: str


class OrganicScoringPrompting(OrganicScoringBase):
    def __init__(self,
        axon: bt.axon,
        synth_dataset: Optional[Union[SynthDatasetBase, Sequence[SynthDatasetBase]]],
        llm_pipeline: vLLMPipeline,
        tokenizer: PreTrainedTokenizerFast,
        update_scores_fn: Callable[[np.ndarray, list[int]], None],
        get_random_uids_fn: Callable[[int, Optional[list[int]]], np.ndarray],
        lock: asyncio.Lock,
        organic_queue: Optional[OrganicQueueBase] = None,
    ):
        super().__init__(
            axon=axon,
            synth_dataset=synth_dataset,
            trigger_frequency=settings.ORGANIC_TRIGGER_FREQUENCY,
            trigger=settings.ORGANIC_TRIGGER,
            trigger_frequency_min=settings.ORGANIC_TRIGGER_FREQUENCY_MIN,
            trigger_scaling_factor=settings.ORGANIC_SCALING_FACTOR,
            organic_queue=organic_queue,
        )
        self._llm_pipeline = llm_pipeline
        self._tokenizer = tokenizer
        self._update_scores_fn = update_scores_fn
        self._get_random_uids_fn = get_random_uids_fn
        self._lock = lock

    async def _generate_rewards(
        self, sample: OrganicEntry, responses: dict[str, SynapseStreamResult], reference: str
    ):
        _, _, rewards = OrganicRewardConfig.apply(responses=responses, reference=reference, query=sample.messages[-1])
        return {
            "rewards": rewards,
            "uids": responses.keys(),
            "organic": sample.is_organic,
        }

    @override
    async def _priority_fn(self, synapse: StreamPromptingSynapse) -> float:
        """Priority function for the axon."""
        return 10000000.0

    @override
    async def _blacklist_fn(self, synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
        """Blacklist function for the axon."""
        # ! DO NOT CHANGE `Tuple` return type to `tuple`, it will break the code (bittensor internal signature checks).
        # We expect the API to be run with one specific hotkey (e.g. OTF).
        return synapse.dendrite.hotkey != settings.ORGANIC_WHITELIST_HOTKEY, ""

    @override
    async def _on_organic_entry(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        """Organic query handle."""
        logger.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")

        # TODO: Query one of the top N incentive miners, keep the rest random.
        uids = list(self._get_random_uids_fn())
        completions: dict[int, dict] = {}
        token_streamer = partial(
            self._stream_miner_response,
            synapse,
            uids,
            completions,
            metagraph=settings.METAGRAPH,
            wallet=settings.WALLET,
        )

        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_queue.add(
            {
                "roles": synapse.roles,
                "messages": synapse.messages,
                "organic": True,
                "synapse": synapse,
                "streaming_response": streaming_response,
                "uids": uids,
                "completions": completions,
            }
        )
        return streaming_response

    async def _stream_miner_response(
        self,
        synapse: StreamPromptingSynapse,
        uids: list[int],
        completions: dict[int, dict],
        send: Send,
    ):
        """Stream back miner's responses."""
        logger.info(f"[Organic] Querying miner UIDs: {uids}")
        try:
            async with dendrite(wallet=settings.WALLET) as dend:
                responses = await dend(
                    axons=[settings.METAGRAPH.axons[uid] for uid in uids],
                    synapse=synapse,
                    timeout=settings.ORGANIC_TIMEOUT,
                    deserialize=False,
                    streaming=True,
                )
        except Exception as e:
            logger.error(f"[Organic] Error querying dendrite: {e}")
            return

        async def stream_miner_chunks(uid: int, chunks: AsyncGenerator):
            accumulated_chunks: list[str] = []
            accumulated_chunks_timings: list[float] = []
            accumulated_tokens_per_chunk: list[int] = []
            synapse: StreamPromptingSynapse | None = None
            completions[uid] = {"completed": False}
            timer_start = time.perf_counter()
            async for chunk in chunks:
                try:
                    if isinstance(chunk, str):
                        accumulated_chunks.append(chunk)
                        accumulated_chunks_timings.append(time.perf_counter() - timer_start)
                        json_chunk = json.dumps({"uid": uid, "chunk": chunk})
                        await send(
                            {
                                "type": "http.response.body",
                                "body": json_chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                    elif isinstance(chunk, StreamPromptingSynapse):
                        synapse = chunk
                except Exception as e:
                    logger.error(f"[Organic] Error while streaming chunks: {e}")
                    break
            # TODO: Do we need to identify the end of each miner's response?
            # json_chunk = json.dumps({"uid": uid, "chunk": b"", "completed": True})
            # await send({"type": "http.response.body", "body": json_chunk, "more_body": False})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            completions[uid]["accumulated_chunks"] = accumulated_chunks
            completions[uid]["accumulated_chunks_timings"] = accumulated_chunks_timings
            completions[uid]["accumulated_tokens_per_chunk"] = accumulated_tokens_per_chunk
            completions[uid]["completed"] = True
            completions[uid]["synapse"] = synapse
            # logger.debug(f"[Organic] Streaming {uid}: {''.join(accumulated_chunks)}")

        logger.info(f"[Organic] Awaiting miner streams UIDs: {uids}")
        await asyncio.gather(
            *[stream_miner_chunks(uid, chunks) for uid, chunks in zip(uids, responses)],
            return_exceptions=True,
        )

    async def _reuse_organic_response(self, sample: OrganicEntry) -> dict[int, SynapseStreamResult]:
        """Return a dictionary where the keys are miner UIDs and the values are their corresponding streaming responses.

        This method reuses miner responses for organic data. It waits for each miner to complete within the
        `neuron.organic_timeout` specified timeout and returns the responses. For miners who exceed the timeout,
        an empty synapse response is returned.

        Args:
            sample: Dict where the keys are miner UIDs and the values are the input streaming synapses.
        """
        if not sample.is_organic:
            return None

        uids = sample["uids"]
        responses: dict[int, SynapseStreamResult] = {}
        logger.info(f"[Organic] Reusing miner responses for organic data, UIDs: {uids}")

        async def _check_completion(sample: dict[str, Any], uid: int):
            while not sample["completions"][uid]["completed"]:
                await asyncio.sleep(0.01)

        async def _wait_for_completion(uid: int):
            try:
                await asyncio.wait_for(
                    _check_completion(sample, uid),
                    settings.ORGANIC_TIMEOUT,
                )
                response = SynapseStreamResult(
                    accumulated_chunks=sample["completions"][uid]["accumulated_chunks"],
                    accumulated_chunks_timings=sample["completions"][uid]["accumulated_chunks_timings"],
                    tokens_per_chunk=sample["completions"][uid]["accumulated_tokens_per_chunk"],
                    synapse=sample["completions"][uid]["synapse"],
                    uid=uid,
                    exception=None,
                )
            except asyncio.TimeoutError:
                response = SynapseStreamResult(
                    accumulated_chunks=[],
                    accumulated_chunks_timings=[],
                    tokens_per_chunk=[],
                    synapse=None,
                    uid=uid,
                    exception=None,
                )
            responses[uid] = response

        await asyncio.gather(*[_wait_for_completion(uid) for uid in uids])
        return responses

    @override
    async def _query_miners(self, sample: OrganicEntry) -> dict[str, SynapseStreamResult]:
        """Query miners with the given synthetic or organic sample."""
        if sample.is_organic and not settings.ORGANIC_REUSE_RESPONSE_DISABLED:
            responses = await self._reuse_organic_response(sample)
            return responses

        # Get the list of uids to query.
        uids = self._get_random_uids_fn()
        logger.info(f"[Organic] Querying miners with synthetic data, UIDs: {uids}")
        streams_responses = await dendrite.forward(
            axons=[settings.METAGRAPH.axons[uid] for uid in uids],
            synapse=StreamPromptingSynapse(roles=sample.roles, messages=sample.messages),
            timeout=settings.ORGANIC_TIMEOUT,
            deserialize=False,
            streaming=True,
        )
        stream_results_dict = dict(zip(uids, streams_responses))
        responses = await handle_response(stream_results_dict, tokenizer=self._tokenizer)
        return dict(zip(uids, responses))

    @override
    async def _set_weights(self, reward_result: dict[str, Any]):
        """Set weights based on the given reward."""
        if not reward_result.get("organic", False):
            reward_result["rewards"] *= settings.ORGANIC_SYNTH_REWARD_SCALE

        self._update_scores_fn(reward_result["rewards"], reward_result["uids"])

    @override
    async def _generate_reference(self, sample: dict[str, Any]) -> str:
        """Generate reference for the given organic or synthetic sample."""
        async with self._lock:
            _, reference = OrganicTask.generate_reference(sample, self._llm_pipeline)
        return reference
