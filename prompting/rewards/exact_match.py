import random

import numpy as np
from loguru import logger
from openai.types.chat import ChatCompletionChunk

from prompting.llms.model_manager import ModelManager
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings
INCORRECT_PENALTY = 0.5
INCOMPLETE_PENALTY = 0.25
MIN_SMOOTH_REWARD = 0.6
VERIFICATION_RATIO = 0.1
VERIFICATION_THRESHOLD = 0.9


def smooth_timings_reward(timings_uid: list[float], min_reward: float = MIN_SMOOTH_REWARD) -> float:
    """Return smooth stream ration based on the deviation between chunks timings.

    Args:
        timings_uid: List of timings for a specific miner.

    Returns:
        float: Smoothed penalty value.
    """
    if not timings_uid:
        return 0.0

    smooth_penalty = np.std(timings_uid)
    return max(min_reward, 1.0 - smooth_penalty)


def normalize_timing(timing: float, timings: float) -> float:
    """Normalize the timing so that a lower timing (i.e. faster response) is closer to 1.
    Ensures the normalized value is between 0 and 1.
    """

    flat_values = [
        x
        for sublist in timings
        if sublist is not None
        for x in (sublist if isinstance(sublist, list) else [sublist])
        if x is not None
    ]
    last_chunk = max(flat_values) if flat_values else shared_settings.INFERENCE_TIMEOUT
    return min(1, max(0, (last_chunk - timing) / last_chunk))


def verify_single_logit(original_logits: dict[str, float], verification_logits: dict[str, float]) -> float:
    """Verify logits by computing cosine similarity between original and verification logits.

    Args:
        original_logits: Original model logits.
        verification_logits: Verification model logits.

    Returns:
        float: Cosine similarity score.
    """
    # Create aligned vectors with same token ordering
    all_tokens = set(original_logits.keys()) | set(verification_logits.keys())

    orig_vec = []
    verif_vec = []
    for token in all_tokens:
        orig_vec.append(original_logits.get(token, -100.0))
        verif_vec.append(verification_logits.get(token, -100.0))

    orig_vec = np.array(orig_vec)
    verif_vec = np.array(verif_vec)

    # Apply softmax to convert logprobs to probabilities
    orig_vec = np.exp(orig_vec) / np.sum(np.exp(orig_vec))
    verif_vec = np.exp(verif_vec) / np.sum(np.exp(verif_vec))

    # Calculate cosine similarity
    orig_vec = orig_vec / np.linalg.norm(orig_vec)
    verif_vec = verif_vec / np.linalg.norm(verif_vec)
    return float(np.dot(orig_vec, verif_vec))


class LogitsRewardModel(BaseRewardModel):
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        task: BaseTextTask,
        model_manager: ModelManager,
        **kwargs,
    ) -> BatchRewardOutput:
        """Calculate rewards based on the logits of the response and verifies them."""
        if model_manager is None:
            raise ValueError("Model manager must be set")

        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_chunk_dicts_raw: list[list[ChatCompletionChunk]] = response_event.stream_results_all_chunk_dicts_raw
        uids: np.ndarray | list[float] = response_event.uids
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timings
        completions: list[str] = response_event.completions
        timeout: float = response_event.timeout
        sampling_parameters: dict = task.sampling_params
        PENALIZE_ALL = BatchRewardOutput(
            rewards=np.array([-INCORRECT_PENALTY] * len(completions)),
            timings=np.array([0.0] * len(completions)),
        )

        if all(not chunk for chunk in all_chunks):
            logger.warning("No chunks to verify, penalizing all miners")
            return PENALIZE_ALL

        if timeout <= 0:
            logger.error(f"Timeout must be greater than 0. Received timeout: {timeout}")
            raise ValueError("Timeout must be greater than 0.")

        # If max_tokens are not provided, always check for eos.
        max_tokens = sampling_parameters.get("max_tokens", float("inf"))
        model = await model_manager.get_model(task.llm_model_id)
        eos_token = model.tokenizer.eos_token
        timing_outputs = []
        rewards = []
        # Iterate over each miner response.
        for chunks, timings, chunk_dicts_raw, uid in zip(all_chunks, all_timings, all_chunk_dicts_raw, uids):
            try:
                # If no response is provided, apply full penalty.
                if not chunks:
                    rewards.append(-INCORRECT_PENALTY)
                    timing_outputs.append(0.0)
                    continue

                completion_length = len(chunks)
                # Sample from 1 to 20 indices for verification.
                num_verify = max(1, min(20, int(completion_length * VERIFICATION_RATIO)))
                # Sample one less to save room for last index.
                verify_indices = random.sample(range(completion_length - 1), num_verify - 1)
                # Always verify the last index.
                last_idx = completion_length - 1
                verify_indices.append(last_idx)
                verify_indices.sort()
                # Verify logits for selected indices.
                verification_scores = []

                for idx in verify_indices:
                    check_idx = min(idx, completion_length - 1)
                    if not chunk_dicts_raw[check_idx].choices[0].logprobs:
                        raise ValueError("No logprobs provided")

                    if chunk_dicts_raw[check_idx].choices[0].logprobs.content is None:
                        raise ValueError("Logprobs content is empty")

                    original_logits = {
                        info.token: info.logprob
                        for info in chunk_dicts_raw[check_idx].choices[0].logprobs.content[0].top_logprobs
                    }

                    verification_output, prompt = await model_manager.generate_logits(
                        model=task.llm_model_id,
                        messages=task.task_messages + [{"role": "assistant", "content": "".join(chunks[:check_idx])}],
                        sampling_params=sampling_parameters,
                        continue_last_message=True,
                    )

                    logit_score = verify_single_logit(original_logits, verification_output)
                    verification_scores.append(logit_score)

                    if idx == last_idx and completion_length < max_tokens:
                        if eos_token and (eos_token not in original_logits or eos_token not in verification_output):
                            # Do not set full penalty, since top_k = 50 and top_lobprobs = 10.
                            # TODO: Make top_k equal to top_logprobs and check for token in top_logprobs.
                            verification_scores = [-INCOMPLETE_PENALTY]

                final_score = float(np.mean(verification_scores))
                if final_score < VERIFICATION_THRESHOLD:
                    rewards.append(0.0)
                    timing_outputs.append(0.0)
                    continue

                valid_chunks: list[float] = []
                for chunk, timing in zip(chunks, timings):
                    if chunk:
                        valid_chunks.append(normalize_timing(timing, all_timings))
                timing_reward = float(np.mean(valid_chunks)) if valid_chunks else 0.0
                smooth_reward = smooth_timings_reward(timings)

                rewards.append(final_score * timing_reward * smooth_reward)
                timing_outputs.append(np.array(valid_chunks).mean())
            except BaseException as e:
                logger.debug(f"Miner {uid} failed to pass logits check: {e}")
                rewards.append(-INCORRECT_PENALTY)
                timing_outputs.append(0.0)

        reward_output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timing_outputs),
        )
        logger.debug(f"Logits rewards: {reward_output.model_dump()}")
        return reward_output
