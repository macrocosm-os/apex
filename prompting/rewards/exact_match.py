import random

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from openai.types.chat import ChatCompletionChunk

from prompting.llms.model_manager import ModelManager
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings

MIN_VERIFY_TOKENS = 10
MAX_VERIFY_TOKENS = 30
NO_EOS_PENALTY = -0.1
INCORRECT_PENALTY = -2
MIN_SMOOTH_PENALTY_SCALE = 0.6
MIN_TIME_PENALTY_SCALE = 0.3
VERIFICATION_THRESH_CONTAINS = 0.96
VERIFICATION_THRESH_SIM = 0.86


class LogitsRewardModel(BaseRewardModel):
    async def reward(  # noqa: C901
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
            rewards=np.array([INCORRECT_PENALTY] * len(completions)),
            timings=np.array([0.0] * len(completions)),
        )

        if all(not chunk for chunk in all_chunks):
            logger.warning("No chunks to verify, penalizing all miners")
            return PENALIZE_ALL

        if timeout <= 0:
            logger.error(f"Timeout must be greater than 0. Received timeout: {timeout}")
            raise ValueError("Timeout must be greater than 0.")

        # If max_tokens are not provided, always check for eos.
        max_tokens = sampling_parameters.get("max_tokens", 64_000)
        model = await model_manager.get_model(task.llm_model_id)
        eos_token = model.tokenizer.eos_token
        timing_verified: list[list[float]] = []
        rewards: list[float] = []
        logger.info(f"Verifying logits with model {task.llm_model_id}")
        # Iterate over each miner response.
        for chunks, timings, chunk_dicts_raw, uid in zip(all_chunks, all_timings, all_chunk_dicts_raw, uids):
            penalty = INCORRECT_PENALTY
            try:
                # If no response is provided, apply full penalty.
                if not chunks:
                    rewards.append(INCORRECT_PENALTY)
                    timing_verified.append([-1.0])
                    continue

                completion_length = len(chunks)
                if completion_length <= 1 and max_tokens > 1:
                    # Response can't be a single token, skip all other checks.
                    rewards.append(INCORRECT_PENALTY)
                    timing_verified.append([-1.0])
                    continue

                if completion_length < MIN_VERIFY_TOKENS:
                    # Not enough tokens to verify, set reward to 0.
                    rewards.append(0)
                    timing_verified.append([-1.0])
                    continue

                eos_idx = completion_length
                verify_indices = self.sample_verification_indices(completion_length)
                scores_sim: list[float] = []
                scores_contains: list[float] = []
                for idx in verify_indices:
                    check_idx = min(idx, completion_length)
                    messages = task.task_messages.copy()
                    to_complete = "".join(chunks[:check_idx])
                    if to_complete:
                        messages.extend([{"role": "assistant", "content": to_complete}])
                    verification_logits, _ = await model_manager.generate_logits(
                        model=task.llm_model_id,
                        messages=messages,
                        sampling_params=sampling_parameters,
                        continue_last_message=len(to_complete) > 0,
                    )
                    if check_idx < eos_idx:
                        if not chunk_dicts_raw[check_idx].choices[0].logprobs:
                            raise ValueError("No logprobs provided")

                        if chunk_dicts_raw[check_idx].choices[0].logprobs.content is None:
                            raise ValueError("Logprobs content is empty")

                        original_logits = {
                            info.token: info.logprob
                            for info in chunk_dicts_raw[check_idx].choices[0].logprobs.content[0].top_logprobs
                        }

                        logit_sim = self.verify_logit_similarity(original_logits, verification_logits)
                        scores_sim.append(logit_sim)

                        logit_contains = self.verify_logit_contains(
                            chunks[check_idx], original_logits, verification_logits
                        )
                        scores_contains.append(logit_contains)

                    elif check_idx == eos_idx and completion_length < max_tokens:
                        if eos_token and eos_token not in verification_logits:
                            penalty = NO_EOS_PENALTY
                            raise ValueError("Partial completion")

                score_sim_mean = float(np.mean(scores_sim))
                score_contains_mean = float(np.mean(scores_contains))

                if score_sim_mean < VERIFICATION_THRESH_SIM:
                    raise ValueError(f"Logits similarity mean score is below threshold: {score_sim_mean:.2f}")

                if score_contains_mean < VERIFICATION_THRESH_CONTAINS:
                    raise ValueError(f"Logits contains mean score is below threshold: {score_contains_mean:.2f}")

                timing_verified.append(timings)
                smooth_reward = self.smooth_timings_reward(timings)
                # Min-max scale logits reward, e.g from [0.95; 1.0] to [0.0, 1.0].
                score_sim_mean = self.rescale(score_sim_mean, min_value=VERIFICATION_THRESH_SIM)
                score_contains_mean = self.rescale(score_contains_mean, min_value=VERIFICATION_THRESH_CONTAINS)
                logits_score = (score_sim_mean + score_contains_mean) / 2
                rewards.append(logits_score * smooth_reward)
            except BaseException as e:
                logger.debug(f"Miner {uid} failed to pass logits check: {e}")
                rewards.append(penalty)
                timing_verified.append([-1.0])

        timing_outputs: list[float] = []
        # Find the fastest response per chunk in the current pool.
        fastest_chunk = self.fastest_timing(timing_verified)
        for idx, (timings, uid) in enumerate(zip(timing_verified, uids)):
            if rewards[idx] < 0:
                timing_outputs.append(0)
                continue

            time_per_chunk = sum(timings) / len(timings)
            if min(timings) < 0 or time_per_chunk <= 0:
                timing_outputs.append(0)
                continue

            # Scale rewards based on how relative current timings to the fastest response.
            timing_reward = float(np.clip(fastest_chunk / time_per_chunk, MIN_TIME_PENALTY_SCALE, 1))
            rewards[idx] *= timing_reward
            timing_outputs.append(timing_reward)

        if len(rewards) != len(timing_outputs) != len(uids):
            raise ValueError(
                f"Rewards, timings or UIDs have different lengths {len(rewards)} {len(timing_outputs)} {len(uids)}"
            )

        rewards = np.array(rewards)
        logger.info(f"Success responses: {len(rewards[rewards > 0])}/{len(rewards)}")

        reward_output = BatchRewardOutput(
            rewards=rewards,
            timings=np.array(timing_outputs),
        )
        logger.debug(f"Logits rewards: {reward_output.model_dump()}")
        return reward_output

    @staticmethod
    def sample_verification_indices(completion_length: int) -> list[int]:
        """Sample random indices for verification, always add 0 and eos_token index."""
        # Sample indices without first and last index.
        num_verify = int(np.clip(completion_length, 1, MAX_VERIFY_TOKENS)) - 2
        verify_indices = random.sample(range(1, completion_length - 1), num_verify)
        # Add first index.
        verify_indices.append(0)
        # Add eos_token index.
        verify_indices.append(completion_length)
        verify_indices.sort()
        return verify_indices

    @staticmethod
    def rescale(value: float, min_value: float = VERIFICATION_THRESH_SIM) -> float:
        """Scale x from the domain [min_value, 1.0] to [0.0, 1.0]."""
        y = (value - min_value) / (1.0 - min_value)
        return max(0.0, min(1.0, y))

    @staticmethod
    def fastest_timing(values: list[list[float]]) -> float:
        """Return the smallest sum of inner list, compute its sum only if the list contains no negative numbers."""
        best = float("+inf")
        for subset in values:
            if subset and min(subset) >= 0.0:
                subset_sum = sum(subset) / len(subset)
                if subset_sum < best:
                    best = subset_sum
        return best if best < float("+inf") else 1e-6

    @staticmethod
    def smooth_timings_reward(timings_uid: list[float], min_reward: float = MIN_SMOOTH_PENALTY_SCALE) -> float:
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

    @staticmethod
    def verify_logit_contains(
        candidate_token: str, candidate_logits: dict[str, float], gt_logits: dict[str, float]
    ) -> float:
        """Verify if the selected token and logprobs are present in the verification output."""
        if not gt_logits:
            return 0.0

        if candidate_token not in candidate_logits.keys():
            return 0.0

        if candidate_token not in gt_logits.keys():
            return 0.0

        return 1.0

    @staticmethod
    def verify_logit_similarity(
        candidate_logits: dict[str, float],
        gt_logits: dict[str, float],
    ) -> float:
        """Similarity between candidate and ground-truth logprobs."""
        if not gt_logits:
            return 0.0

        if len(candidate_logits) != len(gt_logits):
            return 0.0

        # Tokens common to both distributions.
        overlap = set(candidate_logits) & set(gt_logits)
        if not overlap:
            return 0.0

        length = len(gt_logits)
        pred_tensor = torch.zeros(length, dtype=torch.float32)
        gt_tensor = torch.zeros(length, dtype=torch.float32)
        for idx, token in enumerate(overlap):
            pred_tensor[idx] = candidate_logits[token]
            gt_tensor[idx] = gt_logits[token]
        cos = float(F.cosine_similarity(pred_tensor, gt_tensor, dim=0, eps=1e-8).item())

        # Weight by how much of verification is overlapped.
        overlap_frac = len(overlap) / len(gt_logits)

        # Map to [0, 1] and clamp minor numeric drift.
        score = cos * overlap_frac
        return max(0.0, min(1.0, score))
