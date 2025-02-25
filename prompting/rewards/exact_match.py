import numpy as np
from loguru import logger

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings
INCORRECT_PENALTY = 3
INCOMPLETE_PENALTY = 1


def normalize_timing(timing: float, timings: float) -> float:
    """
    Normalize the timing so that a lower timing (i.e. faster response) is closer to 1.
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


class ExactMatchRewardModel(BaseRewardModel):
    async def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """
        Calculates rewards based on an exact match of the response with the reference string.

        If the response exactly matches the reference, rewards are computed from the normalized timings.
        If the response is only a prefix of the reference, a less severe penalty is applied.
        Otherwise, a full penalty is given.

        Rewards are in the range [-3, 1].

        Parameters:
            reference (str): The expected response string.
            response_event (DendriteResponseEvent): Contains completions, chunked results, timings, etc.

        Returns:
            BatchRewardOutput: Contains the computed rewards and average timings.
        """
        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timings
        completions: list[str] = response_event.completions
        timeout: float = response_event.timeout

        if timeout <= 0:
            logger.error("Timeout must be greater than 0. Received timeout: {}", timeout)
            raise ValueError("Timeout must be greater than 0.")

        timing_outputs, rewards = [], []

        # Iterate over each response event.
        for chunks, timings, completion in zip(all_chunks, all_timings, completions):
            # If no response is provided, apply full penalty.
            if chunks == []:
                rewards.append(-INCORRECT_PENALTY)
                timing_outputs.append(0.0)
                continue

            # If the completion is a prefix of the reference, give a less severe penalty
            if len(completion) < len(reference) and reference.startswith(completion):
                rewards.append(-INCOMPLETE_PENALTY)
                timing_outputs.append(0.0)
                continue

            # If the completion does not exactly match the reference, apply full penalty.
            if reference != completion:
                rewards.append(-INCORRECT_PENALTY)
                timing_outputs.append(0.0)
                continue

            # Compute normalized timings for non-empty chunks.
            valid_chunks = []
            for chunk, timing in zip(chunks, timings):
                if chunk:
                    valid_chunks.append(normalize_timing(timing, all_timings))

            # Compute average timings for normalized chunk timings.
            if valid_chunks:
                # If there are valid chunks, compute the average timing.
                final_score = np.mean(valid_chunks)
            else:
                final_score = -INCORRECT_PENALTY

            rewards.append(float(final_score))
            timing_outputs.append(np.array(valid_chunks).mean())

        return BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timing_outputs),
        )
