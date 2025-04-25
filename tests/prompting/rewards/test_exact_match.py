from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from openai.types.chat import ChatCompletionChunk

from prompting.llms.model_manager import ModelManager
from prompting.rewards.exact_match import (
    INCORRECT_PENALTY,
    MAX_VERIFY_TOKENS,
    MIN_SMOOTH_PENALTY_SCALE,
    MIN_VERIFY_TOKENS,
    NO_EOS_PENALTY,
    VERIFICATION_THRESH_SIM,
    LogitsRewardModel,
)
from prompting.rewards.reward import BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent


@pytest.fixture
def model_manager():
    """Mock ModelManager for testing."""
    manager = MagicMock(spec=ModelManager)
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token = "<|endoftext|>"
    model.tokenizer = tokenizer
    manager.get_model.return_value = model

    async def mock_generate_logits(*args, **kwargs):
        return {"token1": -0.1, "token2": -0.5, "<|endoftext|>": -1.0}, "prompt"

    manager.generate_logits = AsyncMock(side_effect=mock_generate_logits)
    return manager


@pytest.fixture
def task():
    """Mock Task for testing."""
    task = MagicMock(spec=BaseTextTask)
    task.llm_model_id = "gpt-4"
    task.task_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    task.sampling_params = {"temperature": 0.7, "max_tokens": 100}
    return task


def create_chat_completion_chunk(content="", logprobs=None):
    """Helper function to create a ChatCompletionChunk object."""
    if logprobs is None:
        logprobs = {content: -0.1, "token2": -0.5, "token3": -0.6, "token4": -0.7, "<|endoftext|>": -1.0}
        print(content, logprobs)

    chunk = MagicMock(spec=ChatCompletionChunk)
    choice = MagicMock()
    choice.index = 0
    choice.delta = MagicMock()
    choice.delta.role = "assistant"
    choice.delta.content = content

    if logprobs:
        choice.logprobs = MagicMock()
        choice.logprobs.content = [MagicMock()]
        choice.logprobs.content[0].top_logprobs = []
        for token, logprob in logprobs.items():
            token_logprob = MagicMock()
            token_logprob.token = token
            token_logprob.logprob = logprob
            choice.logprobs.content[0].top_logprobs.append(token_logprob)
    else:
        choice.logprobs = None

    chunk.choices = [choice]
    chunk.id = "chunk_id"
    chunk.created = 1234567890
    chunk.model = "VeryStronkModel"
    chunk.object = "chat.completion.chunk"
    chunk.usage = None
    return chunk


async def create_response_event_mock(chunks_all, timings_all, timeout: float = 10) -> MagicMock:
    completions = ["".join(chunks) for chunks in chunks_all]
    chunk_dicts_raw = []
    for chunks in chunks_all:
        chunk_dicts_raw.append([create_chat_completion_chunk(chunk) for chunk in chunks])

    response_event = MagicMock(spec=DendriteResponseEvent)
    response_event.stream_results_all_chunks = chunks_all
    response_event.stream_results_all_chunk_dicts_raw = chunk_dicts_raw
    response_event.uids = list(range(len(chunks_all)))
    response_event.stream_results_all_chunks_timings = timings_all
    response_event.completions = completions
    response_event.timeout = timeout
    return response_event


@pytest.mark.asyncio
async def test_correct_completion(model_manager, task):
    """Test case 1: Correct completion with reward >0.5 and ≤1."""
    chunks_all = [["Hello", ", ", "world", "!"]]
    chunks_timings_all = [[0.1, 0.1, 0.1, 0.1]]
    response_event = await create_response_event_mock(chunks_all, chunks_timings_all)
    chunk_dicts_raw = []
    for chunks in chunks_all:
        chunk_dicts_raw.append([create_chat_completion_chunk(chunk) for chunk in chunks])

    with (
        patch("prompting.rewards.exact_match.MIN_VERIFY_TOKENS", 2),
        patch("prompting.rewards.exact_match.LogitsRewardModel.verify_logit_similarity", return_value=1),
        patch("prompting.rewards.exact_match.LogitsRewardModel.verify_logit_contains", return_value=1),
    ):
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="", response_event=response_event, task=task, model_manager=model_manager
        )
        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 1
        assert result.rewards[0] == 1.0


@pytest.mark.asyncio
async def test_mixed_completions(model_manager, task):
    """Test case 2: One ideal completion, one with missing logprobs penalized."""
    chunks_timings_all = [[0.1, 0.2, 0.3, 0.4] for _ in range(3)]
    chunks_all = [["Hello", ", ", "world", "!"], ["Fail", "ed", " ", "completion"], ["Wro", "ng", " ", "completion"]]
    correct_logprobs = []
    for part in chunks_all[0]:
        correct_logprobs.append(create_chat_completion_chunk(part))

    incorrect_logprobs = []
    wrong_logprobs = {"wrong": -0.1, "log": -5.43, "prob": -8.54, "defined": -11, "<|endoftext|>": -3000000}
    for part in chunks_all[1]:
        incorrect_logprobs.append(create_chat_completion_chunk(part, logprobs=wrong_logprobs))

    empty_logprobs = []
    for part in chunks_all[2]:
        empty_logprobs.append(create_chat_completion_chunk(part, logprobs={}))

    chunk_dicts_raw = [correct_logprobs, incorrect_logprobs, empty_logprobs]
    response_event = await create_response_event_mock(chunks_all, chunks_timings_all)
    response_event.stream_results_all_chunk_dicts_raw = chunk_dicts_raw

    def mock_verify_sim(original_logits, verification_logits):
        return VERIFICATION_THRESH_SIM * 0.9 if "wrong" in original_logits else VERIFICATION_THRESH_SIM * 1.1

    with (
        patch("prompting.rewards.exact_match.MIN_VERIFY_TOKENS", 2),
        patch("prompting.rewards.exact_match.LogitsRewardModel.verify_logit_similarity", side_effect=mock_verify_sim),
        patch("prompting.rewards.exact_match.LogitsRewardModel.verify_logit_contains", return_value=1),
    ):
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="", response_event=response_event, task=task, model_manager=model_manager
        )

        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 3
        assert 0.3 < result.rewards[0] <= 0.9
        assert result.rewards[1] == INCORRECT_PENALTY
        assert result.rewards[2] == INCORRECT_PENALTY


@pytest.mark.asyncio
async def test_no_eos_token(model_manager, task):
    """Test case 3: Missing eos_token in logits -> zero reward."""
    chunks = [["Hello", ", ", "world", "!"]]
    timings = [[0.1, 0.2, 0.3, 0.4]]
    response_event = await create_response_event_mock(chunks, timings)

    async def mock_generate_logits_no_eos(*args, **kwargs):
        return {"token1": -0.1, "token2": -0.5}, "prompt"

    model_manager.generate_logits = AsyncMock(side_effect=mock_generate_logits_no_eos)

    # Replace last chunk without eos in its logprobs.
    response_event.stream_results_all_chunk_dicts_raw[0][3] = create_chat_completion_chunk(
        "!", {"token1": -0.1, "token2": -0.5}
    )

    with patch("prompting.rewards.exact_match.LogitsRewardModel.verify_logit_similarity", return_value=0.95):
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="", response_event=response_event, task=task, model_manager=model_manager
        )
        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 1
        assert result.rewards[0] == NO_EOS_PENALTY


def test_verify_logit_similarity():
    """Test the verify_logit_similarity similarity metric."""
    original = {"token1": -0.1, "token2": -0.5, "token3": -1.0, "token4": -1.5, "token5": -2.0}
    # Identical distributions -> 1.0.
    assert LogitsRewardModel.verify_logit_similarity(original, original) == 1.0

    # Disjoint tokens -> near zero.
    disjoint = {"foo": -0.1, "bar": -0.5, "foo1": -1.0, "bar1": -1.5, "foo2": -2.0}
    sim = LogitsRewardModel.verify_logit_similarity(original, disjoint)
    assert 0 <= sim <= 0.01

    # Partial overlap -> between 0 and 1.
    partial = {"token1": -0.1, "token2": -0.5, "token3": -1.0, "foo1": -1.5, "bar1": -2.0}
    sim2 = LogitsRewardModel.verify_logit_similarity(original, partial)
    assert sim2 == 0.6


def test_smooth_reward_scale():
    """Test the smooth_reward_scale function under various conditions."""
    # Test empty timings list.
    assert LogitsRewardModel.smooth_timings_reward([]) == 0.0

    # Test uniform timings (should give maximum reward).
    uniform_timings = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert LogitsRewardModel.smooth_timings_reward(uniform_timings) == 1.0

    # Test high variance timings (should give minimum reward).
    high_var_timings = [0.1, 5.0, 10.0, 0.5, 8.0]
    std_dev = np.std(high_var_timings)
    assert LogitsRewardModel.smooth_timings_reward(high_var_timings) == MIN_SMOOTH_PENALTY_SCALE
    assert 1.0 - std_dev < MIN_SMOOTH_PENALTY_SCALE

    # Test moderate variance timings.
    moderate_var_timings = [0.9, 1.0, 1.1, 0.95, 1.05]
    expected = max(MIN_SMOOTH_PENALTY_SCALE, 1.0 - np.std(moderate_var_timings))
    assert LogitsRewardModel.smooth_timings_reward(moderate_var_timings) == pytest.approx(expected)
    assert MIN_SMOOTH_PENALTY_SCALE < LogitsRewardModel.smooth_timings_reward(moderate_var_timings) < 1.0

    # Test with custom minimum reward.
    custom_min = 0.8
    assert LogitsRewardModel.smooth_timings_reward(high_var_timings, min_reward=custom_min) == custom_min

    # Test with single timing value.
    single_timing = [1.5]
    assert LogitsRewardModel.smooth_timings_reward(single_timing) == 1.0


@pytest.mark.parametrize(
    "value, min_value, expected",
    [
        # Linear mapping.
        (0.6, 0.2, (0.6 - 0.2) / (1.0 - 0.2)),
        # Below min clips to 0.0.
        (0.1, 0.3, 0.0),
        # Above max clips to 1.0.
        (1.2, 0.0, 1.0),
        # At min boundary.
        (0.3, 0.3, 0.0),
        # At max boundary.
        (1.0, 0.3, 1.0),
    ],
)
def test_rescale_various_cases(value, min_value, expected):
    assert LogitsRewardModel.rescale(value, min_value=min_value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        # All valid.
        ([[0.1, 1.0], [5.0, 0.1], [6.5]], 0.55),
        # Mixed values.
        ([[-1.0, 0.5], [2.0, 0.1]], 1.05),
        # All negative.
        ([[-3.0, -0.1], [-2.5]], 1e-6),
        # Empty lists.
        ([[], []], 1e-6),
        # Zeros included.
        ([[0.0, -1.0], [0.0]], 0.0),
    ],
)
def test_fastest_timing_various_cases(values, expected):
    assert LogitsRewardModel.fastest_timing(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    "completion_length",
    [
        5,
        (MIN_VERIFY_TOKENS + MAX_VERIFY_TOKENS) // 2,
        MAX_VERIFY_TOKENS,
        MAX_VERIFY_TOKENS + 5,
    ],
)
def test_sample_verification_indices_properties(completion_length):
    indices = LogitsRewardModel.sample_verification_indices(completion_length)

    # Compute expected number of sampled tokens (before adding EOS)
    expected_k = int(np.clip(completion_length, 1, MAX_VERIFY_TOKENS)) + 1

    # The result should have expected_k samples plus one EOS index
    assert isinstance(indices, list)
    assert len(indices) == expected_k
    assert indices == sorted(indices)
    assert indices[-1] == completion_length
    # All other indices should be in the range [0, completion_length).
    sample_indices = indices[:-1]
    assert all(0 <= idx < completion_length for idx in sample_indices)
    # No duplicates overall.
    assert len(set(indices)) == len(indices)