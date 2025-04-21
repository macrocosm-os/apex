import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage
from openai.types import Completion

from prompting.rewards.exact_match import (
    INCORRECT_PENALTY,
    MIN_SMOOTH_REWARD,
    VERIFICATION_THRESHOLD,
    LogitsRewardModel,
    normalize_timing,
    smooth_timings_reward,
    verify_single_logit,
)
from prompting.rewards.reward import BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from prompting.llms.model_manager import ModelManager
from shared.dendrite import DendriteResponseEvent
from loguru import logger


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
        {"role": "user", "content": "Tell me a joke."}
    ]
    task.sampling_params = {"temperature": 0.7, "max_tokens": 100}
    return task


def create_chat_completion_chunk(content="", logprobs=None):
    """Helper function to create a ChatCompletionChunk object."""
    if logprobs is None:
        logprobs = {"token1": -0.1, "token2": -0.5, "token3": -0.6, "token4": -0.7, "<|endoftext|>": -1.0}
    
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


@pytest.mark.asyncio
async def test_ideal_completion(model_manager, task):
    """Test case 1: Ideal completion with reward >0.5 and ≤1."""
    chunks = [["Hello", ", ", "world", "!"]]
    chunk_dicts_raw = [[
        create_chat_completion_chunk("Hello"),
        create_chat_completion_chunk(", "),
        create_chat_completion_chunk("world"),
        create_chat_completion_chunk("!")
    ]]
    
    with patch('prompting.rewards.exact_match.verify_single_logit', return_value=0.95):
        response_event = MagicMock(spec=DendriteResponseEvent)
        response_event.stream_results_all_chunks = chunks
        response_event.stream_results_all_chunk_dicts_raw = chunk_dicts_raw
        response_event.uids = [1]
        response_event.stream_results_all_chunks_timings = [[0.1, 0.2, 0.3, 0.4]]
        response_event.completions = ["Hello, world!"]
        response_event.timeout = 10.0
        
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="",
            response_event=response_event,
            task=task,
            model_manager=model_manager
        )
        
        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 1
        assert 0.2 < result.rewards[0] <= 0.4


@pytest.mark.asyncio
async def test_mixed_completions(model_manager, task):
    """Test case 2: One ideal completion, one with missing logprobs penalized."""
    chunks = [
        ["Hello", ", ", "world", "!"],
        ["Fail", "ed", " ", "completion"],
        ["Wro", "ng", " ", "completion"]
    ]
    correct_logprobs = []
    for part in chunks[0]:
        correct_logprobs.append(create_chat_completion_chunk(part))
    
    incorrect_logprobs = []
    wrong_logprobs = {"wrong": -0.1, "log": -5.43, "prob": -8.54, "defined": -11, "<|endoftext|>": -3000000}
    for part in chunks[1]:
        incorrect_logprobs.append(
            create_chat_completion_chunk(part, logprobs=wrong_logprobs)
        )
    empty_logprobs = []
    for part in chunks[1]:
        empty_logprobs.append(
            create_chat_completion_chunk(part, logprobs={})
        )
    chunk_dicts_raw = [correct_logprobs, incorrect_logprobs, empty_logprobs]
    
    # Mock verify_single_logit to return different values based on input
    def mock_verify(original_logits, verification_logits):
        # Check if this is the incorrect logprobs case
        if "wrong" in original_logits:
            return VERIFICATION_THRESHOLD * 0.9
        else:
            return VERIFICATION_THRESHOLD * 1.1
    
    with patch("prompting.rewards.exact_match.verify_single_logit", side_effect=mock_verify):
        response_event = MagicMock(spec=DendriteResponseEvent)
        response_event.stream_results_all_chunks = chunks
        response_event.stream_results_all_chunk_dicts_raw = chunk_dicts_raw
        response_event.uids = [1, 2, 3]
        response_event.stream_results_all_chunks_timings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
        response_event.completions = ["Hello, world!", "Missing logprobs", "Empty logprobs"]
        response_event.timeout = 10.0
        
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="",
            response_event=response_event,
            task=task,
            model_manager=model_manager
        )
        
        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 3
        assert 0.2 < result.rewards[0] <= 0.5
        assert result.rewards[1] == 0
        assert result.rewards[2] == -INCORRECT_PENALTY


@pytest.mark.asyncio
async def test_no_eos_token(model_manager, task):
    """Test case 3: Missing eos_token in logits → zero reward."""
    chunks = [["Hello", ", ", "world", "!"]]
    chunk_dicts_raw = [[
        create_chat_completion_chunk("Hello"),
        create_chat_completion_chunk(", "),
        create_chat_completion_chunk("world"),
        create_chat_completion_chunk("!")
    ]]
    
    async def mock_generate_logits_no_eos(*args, **kwargs):
        return {"token1": -0.1, "token2": -0.5}, "prompt"
    model_manager.generate_logits = AsyncMock(side_effect=mock_generate_logits_no_eos)
    
    # Replace last chunk without eos in its logprobs
    chunk_dicts_raw[0][3] = create_chat_completion_chunk("!", {"token1": -0.1, "token2": -0.5})
    
    with patch('prompting.rewards.exact_match.verify_single_logit', return_value=0.95):
        response_event = MagicMock(spec=DendriteResponseEvent)
        response_event.stream_results_all_chunks = chunks
        response_event.stream_results_all_chunk_dicts_raw = chunk_dicts_raw
        response_event.uids = [1]
        response_event.stream_results_all_chunks_timings = [[0.1, 0.2, 0.3, 0.4]]
        response_event.completions = ["Hello, world!"]
        response_event.timeout = 10.0
        
        reward_model = LogitsRewardModel()
        result = await reward_model.reward(
            reference="",
            response_event=response_event,
            task=task,
            model_manager=model_manager
        )
        
        assert isinstance(result, BatchRewardOutput)
        assert len(result.rewards) == 1
        assert result.rewards[0] == 0.0


def test_verify_single_logit():
    """Test the verify_single_logit similarity metric."""
    original = {"token1": -0.1, "token2": -0.5}
    # Identical distributions → 1.0
    assert verify_single_logit(original, original) == 1.0

    # Disjoint tokens → near zero
    disjoint = {"foo": -0.1, "bar": -0.5}
    sim = verify_single_logit(original, disjoint)
    assert 0 <= sim <= 0.01

    # Partial overlap → between 0 and 1
    partial = {"token1": -0.1, "foo": -0.5}
    sim2 = verify_single_logit(original, partial)
    assert 0 < sim2 < 1.0


def test_smooth_reward_scale():
    """Test the smooth_reward_scale function under various conditions."""
    # Test empty timings list.
    assert smooth_timings_reward([]) == 0.0

    # Test uniform timings (should give maximum reward).
    uniform_timings = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert smooth_timings_reward(uniform_timings) == 1.0

    # Test high variance timings (should give minimum reward).
    high_var_timings = [0.1, 5.0, 10.0, 0.5, 8.0]
    std_dev = np.std(high_var_timings)
    assert smooth_timings_reward(high_var_timings) == MIN_SMOOTH_REWARD
    assert 1.0 - std_dev < MIN_SMOOTH_REWARD
    
    # Test moderate variance timings
    moderate_var_timings = [0.9, 1.0, 1.1, 0.95, 1.05]
    expected = max(MIN_SMOOTH_REWARD, 1.0 - np.std(moderate_var_timings))
    assert smooth_timings_reward(moderate_var_timings) == pytest.approx(expected)
    assert MIN_SMOOTH_REWARD < smooth_timings_reward(moderate_var_timings) < 1.0
    
    # Test with custom minimum reward.
    custom_min = 0.8
    assert smooth_timings_reward(high_var_timings, min_reward=custom_min) == custom_min
    
    # Test with single timing value.
    single_timing = [1.5]
    assert smooth_timings_reward(single_timing) == 1.0
