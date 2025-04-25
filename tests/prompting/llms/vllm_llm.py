from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from prompting.llms.vllm_llm import ReproducibleVLLM


def _fake_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.side_effect = (
        lambda conversation, tokenize, add_generation_prompt, continue_final_message:
        f"TEMPLATE::{conversation[-1]['role']}::{conversation[-1]['content']}"
    )
    tok.decode.side_effect = lambda ids: "<s>" if ids == [0] else f"tok{ids[0]}"
    return tok


def _fake_llm(return_logprobs):
    out_obj = SimpleNamespace(
        outputs=[SimpleNamespace(
            text="dummy",
            logprobs=[return_logprobs]
        )]
    )
    llm = MagicMock()
    llm.generate.return_value = [out_obj]
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages, continue_last",
    [
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}], False),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "So"}], True),
    ],
    ids=["new_message", "continue_message"],
)
async def test_generate_logits(monkeypatch, messages, continue_last):
    fake_logprobs = {
        3: SimpleNamespace(logprob=-0.1),
        2: SimpleNamespace(logprob=-0.5),
        1: SimpleNamespace(logprob=-1.0)
    }

    tokenizer_stub = _fake_tokenizer()
    llm_stub = _fake_llm(fake_logprobs)

    with (
        patch("prompting.llms.vllm_llm.LLM", return_value=llm_stub),
        patch("prompting.llms.vllm_llm.SamplingParams", lambda **kw: kw)
    ):
        model = ReproducibleVLLM(model_id="mock-model")
        # Swap tokenizer (LLM stub has none).
        model.tokenizer = tokenizer_stub

        out_dict, rendered_prompt = await model.generate_logits(
            messages=messages,
            sampling_params={"max_tokens": 1, "logprobs": 3},
            top_n=3,
            continue_last_message=continue_last,
        )

    # 1. Tokenizer called exactly once and produced the prompt we got back.
    tokenizer_stub.apply_chat_template.assert_called_once()
    assert rendered_prompt.startswith("TEMPLATE::assistant::")

    # 2. Returned dict is sorted by descending log-prob.
    expected_tokens = ["tok3", "tok2", "tok1"]
    assert list(out_dict.keys()) == expected_tokens
    assert all(a >= b for a, b in zip(out_dict.values(), list(out_dict.values())[1:]))

    # 3. generate() was invoked with that exact prompt.
    llm_stub.generate.assert_called_once_with(rendered_prompt, {'max_tokens': 1, 'logprobs': 3})
