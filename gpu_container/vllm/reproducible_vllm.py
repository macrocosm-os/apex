import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

tqdm.disable = True


class ReproducibleVLLM:
    def __init__(
        self,
        model_id: str = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf",
        device: str = "cuda:0",
        sampling_params: Optional[Dict[str, Union[str, float, int, bool]]] = None,
    ):
        """Deterministic VLLM model."""
        self._device = device
        self.model_id = model_id
        self.sampling_params = {} if sampling_params is None else sampling_params

        self.model = LLM(
            model=model_id,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )

        # Store tokenizer from VLLM for consistency
        self.tokenizer = self.model.get_tokenizer()

    @classmethod
    async def get_max_tokens(
        cls,
        sampling_params: Dict[str, Union[str, float, int, bool]],
        default_value: int = 512,
    ) -> int:
        # Process max tokens with backward compatibility.
        max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_new_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_completion_tokens", default_value)
        return max_tokens

    @classmethod
    async def prepare_sampling_params(
        cls, sampling_params: Optional[Dict[str, Union[str, float, int, bool]]] = None
    ) -> SamplingParams:
        sampling_params = sampling_params or {}
        max_tokens = await cls.get_max_tokens(sampling_params)

        params = SamplingParams(
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            max_tokens=int(max_tokens),
            presence_penalty=float(sampling_params.get("presence_penalty", 0.0)),
            frequency_penalty=float(sampling_params.get("frequency_penalty", 0.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            logprobs=sampling_params.get("logprobs", None),
        )
        return params

    async def generate(
        self,
        messages: Union[List[str], List[Dict[str, str]]],
        sampling_params: Optional[Dict[str, Union[str, float, int, bool]]] = None,
        seed: Optional[int] = None,
        continue_last_message: bool = False,
    ) -> str:
        """Generate text with optimized performance using VLLM."""
        self.set_random_seeds(seed)

        # Convert chat messages to prompt string using tokenizer's chat template
        if isinstance(messages, list) and isinstance(messages[0], dict):
            try:
                # Extract any trailing whitespace before applying template
                trailing_space = ""
                if continue_last_message and messages[-1]["content"]:
                    content = messages[-1]["content"]
                    stripped = content.rstrip()
                    if len(content) > len(stripped):
                        trailing_space = content[len(stripped) :]

                # Try using the tokenizer's chat template
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=not continue_last_message,
                    continue_final_message=continue_last_message,
                )

                # Append back just the trailing whitespace if it was stripped
                if trailing_space:
                    prompt += trailing_space
            except (AttributeError, NotImplementedError):
                raise ValueError(f"Chat template not supported for model {self.model_id}")
        else:
            prompt = messages[0] if isinstance(messages, list) else messages

        # Convert sampling parameters to vLLM format.
        params = sampling_params if sampling_params is not None else self.sampling_params
        vllm_params = await self.prepare_sampling_params(params)
        outputs = self.model.generate(prompt, vllm_params)

        if not outputs:
            return ""

        result = outputs[0].outputs[0].text
        return {"choices": [{"message": {"content": result}}]}

    async def generate_logits(
        self,
        messages: Union[List[str], List[Dict[str, str]]],
        top_logprobs: int = 10,
        sampling_params: Optional[Dict[str, Union[str, float, int, bool]]] = None,
        seed: Optional[int] = None,
        continue_last_message: bool = False,
    ) -> dict[str, float]:
        """Generate logits for the next token prediction.

        Args:
            messages: Input messages or text.
            top_logprobs: Number of top logits to return (default: 10).
            sampling_params: Generation parameters.
            seed: Random seed for reproducibility.
            continue_last_message: Whether to continue the last message in chat format.

        Returns:
            Dictionary mapping tokens to their log probabilities.
        """
        self.set_random_seeds(seed)
        params = sampling_params if sampling_params is not None else self.sampling_params
        params = params.copy()
        params["max_tokens"] = 1
        params["logprobs"] = top_logprobs
        vllm_params = await self.prepare_sampling_params(params)

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=not continue_last_message,
            continue_final_message=continue_last_message,
        )

        outputs = self.model.generate(prompt, vllm_params)

        if not outputs or not outputs[0].outputs[0].logprobs:
            return {}

        logprobs = outputs[0].outputs[0].logprobs[0]
        token_logprobs = {self.tokenizer.decode([token]): logprob.logprob for token, logprob in logprobs.items()}
        sorted_token_logprobs = dict(sorted(token_logprobs.items(), key=lambda item: item[1], reverse=True))
        return sorted_token_logprobs, prompt

    def set_random_seeds(self, seed: Optional[int] = 42):
        """Set random seeds for reproducibility across all relevant libraries."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def format_messages(
        messages: Union[List[str], List[Dict[str, str]]],
    ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        return messages
