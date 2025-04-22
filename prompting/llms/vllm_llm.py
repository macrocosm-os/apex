import gc
import random

import numpy as np
import torch
from loguru import logger
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel


class ReproducibleVLLM:
    def __init__(
        self,
        model_id: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ",
        device: str = "cuda:0",
        sampling_params: dict[str, str | float | int | bool] | None = None,
    ):
        """Deterministic VLLM model."""
        self._device = device
        self.model_id = model_id
        self.sampling_params = {} if sampling_params else sampling_params

        # VLLM specific initialization
        # gpu_memory_utilization = 0.9  # Default high utilization since VLLM is memory efficient
        self.model = LLM(
            model=model_id,
            # tensor_parallel_size=1,  # Single GPU by default
            # dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )

        # Store tokenizer from VLLM for consistency
        self.tokenizer = self.model.get_tokenizer()

    async def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
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

        # Convert sampling parameters to VLLM format
        params = sampling_params if sampling_params else self.sampling_params
        vllm_params = SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_new_tokens", 100),
            presence_penalty=params.get("presence_penalty", 0.0),
            frequency_penalty=params.get("frequency_penalty", 0.0),
            top_k=params.get("top_k", -1),
            logprobs=params.get("logprobs", None),
        )

        # Generate using VLLM
        outputs = self.model.generate(prompt, vllm_params)

        if not outputs:
            return ""

        # Return just the generated text without the prompt
        result = outputs[0].outputs[0].text
        return result

    async def generate_logits(
        self,
        messages: list[str] | list[dict[str, str]],
        top_n: int = 10,
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> dict[str, float]:
        """
        Generate logits for the next token prediction.

        Args:
            messages: Input messages or text
            top_n: Number of top logits to return (default: 10)
            sampling_params: Generation parameters
            seed: Random seed for reproducibility
            continue_last_message: Whether to continue the last message in chat format

        Returns:
            dict: Dictionary mapping tokens to their log probabilities
        """
        self.set_random_seeds(seed)
        # Set up sampling parameters for logit generation
        params = sampling_params if sampling_params else self.sampling_params
        params["max_tokens"] = 1
        params["logprobs"] = 10
        vllm_params = SamplingParams(
            **params,
        )

        # Generate using VLLM
        trailing_space = ""

        if continue_last_message and messages[-1]["content"]:
            content = messages[-1]["content"]
            stripped = content.rstrip()
            if len(content) > len(stripped):
                trailing_space = content[len(stripped) :]

        if not messages[-1]["content"]:
            continue_last_message = False
            messages = messages[:-1]
        bot_token_id = self.tokenizer.bos_token_id
        if not continue_last_message:
            messages[-1]["content"] = f"{self.tokenizer.decode([bot_token_id])}{messages[-1]['content']}"

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=not continue_last_message,
            continue_final_message=continue_last_message,
        )
        if trailing_space:
            prompt += trailing_space

        outputs = self.model.generate(prompt, vllm_params)

        if not outputs or not outputs[0].outputs[0].logprobs:
            return {}

        # Extract logprobs from the first token
        logprobs = outputs[0].outputs[0].logprobs[0]

        logprobs_list = [(k, v.logprob) for k, v in logprobs.items()]
        sorted_logprobs = sorted(logprobs_list, key=lambda x: x[1], reverse=True)

        top_token_ids = [x[0] for x in sorted_logprobs]
        top_logprob_values = [x[1] for x in sorted_logprobs]

        step_logprobs = {
            "top_tokens": [self.tokenizer.decode([tid]) for tid in top_token_ids],
            "top_logprobs": top_logprob_values,
        }

        # Create dictionary of token to logprob mapping
        token_logprobs = {
            token: logprob for token, logprob in zip(step_logprobs["top_tokens"], step_logprobs["top_logprobs"])
        }

        return token_logprobs, prompt

    def set_random_seeds(self, seed: int | None = 42):
        """Set random seeds for reproducibility across all relevant libraries."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def unload_model(self):
        try:
            destroy_model_parallel()
            if hasattr(self.model, "llm_engine") and hasattr(self.model.llm_engine, "driver_worker"):
                del self.model.llm_engine.driver_worker
            if hasattr(self.model, "model"):
                del self.model
            if hasattr(self.model, "tokenizer"):
                del self.tokenizer

            gc.collect()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Successfully deleted the LLM pipeline and freed GPU memory")
        except BaseException as e:
            logger.error(f"An error occurred during model unloading: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        self.unload_model()

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        return messages
