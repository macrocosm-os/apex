from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    content: str
    role: Literal["user", "assistant", "system"]


class SamplingParameters(BaseModel):
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    top_k: Optional[int] = -1
    logprobs: Optional[int] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    seed: Optional[int]
    sampling_parameters: Optional[SamplingParameters] = SamplingParameters()
    continue_last_message: Optional[bool] = False


class LogitsRequest(ChatRequest):
    top_logprobs: Optional[int] = 10
