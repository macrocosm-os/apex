from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    content: str
    role: Literal["user", "assistant", "system"]

class SamplingParameters(BaseModel):
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    max_new_tokens: Optional[int] = 1024
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[int] = 10

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    seed: Optional[int] = 42
    task: Optional[str] = "InferenceTask"
    mixture: Optional[bool] = False
    sampling_parameters: Optional[SamplingParameters] = SamplingParameters()
    json_format: Optional[bool] = False
    stream: Optional[bool] = False
    continue_last_message: Optional[bool] = False