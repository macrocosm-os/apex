import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from validator_api.job_store import JobStatus


class CompletionsRequest(BaseModel):
    """Request model for the /v1/chat/completions endpoint."""

    @model_validator(mode="after")
    def add_tools(self):
        if self.tools:
            self.messages.append({"role": "tool", "content": json.dumps(self.tools)})
        return self

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        example=[1, 2, 3],
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description="List of message objects with 'role' and 'content' keys. Roles can be 'system', 'user', or 'assistant'.",
        example=[{"role": "user", "content": "Tell me about neural networks"}],
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results. If not provided, a random seed will be generated.",
        example=42,
    )
    task: Optional[str] = Field(
        default="InferenceTask", description="Task identifier to choose the inference type.", example="InferenceTask"
    )
    model: Optional[str] = Field(
        default="Default",
        description="Model identifier to filter available miners.",
        example="Default",
    )
    test_time_inference: bool = Field(
        default=False, description="Enable step-by-step reasoning mode that shows the model's thinking process."
    )
    mixture: bool = Field(
        default=False, description="Enable mixture of miners mode that combines responses from multiple miners."
    )
    sampling_parameters: Optional[Dict[str, Any]] = Field(
        default={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 1024,
            "do_sample": True,
        },
        description="Parameters to control text generation, such as temperature, top_p, etc.",
        example={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 512,
            "do_sample": True,
        },
    )
    inference_mode: Optional[str] = Field(
        default=None,
        description="Inference mode to use for the task.",
        example="Chain-of-Thought",
    )
    json_format: bool = Field(default=False, description="Enable JSON format for the response.", example=True)
    stream: bool = Field(default=False, description="Enable streaming for the response.", example=True)
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of tools to use for the task.",
        # TODO: Add example that's not just from claude
        example=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {"timezone": {"type": "string", "description": "The timezone to get the time in"}},
                },
            }
        ],
    )


class WebRetrievalRequest(BaseModel):
    """Request model for the /web_retrieval endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        example=[1, 2, 3],
    )
    search_query: str = Field(
        ..., description="The query to search for on the web.", example="latest advancements in quantum computing"
    )
    n_miners: int = Field(default=3, description="Number of miners to query for results.", example=15, ge=1)
    n_results: int = Field(
        default=1, description="Maximum number of results to return in the response.", example=5, ge=1
    )
    max_response_time: int = Field(
        default=10, description="Maximum time to wait for responses in seconds.", example=15, ge=1
    )


class WebSearchResult(BaseModel):
    """Model for a single web search result."""

    url: str = Field(..., description="The URL of the web page.", example="https://example.com/article")
    content: Optional[str] = Field(
        default=None,
        description="The relevant content extracted from the page.",
        example="Quantum computing has seen significant advancements in the past year...",
    )
    relevant: Optional[str] = Field(
        default=None,
        description="Information about why this result is relevant to the query.",
        example="This article discusses the latest breakthroughs in quantum computing research.",
    )


class WebRetrievalResponse(BaseModel):
    """Response model for the /web_retrieval endpoint."""

    results: List[WebSearchResult] = Field(..., description="List of unique web search results.")

    def to_dict(self):
        return self.model_dump().update({"results": [r.model_dump() for r in self.results]})


class TestTimeInferenceRequest(BaseModel):
    """Request model for the /test_time_inference endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        example=[1, 2, 3],
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description="List of message objects with 'role' and 'content' keys. Roles can be 'system', 'user', or 'assistant'.",
        example=[{"role": "user", "content": "Solve the equation: 3x + 5 = 14"}],
    )
    model: Optional[str] = Field(default=None, description="Model identifier to use for inference.", example="gpt-4")
    json_format: bool = Field(default=False, description="Enable JSON format for the response.", example=True)

    def to_dict(self):
        return self.model_dump().update({"messages": [m.model_dump() for m in self.messages]})


class JobResponse(BaseModel):
    """Response model for the /v1/chat/completions/jobs endpoint."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    created_at: str = Field(..., description="Timestamp when the job was created")
    updated_at: str = Field(..., description="Timestamp when the job was last updated")


class JobResultResponse(JobResponse):
    """Response model for the /v1/chat/completions/jobs/{job_id} endpoint."""

    result: Optional[List[str]] = Field(None, description="Result of the job if completed")
    error: Optional[str] = Field(None, description="Error message if the job failed")
