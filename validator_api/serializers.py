from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from validator_api.messages import Message, Messages


class CompletionsRequest(BaseModel):
    """Request model for the /v1/chat/completions endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        example=[1, 2, 3],
    )
    messages: Union[List[Dict[str, Any]], Messages] = Field(
        ...,
        description="List of message objects with 'role' and 'content' keys. Content can be a string or a list of ContentItems for multimodal inputs.",
        example=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me about this image"},
                    {"type": "image", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ],
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

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Custom validation to handle messages conversion."""
        if isinstance(obj, dict) and "messages" in obj:
            # Convert messages to a Messages object if it's a dictionary
            if isinstance(obj["messages"], list):
                # Process each message to ensure it uses the new format
                processed_messages = []
                for msg in obj["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        # If content is a string, convert to new format
                        if isinstance(msg["content"], str):
                            msg = {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                    processed_messages.append(msg)
                obj["messages"] = Messages(messages=processed_messages)
            elif isinstance(obj["messages"], dict):
                obj["messages"] = Messages.from_dict(obj["messages"])
        return super().model_validate(obj, *args, **kwargs)

    def get_legacy_messages(self) -> List[Dict[str, str]]:
        """Get messages in legacy format."""
        if isinstance(self.messages, Messages):
            return self.messages.to_legacy_format()

        # If we have a list of messages, convert each to legacy format as needed
        legacy_messages = []
        for msg in self.messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if isinstance(msg["content"], list):
                    # Convert content list to string (join text items)
                    text_content = " ".join(
                        [
                            item.get("text", "")
                            for item in msg["content"]
                            if item.get("type") == "text" and item.get("text")
                        ]
                    )
                    legacy_messages.append({"role": msg["role"], "content": text_content})
                else:
                    # Already in legacy format
                    legacy_messages.append(msg)
            elif isinstance(msg, Message):
                legacy_messages.append(msg.to_legacy_format())

        return legacy_messages

    # Add model_validator to ensure messages are always a Messages object
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if isinstance(v, list):
            # Process each message to ensure it uses the new format
            processed_messages = []
            for msg in v:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # If content is a string, convert to new format
                    if isinstance(msg["content"], str):
                        msg = {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                processed_messages.append(msg)
            return Messages(messages=processed_messages)
        return v


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


if __name__ == "__main__":
    request = CompletionsRequest(messages=[{"role": "user", "content": "Tell me about neural networks"}])
    print(request.get_legacy_messages())
    print(request.messages)
    print(request.model_dump())
