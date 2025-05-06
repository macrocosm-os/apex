from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class ContentItem(BaseModel):
    """Model for content items in the new message format."""

    type: Literal["text", "image"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """Model for chat messages with support for both legacy and new formats."""

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentItem]]

    def to_legacy_format(self) -> Dict[str, str]:
        """Convert to legacy format with string content."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        # For list of content items, concatenate text content
        text_content = " ".join([item.text for item in self.content if item.type == "text" and item.text])
        return {"role": self.role, "content": text_content}

    @classmethod
    def from_legacy_format(cls, message: Dict[str, str]) -> "Message":
        """Convert from legacy format (simple dict with role/content) to new format with ContentItem list."""
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("Invalid legacy message format")

        # Convert string content to a list with a single text ContentItem
        if isinstance(message["content"], str):
            content = [ContentItem(type="text", text=message["content"])]
            return cls(role=message["role"], content=content)

        # If content is already a list, assume it's already in the new format
        return cls(**message)


class Messages(BaseModel):
    """Model for a list of messages with support for both legacy and new formats."""

    messages: List[Union[Dict[str, Any], Message]]

    def to_legacy_format(self) -> List[Dict[str, str]]:
        """Convert all messages to legacy format for backward compatibility."""
        legacy_messages = []
        for message in self.messages:
            if isinstance(message, Message):
                legacy_messages.append(message.to_legacy_format())
            elif isinstance(message, dict) and "role" in message and "content" in message:
                legacy_messages.append(message)
        return legacy_messages

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Messages":
        """Create a Messages object from a dictionary."""
        if "messages" in data:
            messages_data = data["messages"]
        else:
            messages_data = data.get("messages", [])

        processed_messages = []
        for msg in messages_data:
            if isinstance(msg, dict):
                if "role" in msg and "content" in msg:
                    # Check if this is a legacy format message (content is string)
                    if isinstance(msg["content"], str):
                        processed_messages.append(Message.from_legacy_format(msg))
                    else:
                        # Already in new format
                        processed_messages.append(Message(**msg))
                else:
                    # Skip invalid message formats
                    continue
            elif isinstance(msg, Message):
                processed_messages.append(msg)

        return cls(messages=processed_messages)
