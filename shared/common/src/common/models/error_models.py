from typing import Optional

from pydantic import BaseModel, model_validator


class FileSizeError(Exception):
    def __init__(self, message: str = "File size validation failed", **kwargs):
        super().__init__(message)
        self.message = message


class FileTooOldError(Exception):
    def __init__(self, message: str = "File is too old", **kwargs):
        super().__init__(message)
        self.message = message


class S3FileNotFoundError(Exception):
    def __init__(self, message: str = "S3 file not found", **kwargs):
        super().__init__(message)
        self.message = message


class EntityNotRegisteredError(BaseModel):
    message: str = "Entity not registered"
    name: Optional[str] = None


class BaseErrorModel(BaseModel):
    error_name: str | None = None
    error_dict: dict | None = None


class SpecVersionError(BaseModel):
    expected_version: int
    actual_version: str
    message: str = "Spec version mismatch"

    @model_validator(mode="after")
    def make_message(self):
        self.message = f"Spec version mismatch. Expected: {self.expected_version}, Received: {self.actual_version}"
        return self
