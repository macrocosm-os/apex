from datetime import datetime
from pydantic import BaseModel, model_validator
from typing import Optional


class SubmitRequest(BaseModel):
    competition_id: int
    round_number: int = 0
    raw_code: Optional[str] = None  # Text-based submissions (e.g., .py files)
    raw_binary: Optional[str] = None  # Base64-encoded binary submissions (e.g., .pt files)
    file_extension: str = ".py"  # File extension for the submission

    @model_validator(mode="after")
    def validate_content(self) -> "SubmitRequest":
        if self.raw_code is None and self.raw_binary is None:
            raise ValueError("Either raw_code or raw_binary must be provided")
        if self.raw_code is not None and self.raw_binary is not None:
            raise ValueError("Only one of raw_code or raw_binary should be provided")
        return self

    @property
    def is_binary(self) -> bool:
        return self.raw_binary is not None


class SubmitResponse(BaseModel):
    submission_id: int


class SubmissionRequest(BaseModel):
    submission_id: Optional[int] = None
    competition_id: Optional[int] = None
    hotkey: Optional[str] = None
    start_idx: int = 0
    count: int = 10
    filter_mode: str = "all"
    sort_mode: str = "score"


class SubmissionRecord(BaseModel):
    id: int
    competition_id: int
    round_number: int
    state: str
    hotkey: str
    version: int
    top_score: bool
    submit_at: datetime
    eval_at: Optional[datetime] = None
    reveal_at: Optional[datetime] = None
    eval_raw_score: Optional[float] = None
    eval_score: Optional[float] = None
    eval_time_in_seconds: Optional[float] = None
    eval_error: Optional[str] = None


class SubmissionDetail(BaseModel):
    id: int
    submit_metadata: dict | None = None
    eval_metadata: dict | None = None
    eval_file_paths: dict | None = None
    code_path: str | None = None
    round_number: int
    reveal_at: Optional[datetime] = None
    is_binary: bool = False
    language: str | None = None


class SubmissionPagination(BaseModel):
    start_idx: int
    count: int
    total: int
    has_more: bool


class SubmissionResponse(BaseModel):
    submissions: list[SubmissionRecord]
    pagination: SubmissionPagination


class FileRequest(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    start_idx: int = 0
    reverse: bool = False
