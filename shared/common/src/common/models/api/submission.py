from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class SubmitRequest(BaseModel):
    competition_id: int
    round_number: int
    raw_code: str


class SubmissionRequest(BaseModel):
    submission_id: Optional[int] = None
    competition_id: Optional[int] = None
    hotkey: Optional[str] = None
    start_idx: int = 0
    count: int = 10


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
