from pydantic import BaseModel
from enum import Enum


class FileType(str, Enum):
    CODE = "code"
    LOG = "log"
    HISTORY = "history"


class JobResponse(BaseModel):
    submission_id: list[int]
    competition_id: int
    competition_name: str
    competition_pkg: str
    round_number: int
    hotkey: list[str]
    version: list[int]
    input_data: dict
    language: list[str]
    submit_metadata: list[dict]
    raw_code: list[str]


class JobResults(BaseModel):
    submission_id: int
    eval_metadata: dict = {}
    eval_error: str | None = None
    eval_time_in_seconds: float | None = None
    eval_raw_score: float | None = None
    eval_score: float | None = None
    eval_at: str | None = None


class JobFile(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    file_content: str
