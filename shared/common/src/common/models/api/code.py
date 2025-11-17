from datetime import datetime
from pydantic import BaseModel

from common.models.api.file import FilePagination


class CodeRequest(BaseModel):
    competition_id: int
    round_number: int | None = None
    hotkey: str
    version: int | None = None
    start_idx: int = 0


class CodeResponse(BaseModel):
    version: int
    language: str
    code: str
    submit_at: datetime
    pagination: FilePagination
