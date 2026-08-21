from typing import Optional

from pydantic import BaseModel


class SubmissionRankRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 100
    hotkey: Optional[str] = None
    round_number: Optional[int] = None
