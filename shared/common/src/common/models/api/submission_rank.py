from pydantic import BaseModel  # type: ignore
from typing import Optional
from datetime import datetime
from common.models.api.submission import SubmissionPagination  # type: ignore


class SubmissionRankRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 100
    hotkey: Optional[str] = None
    round_number: Optional[int] = None


class SubmissionRankMiner(BaseModel):
    rank: int
    top_scorer: bool
    hotkey: str
    score: float
    score_render: float
    version: int
    round_number: int
    submission_date: datetime
    join_date: datetime
    submissions_count: int


class SubmissionRankResponse(BaseModel):
    competition_id: int
    incentive_weight_render: float
    curr_top_scorer_hotkey: Optional[str] = None
    miners: list[SubmissionRankMiner]
    pagination: SubmissionPagination
    total_submissions: int


class SubmissionRankCache(BaseModel):
    comp_row: dict
    miners: list[SubmissionRankMiner]
    scaled_incentive: float
    total_submissions: int
