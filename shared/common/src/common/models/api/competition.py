from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from common.settings import DEFAULT_BASE_BURN_RATE
from common.models.api.submission import SubmissionPagination


class CompetitionRequest(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    pkg: Optional[str] = None
    ptype: Optional[str] = None
    ctype: Optional[str] = None
    state: Optional[str] = None
    start_idx: int = 0
    count: int = 10


class RoundRecord(BaseModel):
    id: int
    competition_id: int
    round_number: int
    state: str
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    competed_at: Optional[datetime] = None
    submit_at: Optional[datetime] = None  # For ordering purposes


class CompetitionRecord(BaseModel):
    id: int
    name: str
    description: str
    state: str
    pkg: str
    ptype: str
    ctype: str
    baseline_score: float
    baseline_raw_score: float
    incentive_weight: float
    burn_factor: float
    burn_factor_reset_at: Optional[datetime] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Current round details
    curr_round_id: Optional[int] = None
    curr_round_number: Optional[int] = None
    curr_round: Optional[RoundRecord] = None
    # Top score details
    curr_top_score_id: Optional[int] = None
    top_score_value: Optional[float] = None
    top_scorer_hotkey: Optional[str] = None
    score_to_beat: Optional[float] = None
    total_submissions: int = 0
    image_url: Optional[str] = None
    base_burn_rate: float = DEFAULT_BASE_BURN_RATE


class CompetitionResponse(BaseModel):
    competitions: list[CompetitionRecord]
    pagination: SubmissionPagination
