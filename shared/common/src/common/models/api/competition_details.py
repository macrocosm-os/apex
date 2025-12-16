from pydantic import BaseModel  # type: ignore
from typing import Optional, List
from datetime import datetime
from common.models.api.submission import SubmissionPagination


class CompetitionDetailsRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 100


class CompetitionInfo(BaseModel):
    id: int
    name: str
    description: str
    state: str
    pkg: str
    baseline_score: float
    baseline_raw_score: float
    incentive_weight: float
    burn_factor: float
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    image_url: Optional[str] = None


class RoundInfo(BaseModel):
    id: int
    competition_id: int
    round_number: int
    state: str
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    competed_at: Optional[datetime] = None
    submit_at: Optional[datetime] = None


class ScorePoint(BaseModel):
    date: datetime
    score: float
    round_number: int
    hotkey: str


class RoundAnnotation(BaseModel):
    round_number: int
    start_at: Optional[datetime] = None


class CompetitionDetailsResponse(BaseModel):
    top_score: float
    score_to_beat: float
    competition: CompetitionInfo
    curr_round: Optional[RoundInfo] = None
    top_scores: List[ScorePoint]
    rounds: List[RoundAnnotation]
    # Top scores pagination.
    pagination: SubmissionPagination
    total_submissions: int


class CompetitionDetailsCache(BaseModel):
    competition: CompetitionInfo
    current_round: Optional[RoundInfo] = None
    top_score_value: float
    score_to_beat: float
    all_scores: List[ScorePoint]
    rounds: List[RoundAnnotation]
    total_submissions: int
