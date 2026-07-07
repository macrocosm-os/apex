from pydantic import BaseModel  # type: ignore
from typing import Optional, List
from datetime import datetime
from common.models.api.competition import SponsorMetadata
from common.models.api.submission import SubmissionPagination


class CompetitionDetailsRequest(BaseModel):
    competition_id: int


class TopScoresRequest(BaseModel):
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
    sponsor: Optional[SponsorMetadata] = None
    active_miners: int = 0
    total_alpha_earned: Optional[float] = None
    daily_alpha_earned: Optional[float] = None
    total_rounds: Optional[int] = None
    # Miner submission metadata surfaced on the competition page.
    round_length_in_days: Optional[float] = None
    submission_fee_usd: Optional[float] = None
    submission_reveal_days: Optional[float] = None
    submission_rate_limit: Optional[str] = None  # global, e.g. "4/day"
    notes: List[str] = []


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
    raw_score: float
    round_number: int
    version: int
    hotkey: str
    coldkey: Optional[str] = None
    has_metadata: bool = False


class RoundAnnotation(BaseModel):
    round_number: int
    start_at: Optional[datetime] = None


class CompetitionDetailsResponse(BaseModel):
    top_score: float
    score_to_beat: Optional[float] = None
    competition: CompetitionInfo
    curr_round: Optional[RoundInfo] = None
    rounds: List[RoundAnnotation]
    total_submissions: int
    daily_submissions: List[int] = []


class TopScoresResponse(BaseModel):
    top_scores: List[ScorePoint]
    pagination: SubmissionPagination
    current_competition_submissions: int = 0
    current_round_submissions: int = 0
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0
    daily_submissions: List[int] = []


class CompetitionDetailsCache(BaseModel):
    competition: CompetitionInfo
    current_round: Optional[RoundInfo] = None
    top_score_value: float
    score_to_beat: Optional[float] = None
    rounds: List[RoundAnnotation]
    total_submissions: int
    daily_submissions: List[int] = []


class TopScoresCache(BaseModel):
    top_scores: List[ScorePoint]
    daily_submission_counts: dict[str, int] = {}
    current_competition_submissions: int = 0
    current_round_submissions: int = 0
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0
