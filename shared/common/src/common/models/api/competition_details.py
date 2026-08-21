from pydantic import BaseModel, Field  # type: ignore
from typing import Literal, Optional, List
from datetime import datetime
from common.models.api.competition import SponsorMetadata
from common.models.api.pagination import Pagination


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
    # Score semantics (APEX-108): what eval_score/baseline_score mean for this
    # competition. Registry-backed packages normalize to 0-1; spec-driven ones
    # report the referee's raw score unchanged. baseline_valid gates whether
    # baseline_score is a meaningful same-scale comparison target for charts.
    score_scale: Literal["normalized_0_1", "raw"] = "normalized_0_1"
    score_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    baseline_valid: bool = False
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
    # Whether this point's round has completed, i.e. the reveal gate has
    # lifted for its submission detail (metadata, artifacts). Formerly the
    # misleadingly named `has_metadata`.
    revealed: bool = False


class RoundAnnotation(BaseModel):
    round_number: int
    start_at: Optional[datetime] = None


class CompetitionDetailsResponse(BaseModel):
    top_score: float
    score_to_beat: Optional[float] = Field(
        default=None,
        description=(
            "Threshold-adjusted score a new submission must clear to take the top spot, on the "
            "competition's score_scale. Null between rounds: when the reigning top scorer is not "
            "from the current round, its raw score was produced under different round conditions "
            "(e.g. max_epoch_time), so no comparable target exists until the new round's "
            "auto-submission is evaluated."
        ),
    )
    competition: CompetitionInfo
    curr_round: Optional[RoundInfo] = None
    rounds: List[RoundAnnotation]
    total_submissions: int
    daily_submissions: List[int] = []


class TopScoresResponse(BaseModel):
    top_scores: List[ScorePoint]
    pagination: Pagination
    current_competition_submissions: int = 0
    current_round_submissions: int = 0
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0
    daily_submissions: List[int] = []


class TopScoresCache(BaseModel):
    top_scores: List[ScorePoint]
    daily_submission_counts: dict[str, int] = {}
    current_competition_submissions: int = 0
    current_round_submissions: int = 0
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0
