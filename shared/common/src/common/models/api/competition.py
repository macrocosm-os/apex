from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, Field
from typing import Literal, Optional

from common.settings import DEFAULT_BASE_BURN_RATE
from common.models.api.pagination import Pagination


class CompetitionRequest(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    pkg: Optional[str] = None
    ptype: Optional[str] = None
    ctype: Optional[str] = None
    state: Optional[str] = None
    show_completed: bool = False
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


class SponsorMetadata(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None


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
    # Score semantics (APEX-108): what eval_score/baseline_score mean for this
    # competition. Registry-backed packages normalize to 0-1; spec-driven ones
    # report the referee's raw score unchanged. baseline_valid gates whether
    # baseline_score is a meaningful same-scale comparison target for charts.
    score_scale: Literal["normalized_0_1", "raw"] = "normalized_0_1"
    score_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    baseline_valid: bool = False
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
    total_submissions: int = 0
    active_miners: int = 0
    total_alpha_earned: Optional[float] = None
    daily_alpha_earned: Optional[float] = None
    total_rounds: Optional[int] = None
    daily_submissions: list[int] = []
    image_url: Optional[str] = None
    doc_url: Optional[str] = None
    sponsor: Optional[SponsorMetadata] = None
    base_burn_rate: float = DEFAULT_BASE_BURN_RATE
    submission_fee_usd: Decimal = Decimal("0.0")


class CompetitionResponse(BaseModel):
    competitions: list[CompetitionRecord]
    pagination: Pagination
    total_alpha_earned: Optional[float] = None
    total_agents: int = 0
    daily_submissions: list[int] = []


class ComingSoonCompetition(BaseModel):
    name: str
    description: str
    image_url: Optional[str] = None
    ctype: str
    ptype: str
    start_at: Optional[date] = None


class ComingSoonResponse(BaseModel):
    competitions: list[ComingSoonCompetition]
