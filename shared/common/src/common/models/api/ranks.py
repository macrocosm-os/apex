from typing import Optional

from pydantic import BaseModel

from common.models.api.pagination import Pagination
from common.models.api.submission import RankRecord


class MinerRanksRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 50
    round_number: Optional[int] = None


class CompetitionMeta(BaseModel):
    """Typed replacement for the untyped comp_row dict formerly cached."""

    competition_id: int
    curr_top_scorer_hotkey: Optional[str] = None
    curr_top_scorer_coldkey: Optional[str] = None


class RanksResponse(BaseModel):
    """Envelope for both rank endpoints (/miners and /submissions...)."""

    competition_id: int
    # Deprecated (FE marks it @deprecated) — remove in the APEX-106 cleanup PR.
    incentive_weight_render: float
    curr_top_scorer_hotkey: Optional[str] = None
    curr_top_scorer_coldkey: Optional[str] = None
    miners: list[RankRecord]
    pagination: Pagination
    total_submissions: int


class RanksCache(BaseModel):
    """Full (unsliced) rank listing cached in Redis; responses slice a page."""

    meta: CompetitionMeta
    records: list[RankRecord]
    scaled_incentive: float
    total_submissions: int
