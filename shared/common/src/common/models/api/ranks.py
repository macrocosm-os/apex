from pydantic import BaseModel  # type: ignore
from typing import Optional


class MinerRanksRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 50
    round_number: Optional[int] = None


class MinerRankRecord(BaseModel):
    # Miner rank in the leaderboard
    rank: int
    # True if current submission is a winner
    top_scorer: bool
    hotkey: str
    coldkey: Optional[str] = None
    # Miner's best score
    score: float
    # Raw (un-normalized) eval score of the ranked submission
    raw_score: float
    # Score multiplied by incentive weight (for rendering)
    score_render: float
    # Miner's last submission version
    version: int
    # Round number of the best submission
    round_number: int
    submission_date: str | int | float | None = None
    join_date: str | int | float | None = None
    # Number of submissions by the miner
    submissions_count: int
    # True if any of this miner's submissions has a browser-playable artifact
    # (currently: ONNX-converted Tron round winners). Generic across competitions.
    can_play: bool = False
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0


class RanksPagination(BaseModel):
    start_idx: int
    count: int
    total: int
    has_more: bool


class MinerRanksResponse(BaseModel):
    competition_id: int
    incentive_weight_render: float
    curr_top_scorer_hotkey: Optional[str] = None
    curr_top_scorer_coldkey: Optional[str] = None
    miners: list[MinerRankRecord]
    pagination: RanksPagination
    total_submissions: int


class MinerRanksCache(BaseModel):
    comp_row: dict
    miners: list[MinerRankRecord]
    scaled_incentive: float
    total_submissions: int
