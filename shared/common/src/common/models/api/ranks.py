from pydantic import BaseModel  # type: ignore
from typing import Optional


class MinerRanksRequest(BaseModel):
    competition_id: int
    start_idx: int = 0
    count: int = 50


class MinerRankRecord(BaseModel):
    # Miner rank in the leaderboard
    rank: int
    # True if current submission is a winner
    top_scorer: bool
    hotkey: str
    # Miner's best score
    score: float
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


class RanksPagination(BaseModel):
    start_idx: int
    count: int
    total: int
    has_more: bool


class MinerRanksResponse(BaseModel):
    competition_id: int
    incentive_weight_render: float
    curr_top_scorer_hotkey: Optional[str] = None
    miners: list[MinerRankRecord]
    pagination: RanksPagination


class MinerRanksCache(BaseModel):
    comp_row: dict
    miners: list[MinerRankRecord]
    scaled_incentive: float
