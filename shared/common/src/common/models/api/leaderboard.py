from datetime import datetime
from enum import Enum

from pydantic import BaseModel  # type: ignore

from common.models.api.ranks import RanksPagination


class LeaderboardMode(str, Enum):
    """Dimension the leaderboard is ranked by. Every entry returns the full
    metric bundle regardless of mode; the mode only decides the ``rank`` order."""

    FIRST_PLACES = "first_places"
    MEAN_SCORE = "mean_score"
    MEDIAN_SCORE = "median_score"
    MEAN_PLACE = "mean_place"
    MEDIAN_PLACE = "median_place"
    TOTAL_SUBMISSIONS = "total_submissions"
    EARNINGS = "earnings"


class LeaderboardRequest(BaseModel):
    mode: LeaderboardMode = LeaderboardMode.EARNINGS
    start_idx: int = 0
    count: int = 50


class LeaderboardEntry(BaseModel):
    coldkey: str
    # 1-based position under the selected mode. 0 in the cache before ranking.
    rank: int = 0
    # Competitions this coldkey currently holds the top score in (snapshot).
    first_places: int = 0
    # Total all-time alpha earned by this coldkey.
    total_earned: float = 0.0
    # Total all-time submissions across all hotkeys owned by this coldkey.
    total_submissions: int = 0
    # Total competitions entered across all hotkeys owned by this coldkey.
    competitions_entered: int = 0
    first_submission_at: datetime | None = None
    # Across competitions entered, over the coldkey's best score per competition.
    mean_score: float = 0.0
    median_score: float = 0.0
    # Across competitions entered, over the coldkey's best place per competition
    # (lower is better).
    mean_place: float = 0.0
    median_place: float = 0.0


class LeaderboardResponse(BaseModel):
    mode: LeaderboardMode
    miners: list[LeaderboardEntry]
    pagination: RanksPagination


class LeaderboardCache(BaseModel):
    # Per-coldkey metrics, unranked (rank=0). Ranking is applied per-request so
    # all modes share a single cache entry.
    entries: list[LeaderboardEntry]
