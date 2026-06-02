from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel

from common.models.api.submission import SubmissionPagination


class SubmissionHistoryRecord(BaseModel):
    """A single submission entry shown on the miner profile page."""

    submission_id: int
    hotkey: str
    round_number: int
    submitted_at: datetime
    score: Optional[float] = None
    state: str
    version: int


class CompetitionHistory(BaseModel):
    """Per-competition rollup with nested submission history."""

    competition_id: int
    competition_name: str
    submission_count: int
    best_score: Optional[float] = None
    last_submission_at: Optional[datetime] = None
    alpha_earned: float = 0.0
    submissions: list[SubmissionHistoryRecord] = []


class DailyActivity(BaseModel):
    """One bucket of the contribution-style activity timeline."""

    date: date
    count: int


class ActivityTimeline(BaseModel):
    """Activity timeline used by the profile heatmap."""

    daily_submissions: list[DailyActivity] = []


class ProfileSummary(BaseModel):
    """High-level metrics rendered in the profile header."""

    competition_count: int
    submission_count: int
    scored_submission_count: int
    best_score: Optional[float] = None
    last_submission_at: Optional[datetime] = None
    alpha_earned_total: float = 0.0


class MinerProfileResponse(BaseModel):
    """Aggregate response for `GET /public/miners/by-coldkey/{coldkey}/profile`."""

    coldkey: str
    hotkeys: list[str] = []
    summary: ProfileSummary
    activity: ActivityTimeline
    competitions: list[CompetitionHistory] = []
    pagination: SubmissionPagination
