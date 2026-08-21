from decimal import Decimal
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, computed_field

from common.models.api.pagination import Pagination
from common.models.api.submission import SubmissionBase


class SubmissionHistoryRecord(SubmissionBase):
    """A single submission entry shown on the miner profile page."""

    rank: Optional[int] = None

    # Deprecated dual-emitted name — remove in the APEX-106 cleanup PR.
    @computed_field
    @property
    def submission_id(self) -> int:
        return self.id


class ProfileHotkey(BaseModel):
    """A hotkey linked to the profile with its alpha earnings breakdown."""

    hotkey: str
    alpha_earned_total: Decimal = Decimal("0.0")


class CompetitionHistory(BaseModel):
    """Per-competition rollup with nested submission history."""

    competition_id: int
    competition_name: str
    submission_count: int
    best_score: Optional[float] = None
    last_submission_at: Optional[datetime] = None
    alpha_earned: Decimal = Decimal("0.0")
    submissions: list[SubmissionHistoryRecord] = []


class DailyActivity(BaseModel):
    """One bucket of the contribution-style activity timeline."""

    date: date
    count: int


class DailyEarnings(BaseModel):
    """One bucket of daily alpha earnings."""

    date: date
    alpha_earned: Decimal


class ActivityTimeline(BaseModel):
    """Activity timeline used by the profile heatmap."""

    daily_submissions: list[DailyActivity] = []
    daily_earnings: list[DailyEarnings] = []


class ProfileSummary(BaseModel):
    """High-level metrics rendered in the profile header."""

    competition_count: int
    submission_count: int
    scored_submission_count: int
    best_score: Optional[float] = None
    last_submission_at: Optional[datetime] = None
    alpha_earned_total: Decimal = Decimal("0.0")


class MinerProfileResponse(BaseModel):
    """Aggregate response for `GET /public/miners/by-coldkey/{coldkey}/profile`."""

    coldkey: str
    hotkeys: list[ProfileHotkey] = []
    summary: ProfileSummary
    activity: ActivityTimeline
    competitions: list[CompetitionHistory] = []
    pagination: Pagination


class ColdkeyExistsResponse(BaseModel):
    """Response for `GET /public/miners/by-coldkey/{coldkey}/exists`."""

    coldkey: str
    exists: bool
