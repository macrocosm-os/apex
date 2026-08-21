from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, computed_field, model_validator
from typing import Optional

from common.models.api.eval_metadata import StandardEvalMetadata
from common.models.api.pagination import Pagination


class SubmitRequest(BaseModel):
    competition_id: int
    round_number: int = 0
    raw_code: Optional[str] = None  # Text-based submissions (e.g., .py files)
    raw_binary: Optional[str] = None  # Base64-encoded binary submissions (e.g., .pt files)
    file_extension: str = ".py"  # File extension for the submission
    payment_block_hash: Optional[str] = None  # Block hash of on-chain fee payment (hex string)
    payment_extrinsic_index: Optional[int] = None  # Extrinsic index within the block

    @model_validator(mode="after")
    def validate_content(self) -> "SubmitRequest":
        if self.raw_code is None and self.raw_binary is None:
            raise ValueError("Either raw_code or raw_binary must be provided")
        if self.raw_code is not None and self.raw_binary is not None:
            raise ValueError("Only one of raw_code or raw_binary should be provided")
        return self

    @property
    def is_binary(self) -> bool:
        return self.raw_binary is not None


class SubmitResponse(BaseModel):
    submission_id: int


class SubmissionRequest(BaseModel):
    submission_id: Optional[int] = None
    competition_id: Optional[int] = None
    hotkey: Optional[str] = None
    start_idx: int = 0
    count: int = 10
    filter_mode: str = "all"
    sort_mode: str = "score"


class SubmissionBase(BaseModel):
    """Canonical submission shape (APEX-106). Every submission-bearing
    response model subclasses this so names and types cannot drift."""

    id: int
    competition_id: int
    round_number: int
    state: str
    hotkey: str
    coldkey: Optional[str] = None
    version: int
    submitted_at: datetime
    score: Optional[float] = None  # normalized eval score
    raw_score: Optional[float] = None
    top_score: bool = False


class SubmissionRecord(SubmissionBase):
    eval_at: Optional[datetime] = None
    reveal_at: Optional[datetime] = None
    eval_time_in_seconds: Optional[float] = None
    eval_error: Optional[str] = None

    # Deprecated dual-emitted names — remove in the APEX-106 cleanup PR.
    @computed_field
    @property
    def submit_at(self) -> datetime:
        return self.submitted_at

    @computed_field
    @property
    def eval_score(self) -> Optional[float]:
        return self.score

    @computed_field
    @property
    def eval_raw_score(self) -> Optional[float]:
        return self.raw_score


class RankRecord(SubmissionBase):
    """One row of a competition rank listing. Replaces MinerRankRecord and
    SubmissionRankMiner — both rank endpoints serve this shape."""

    rank: int
    # Number of scored submissions by the miner (competition- or round-scoped)
    submissions_count: int
    # Miner's first scored submission time in scope
    join_date: Optional[datetime] = None
    # True if any of this miner's submissions has a browser-playable artifact
    # (for example, ONNX-converted round winners). Generic across competitions.
    can_play: bool = False
    estimated_current_competition_alpha_earned: float = 0.0
    estimated_current_round_alpha_earned: float = 0.0
    # Deprecated presentation-layer value (score * scaled_incentive); the FE
    # never reads it. Plain field (not computed) because it needs the
    # competition's incentive weight. Remove in the APEX-106 cleanup PR.
    score_render: float = 0.0

    # Deprecated dual-emitted names — remove in the APEX-106 cleanup PR.
    @computed_field
    @property
    def top_scorer(self) -> bool:
        return self.top_score

    @computed_field
    @property
    def submission_date(self) -> datetime:
        return self.submitted_at


class SubmissionDetail(SubmissionBase):
    submit_metadata: dict | None = None
    # Typed so the envelope lands in the OpenAPI schema and the dashboard gets
    # generated types instead of `any`.
    eval_metadata: StandardEvalMetadata | None = None
    eval_file_paths: dict | None = None
    code_path: str | None = None
    reveal_at: Optional[datetime] = None
    # Miner's current best-per-miner rank in this competition (same semantics
    # as /dashboard/competitions/{id}/miners). None while unrevealed or unscored.
    rank: Optional[int] = None
    eval_error: Optional[str] = None
    eval_time_in_seconds: Optional[float] = None
    # Reveal gate: eval_metadata / score / raw_score / eval_file_paths / rank /
    # eval_error / eval_time_in_seconds are nulled until the round completes.
    # `revealed` tells clients the nulls are gating (not missing data);
    # round_state/round_end_at say when it lifts.
    revealed: bool = True
    round_state: Optional[str] = None
    round_end_at: Optional[datetime] = None
    is_binary: bool = False
    language: str | None = None
    # True if this submission has a browser-playable artifact (e.g. an
    # ONNX-converted model). Generic across competitions; derived from
    # `submit_metadata.onnx`.
    can_play: bool = False

    # Deprecated dual-emitted names — remove in the APEX-106 cleanup PR.
    @computed_field
    @property
    def eval_score(self) -> Optional[float]:
        return self.score

    @computed_field
    @property
    def eval_raw_score(self) -> Optional[float]:
        return self.raw_score


class SubmissionResponse(BaseModel):
    submissions: list[SubmissionRecord]
    pagination: Pagination


class SubmissionFeeResponse(BaseModel):
    amount_rao: int
    send_address: str
    competition_id: int
    fee_usd: Decimal  # USD equivalent of amount_rao at the current TAO price


class FileRequest(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    start_idx: int = 0
    reverse: bool = False
