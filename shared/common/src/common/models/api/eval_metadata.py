"""Standard envelope for `eval_metadata`.

Every competition fills in the same top level so the dashboard can render a
generic table and a known set of charts without a parser per competition. The
competition-specific payload keeps its existing shape under `details`.

See docs/superpowers/specs/2026-08-04-standard-eval-metadata-design.md.
"""

from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError

MetricFormat = Literal["number", "percent", "duration_s", "score", "count", "text"]
MetricDirection = Literal["higher_is_better", "lower_is_better", "neutral"]
UnitType = Literal["match", "game", "task", "subset", "instance", "completion", "other"]


class EvalMetric(BaseModel):
    """One named measurement, with enough metadata that a chart never has to
    infer semantics from the property name."""

    model_config = ConfigDict(extra="forbid")

    key: str  # stable machine id, e.g. "win_rate"
    label: str  # human label, e.g. "Win rate"
    value: float | int | str
    unit: str | None = None  # "s", "%", "tao"
    format: MetricFormat = "number"
    direction: MetricDirection = "higher_is_better"
    baseline_value: float | None = None
    # Set only on entries in `metrics`; never on `summary` (see StandardEvalMetadata).
    unit_id: str | None = None


class EvalUnit(BaseModel):
    """A repeated sub-run within one evaluation: a match, game, task, subset..."""

    model_config = ConfigDict(extra="forbid")

    id: str  # "match-1", "task-alpha"
    parent_id: str | None = None
    type: UnitType
    index: int
    label: str
    category: str | None = None  # scenario / pool / opponent / concept
    outcome: str | None = None  # won / lost / hit / miss — vocabulary is package-local
    duration_seconds: float | None = None
    steps: int | None = None
    sample_count: int | None = None


class EvalCapabilities(BaseModel):
    """Which shared visualisations this payload can actually back.

    Declared by the producer and validated against the payload, so a flag can
    never lie and leave the dashboard rendering an empty chart.
    """

    model_config = ConfigDict(extra="forbid")

    score_distribution: bool = False
    baseline_comparison: bool = False
    runtime_breakdown: bool = False
    outcome_breakdown: bool = False
    category_breakdown: bool = False
    timeline: bool = False


class StandardEvalMetadata(BaseModel):
    """The envelope. `schema_version == 0` means a legacy payload wrapped on read."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    score: float | None = None  # mirrors submission.eval_score
    raw_score: float | None = None  # mirrors submission.eval_raw_score
    baseline: float | None = None
    # Run-level headline metrics. Array order IS display order. No unit_id here.
    summary: list[EvalMetric] = []
    # Per-unit series feeding distribution / runtime charts. unit_id required.
    metrics: list[EvalMetric] = []
    units: list[EvalUnit] = []
    capabilities: EvalCapabilities = Field(default_factory=EvalCapabilities)
    # Competition-specific payload, verbatim.
    details: dict = {}

    @model_validator(mode="after")
    def _validate_envelope(self) -> "StandardEvalMetadata":
        unit_ids = {u.id for u in self.units}

        for m in self.summary:
            if m.unit_id:
                raise ValueError(f"summary metric {m.key!r} must not set unit_id — put it in `metrics`")
        for m in self.metrics:
            if not m.unit_id:
                raise ValueError(f"metrics entry {m.key!r} must set unit_id")
            if m.unit_id not in unit_ids:
                raise ValueError(f"metrics entry {m.key!r} references unknown unit {m.unit_id!r}")

        c = self.capabilities
        if c.score_distribution and not self.metrics:
            raise ValueError("score_distribution declared but no per-unit metrics")
        if (
            c.baseline_comparison
            and self.baseline is None
            and not any(m.baseline_value is not None for m in (*self.summary, *self.metrics))
        ):
            raise ValueError("baseline_comparison declared but no baseline values")
        if c.runtime_breakdown and not any(u.duration_seconds is not None for u in self.units):
            raise ValueError("runtime_breakdown declared but no unit durations")
        if c.outcome_breakdown and not any(u.outcome for u in self.units):
            raise ValueError("outcome_breakdown declared but no unit outcomes")
        if c.category_breakdown and not any(u.category for u in self.units):
            raise ValueError("category_breakdown declared but no unit categories")
        # `timeline` depends on eval_file_paths history artifacts, which this
        # envelope does not describe — trusted as declared.
        return self


def coerce_eval_metadata(raw: Any) -> StandardEvalMetadata:
    """Best-effort legacy/author blob -> envelope. Never raises.

    Used at the read boundary and by the spec-driven path, where the payload
    comes from competition-author referee code and a hard failure would zero
    out a real evaluation. `raw` is genuinely untrusted: authors can emit any
    JSON value under the metadata key, not just a dict.
    """
    if isinstance(raw, StandardEvalMetadata):
        return raw
    if not raw:
        return StandardEvalMetadata(schema_version=0)
    if not isinstance(raw, dict):
        # A list/str/int/etc can't be interpreted as a legacy blob (`details`
        # is itself a dict) or examined for `schema_version`. Wrap with no
        # details rather than let `StandardEvalMetadata(details=raw)` raise.
        logger.warning(f"non-dict eval_metadata ({type(raw).__name__}), wrapping as legacy with no details")
        return StandardEvalMetadata(schema_version=0)
    # Gate on PRESENCE, not truthiness: a stored legacy envelope has
    # schema_version == 0, and a truthiness check would re-wrap it, nesting
    # details inside details on every read.
    if "schema_version" in raw:
        try:
            return StandardEvalMetadata.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"non-conforming eval_metadata, wrapping as legacy: {e}")
            return StandardEvalMetadata(schema_version=0, details=raw)
    return StandardEvalMetadata(schema_version=0, details=raw)
