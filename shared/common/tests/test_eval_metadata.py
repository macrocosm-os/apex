"""Contract tests for the StandardEvalMetadata envelope.

The validator tests matter more than the happy path: each one pins a way the
envelope could lie to the dashboard (a capability flag with no data behind it,
a per-unit metric pointing at a unit that doesn't exist).
"""

import pytest
from pydantic import ValidationError

from common.models.api.eval_metadata import (
    EvalCapabilities,
    EvalMetric,
    EvalUnit,
    StandardEvalMetadata,
)


def _unit(uid="match-1", **over):
    kwargs = {"id": uid, "type": "match", "index": 1, "label": "Match 1"}
    kwargs.update(over)
    return EvalUnit(**kwargs)


def test_minimal_envelope_defaults_to_v1():
    meta = StandardEvalMetadata()
    assert meta.schema_version == 1
    assert meta.summary == []
    assert meta.metrics == []
    assert meta.units == []
    assert meta.details == {}
    assert meta.capabilities.score_distribution is False


def test_metric_defaults():
    m = EvalMetric(key="win_rate", label="Win rate", value=0.5)
    assert m.format == "number"
    assert m.direction == "higher_is_better"
    assert m.unit is None
    assert m.unit_id is None


def test_run_level_summary_renders_without_units():
    meta = StandardEvalMetadata(
        score=0.81,
        raw_score=124.3,
        summary=[
            EvalMetric(key="avg_quality", label="Avg quality", value=124.3),
            EvalMetric(key="num_instances", label="Instances", value=12, format="count", direction="neutral"),
        ],
    )
    assert [m.key for m in meta.summary] == ["avg_quality", "num_instances"]


def test_extra_keys_are_rejected_so_legacy_blobs_cannot_validate():
    # Without extra="forbid" this SUCCEEDS and silently discards the payload.
    with pytest.raises(ValidationError):
        StandardEvalMetadata.model_validate({"master_seed": 1, "avg_quality": 0.8})


def test_summary_metric_may_not_carry_unit_id():
    with pytest.raises(ValidationError, match="must not set unit_id"):
        StandardEvalMetadata(
            units=[_unit()],
            summary=[EvalMetric(key="s", label="S", value=1.0, unit_id="match-1")],
        )


def test_per_unit_metric_requires_unit_id():
    with pytest.raises(ValidationError, match="must set unit_id"):
        StandardEvalMetadata(units=[_unit()], metrics=[EvalMetric(key="s", label="S", value=1.0)])


def test_per_unit_metric_must_reference_a_real_unit():
    with pytest.raises(ValidationError, match="unknown unit"):
        StandardEvalMetadata(
            units=[_unit("match-1")],
            metrics=[EvalMetric(key="s", label="S", value=1.0, unit_id="match-9")],
        )


def test_score_distribution_requires_per_unit_metrics():
    with pytest.raises(ValidationError, match="score_distribution"):
        StandardEvalMetadata(
            units=[_unit()],
            capabilities=EvalCapabilities(score_distribution=True),
        )


def test_baseline_comparison_requires_a_baseline_value():
    with pytest.raises(ValidationError, match="baseline_comparison"):
        StandardEvalMetadata(capabilities=EvalCapabilities(baseline_comparison=True))


def test_baseline_comparison_satisfied_by_envelope_baseline():
    meta = StandardEvalMetadata(baseline=0.5, capabilities=EvalCapabilities(baseline_comparison=True))
    assert meta.capabilities.baseline_comparison is True


def test_baseline_comparison_satisfied_by_metric_baseline_value():
    meta = StandardEvalMetadata(
        summary=[EvalMetric(key="profit", label="Profit", value=10.0, baseline_value=8.0)],
        capabilities=EvalCapabilities(baseline_comparison=True),
    )
    assert meta.capabilities.baseline_comparison is True


def test_runtime_breakdown_requires_unit_durations():
    with pytest.raises(ValidationError, match="runtime_breakdown"):
        StandardEvalMetadata(units=[_unit()], capabilities=EvalCapabilities(runtime_breakdown=True))


def test_outcome_breakdown_requires_unit_outcomes():
    with pytest.raises(ValidationError, match="outcome_breakdown"):
        StandardEvalMetadata(units=[_unit()], capabilities=EvalCapabilities(outcome_breakdown=True))


def test_category_breakdown_requires_unit_categories():
    with pytest.raises(ValidationError, match="category_breakdown"):
        StandardEvalMetadata(units=[_unit()], capabilities=EvalCapabilities(category_breakdown=True))


def test_timeline_is_trusted_because_artifacts_live_outside_the_envelope():
    meta = StandardEvalMetadata(capabilities=EvalCapabilities(timeline=True))
    assert meta.capabilities.timeline is True


def test_fully_populated_envelope_round_trips():
    meta = StandardEvalMetadata(
        score=0.66,
        raw_score=2.0,
        summary=[EvalMetric(key="matches_won", label="Won", value=2, format="count")],
        units=[
            _unit("match-1", outcome="won", duration_seconds=42.1, category="opponent-b"),
            _unit("match-2", index=2, label="Match 2", outcome="lost", duration_seconds=38.7),
        ],
        metrics=[
            EvalMetric(key="match_raw_score", label="Match score", value=1.0, unit_id="match-1", format="score"),
            EvalMetric(key="match_raw_score", label="Match score", value=0.0, unit_id="match-2", format="score"),
        ],
        capabilities=EvalCapabilities(
            score_distribution=True, outcome_breakdown=True, runtime_breakdown=True, category_breakdown=True
        ),
        details={"Match 1": {"seed": 100}},
    )
    dumped = meta.model_dump(mode="json")
    assert StandardEvalMetadata.model_validate(dumped) == meta


from common.models.api.eval_metadata import coerce_eval_metadata


def test_coerce_none_and_empty_give_a_legacy_envelope():
    for raw in (None, {}):
        meta = coerce_eval_metadata(raw)
        assert meta.schema_version == 0
        assert meta.details == {}


def test_coerce_passes_through_an_envelope_instance():
    original = StandardEvalMetadata(score=0.5)
    assert coerce_eval_metadata(original) is original


def test_coerce_wraps_a_legacy_blob_verbatim():
    legacy = {"master_seed": 7, "num_instances": 12, "avg_quality": 0.81}
    meta = coerce_eval_metadata(legacy)
    assert meta.schema_version == 0
    assert meta.summary == []
    assert meta.details == legacy


def test_coerce_parses_a_valid_v1_payload():
    payload = StandardEvalMetadata(score=0.4, summary=[EvalMetric(key="k", label="K", value=1.0)]).model_dump(
        mode="json"
    )
    meta = coerce_eval_metadata(payload)
    assert meta.schema_version == 1
    assert [m.key for m in meta.summary] == ["k"]


def test_coerce_wraps_a_malformed_v1_payload_instead_of_raising():
    # schema_version says v1, but `summary` is the wrong type.
    meta = coerce_eval_metadata({"schema_version": 1, "summary": "not-a-list"})
    assert meta.schema_version == 0
    assert meta.details == {"schema_version": 1, "summary": "not-a-list"}


def test_coerce_is_idempotent_on_a_stored_legacy_envelope():
    """Regression: gating on truthiness of schema_version re-wraps a stored v0
    envelope, nesting details inside details on every read. Unmigrated runners
    persist v0 envelopes, so this is the common path."""
    legacy = {"master_seed": 7}
    once = coerce_eval_metadata(legacy).model_dump(mode="json")
    twice = coerce_eval_metadata(once)
    assert twice.schema_version == 0
    assert twice.details == legacy  # NOT {"schema_version": 0, "details": {...}}


@pytest.mark.parametrize("raw", [[1, 2, 3], "not-a-dict", 5])
def test_coerce_does_not_raise_on_non_dict_author_payloads(raw):
    """Regression: authors can emit any JSON value under the metadata key.
    A truthy non-dict used to fall through to
    StandardEvalMetadata(details=raw), and `details` is itself a dict, so
    pydantic raised ValidationError outside the try/except."""
    meta = coerce_eval_metadata(raw)
    assert meta.schema_version == 0


from common.models.api.job import JobResults


def test_job_results_defaults_to_a_legacy_envelope():
    assert JobResults().eval_metadata.schema_version == 0


def test_job_results_rejects_a_bare_legacy_dict():
    """APEX-103 end state: every deployed worker posts dumped envelopes, so a
    legacy dict over the wire is a bug to surface, not tolerate."""
    with pytest.raises(ValidationError):
        JobResults(submission_id=1, eval_metadata={"win_rate": 0.7})


def test_job_results_rejects_a_present_but_invalid_v1_payload():
    with pytest.raises(ValidationError):
        JobResults(submission_id=1, eval_metadata={"schema_version": 1, "summary": "not-a-list"})


def test_job_results_round_trips_an_envelope_as_json():
    """The worker posts JSON; the orchestrator parses it back. Both sides must
    see the same envelope."""
    sent = JobResults(
        submission_id=1,
        eval_metadata=StandardEvalMetadata(score=0.5, summary=[EvalMetric(key="k", label="K", value=1)]),
        match_outcome="won",
        opponent_submission_id=20,
    )
    received = JobResults.model_validate(sent.model_dump(mode="json"))
    assert received.eval_metadata.schema_version == 1
    assert [m.key for m in received.eval_metadata.summary] == ["k"]
    assert received.match_outcome == "won"
    assert received.opponent_submission_id == 20


def test_job_results_control_fields_default_to_none():
    jr = JobResults()
    assert jr.match_outcome is None
    assert jr.opponent_submission_id is None
    assert jr.screening_status is None


from common.models.api.submission import SubmissionDetail


def test_submission_detail_rejects_a_bare_legacy_dict():
    """SubmissionDetail's tolerant validator existed for the 60s Redis detail
    cache and old-orchestrator responses. Prod has served envelope-shaped
    payloads since v4.2.20, and the read path coerces explicitly before
    constructing the model, so a legacy dict reaching validation is a bug."""
    with pytest.raises(ValidationError):
        SubmissionDetail.model_validate({"id": 1, "round_number": 1, "eval_metadata": {"master_seed": 7}})


def test_submission_detail_eval_metadata_none_is_still_allowed():
    """The reveal gate returns `eval_metadata=None` for a live round; the
    tolerant validator must not turn that into a legacy-wrapped envelope."""
    detail = SubmissionDetail.model_validate({"id": 1, "round_number": 1, "eval_metadata": None})
    assert detail.eval_metadata is None
