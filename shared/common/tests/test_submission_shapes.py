"""APEX-106: canonical submission shapes and wire compatibility.

Wire-compat tests assert the deprecated dual-emitted names stay on the wire
during the deprecation window; delete them in the cleanup PR that drops the
computed fields.
"""

from common.models.api.pagination import Pagination


def test_pagination_shape():
    p = Pagination(start_idx=0, count=10, total=25, has_more=True)
    assert p.model_dump() == {"start_idx": 0, "count": 10, "total": 25, "has_more": True}


from datetime import datetime

from common.models.api.submission import SubmissionBase, SubmissionRecord

CORE = dict(
    id=7,
    competition_id=3,
    round_number=2,
    state="scored",
    hotkey="hk1",
    coldkey="ck1",
    version=1,
    submitted_at=datetime(2026, 8, 1, 12, 0, 0),
    score=0.5,
    raw_score=1.5,
    top_score=True,
)


def test_submission_base_canonical_fields():
    base = SubmissionBase(**CORE)
    assert base.submitted_at == datetime(2026, 8, 1, 12, 0, 0)
    assert base.score == 0.5
    assert base.raw_score == 1.5
    assert base.top_score is True


def test_submission_record_dual_emits_old_names():
    rec = SubmissionRecord(**CORE, eval_error="boom", eval_time_in_seconds=1.25)
    dumped = rec.model_dump(mode="json")
    # canonical names
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["score"] == 0.5
    assert dumped["raw_score"] == 1.5
    # deprecated names still on the wire (drop in cleanup PR)
    assert dumped["submit_at"] == "2026-08-01T12:00:00"
    assert dumped["eval_score"] == 0.5
    assert dumped["eval_raw_score"] == 1.5
    # untouched extras
    assert dumped["eval_error"] == "boom"
    assert dumped["eval_time_in_seconds"] == 1.25


from common.models.api.submission import SubmissionDetail


def test_submission_detail_carries_core_identity_and_dual_emits():
    detail = SubmissionDetail(**CORE, rank=4, code_path="a/b.py")
    dumped = detail.model_dump(mode="json")
    # previously-missing core fields the FE had to join client-side
    assert dumped["hotkey"] == "hk1"
    assert dumped["competition_id"] == 3
    assert dumped["state"] == "scored"
    assert dumped["version"] == 1
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["rank"] == 4
    # deprecated names still on the wire (drop in cleanup PR)
    assert dumped["eval_score"] == 0.5
    assert dumped["eval_raw_score"] == 1.5


from common.models.api.submission import RankRecord


def test_rank_record_dual_emits_old_names():
    rec = RankRecord(
        **CORE,
        rank=1,
        submissions_count=5,
        join_date=datetime(2026, 7, 1, 0, 0, 0),
        score_render=0.25,
    )
    dumped = rec.model_dump(mode="json")
    # canonical
    assert dumped["rank"] == 1
    assert dumped["top_score"] is True
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["id"] == 7 and dumped["state"] == "scored"
    # deprecated names still on the wire (drop in cleanup PR)
    assert dumped["top_scorer"] is True
    assert dumped["submission_date"] == "2026-08-01T12:00:00"
    assert dumped["score_render"] == 0.25


from common.models.api.miner_profile import SubmissionHistoryRecord


def test_history_record_dual_emits_submission_id():
    rec = SubmissionHistoryRecord(**CORE, rank=2)
    dumped = rec.model_dump(mode="json")
    assert dumped["id"] == 7
    assert dumped["submission_id"] == 7  # deprecated — drop in cleanup PR
    assert dumped["rank"] == 2
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"


def test_ranks_response_keeps_deprecated_envelope_field():
    from common.models.api.ranks import RanksResponse
    from common.models.api.pagination import Pagination

    resp = RanksResponse(
        competition_id=3,
        incentive_weight_render=0.5,
        miners=[],
        pagination=Pagination(start_idx=0, count=0, total=0, has_more=False),
        total_submissions=0,
    )
    assert resp.model_dump()["incentive_weight_render"] == 0.5
