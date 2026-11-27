"""Inferred `started` must use the same strict bool parsing as `complete`.

`SourceReleaseEvidence.from_dict` inferred `started` with
`bool(raw.get("complete"))`. An explicit `"complete": "false"` therefore
produced `complete=False` but `started=True`, marking a release as begun when
the evidence said it was not.
"""
from __future__ import annotations

from auto_ingest.custody.evidence import SourceReleaseEvidence


def test_string_false_complete_does_not_infer_started():
    ev = SourceReleaseEvidence.from_dict({"complete": "false"}, 8)
    assert ev.complete is False
    assert ev.started is False


def test_explicit_true_or_release_activity_still_infers_started():
    assert SourceReleaseEvidence.from_dict({"complete": True}, 8).started is True
    assert SourceReleaseEvidence.from_dict(
        {"released": {"files": 1, "bytes": 10}}, 8).started is True


def test_explicit_started_flag_wins():
    assert SourceReleaseEvidence.from_dict(
        {"complete": True, "started": "false"}, 8).started is False
