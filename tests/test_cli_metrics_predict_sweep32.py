"""`shorts publish metrics predict` must persist predictions, not crash.

The predict sub-action rebuilt each queued short with
``Plan.from_dict({"shorts": [sd]})``. That dict has no ``topic``/``brief``
keys, which ``Plan.from_dict`` requires, so every short raised ``KeyError``
before a single prediction was stored.

Hermetic and offline: a tiny in-memory plan written to pytest's tmp_path and
an in-process args namespace. No Neo4j, network, credentials, model or media.
"""
from __future__ import annotations

import json
from pathlib import Path

from auto_ingest.shorts import cli


class _Args:
    """Only the attributes `_cmd_publish_metrics` reads."""

    def __init__(self, target, **overrides):
        self.action = "metrics"
        self.target = target
        self.metrics_store = None
        self.metrics_file = None
        self.metrics_format = "csv"
        self.platform = "youtube_shorts"
        self.plan_file = None
        self.as_json = False
        self.platforms = None
        self.plans_dir = None
        self.dry_run = True
        for key, value in overrides.items():
            setattr(self, key, value)


def _short(short_id: str) -> dict:
    return {
        "id": short_id,
        "brief_topic": "sweep32_topic",
        "title": f"Short {short_id}",
        "cues": [{"start": 0.4, "end": 3.4, "text": "a surprising hook",
                  "kind": "hook"}],
        "shots": [],
        "status": "planned",
    }


def _write_plan(path: Path, shorts: list) -> Path:
    path.write_text(json.dumps({"shorts": shorts}), encoding="utf-8")
    return path


def _rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_predict_stores_one_prediction_per_short(tmp_path):
    """DEFECT: base raises KeyError('topic'); a fix must write every short."""
    plan = _write_plan(tmp_path / "plan.json", [_short("s1"), _short("s2")])
    store = tmp_path / "metrics.jsonl"

    rc = cli._cmd_publish_metrics(
        _Args("predict", plan_file=plan, metrics_store=store))

    assert rc == 0
    assert store.exists()
    rows = _rows(store)
    assert [r["short_id"] for r in rows] == ["s1", "s2"]
    assert all(r["platform"] == "youtube_shorts" for r in rows)
    assert all(isinstance(r["pred_virality"], (int, float)) for r in rows)


def test_predict_empty_plan_stores_nothing(tmp_path):
    """Control: an empty plan is a no-op on both base and fixed code."""
    plan = _write_plan(tmp_path / "empty-plan.json", [])
    store = tmp_path / "metrics.jsonl"

    assert cli._cmd_publish_metrics(
        _Args("predict", plan_file=plan, metrics_store=store)) == 0
    assert not store.exists()


def test_predict_requires_plan_file(tmp_path):
    """Control: a missing file is a usage error, not a traceback."""
    store = tmp_path / "metrics.jsonl"
    assert cli._cmd_publish_metrics(
        _Args("predict", plan_file=None, metrics_store=store)) == 2


def test_report_on_empty_store_is_a_noop(tmp_path):
    """Control: the report sub-action is untouched by the fix."""
    store = tmp_path / "absent.jsonl"
    assert cli._cmd_publish_metrics(_Args("report", metrics_store=store)) == 0


def test_unknown_metrics_sub_action_returns_two(tmp_path):
    """Control: unknown sub-actions still report usage error 2."""
    assert cli._cmd_publish_metrics(_Args("bogus")) == 2


def test_ingest_json_still_works(tmp_path):
    """Control: the already-covered ingest branch keeps accepting JSON rows."""
    src = tmp_path / "rows.json"
    src.write_text(json.dumps([{"short_id": "x", "views": 3}]), encoding="utf-8")
    store = tmp_path / "metrics.jsonl"

    rc = cli._cmd_publish_metrics(_Args(
        "ingest", metrics_file=src, metrics_format="json", metrics_store=store))

    assert rc == 0
    assert [r["short_id"] for r in _rows(store)] == ["x"]
