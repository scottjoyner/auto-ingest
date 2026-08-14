from __future__ import annotations

import json
import tarfile
from pathlib import Path

from auto_ingest.diagnostic_diff import compare, load_report


def test_compare_ignores_capture_time_and_reports_environment_drift():
    left = {
        "generated_at_epoch": 1,
        "platform": {"python": "3.12.1", "machine": "x86_64"},
        "packages": {"numpy": "2.1.0"},
        "tools": {"ffmpeg": {"available": True}},
    }
    right = {
        "generated_at_epoch": 2,
        "platform": {"python": "3.12.2", "machine": "x86_64"},
        "packages": {"numpy": "2.2.0"},
        "tools": {"ffmpeg": {"available": False}},
    }
    changes = compare(left, right)
    fields = {row["field"] for row in changes}
    assert fields == {"packages.numpy", "platform.python", "tools.ffmpeg.available"}


def test_load_report_accepts_json_and_bundle(tmp_path: Path):
    payload = {"platform": {"python": "3.12"}}
    json_path = tmp_path / "diagnostics.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_report(json_path) == payload

    root = tmp_path / "bundle-root" / "auto-ingest-diagnostics"
    root.mkdir(parents=True)
    (root / "diagnostics.json").write_text(json.dumps(payload), encoding="utf-8")
    bundle = tmp_path / "diag.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(root, arcname="auto-ingest-diagnostics")
    assert load_report(bundle) == payload


def test_compare_reports_missing_fields_explicitly():
    changes = compare({"packages": {"numpy": "1"}}, {"packages": {}})
    assert changes == [{"field": "packages.numpy", "left": "1", "right": "<missing>"}]
