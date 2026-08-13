from __future__ import annotations

import json
from pathlib import Path

from tools.coverage_audit import audit, failures


def _write_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "sample.py"
    source.write_text(
        """\ndef covered(x):\n    if x:\n        return 1\n    return 0\n\nclass Thing:\n    def partial(self, y):\n        if y > 2:\n            return y\n        return -1\n""",
        encoding="utf-8",
    )
    payload = {
        "totals": {"percent_covered": 75.0},
        "files": {
            "sample.py": {
                "executed_lines": [2, 3, 4, 7, 8, 9],
                "missing_lines": [5, 10],
                "executed_branches": [[3, 4], [9, 10]],
                "missing_branches": [[3, 5], [9, 11]],
            }
        },
    }
    cov = tmp_path / "coverage.json"
    cov.write_text(json.dumps(payload), encoding="utf-8")
    return cov


def test_audit_reports_nested_method_and_branch_aware_percent(tmp_path: Path):
    cov = _write_fixture(tmp_path)
    report = audit(cov, tmp_path)
    rows = {r["qualname"]: r for r in report["functions"]}
    assert set(rows) == {"covered", "Thing.partial"}
    assert rows["covered"]["statements"] == 4
    assert rows["covered"]["covered_statements"] == 3
    assert rows["covered"]["branches"] == 1
    assert rows["covered"]["covered_branches"] == 1
    assert rows["covered"]["percent"] == 80.0


def test_failures_enforces_total_and_function_floors(tmp_path: Path):
    report = audit(_write_fixture(tmp_path), tmp_path)
    problems = failures(report, min_total=90.0, min_function=90.0)
    assert problems[0] == "repository coverage 75.00% < 90.00%"
    assert any("covered" in problem for problem in problems)
    assert any("Thing.partial" in problem for problem in problems)


def test_failures_passes_when_thresholds_are_met(tmp_path: Path):
    report = audit(_write_fixture(tmp_path), tmp_path)
    assert failures(report, min_total=70.0, min_function=50.0) == []


def test_audit_skips_missing_or_unparseable_files(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("def nope(:\n", encoding="utf-8")
    cov = tmp_path / "coverage.json"
    cov.write_text(
        json.dumps(
            {
                "totals": {"percent_covered": 100.0},
                "files": {
                    "missing.py": {"executed_lines": [], "missing_lines": []},
                    "bad.py": {"executed_lines": [1], "missing_lines": []},
                },
            }
        ),
        encoding="utf-8",
    )
    report = audit(cov, tmp_path)
    assert report["functions"] == []
