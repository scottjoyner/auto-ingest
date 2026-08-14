from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from auto_ingest import diagnostics


def test_environment_presence_never_returns_secret_values(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://example")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-secret-value")
    report = diagnostics.environment_presence()
    encoded = json.dumps(report)
    assert report["configured"]["NEO4J_URI"] is True
    assert report["sensitive_present"]["NEO4J_PASSWORD"] is True
    assert "super-secret-value" not in encoded
    assert "bolt://example" not in encoded


def test_storage_and_tool_probes(monkeypatch, tmp_path: Path):
    rows = diagnostics.storage_status([tmp_path, tmp_path / "missing"])
    assert rows[0]["exists"] is True
    assert rows[0]["total_gb"] >= rows[0]["free_gb"]
    assert rows[1] == {"path": str(tmp_path / "missing"), "exists": False}

    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    assert diagnostics._safe_command(("missing", "--version")) == {"available": False}

    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "/bin/tool")
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="tool 1.2\nmore", stderr=""),
    )
    status = diagnostics._safe_command(("tool", "--version"))
    assert status == {"available": True, "returncode": 0, "summary": "tool 1.2"}


def test_neo4j_status_is_presence_only_when_unconfigured(monkeypatch):
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert diagnostics.neo4j_status() == {"configured": False}


def test_bundle_contains_sanitized_json(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        diagnostics,
        "collect",
        lambda: {"platform": {"python": "3.12"}, "environment": {"sensitive_present": {}}},
    )
    target = diagnostics.write_bundle(tmp_path / "diag.tar.gz")
    assert target.exists()
    with tarfile.open(target, "r:gz") as archive:
        member = archive.extractfile("auto-ingest-diagnostics/diagnostics.json")
        assert member is not None
        payload = json.loads(member.read().decode("utf-8"))
    assert payload["platform"]["python"] == "3.12"


def test_collect_has_expected_sections(monkeypatch):
    monkeypatch.setattr(diagnostics, "tool_status", lambda: {"ffmpeg": {"available": True}})
    monkeypatch.setattr(diagnostics, "package_versions", lambda: {"numpy": "1"})
    monkeypatch.setattr(diagnostics, "storage_status", lambda paths: [])
    monkeypatch.setattr(diagnostics, "neo4j_status", lambda: {"configured": False})
    report = diagnostics.collect()
    assert {"platform", "environment", "packages", "tools", "storage", "neo4j"} <= set(report)
