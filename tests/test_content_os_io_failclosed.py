"""Fail-closed metadata parsing in ``content_os.io.load_yaml``.

Regression guard for the PyYAML-absent fallback: a corrupt JSON metadata
document must raise, not be silently converted into a plausible-looking dict
by the naive line parser.  When it does not raise, ``core.doctor`` reports a
corrupt ``config.yaml`` as healthy and ``record_source_manifest`` silently
drops every previously recorded source before rewriting the file.

Hermetic: tmp_path only, no live pipeline, no model, no Neo4j, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from content_os import core, io


def _no_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the documented PyYAML-absent fallback deterministically."""
    monkeypatch.setattr(io, "yaml", None)


def test_load_yaml_rejects_malformed_json_metadata(monkeypatch, tmp_path: Path):
    _no_pyyaml(monkeypatch)
    meta = tmp_path / "source-manifest.json"
    meta.write_text(
        '{"sources": [{"source_id": "abc"}], "truncated":', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="[Mm]alformed JSON"):
        io.load_yaml(meta)


def test_doctor_flags_malformed_json_config_in_fallback_mode(
    monkeypatch, tmp_path: Path
):
    root = tmp_path / "content-os"
    core.init(root)
    (root / "config.yaml").write_text('{"llm": {"provider":', encoding="utf-8")
    _no_pyyaml(monkeypatch)

    issues = core.doctor(root)

    assert any("Malformed" in issue and "config.yaml" in issue for issue in issues), (
        f"doctor silently accepted a corrupt config.yaml: {issues!r}"
    )


def test_load_yaml_fallback_still_parses_simple_yaml(monkeypatch, tmp_path: Path):
    """Control: the line-oriented fallback must keep working for real YAML."""
    _no_pyyaml(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider: ollama\ntemperature: 0.2\n", encoding="utf-8")

    assert io.load_yaml(cfg) == {"provider": "ollama", "temperature": 0.2}


def test_load_yaml_fallback_still_parses_valid_json(monkeypatch, tmp_path: Path):
    """Control: valid JSON metadata is still returned as a mapping."""
    _no_pyyaml(monkeypatch)
    meta = tmp_path / "source-manifest.json"
    meta.write_text('{"sources": [1, 2], "ok": true}', encoding="utf-8")

    assert io.load_yaml(meta) == {"sources": [1, 2], "ok": True}
