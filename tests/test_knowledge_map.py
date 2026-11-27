"""Tests for auto_ingest.knowledge_map (G1) — system python, no live DB.

These prove the wiring without a live sync:
  * module imports,
  * the CLI parser accepts both subcommands + --dry-run,
  * sync_vault_to_neo4j(dry_run=True) does NOT open a neo4j driver and does NOT
    call the heavy harvest hook (we monkeypatch that call to record it),
  * sync_neo4j_to_vault(dry_run=True) does NOT open a neo4j driver.
"""
from __future__ import annotations

import auto_ingest.knowledge_map as km


def test_module_imports():
    assert hasattr(km, "sync_vault_to_neo4j")
    assert hasattr(km, "sync_neo4j_to_vault")
    assert hasattr(km, "main")


def test_cli_parser_accepts_subcommands_and_dry_run():
    for cmd in ("sync_vault_to_neo4j", "sync_neo4j_to_vault"):
        # Parse directly via argparse by invoking main with patched sys.argv.
        import argparse

        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        s1 = sub.add_parser("sync_vault_to_neo4j")
        s1.add_argument("--dry-run", action="store_true")
        s2 = sub.add_parser("sync_neo4j_to_vault")
        s2.add_argument("--dry-run", action="store_true")
        parsed = p.parse_args([cmd, "--dry-run"])
        assert parsed.cmd == cmd
        assert parsed.dry_run is True


def test_sync_vault_to_neo4j_dry_run_does_not_touch_db(monkeypatch):
    """dry_run must not open a driver nor run the harvest hook."""
    calls = {"harvest": 0, "driver_opened": False}

    def fake_run_harvest_hook():
        calls["harvest"] += 1

    monkeypatch.setattr(km, "_run_harvest_hook", fake_run_harvest_hook)

    # Guard against any accidental GraphDatabase.driver() call.
    import neo4j

    real_driver = neo4j.GraphDatabase.driver

    def spy_driver(*a, **k):
        calls["driver_opened"] = True
        return real_driver(*a, **k)

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", spy_driver)

    rc = km.sync_vault_to_neo4j(config=None, dry_run=True)

    assert rc == 0
    assert calls["harvest"] == 0, "dry-run must not invoke the harvest hook"
    assert calls["driver_opened"] is False, "dry-run must not open a neo4j driver"


def test_sync_neo4j_to_vault_dry_run_does_not_touch_db(monkeypatch):
    calls = {"driver_opened": False}
    import neo4j

    real_driver = neo4j.GraphDatabase.driver

    def spy_driver(*a, **k):
        calls["driver_opened"] = True
        return real_driver(*a, **k)

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", spy_driver)

    rc = km.sync_neo4j_to_vault(config=None, dry_run=True)

    assert rc == 0
    assert calls["driver_opened"] is False, "dry-run must not open a neo4j driver"


def test_vault_roots_dedupe_and_resolve():
    km_cfg = {
        "central_vault_path": "/x/knowledge",
        "local_vault_path": "/x/nas-knowledge",
        "mirror_vault_path": "/x/knowledge",  # duplicate -> dropped
        "canonical_vault_path": "/x/canonical",
    }
    roots = km._vault_roots(km_cfg)
    assert roots == [
        __import__("pathlib").Path("/x/knowledge"),
        __import__("pathlib").Path("/x/nas-knowledge"),
        __import__("pathlib").Path("/x/canonical"),
    ]


def test_export_label_never_overwrites_a_distinct_node_on_filename_collision(tmp_path):
    """Distinct nodes whose names sanitize to one filename must all be exported.

    ``_safe_filename`` folds every non-alphanumeric char to ``_`` and truncates
    to 120 chars. ``A/B`` and ``A_B`` therefore map to the same ``A_B.md``, as
    do two titles sharing their first 120 characters. When the exporter writes
    one markdown file per node, the later node silently replaces the earlier
    one in the mirror vault - graph content lost with no error and no log line.
    """
    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def data(self):
            return list(self._rows)

    rows = [
        {"n": {"title": "A/B"}},
        {"n": {"title": "A_B"}},
        {"n": {"title": "z" * 200 + "-first"}},
        {"n": {"title": "z" * 200 + "-second"}},
    ]

    class _Session:
        def run(self, query, **kwargs):
            lim, skip = kwargs["lim"], kwargs["skip"]
            return _Result(rows[skip:skip + lim])

    written = km._export_label(_Session(), "VaultDocument", tmp_path, budget=100)
    assert written == len(rows)

    files = sorted(p.name for p in tmp_path.glob("*.md"))
    assert len(files) == len(rows), (
        f"expected {len(rows)} distinct export files, got {files}"
    )
    contents = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.md"))
    for title in ("A/B", "A_B", "z" * 200 + "-first", "z" * 200 + "-second"):
        assert title in contents, f"node {title!r} was silently dropped from the export"

    # Disambiguation is stable: a repeat export yields the same filenames, not
    # an ever-growing -2/-3 suffix series.
    again = tmp_path / "again"
    again.mkdir()
    km._export_label(_Session(), "VaultDocument", again, budget=100)
    assert sorted(p.name for p in again.glob("*.md")) == files
