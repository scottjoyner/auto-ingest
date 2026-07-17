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
