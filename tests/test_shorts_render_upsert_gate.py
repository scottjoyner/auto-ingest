"""The ``shorts render --upsert`` flag must actually gate Neo4j manifest writes.

``render_plan(upsert=...)`` is the documented place that persists a :ShortPlan /
:Short manifest, and its help text calls ``--upsert`` the switch that enables it.
The CLI used to call ``upsert_manifest`` precisely when ``--upsert`` was *absent*,
so a plain ``shorts render`` wrote to Neo4j and there was no invocation that
rendered without persisting a manifest. These tests pin the gate with fully
mocked I/O (no Neo4j, no filesystem, no model).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_ingest.shorts.cli as cli


def _args(**overrides):
    values = {
        "plan": "plan.json",
        "only": None,
        "out_dir": None,
        "pool": 10,
        "tts": False,
        "upsert": False,
        "skip_history": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def render_env(monkeypatch):
    calls: dict = {"rendered_paths": [Path("s1.mp4")]}

    shot = SimpleNamespace(clip_key="clip_A")
    short = SimpleNamespace(id="s1", shots=[shot])
    plan = SimpleNamespace(
        shorts=[short],
        save=lambda path: calls.setdefault("saved", str(path)),
    )
    monkeypatch.setattr(cli.Plan, "load", staticmethod(lambda path: plan))

    driver = SimpleNamespace(close=lambda: calls.setdefault("closed", True))
    monkeypatch.setattr(cli, "_driver", lambda: (driver, "neo4j"))
    monkeypatch.setattr(cli.backdrop, "select_highway_pool", lambda d, limit: [])

    def _no_iterate(*a, **k):
        raise AssertionError("iterate_plan must not run: plan already has shots")

    monkeypatch.setattr(cli.planner, "iterate_plan", _no_iterate)

    def _fake_render_plan(plan, out_dir, **kw):
        calls["render_plan_kwargs"] = kw
        return calls["rendered_paths"]

    monkeypatch.setattr(cli.render, "render_plan", _fake_render_plan)

    def _fake_upsert(driver, plan, out_dir):
        calls["upsert_count"] = calls.get("upsert_count", 0) + 1

    monkeypatch.setattr(cli.render, "upsert_manifest", _fake_upsert)
    return calls


def test_default_render_does_not_write_manifest(render_env):
    """Without --upsert, no manifest write may happen (the fail-open case)."""
    assert cli._cmd_render(_args(upsert=False)) == 0
    assert render_env["render_plan_kwargs"]["upsert"] is False
    assert render_env.get("upsert_count", 0) == 0


def test_upsert_with_rendered_shorts_is_owned_by_render_plan(render_env):
    """With --upsert and clips rendered, render_plan performs the single write."""
    assert cli._cmd_render(_args(upsert=True)) == 0
    assert render_env["render_plan_kwargs"]["upsert"] is True
    assert render_env.get("upsert_count", 0) == 0


def test_upsert_with_no_rendered_shorts_still_persists_plan(render_env):
    """With --upsert but zero renders, the CLI fallback persists the plan once."""
    render_env["rendered_paths"] = []
    assert cli._cmd_render(_args(upsert=True)) == 0
    assert render_env.get("upsert_count", 0) == 1
