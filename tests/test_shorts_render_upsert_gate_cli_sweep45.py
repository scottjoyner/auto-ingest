"""Offline CLI test: `shorts render --upsert` must gate Neo4j manifest writes.

Hermetic by construction: `_cmd_render` is exercised with a fake driver and
fake plan/render collaborators. No Neo4j, network, credentials, moviepy, TTS,
rendered MP4, NAS mount or filesystem fixture is used (the fake plan's `save`
records the call instead of writing).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_ingest.shorts import cli


class _FakeDriver:
    def close(self):
        pass


class _FakeShort:
    def __init__(self):
        self.id = "s1"
        self.status = "planned"
        self.out_path = None
        self.shots = [SimpleNamespace(clip_key="clip-1")]


class _FakePlan:
    def __init__(self, calls):
        self._calls = calls
        self.shorts = [_FakeShort()]
        self.plan_id = "plan-1"
        self.topic = "topic"
        self.brief = SimpleNamespace(title="Title")

    def save(self, path):
        self._calls.append(("save", str(path)))


def _args(*, upsert: bool):
    return SimpleNamespace(
        plan="plan.json",
        out_dir=None,
        only=None,
        tts=False,
        upsert=upsert,
        skip_history=False,
        skip_used=False,
        pool=10,
    )


def _run_cmd(monkeypatch, *, upsert: bool):
    calls: list[tuple] = []
    plan = _FakePlan(calls)

    monkeypatch.setattr(cli, "_driver", lambda: (_FakeDriver(), "neo4j"))
    monkeypatch.setattr(cli, "Plan", SimpleNamespace(load=lambda path: plan))
    monkeypatch.setattr(cli.backdrop, "select_highway_pool", lambda *a, **k: [])
    monkeypatch.setattr(
        cli.planner, "iterate_plan",
        lambda *a, **k: pytest.fail("iterate_plan must not run: every short has a shot"),
    )
    render_calls: list[dict] = []
    monkeypatch.setattr(
        cli.render, "render_plan",
        lambda p, out_dir, **kwargs: (render_calls.append(kwargs) or ["out.mp4"]),
    )
    upsert_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.render, "upsert_manifest",
        lambda driver, p, out_dir: upsert_calls.append((driver, p, out_dir)),
    )

    rc = cli._cmd_render(_args(upsert=upsert))
    return rc, render_calls, upsert_calls


def test_render_without_upsert_does_not_write_manifest(monkeypatch):
    """DEFECT: base always called upsert_manifest because of an inverted guard."""
    rc, render_calls, upsert_calls = _run_cmd(monkeypatch, upsert=False)
    assert rc == 0
    assert render_calls and render_calls[0]["upsert"] is False
    assert upsert_calls == [], (
        "shorts render wrote a Neo4j manifest with --upsert absent"
    )


def test_render_with_upsert_still_enables_manifest_persistence(monkeypatch):
    """Regression guard: the opt-in flag must still reach render_plan."""
    rc, render_calls, upsert_calls = _run_cmd(monkeypatch, upsert=True)
    assert rc == 0
    assert render_calls and render_calls[0]["upsert"] is True
    assert upsert_calls == [], "render_plan handles the upsert; no duplicate CLI write"


def test_parser_exposes_upsert_as_opt_in():
    parser = cli.build_parser()
    assert parser.parse_args(["render", "p.json"]).upsert is False
    assert parser.parse_args(["render", "p.json", "--upsert"]).upsert is True
