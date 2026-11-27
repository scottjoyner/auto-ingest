"""Regression: render_plan must not report a no-footage 'rejected' short as
a rendered output (r25, 2026-11-26).

Hermetic: render_short is monkeypatched to emulate the real early-return path;
no moviepy, Neo4j, network, credentials, model load or clock access.
"""
from __future__ import annotations

from pathlib import Path

from auto_ingest.shorts import render
from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort, Shot


def _short(id_: str) -> PlannedShort:
    return PlannedShort(
        id=id_, title=id_, brief_topic="demo",
        shots=[Shot(clip_key=id_ + "_clip", fr_path="/no/such.mp4", t_sec=0, dur=6)],
        cues=[Cue(text="x", start=0, end=2)],
    )


def _plan(*items: PlannedShort) -> Plan:
    return Plan(
        topic="demo",
        brief=Brief(topic="demo", title="Demo", hook="hi"),
        shorts=list(items),
    )


def _rejected(it: PlannedShort, out_dir, **kw):
    """Emulate render_short's no-resolvable-footage early return."""
    it.status = "rejected"
    return Path(out_dir) / f"{it.id}.mp4"  # would-be path; no MP4 is written


def _rendered(it: PlannedShort, out_dir, **kw):
    """Emulate a successful render_short."""
    p = Path(out_dir) / f"{it.id}.mp4"
    it.status = "rendered"
    it.out_path = str(p)
    return p


class _UpsertAttempted(Exception):
    """Sentinel: the manifest-upsert branch was entered."""


def test_render_plan_excludes_rejected_short_from_outputs(monkeypatch, tmp_path):
    item = _short("nope")
    monkeypatch.setattr(render, "render_short", _rejected)
    out = render.render_plan(_plan(item), tmp_path)
    assert out == [], out
    assert item.status == "rejected"
    assert item.out_path is None
    assert not (tmp_path / "nope.mp4").exists()


def test_render_plan_rejected_only_does_not_attempt_manifest_upsert(monkeypatch, tmp_path):
    item = _short("nope")
    monkeypatch.setattr(render, "render_short", _rejected)

    def _creds():
        raise _UpsertAttempted()

    monkeypatch.setattr(render, "_neo4j_creds", _creds)
    warnings = []
    monkeypatch.setattr(render.log, "warning", lambda *a, **k: warnings.append(a))

    out = render.render_plan(_plan(item), tmp_path, upsert=True)
    assert out == [], out
    assert warnings == [], warnings  # upsert branch never entered


def test_render_plan_keeps_rendered_outputs_control(monkeypatch, tmp_path):
    item = _short("yes")
    monkeypatch.setattr(render, "render_short", _rendered)
    out = render.render_plan(_plan(item), tmp_path)
    assert [Path(p).name for p in out] == ["yes.mp4"]
    assert item.status == "rendered"


def test_render_plan_mixed_plan_returns_only_rendered(monkeypatch, tmp_path):
    bad, good = _short("bad"), _short("good")

    def _fake(it, out_dir, **kw):
        return _rejected(it, out_dir, **kw) if it.id == "bad" else _rendered(it, out_dir, **kw)

    monkeypatch.setattr(render, "render_short", _fake)
    out = render.render_plan(_plan(bad, good), tmp_path)
    assert [Path(p).name for p in out] == ["good.mp4"]
    assert bad.status == "rejected"
    assert good.status == "rendered"
