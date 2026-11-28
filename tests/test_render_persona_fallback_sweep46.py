"""Sweep46 — persona-overlay failure must keep the base render as the final MP4.

Hermetic: no moviepy, no Neo4j, no model, no network. The compose step and the
persona overlay are replaced with tiny in-process stubs, and all files live in
pytest's scratch tmp_path.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from auto_ingest.shorts import persona as persona_mod
from auto_ingest.shorts import render as render_mod
from auto_ingest.shorts.models import Cue, PlannedShort, Shot


class _FakeClip:
    def close(self) -> None:
        pass


@pytest.fixture()
def stub_render_stack(monkeypatch):
    """Replace the video stack with deterministic in-process stand-ins."""
    def fake_compose(shots, cues, out_path, **kwargs):
        Path(out_path).write_bytes(b"BASE-MP4")

    monkeypatch.setattr(
        "auto_ingest.shorts.compose.compose_scripted_short", fake_compose,
        raising=False,
    )

    editor = types.ModuleType("moviepy.editor")
    editor.VideoFileClip = lambda _path: _FakeClip()
    moviepy = types.ModuleType("moviepy")
    moviepy.__path__ = []  # mark as package so `moviepy.editor` resolves
    moviepy.editor = editor
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    monkeypatch.setitem(sys.modules, "moviepy.editor", editor)

    def fake_thumbnail(mp4, out_path, **kwargs):
        Path(out_path).write_bytes(b"THUMB")

    monkeypatch.setattr(
        "auto_ingest.shorts.thumbnail.make_thumbnail", fake_thumbnail,
        raising=False,
    )
    return monkeypatch


def _item(tmp_path: Path) -> PlannedShort:
    footage = tmp_path / "footage.mp4"
    footage.write_bytes(b"stub-footage")
    return PlannedShort(
        id="sweep46",
        brief_topic="research",
        title="Test short",
        cues=[Cue(start=0.0, end=1.0, text="hello")],
        shots=[Shot(clip_key="clip-1", fr_path=str(footage), t_sec=0.0, dur=1.0)],
    )


def _cfg() -> persona_mod.PersonaConfig:
    return persona_mod.PersonaConfig(source="stylized")


# ---------------------------------------------------------------------------
# Defect cases — persona overlay fails after the base render succeeded.
# ---------------------------------------------------------------------------

def test_persona_failure_keeps_base_render_as_final_mp4(tmp_path, stub_render_stack):
    def boom(*args, **kwargs):
        raise RuntimeError("persona engine unavailable")

    stub_render_stack.setattr(persona_mod, "compose_with_persona", boom)

    out_dir = tmp_path / "out"
    item = _item(tmp_path)
    returned = render_mod.render_short(item, out_dir, persona=_cfg())

    final = out_dir / "sweep46.mp4"
    base = out_dir / "sweep46.base.mp4"
    assert returned == final
    assert final.is_file(), (
        "render_short reported a rendered short but deleted the base render "
        "after the persona overlay failed"
    )
    assert final.read_bytes() == b"BASE-MP4"
    assert not base.exists(), "base temp must not be left behind after promotion"


def test_persona_failure_status_rendered_implies_a_file(tmp_path, stub_render_stack):
    def boom(*args, **kwargs):
        raise RuntimeError("persona engine unavailable")

    stub_render_stack.setattr(persona_mod, "compose_with_persona", boom)

    out_dir = tmp_path / "out"
    item = _item(tmp_path)
    render_mod.render_short(item, out_dir, persona=_cfg())

    assert item.status == "rendered"
    assert item.out_path is not None
    assert Path(item.out_path).is_file(), "status 'rendered' with no MP4 on disk"


# ---------------------------------------------------------------------------
# Controls — behaviour that must not change.
# ---------------------------------------------------------------------------

def test_successful_persona_overlay_still_writes_final_and_removes_base(
        tmp_path, stub_render_stack):
    def compose_persona(broll_mp4, cues, cfg, out_mp4, **kwargs):
        Path(out_mp4).write_bytes(b"PERSONA-MP4")
        return Path(out_mp4)

    stub_render_stack.setattr(
        persona_mod, "compose_with_persona", compose_persona,
    )

    out_dir = tmp_path / "out"
    item = _item(tmp_path)
    render_mod.render_short(item, out_dir, persona=_cfg())

    assert (out_dir / "sweep46.mp4").read_bytes() == b"PERSONA-MP4"
    assert not (out_dir / "sweep46.base.mp4").exists()


def test_plain_render_has_no_base_temp(tmp_path, stub_render_stack):
    out_dir = tmp_path / "out"
    item = _item(tmp_path)
    render_mod.render_short(item, out_dir, persona=None)

    assert (out_dir / "sweep46.mp4").read_bytes() == b"BASE-MP4"
    assert not (out_dir / "sweep46.base.mp4").exists()
