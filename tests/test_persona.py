"""
Persona / on-screen personality tests — pure + graceful-degradation paths.

The real talking-head (ONNX/Wav2Lip ONNX via onnxruntime) and XTTS voice clone
owner media, so those raise PersonaUnavailable here. We test the parts that
run anywhere: config resolution, the stylized-avatar fallback composite, and
that compose_with_persona degrades to a valid video when voice/face are absent.
"""
from pathlib import Path

import pytest

from auto_ingest.shorts import persona
from auto_ingest.shorts.models import PlannedShort

HAS_MOVIEPY = True
SMOKE = Path("/tmp/opencode/smoke.mp4")
try:
    import moviepy  # noqa: F401
except Exception:
    HAS_MOVIEPY = False


def test_persona_config_from_env(monkeypatch):
    monkeypatch.setenv("PERSONA_SOURCE", "photo")
    monkeypatch.setenv("PERSONA_FACE_PATH", "/tmp/face.jpg")
    cfg = persona.PersonaConfig.from_env()
    assert cfg.source == "photo"
    assert cfg.face_path == "/tmp/face.jpg"
    # invalid source falls back to stylized
    monkeypatch.setenv("PERSONA_SOURCE", "bogus")
    assert persona.PersonaConfig.from_env().source == "stylized"


def test_resolve_persona_variants():
    # render.py helper isn't imported here; test PersonaConfig paths directly.
    assert persona.PersonaConfig.from_env().source in ("stylized", "photo", "video")
    c = persona.PersonaConfig("video")
    assert c.source == "video"


def test_make_talking_head_unavailable_without_model():
    # No GPU/model on this host -> PersonaUnavailable, not a crash.
    try:
        persona.make_talking_head(Path("/tmp/x.audio.wav"), Path("/tmp/x.face.jpg"),
                             Path("/tmp/x.face.mp4"))
    except persona.PersonaUnavailable:
        pass
    else:
        raise AssertionError("expected PersonaUnavailable (no model on host)")


def _make_smoke_clip(path):
    """Generate a tiny synthetic base clip so the persona test runs without an
    external asset (S-G15: the old test silently skipped without SMOKE)."""
    from moviepy.editor import ColorClip
    clip = ColorClip(size=(1080, 1920), color=(20, 20, 30), duration=1.0)
    clip.fps = 24
    clip.write_videofile(str(path), codec="libx264", audio=False, fps=24)
    clip.close()


def test_compose_with_persona_falls_back_to_stylized(tmp_path):
    # With a base video + stylized config, compose_with_persona should produce
    # a valid output (voice clone absent -> base audio; face -> stylized avatar).
    if not HAS_MOVIEPY:
        pytest.skip("moviepy not installed in this interpreter")
    base = tmp_path / "base.mp4"
    _make_smoke_clip(base)
    out = tmp_path / "persona_out.mp4"
    cfg = persona.PersonaConfig(source="stylized")
    # narrate_script is environment-dependent; compose_with_persona falls back
    # to the base B-roll audio when no narration is produced.
    result = persona.compose_with_persona(base, [{"text": "hi"}], cfg, out)
    assert result.exists()
    assert result.stat().st_size > 1000


def test_planned_short_persona_field_roundtrip(tmp_path):
    s = PlannedShort(id="x", brief_topic="t", title="T", persona="stylized")
    p = tmp_path / "plan.json"
    # round-trip via a minimal Plan
    from auto_ingest.shorts.models import Brief, Plan
    plan = Plan(topic="t", brief=Brief(topic="t", title="T", hook="h"),
                shorts=[s])
    plan.save(p)
    loaded = Plan.load(p)
    assert loaded.shorts[0].persona == "stylized"
