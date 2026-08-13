from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from auto_ingest.shorts import compose as c


class FakeClip:
    def __init__(self, duration=8.0, w=1920, h=1080, fps=30.0):
        self.duration = duration
        self.w = w
        self.h = h
        self.fps = fps
        self.audio = None
        self.closed = False

    def without_audio(self): self.audio = None; return self
    def subclip(self, start, end): return FakeClip(max(0, end-start), self.w, self.h, self.fps)
    def crop(self, **kw): return self
    def resize(self, size): self.w, self.h = size; return self
    def crossfadein(self, _n): return self
    def crossfadeout(self, _n): return self
    def set_start(self, _n): return self
    def set_duration(self, n): self.duration = n; return self
    def set_position(self, _p): return self
    def fadein(self, _n): return self
    def set_audio(self, a): self.audio = a; return self
    def fx(self, _fn, **_kw): return self
    def close(self): self.closed = True


class FakeAudio(FakeClip):
    pass


def install_moviepy(monkeypatch, *, duration=8.0):
    editor = ModuleType("moviepy.editor")
    editor.VideoFileClip = lambda _p: FakeClip(duration=duration)
    editor.ImageClip = lambda _a: FakeClip(duration=0, w=100, h=100)
    editor.CompositeVideoClip = lambda clips: FakeClip(duration=max((x.duration for x in clips), default=0))
    editor.concatenate_videoclips = lambda clips, method=None: FakeClip(duration=sum(x.duration for x in clips))
    editor.AudioFileClip = lambda _p: FakeAudio(duration=2.0)
    moviepy = ModuleType("moviepy"); moviepy.editor = editor
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    monkeypatch.setitem(sys.modules, "moviepy.editor", editor)
    fxall = ModuleType("moviepy.audio.fx.all"); fxall.audio_loop = object()
    monkeypatch.setitem(sys.modules, "moviepy.audio", ModuleType("moviepy.audio"))
    monkeypatch.setitem(sys.modules, "moviepy.audio.fx", ModuleType("moviepy.audio.fx"))
    monkeypatch.setitem(sys.modules, "moviepy.audio.fx.all", fxall)


def patch_drawers(monkeypatch):
    profile = {
        "kenburns": True, "show_speed_hud": True, "karaoke": True,
        "anim_sec": 0.2, "font_size": 20, "font_path": "x", "line_height_px": 40,
    }
    monkeypatch.setattr(c, "DEFAULT_PROFILES", {"clean": profile})
    monkeypatch.setattr(c, "_normalize_profile_from_json", lambda p: dict(p))
    monkeypatch.setattr(c, "_apply_kenburns", lambda clip: clip)
    monkeypatch.setattr(c, "_draw_speed_hud", lambda *a: Image.new("RGBA", (20, 10)))
    monkeypatch.setattr(c, "_draw_scrim", lambda w, h: Image.new("RGBA", (w, max(1, h))))
    monkeypatch.setattr(c, "_build_caption_image", lambda *a: Image.new("RGBA", (80, 30)))
    monkeypatch.setattr(c, "_draw_sentence_highlight", lambda *a: Image.new("RGBA", (80, 30)))
    monkeypatch.setattr(c, "_cue_kind_style", lambda kind: {"font_scale": 1.0, "tag": "TAG" if kind == "hook" else "", "accent": "x"})
    monkeypatch.setattr(c, "_load_font", lambda *a: object())
    monkeypatch.setattr(c, "_render_tag", lambda *a: Image.new("RGBA", (30, 10)))
    monkeypatch.setattr(c, "_draw_end_card", lambda *a: Image.new("RGBA", (80, 100)))
    monkeypatch.setattr(c, "_cue_word_timings", lambda text, start, end, words: [(w, start+i*.4, min(end, start+(i+1)*.4)) for i, w in enumerate(text.split())])
    return profile


def test_compose_rejects_missing_shots(monkeypatch, tmp_path):
    install_moviepy(monkeypatch)
    patch_drawers(monkeypatch)
    with pytest.raises(RuntimeError):
        c.compose_scripted_short([{"fr_path": str(tmp_path / "missing.mp4")}], [], tmp_path / "out.mp4")


def test_compose_executes_karaoke_hud_endcard_and_audio(monkeypatch, tmp_path):
    install_moviepy(monkeypatch, duration=5.0)
    profile = patch_drawers(monkeypatch)
    a = tmp_path / "a.mp4"; a.write_bytes(b"x")
    b = tmp_path / "b.mp4"; b.write_bytes(b"x")
    narration = tmp_path / "n.wav"; narration.write_bytes(b"x")
    writes = []
    monkeypatch.setattr(c, "_write_videofile_safely", lambda clip, out, **kw: writes.append((clip, out, kw)))
    c.compose_scripted_short(
        [
            {"fr_path": str(a), "t_sec": 4.0, "dur": 3.0, "mph": 45},
            {"fr_path": str(b), "t_sec": 0.0, "dur": 2.0, "mph": None},
        ],
        [
            {"start": 0.0, "end": 1.5, "text": "two words", "kind": "hook"},
            {"start": 1.5, "end": 1.0, "text": "single", "kind": "line"},
            {"start": 99, "end": 100, "text": "late"},
            {"start": 2, "end": 3, "text": "   "},
        ],
        tmp_path / "out.mp4", profile=profile, narration_audio=narration,
        end_card=True, hashtag="",
    )
    assert writes and writes[0][2]["codec"] == "libx264"
    assert writes[0][2]["audio_codec"] == "aac"


def test_compose_non_karaoke_and_silent_write_fallback(monkeypatch, tmp_path):
    install_moviepy(monkeypatch, duration=6.0)
    profile = patch_drawers(monkeypatch)
    profile.update({"karaoke": False, "kenburns": False, "show_speed_hud": False, "anim_sec": 0})
    src = tmp_path / "a.mp4"; src.write_bytes(b"x")
    calls = []
    def writer(clip, out, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("first write fails")
    monkeypatch.setattr(c, "_write_videofile_safely", writer)
    c.compose_scripted_short(
        [{"fr_path": str(src), "t_sec": 0, "dur": 4}],
        [{"start": 0, "end": 2, "text": "one line", "kind": "line"}],
        tmp_path / "out.mp4", profile=profile, end_card=False,
    )
    assert len(calls) == 2 and calls[1]["audio_codec"] is None
