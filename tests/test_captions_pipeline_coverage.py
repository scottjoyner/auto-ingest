from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from auto_ingest.shorts import captions as c


def profile():
    return {
        "font_path": "/definitely/missing.ttf",
        "font_size": 22,
        "line_height_px": 32,
        "stroke_px": 1,
        "max_width_px": 220,
        "colors": {
            "text": [255, 255, 255, 255],
            "stroke": [0, 0, 0, 255],
            "highlight_text": [0, 0, 0, 255],
        },
        "speaker_overrides": {
            "g1": {"font_size": 28, "colors": {"text": [1, 2, 3, 255]}}
        },
    }


def test_profiles_speakers_normalization_and_colors(tmp_path):
    pp = tmp_path / "profiles.json"
    pp.write_text(json.dumps({"x": profile()}))
    loaded = c.load_profiles_from_json(pp)
    assert isinstance(loaded["x"]["colors"]["text"], tuple)
    smp = tmp_path / "speakers.json"
    smp.write_text(json.dumps({"A": {"global_id": "g1", "name": "Alice"}}))
    smap = c.load_speaker_map(smp)
    assert smap["A"].name == "Alice" and c.load_speaker_map(None) == {}
    p = c._normalize_profile_from_json(profile())
    assert isinstance(p["speaker_overrides"]["g1"]["colors"]["text"], tuple)
    assert c._normalize_profile_from_json("x") == "x"
    assert c._normalize_colors_dict("x") == "x"
    merged = c._apply_speaker_override(p, "A", smap)
    assert merged["font_size"] == 28 and merged["colors"]["text"] == (1, 2, 3, 255)
    assert c._apply_speaker_override(p, None, smap) is p
    assert c._speaker_display_name("A", smap) == "Alice"
    assert c._speaker_display_name("B", smap) == "B"
    assert c._speaker_display_name(None, smap) is None
    assert c._speaker_color(None, smap) == (255, 235, 59, 230)
    assert c._speaker_color("A", smap) == c._hash_color("g1")
    assert len({c._hash_color(f"key-{i}") for i in range(20)}) > 10


def test_wrap_tags_kind_and_caption_images():
    font = c._load_font("/missing.ttf", 18)
    img = Image.new("RGBA", (240, 200))
    draw = ImageDraw.Draw(img)
    lines = c._wrap_text(
        draw, "one two three supercalifragilisticexpialidocious", font, 70
    )
    assert len(lines) > 2
    tag = c._render_tag("Topic", font, (255, 0, 0, 255))
    assert tag.width > 0 and tag.height > 0
    assert c._cue_kind_style("hook")["tag"] == "Topic"
    assert c._cue_kind_style("unknown")["font_scale"] == 1.0
    cap = c._build_caption_image("hello world", 320, profile(), "hook")
    assert cap.width == 320 and cap.height > 0


def test_sentence_highlight_wordgrid_and_render_cache(tmp_path):
    p = c._normalize_profile_from_json(profile())
    smap = {"A": c.SpeakerMapEntry("A", "g1", "Alice")}
    a = c._draw_sentence("hello world", 320, p, "A", smap)
    b = c._draw_sentence_highlight("hello world", "world", 320, p, "A", smap)
    words = [{"word": "hello", "speaker": "A"}, {"word": "world", "speaker": "B"}]
    g = c._draw_wordgrid(words, 0, 320, p, smap)
    assert a.width == b.width == g.width == 320
    key = c._cap_key(
        "sentence",
        sentence="cached",
        width=320,
        profile=p,
        speaker_label=None,
        speaker_map={},
    )
    arr = c._render_cached(key)
    assert arr.shape[1] == 320
    key2 = c._cap_key(
        "karaoke",
        sentence="hello world",
        word="world",
        width=320,
        profile=p,
        speaker_label=None,
        speaker_map={},
    )
    assert c._render_cached(key2).shape[1] == 320
    key3 = c._cap_key(
        "wordgrid", words=words, active_idx=1, width=320, profile=p, speaker_map={}
    )
    assert c._render_cached(key3).shape[1] == 320
    bad = json.dumps({"kind": "bad", "profile": p, "speaker_map": {}})
    try:
        c._render_cached(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("bad kind must fail")


def test_cue_timings_scrim_hud_endcard_and_kenburns():
    assert c._cue_word_timings("", 0, 1, None) == []
    explicit = c._cue_word_timings(
        "a b", 0, 1, [{"word": "a", "start": 0, "end": 0.4}, {"word": "b"}]
    )
    assert explicit == [("a", 0.0, 0.4)]
    even = c._cue_word_timings("a b c", 0, 0.3, None)
    assert len(even) == 3 and all(end - start >= 0.18 for _, start, end in even)
    scrim = c._draw_scrim(20, 10)
    hud = c._draw_speed_hud(54.6, 400, 200)
    end = c._draw_end_card(400, 700, "topic")
    assert scrim.size == (20, 10) and hud.size == (400, 200) and end.size == (400, 700)

    class Clip:
        w = 4
        h = 4
        duration = 2

        def fl(self, fn):
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            out = fn(lambda t: frame, 1.0)
            assert out.shape == frame.shape
            return self

        def resize(self, size):
            assert size == (4, 4)
            return self

    assert c._apply_kenburns(Clip(), zoom=1.2) is not None


def test_atomic_paths_and_safe_video_write(monkeypatch, tmp_path):
    final = tmp_path / "nested" / "out.mp4"
    c._ensure_parent_dir(final)
    assert final.parent.exists()
    temp = c._temp_path_with_same_ext(final)
    assert temp.suffix == ".mp4" and ".__tmp__" in temp.name
    src = tmp_path / "source.mp4"
    src.write_bytes(b"x")
    c._atomic_replace(src, final)
    assert final.read_bytes() == b"x"

    class Clip:
        def write_videofile(self, path, **kw):
            assert kw["codec"] == "libx264"
            assert "-movflags" in kw["ffmpeg_params"]
            with open(path, "wb") as fh:
                fh.write(b"video")

    final2 = tmp_path / "out2.mp4"
    c._write_videofile_safely(
        Clip(), final2, fps=30, extra_ffmpeg_params=["-metadata", "x=y"]
    )
    assert final2.read_bytes() == b"video"
