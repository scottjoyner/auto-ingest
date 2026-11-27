"""Round-37: hermetic tests for caption profile normalization + speaker overrides.

This pins a previously uncovered, purely-local invariant in
``auto_ingest.shorts.captions``: JSON-serialized caption profiles carry their
RGBA colors as *lists*, but the renderer needs PIL-compatible *tuples*, and
per-speaker overrides must be recast the same way and merged with well-defined
precedence before rendering. Nothing here renders an image, touches the
network, a graph database, the GPU, or any credential; fixtures are small JSON
files under pytest's ``tmp_path``.
"""

from __future__ import annotations

import json

from auto_ingest.shorts import captions
from auto_ingest.shorts.captions import SpeakerMapEntry


def test_load_profiles_from_json_recasts_colors(tmp_path):
    raw = {
        "clean": {
            "font_size": 64,
            "colors": {
                "text": [255, 255, 255, 255],
                "stroke": [0, 0, 0, 255],
            },
            "karaoke": True,
        }
    }
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    profs = captions.load_profiles_from_json(p)

    assert set(profs) == {"clean"}
    prof = profs["clean"]
    assert prof["colors"]["text"] == (255, 255, 255, 255)
    assert prof["colors"]["stroke"] == (0, 0, 0, 255)
    assert isinstance(prof["colors"]["text"], tuple)
    # Non-color keys are carried through unchanged.
    assert prof["font_size"] == 64
    assert prof["karaoke"] is True


def test_load_speaker_map_missing_path_is_empty(tmp_path):
    assert captions.load_speaker_map(None) == {}
    assert captions.load_speaker_map(tmp_path / "does-not-exist.json") == {}


def test_load_speaker_map_builds_entries_with_optional_fields(tmp_path):
    p = tmp_path / "speakers.json"
    p.write_text(
        json.dumps(
            {
                "SPEAKER_00": {"global_id": "gs-1", "name": "Alice"},
                "SPEAKER_01": {"global_id": None, "name": None},
                "SPEAKER_02": {},
            }
        ),
        encoding="utf-8",
    )

    smap = captions.load_speaker_map(p)

    assert set(smap) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert smap["SPEAKER_00"] == SpeakerMapEntry(
        local_label="SPEAKER_00", global_id="gs-1", name="Alice"
    )
    assert smap["SPEAKER_01"].global_id is None
    assert smap["SPEAKER_02"].name is None
    # local_label is always derived from the JSON key, never from the value.
    assert smap["SPEAKER_02"].local_label == "SPEAKER_02"


def test_normalize_colors_dict_recasts_lists_only():
    src = {
        "text": [1, 2, 3, 4],
        "stroke": (0, 0, 0, 255),
        "alpha": 128,
        "note": "keep-me",
    }
    out = captions._normalize_colors_dict(src)

    assert out["text"] == (1, 2, 3, 4)
    assert isinstance(out["text"], tuple)
    assert out["stroke"] == (0, 0, 0, 255)
    assert out["alpha"] == 128
    assert out["note"] == "keep-me"
    # Input is never mutated.
    assert isinstance(src["text"], list)


def test_normalize_colors_dict_passthrough_for_non_dict():
    assert captions._normalize_colors_dict([1, 2, 3]) == [1, 2, 3]
    assert captions._normalize_colors_dict(None) is None


def test_normalize_profile_recasts_top_level_and_nested_override_colors():
    prof = {
        "font_size": 66,
        "colors": {"text": [1, 2, 3, 4]},
        "speaker_overrides": {
            "gs-1": {
                "colors": {"text": [9, 8, 7, 6], "stroke": [0, 0, 0, 200]},
                "font_size": 80,
            },
            "gs-2": {"font_path": "/some/font.ttf"},
        },
    }
    out = captions._normalize_profile_from_json(prof)

    assert out["colors"]["text"] == (1, 2, 3, 4)
    ov = out["speaker_overrides"]["gs-1"]
    assert ov["colors"]["text"] == (9, 8, 7, 6)
    assert isinstance(ov["colors"]["stroke"], tuple)
    assert ov["font_size"] == 80
    # An override without colors is left structurally intact.
    assert out["speaker_overrides"]["gs-2"] == {"font_path": "/some/font.ttf"}
    # Input is never mutated.
    assert isinstance(prof["colors"]["text"], list)
    assert isinstance(prof["speaker_overrides"]["gs-1"]["colors"]["text"], list)


def test_normalize_profile_passthrough_for_non_dict():
    assert captions._normalize_profile_from_json("not-a-profile") == "not-a-profile"


def test_apply_speaker_override_noop_cases_return_same_object():
    profile = {"colors": {"text": (255, 255, 255, 255)}, "speaker_overrides": {
        "gs-1": {"colors": {"text": [1, 2, 3, 4]}}
    }}
    smap = {"SPEAKER_00": SpeakerMapEntry("SPEAKER_00", "gs-1", "Alice")}

    assert captions._apply_speaker_override(profile, None, smap) is profile
    assert captions._apply_speaker_override(profile, "UNKNOWN", smap) is profile
    # A mapped speaker with no global id cannot select an override.
    no_id = {"SPEAKER_01": SpeakerMapEntry("SPEAKER_01", None, "Bob")}
    assert captions._apply_speaker_override(profile, "SPEAKER_01", no_id) is profile
    # A global id with no override entry is also a no-op.
    no_ov = {"SPEAKER_02": SpeakerMapEntry("SPEAKER_02", "gs-9", "Carol")}
    assert captions._apply_speaker_override(profile, "SPEAKER_02", no_ov) is profile


def test_apply_speaker_override_merges_colors_and_scalar_fields():
    profile = {
        "colors": {
            "text": (255, 255, 255, 255),
            "stroke": (0, 0, 0, 255),
            "highlight_text": (0, 0, 0, 255),
        },
        "font_size": 64,
        "karaoke": True,
        "speaker_overrides": {
            "gs-1": {
                "colors": {"text": [10, 20, 30, 40]},
                "font_size": 80,
                "stroke_px": 6,
                "karaoke": False,  # not a mergeable scalar -> ignored
            }
        },
    }
    smap = {"SPEAKER_00": SpeakerMapEntry("SPEAKER_00", "gs-1", "Alice")}

    merged = captions._apply_speaker_override(profile, "SPEAKER_00", smap)

    # Override wins on the shared color key ...
    assert merged["colors"]["text"] == (10, 20, 30, 40)
    assert isinstance(merged["colors"]["text"], tuple)
    # ... while baseline color keys that the override did not mention survive.
    assert merged["colors"]["stroke"] == (0, 0, 0, 255)
    assert merged["colors"]["highlight_text"] == (0, 0, 0, 255)
    # Only the four named scalar fields are merged.
    assert merged["font_size"] == 80
    assert merged["stroke_px"] == 6
    assert merged["karaoke"] is True
    # The source profile is not mutated by the merge.
    assert profile["colors"]["text"] == (255, 255, 255, 255)
    assert profile["font_size"] == 64
    assert "stroke_px" not in profile
