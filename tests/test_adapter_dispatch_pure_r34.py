"""Round 34 contract pin: `content_os.adapters` source-adapter dispatch.

Hermetic/offline. The dispatch surface is the pair of pure helpers

    adapters_for_rules(rules) -> list[SourceAdapter]
    detect_adapter(path, rules) -> SourceAdapter | None

plus the ``can_handle`` suffix predicates they delegate to.  Together they
decide *which* adapter extracts a source document, and therefore its
``source_type`` / ``adapter`` fields and whether any text is extracted at all.
No clock, filesystem read, DB, network or service is involved: ``can_handle``
for every built-in adapter is a pure suffix-set membership test, so dispatch
works on a path that does not exist.

This file pins the observed dispatch contract only.  It asserts no new
behavior and changes no production source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from content_os.adapters import (
    ADAPTERS,
    MEDIA_EXTENSIONS,
    TEXT_EXTENSIONS,
    AdapterRule,
    RuleBasedAdapter,
    adapters_for_rules,
    detect_adapter,
)

BUILTIN_NAMES = ["json", "csv", "text", "pdf", "media_manifest"]


def _names(adapters):
    return [getattr(a, "name", None) for a in adapters]


# ---------------------------------------------------------------------------
# adapters_for_rules: ordering / composition contract
# ---------------------------------------------------------------------------


def test_builtin_adapter_order_is_stable():
    assert _names(adapters_for_rules()) == BUILTIN_NAMES


def test_module_level_adapters_matches_builtin_order():
    assert _names(ADAPTERS) == BUILTIN_NAMES


def test_empty_rules_equals_none():
    assert _names(adapters_for_rules([])) == _names(adapters_for_rules(None))
    assert _names(adapters_for_rules([])) == BUILTIN_NAMES


def test_rules_are_prepended_in_input_order():
    r1 = AdapterRule(name="a", extensions={".x"}, source_type="sx")
    r2 = AdapterRule(name="b", extensions={".y"}, source_type="sy")
    assert _names(adapters_for_rules([r1, r2])) == [
        "rule:a",
        "rule:b",
        *BUILTIN_NAMES,
    ]


def test_rule_based_adapter_name_is_prefixed():
    r = AdapterRule(name="custom", extensions={".z"}, source_type="sz")
    first = adapters_for_rules([r])[0]
    assert isinstance(first, RuleBasedAdapter)
    assert first.name == "rule:custom"
    assert first.rule is r


# ---------------------------------------------------------------------------
# detect_adapter: built-in suffix dispatch (case-insensitive, pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("doc.json", "json"),
        ("doc.jsonl", "json"),
        ("DOC.JSON", "json"),
        ("doc.JSONL", "json"),
        ("table.csv", "csv"),
        ("TABLE.CSV", "csv"),
        ("notes.txt", "text"),
        ("notes.md", "text"),
        ("notes.markdown", "text"),
        ("run.log", "text"),
        ("subs.srt", "text"),
        ("subs.vtt", "text"),
        ("SUBS.SRT", "text"),
        ("paper.pdf", "pdf"),
        ("PAPER.PDF", "pdf"),
        ("song.mp3", "media_manifest"),
        ("song.wav", "media_manifest"),
        ("song.m4a", "media_manifest"),
        ("song.flac", "media_manifest"),
        ("clip.mp4", "media_manifest"),
        ("clip.mov", "media_manifest"),
        ("clip.mkv", "media_manifest"),
        ("clip.avi", "media_manifest"),
        ("photo.jpg", "media_manifest"),
        ("photo.jpeg", "media_manifest"),
        ("photo.png", "media_manifest"),
        ("photo.webp", "media_manifest"),
        ("PHOTO.JPG", "media_manifest"),
    ],
)
def test_builtin_suffix_dispatch(name, expected):
    assert getattr(detect_adapter(Path(name)), "name", None) == expected


@pytest.mark.parametrize(
    "name",
    [
        "blob.bin",
        "README",
        "noext",
        "doc.docx",
        "data.parquet",
        "script.py",
        ".hidden",
        "archive.tar.gz",
    ],
)
def test_unknown_suffix_returns_none(name):
    assert detect_adapter(Path(name)) is None


def test_dispatch_ignores_directory_and_existence():
    # can_handle only inspects the suffix: a non-existent path still dispatches.
    missing = Path("/does/not/exist/adapter-dispatch-r34/doc.json")
    assert not missing.exists()
    assert getattr(detect_adapter(missing), "name", None) == "json"
    assert detect_adapter(Path("/does/not/exist/x.unknown")) is None


def test_every_builtin_is_reachable_by_dispatch():
    # Invariant: each built-in adapter returned by adapters_for_rules has a
    # representative suffix that detect_adapter routes back to that adapter.
    representatives = {
        "json": "doc.json",
        "csv": "table.csv",
        "text": "notes.txt",
        "pdf": "paper.pdf",
        "media_manifest": "clip.mp4",
    }
    builtins = adapters_for_rules()
    assert len(builtins) == len(representatives)
    for adapter in builtins:
        chosen = detect_adapter(Path(representatives[adapter.name]))
        assert type(chosen) is type(adapter)
        assert chosen.name == adapter.name


def test_extension_sets_are_case_normalized_targets():
    # Pin the exact declared suffix sets the built-ins rely on (lowercase).
    assert TEXT_EXTENSIONS == {".txt", ".md", ".markdown", ".log", ".srt", ".vtt"}
    assert MEDIA_EXTENSIONS == {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }


# ---------------------------------------------------------------------------
# detect_adapter: rule precedence / fallthrough contract
# ---------------------------------------------------------------------------


def test_rule_adapter_wins_over_builtin_json():
    r = AdapterRule(name="custom", extensions={".json"}, source_type="custom")
    chosen = detect_adapter(Path("doc.json"), [r])
    assert isinstance(chosen, RuleBasedAdapter)
    assert chosen.name == "rule:custom"


def test_rule_adapter_wins_over_builtin_csv():
    r = AdapterRule(name="custom", extensions={".csv"}, source_type="custom")
    chosen = detect_adapter(Path("table.csv"), [r])
    assert isinstance(chosen, RuleBasedAdapter)
    assert chosen.name == "rule:custom"


def test_rule_adapter_wins_over_builtin_media():
    r = AdapterRule(name="custom", extensions={".mp4"}, source_type="custom")
    chosen = detect_adapter(Path("clip.mp4"), [r])
    assert isinstance(chosen, RuleBasedAdapter)
    assert chosen.name == "rule:custom"


def test_first_matching_rule_wins():
    r1 = AdapterRule(name="first", extensions={".csv"}, source_type="s1")
    r2 = AdapterRule(name="second", extensions={".csv"}, source_type="s2")
    chosen = detect_adapter(Path("table.csv"), [r1, r2])
    assert chosen.name == "rule:first"
    assert chosen.rule is r1


def test_non_matching_rule_falls_through_to_builtin():
    r = AdapterRule(name="custom", extensions={".zzz"}, source_type="custom")
    assert getattr(detect_adapter(Path("doc.json"), [r]), "name", None) == "json"
    assert detect_adapter(Path("doc.zzz"), [r]).name == "rule:custom"


def test_rule_suffix_match_is_case_insensitive():
    r = AdapterRule(name="custom", extensions={".csv"}, source_type="custom")
    assert detect_adapter(Path("TABLE.CSV"), [r]).name == "rule:custom"


def test_rule_order_does_not_affect_builtin_fallback():
    r1 = AdapterRule(name="a", extensions={".a"}, source_type="sa")
    r2 = AdapterRule(name="b", extensions={".b"}, source_type="sb")
    assert getattr(detect_adapter(Path("doc.pdf"), [r1, r2]), "name", None) == "pdf"
    assert getattr(detect_adapter(Path("doc.pdf"), [r2, r1]), "name", None) == "pdf"
