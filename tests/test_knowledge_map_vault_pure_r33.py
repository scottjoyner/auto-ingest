"""Hermetic contract pin for ``auto_ingest.knowledge_map`` vault-sync helpers (r33).

Scope: the three pure functions that (a) turn a markdown vault document into a
content-addressed resumability key and a graph title, and (b) turn an exported
graph node back into mirror-vault markdown:

  * ``_doc_hash``        -- SHA-256 content hash used to skip unchanged docs
  * ``_first_heading``   -- first-H1 title derivation for an indexed doc
  * ``_node_to_markdown``-- deterministic mirror-vault renderer for a node

All three are pure: they read only their arguments, touch no clock, filesystem,
database, network or service, and mutate no input. This file pins the observed
contract only; it asserts no defect and changes no production behavior.

Fresh surface: none of these helpers had a test in any ref at pin time; the
existing ``tests/test_knowledge_map.py`` covers only module import, CLI parsing,
the two dry-run paths, and ``_vault_roots``. ``_safe_filename`` was already
covered in sweep21 and is deliberately excluded here.
"""

from __future__ import annotations

import hashlib

import auto_ingest.knowledge_map as km

FALLBACK = "fallback-title"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------------------------
# _doc_hash -- content-addressed resumability key
# ---------------------------------------------------------------------------


def test_doc_hash_is_sha256_hex_of_utf8_replace():
    for text in ("", "hello", "line1\nline2", "caf\u00e9 \u2014 \u65e5\u672c\u8a9e", "a" * 5000):
        expected = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        assert km._doc_hash(text) == expected
        assert len(expected) == 64
        assert set(expected) <= set("0123456789abcdef")


def test_doc_hash_empty_string_matches_known_digest():
    assert km._doc_hash("") == EMPTY_SHA256


def test_doc_hash_is_deterministic_and_byte_order_sensitive():
    assert km._doc_hash("ab") == km._doc_hash("ab")
    assert km._doc_hash("ab") != km._doc_hash("ba")
    assert km._doc_hash("a b") != km._doc_hash("a  b")


def test_doc_hash_encodes_lone_surrogate_with_replace():
    text = "x\ud800y"
    expected = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    assert km._doc_hash(text) == expected
    assert km._doc_hash(text) != km._doc_hash("xy")


# ---------------------------------------------------------------------------
# _first_heading -- title derivation for an indexed vault document
# ---------------------------------------------------------------------------


def test_first_heading_returns_first_h1_title_stripped():
    text = "intro\n\n#   My Title  \nmore\n# Second\n"
    assert km._first_heading(text, FALLBACK) == "My Title"


def test_first_heading_only_accepts_hash_space_not_h2_or_attached():
    text = "## H2\n### H3\n#H4\n#\tTab\n"
    assert km._first_heading(text, FALLBACK) == FALLBACK


def test_first_heading_accepts_indented_h1_after_strip():
    assert km._first_heading("body\n   # Indented\n", FALLBACK) == "Indented"


def test_first_heading_handles_crlf():
    assert km._first_heading("pre\r\n# CRLF title\r\n", FALLBACK) == "CRLF title"


def test_first_heading_hash_only_line_falls_back_after_strip():
    # strip() removes the trailing space, so a bare "# " is not a "# " match.
    assert km._first_heading("# \n", FALLBACK) == FALLBACK
    assert km._first_heading("#  \n", FALLBACK) == FALLBACK


def test_first_heading_fallback_for_empty_and_no_h1():
    assert km._first_heading("", FALLBACK) == FALLBACK
    assert km._first_heading("no headings here", FALLBACK) == FALLBACK


def test_first_heading_unicode_and_does_not_mutate_input():
    text = "# Caf\u00e9 \u2615 \u65e5\u672c\u8a9e\n"
    original = text
    assert km._first_heading(text, FALLBACK) == "Caf\u00e9 \u2615 \u65e5\u672c\u8a9e"
    assert text == original


# ---------------------------------------------------------------------------
# _node_to_markdown -- mirror-vault renderer for an exported graph node
# ---------------------------------------------------------------------------


def test_node_to_markdown_empty_props_is_heading_and_blank_line():
    assert km._node_to_markdown("Entity", {}) == "# Entity\n\n"


def test_node_to_markdown_bullets_in_insertion_order():
    md = km._node_to_markdown("Entity", {"key": "k1", "name": "Alpha", "n": 7})
    assert md == "# Entity\n\n- **key**: k1\n- **name**: Alpha\n- **n**: 7\n"


def test_node_to_markdown_hoists_text_to_body_never_as_bullet():
    md = km._node_to_markdown(
        "VaultDocument", {"key": "notes/a.md", "title": "A", "text": "body line"}
    )
    assert "- **text**:" not in md
    assert md == (
        "# VaultDocument\n\n- **key**: notes/a.md\n- **title**: A\n\n## Body\n\nbody line\n"
    )


def test_node_to_markdown_text_key_emitted_even_when_empty():
    md = km._node_to_markdown("Entity", {"text": ""})
    assert md == "# Entity\n\n\n## Body\n\n\n"


def test_node_to_markdown_coerces_non_string_values():
    md = km._node_to_markdown("E", {"n": None, "vals": [1, 2], "flag": True, "text": 123})
    assert "- **n**: None\n" in md
    assert "- **vals**: [1, 2]\n" in md
    assert "- **flag**: True\n" in md
    assert md.endswith("## Body\n\n123\n")


def test_node_to_markdown_is_insertion_ordered_not_sorted():
    z_first = km._node_to_markdown("E", {"z": "1", "a": "2"})
    a_first = km._node_to_markdown("E", {"a": "2", "z": "1"})
    assert z_first == "# E\n\n- **z**: 1\n- **a**: 2\n"
    assert a_first == "# E\n\n- **a**: 2\n- **z**: 1\n"
    assert z_first != a_first


def test_node_to_markdown_preserves_adversarial_values_verbatim():
    md = km._node_to_markdown(
        "E; drop", {"k\n": "v\r\n# injected", "text": "# body heading\n- bullet"}
    )
    assert "# E; drop\n\n" in md
    assert "- **k\n**: v\r\n# injected\n" in md
    assert md.endswith("## Body\n\n# body heading\n- bullet\n")


def test_node_to_markdown_does_not_mutate_props():
    props = {"key": "k", "text": "body"}
    original = dict(props)
    km._node_to_markdown("E", props)
    assert props == original


def test_node_to_markdown_single_trailing_newline_for_nonempty_props():
    for props in ({"k": "v"}, {"k": "v", "text": "b"}):
        md = km._node_to_markdown("E", props)
        assert md.endswith("\n")
        assert not md.endswith("\n\n")


# ---------------------------------------------------------------------------
# cross-function invariants: exported nodes stay re-indexable and resumable
# ---------------------------------------------------------------------------


def test_rendered_node_title_roundtrips_through_first_heading():
    props = {"key": "notes/a.md", "title": "Not the label", "text": "# body H1\nignored"}
    md = km._node_to_markdown("VaultDocument", props)
    assert km._first_heading(md, FALLBACK) == "VaultDocument"


def test_identical_render_yields_identical_doc_hash():
    props = {"key": "k", "title": "T", "text": "body"}
    first = km._node_to_markdown("VaultDocument", props)
    second = km._node_to_markdown("VaultDocument", dict(props))
    assert first == second
    assert km._doc_hash(first) == km._doc_hash(second)
    assert km._doc_hash(first) != km._doc_hash(second + "\n")
