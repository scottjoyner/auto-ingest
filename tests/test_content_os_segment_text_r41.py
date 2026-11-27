"""Hermetic contract tests for ``content_os.adapters._segment_text`` (r41).

The helper is the text-recovery fallback shared by the JSON/CSV/text source
adapters. It must never let an *empty or textless* ``segments`` list mask a
usable top-level text field: a payload like ``{"segments": [], "text": "..."}``
is common in ASR exports that finished with no segments but kept a transcript.

No filesystem, network, credentials, DB, model or service is touched: every
case uses literal Python objects and imports only the pure stdlib-based module.
"""

from content_os.adapters import _segment_text


def test_empty_segments_falls_back_to_text():
    """Empty ``segments`` must not hide a valid top-level ``text`` field."""
    assert _segment_text({"segments": [], "text": "fallback transcript"}) == ("fallback transcript")


def test_textless_segments_fall_back_to_transcript():
    """Segments with no ``text`` must not hide a ``transcript`` field."""
    data = {
        "segments": [{"start": 0.0, "end": 1.0}, {"start": 1.0}],
        "transcript": "recovered from transcript key",
    }
    assert _segment_text(data) == "recovered from transcript key"


def test_textless_segments_fall_back_to_summary():
    """Fallback key order is preserved once empty segments are skipped."""
    data = {"segments": [{}], "summary": "a summary", "text": "a text"}
    assert _segment_text(data) == "a text"


def test_nonempty_segments_still_win_with_start_prefix():
    """Control: real segments keep their timestamp-prefixed, ordered text."""
    data = {
        "segments": [
            {"start": 1.5, "text": "second"},
            {"start": 0.0, "text": "first"},
        ],
        "text": "should not be used",
    }
    assert _segment_text(data) == "[1.5] second\n[0.0] first"


def test_segments_without_start_have_no_prefix():
    """Control: a segment with no ``start`` is emitted without a prefix."""
    assert _segment_text({"segments": [{"text": "bare"}]}) == "bare"


def test_generic_key_order_unchanged():
    """Control: without ``segments`` the first non-empty text key still wins."""
    data = {"transcript": "t", "text": "x", "summary": "s"}
    assert _segment_text(data) == "t"
    assert _segment_text({"text": 123, "summary": "s"}) == "s"


def test_list_input_unchanged():
    """Control: list payloads still join their text items verbatim."""
    assert _segment_text(["alpha", {"text": "beta"}, {"text": ""}, 9]) == ("alpha\nbeta")


def test_non_text_payload_is_empty():
    """Control: unsupported payload shapes still return an empty string."""
    assert _segment_text(None) == ""
    assert _segment_text({"segments": "not a list", "text": 5}) == ""
