"""Hermetic tests for content_miner's top-paper excerpt selection (r24).

No live Neo4j: the driver/session is a tiny in-memory fake that returns a
fixed ``.data()`` payload, matching the shape ``_top_paper_excerpt_impl``
consumes. The Cypher uses ``OPTIONAL MATCH`` with a ``WHERE ch.text IS NOT
NULL AND size(ch.text) > 50``, so a paper with no qualifying Chunk is still
returned as a row with ``text is None``. Ordering is by paper metadata
(``published DESC, keyword_count DESC``), not by excerpt presence.
"""

from auto_ingest.shorts import content_miner


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return [dict(r) for r in self._rows]


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **kwargs):
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self, **kwargs):
        return _FakeSession(self._rows)


def test_top_paper_excerpt_skips_row_with_no_chunk():
    """A top-ranked paper with no Chunk must not mask a real excerpt lower down."""
    rows = [
        {"text": None},  # top by published DESC, but no qualifying Chunk
        {"text": "  An actual citable finding from the second paper.  "},
    ]
    out = content_miner.top_paper_excerpt(_FakeDriver(rows), "topic", max_len=170)
    assert out is not None, "must fall through to the first paper that has an excerpt"
    assert out == "An actual citable finding from the second paper."


def test_top_paper_excerpt_keeps_real_top_row():
    """Control: when the top row has text, it is still preferred."""
    rows = [
        {"text": "Top paper excerpt here."},
        {"text": "Second paper excerpt here."},
    ]
    assert content_miner.top_paper_excerpt(_FakeDriver(rows), "topic") == "Top paper excerpt here."


def test_top_paper_excerpt_all_null_returns_none():
    """Control: no row carries an excerpt -> clean None (planner falls back)."""
    rows = [{"text": None}, {"text": ""}]
    assert content_miner.top_paper_excerpt(_FakeDriver(rows), "topic") is None


def test_top_paper_excerpt_truncates_on_word_boundary():
    """Control: long excerpts are ellipsised at a word boundary, not mid-word."""
    text = " ".join(["word"] * 60)
    out = content_miner.top_paper_excerpt(_FakeDriver([{"text": text}]), "topic", max_len=40)
    assert out is not None
    assert out.endswith("...")
    assert len(out) <= 40
    assert "wor..." not in out  # did not cut in the middle of a word
