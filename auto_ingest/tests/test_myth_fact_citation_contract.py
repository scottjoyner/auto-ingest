"""A myth/fact pair must never present a fabricated citation as citable text.

``content_miner.myth_fact_for_topic`` documents two promises: it returns
``None`` when the graph has no citable material, and its ``fact`` is a real
correction "sourced verbatim from a top paper's Chunk". A graph with papers but
no qualifying chunks is the boundary case: the OPTIONAL MATCH still returns
rows, but there is nothing to quote. The old fallback minted
``"Research (<title>) shows otherwise"`` - an uncited claim dressed as a
citation - instead of returning ``None`` so the planner can use its templated
fallback.

Pure in-memory fake driver: no Neo4j, network, model, credentials or fixtures.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import content_miner


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return list(self._rows)


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        return _Result(self._rows)


class _Driver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _Session(self._rows)


def _call(rows):
    return content_miner.myth_fact_for_topic(_Driver(rows), "large_language_models")


_PAPER_ROWS = [
    {"ptitle": "An Old Paper", "pyear": 2019, "cites": 99, "excerpt": ""},
    {"ptitle": "A New Paper", "pyear": 2024, "cites": 3, "excerpt": None},
]

_VERBATIM = "The model outperforms prior work without extra data."


@pytest.mark.parametrize(
    "rows",
    [
        _PAPER_ROWS,
        [dict(_PAPER_ROWS[0], excerpt="   \n\t"), dict(_PAPER_ROWS[1])],
        [dict(_PAPER_ROWS[0], excerpt=None), dict(_PAPER_ROWS[1], excerpt="")],
        [dict(_PAPER_ROWS[0], excerpt=""), dict(_PAPER_ROWS[1], excerpt="")],
    ],
    ids=["empty-and-null", "whitespace", "null-and-empty", "all-empty"],
)
def test_papers_without_a_citable_chunk_return_none(rows):
    """No excerpt anywhere => no myth/fact pair, not an invented citation."""
    assert _call(rows) is None


def test_real_chunk_is_still_quoted_verbatim():
    """Control: a real excerpt keeps the contract's citable fact unchanged."""
    rows = [dict(_PAPER_ROWS[0]), dict(_PAPER_ROWS[1], excerpt=_VERBATIM)]
    result = _call(rows)
    assert result is not None
    myth, fact = result
    assert "large language models" in myth
    assert fact == _VERBATIM


def test_empty_graph_still_returns_none():
    """Control: the pre-existing zero-row path must not regress."""
    assert _call([]) is None
