"""Hermetic tests for link_media checkpoint resume batch sizing (r26 sweep).

No Neo4j, no files, no network: a fake driver only records the Cypher ``limit``
it is asked for and slices an in-memory row pool accordingly.
"""
from __future__ import annotations

import auto_ingest.personal.link_media as lm


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    def __init__(self, pool, calls):
        self._pool = pool
        self._calls = calls

    def run(self, cypher, **params):
        n = params.get("limit")
        self._calls.append(n)
        rows = self._pool if n is None else self._pool[:n]
        return _Result([dict(r) for r in rows])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    def __init__(self, pool):
        self.pool = pool
        self.fetch_limits = []

    def session(self, database=None):
        return _Session(self.pool, self.fetch_limits)


def _pool(n=8):
    return [
        {"sha": "sha-%d" % i, "lat": 42.0 + i * 0.001, "lon": -71.0}
        for i in range(n)
    ]


def test_checkpoint_rows_do_not_starve_resume_batch():
    """Checkpointed SHAs at the head of the queue must not zero out a resume pass."""
    drv = _Driver(_pool())
    done = {"sha-0", "sha-1", "sha-2", "sha-3"}
    got = lm._fetch_unlinked(drv, limit=2, done=done)
    assert [r["sha"] for r in got] == ["sha-4", "sha-5"]
    # Cypher must be padded by the checkpoint size; the Python filter runs after LIMIT.
    assert drv.fetch_limits == [2 + len(done)]


def test_no_checkpoint_keeps_exact_limit():
    """With an empty checkpoint the fetch size and result count stay unchanged."""
    drv = _Driver(_pool())
    got = lm._fetch_unlinked(drv, limit=3, done=set())
    assert [r["sha"] for r in got] == ["sha-0", "sha-1", "sha-2"]
    assert drv.fetch_limits == [3]


def test_checkpoint_larger_than_batch_still_advances():
    """Even when every head row is checkpointed, fresh rows beyond the batch are returned."""
    drv = _Driver(_pool(10))
    done = {"sha-%d" % i for i in range(6)}
    got = lm._fetch_unlinked(drv, limit=2, done=done)
    assert [r["sha"] for r in got] == ["sha-6", "sha-7"]
