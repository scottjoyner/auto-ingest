"""Regression: a successful non-terminal stage update must return the job row.

``ingest_claim.update_stage`` documents that it returns the updated job dict,
or ``None`` only when the node is missing or the owner guard fails.  Its Cypher
used to append ``WITH j WHERE $stage = 'graph_written'`` *after* the stage
``SET``.  In Cypher that is a row filter: for every stage other than the
terminal one the row is dropped before ``RETURN``, so ``.single()`` is ``None``
even though the ``SET j.stages[$stage] = true`` already executed.  Callers that
treat ``None`` as "stage update failed" then abort a job that actually advanced.

The repo's existing ``tests/test_ingest_claim.py`` fake driver ignores the
post-SET gate, which is why the defect shipped.  This test's fake emulates that
row-pipeline step faithfully, so it fails on the pristine query and passes once
the status transition is expressed as a ``CASE`` inside the single ``SET``.

Hermetic: in-memory store only; no Neo4j, no filesystem, no network.
"""
from __future__ import annotations

from auto_ingest import ingest_claim


class _Rows:
    """Minimal neo4j.Result stand-in."""

    def __init__(self, rows):
        self._rows = list(rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def data(self):
        return [dict(r) for r in self._rows]


_UPDATE_PREFIX = (
    "MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner OR $owner = ''"
)


class _GateAwareSession:
    """Emulate the update-stage query's row pipeline, including WITH/WHERE.

    The one semantic that matters here: a ``WITH j WHERE <pred>`` clause that
    follows the stage ``SET`` filters rows *after* the write, so a false
    predicate yields no returned record from ``.single()``.
    """

    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        if not q.startswith(_UPDATE_PREFIX):
            raise AssertionError(f"unexpected query shape: {q[:100]!r}")

        key = params["key"]
        stage = params["stage"]
        owner = params["owner"]
        job = self._store.get(key)

        # MATCH {key} — no row => no result.
        if job is None:
            return _Rows([])
        # WHERE j.owner = $owner OR $owner = '' — owner guard.
        if owner != "" and job["owner"] != owner:
            return _Rows([])

        # SET j.stages[$stage] = true — executes before any WITH/WHERE gate.
        job["stages"] = dict(job["stages"])
        job["stages"][stage] = True

        # Faithful emulation of the buggy shape: `WITH j WHERE $stage =
        # 'graph_written'` drops the row for every non-terminal stage, AFTER
        # the SET above has already committed.
        if "WITH j WHERE $stage = 'graph_written'" in q and stage != "graph_written":
            return _Rows([])

        if stage == "graph_written":
            job["status"] = "done"
        return _Rows([
            {"key": key, "stages": job["stages"], "status": job["status"]}
        ])


class _GateAwareDriver:
    def __init__(self):
        self.store = {}

    def session(self):
        return _GateAwareSession(self.store)


def _job(store, key="k1"):
    store[key] = {
        "key": key,
        "owner": "",
        "claimed_at": 0,
        "status": "pending",
        "stages": {s: False for s in ingest_claim.STAGES},
    }


def test_non_terminal_stage_update_returns_the_updated_job():
    """The write succeeded, so the caller must get the row back."""
    d = _GateAwareDriver()
    _job(d.store)
    res = ingest_claim.update_stage(d, "k1", "copied", owner="")

    assert res is not None, "successful non-terminal stage update returned None"
    assert res["key"] == "k1"
    assert res["stages"]["copied"] is True
    assert res["status"] == "pending", "non-terminal stage must not flip status"


def test_every_non_terminal_stage_returns_a_row():
    """Not just 'copied' — all five non-terminal stages were filtered out."""
    d = _GateAwareDriver()
    for stage in [s for s in ingest_claim.STAGES if s != "graph_written"]:
        _job(d.store, key=stage)
        res = ingest_claim.update_stage(d, stage, stage, owner="")
        assert res is not None, f"{stage} update returned None"
        assert res["stages"][stage] is True


def test_terminal_stage_still_flips_done_and_returns_row():
    """Parity guard: the terminal stage was the only one that worked."""
    d = _GateAwareDriver()
    _job(d.store)
    res = ingest_claim.update_stage(d, "k1", "graph_written", owner="")
    assert res is not None
    assert res["status"] == "done"
    assert res["stages"]["graph_written"] is True


def test_owner_guard_still_filters_the_row():
    """A genuine owner-guard failure must still yield None."""
    d = _GateAwareDriver()
    _job(d.store)
    d.store["k1"]["owner"] = "hostA"
    assert ingest_claim.update_stage(d, "k1", "copied", owner="hostB") is None
    assert d.store["k1"]["stages"]["copied"] is False


def test_query_has_no_post_set_row_filter():
    """Static contract: RETURN must not be gated on the terminal stage."""
    q = ingest_claim._UPDATE_STAGE_QUERY
    assert "WITH j WHERE $stage = 'graph_written'" not in q
    assert "CASE WHEN $stage = 'graph_written'" in q
