"""Hermetic regression: the claim reaper's LIMIT must bound the mutation.

``reap()`` is documented as a bounded reaper for the live 21M-node graph. Its
Cypher used to place ``LIMIT $limit`` *after* ``RETURN count(j) AS cleared``.
An aggregation returns a single row, so that LIMIT only truncates the report;
every expired ``IngestJob`` is still cleared. This module proves the bound is
on the SET, without a live database.
"""
from __future__ import annotations

import re

from auto_ingest import ingest_claim


def _flat(query: str) -> str:
    return " ".join(query.split())


class _Single:
    def __init__(self, record):
        self._record = record

    def single(self):
        return self._record


class _BoundAwareDriver:
    """Fake driver that applies LIMIT only where Cypher actually would.

    Cypher applies ``LIMIT`` to rows being returned. ``RETURN count(j)`` is an
    aggregation and produces one row, so a trailing ``LIMIT $limit`` cannot
    bound the preceding ``SET``. Only a ``WITH j LIMIT $limit`` before the
    ``SET`` bounds which matched nodes get mutated.
    """

    def __init__(self, expired_keys, captured):
        self.expired_keys = list(expired_keys)
        self.captured = captured

    def session(self):
        outer = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, query, **params):
                q = _flat(query)
                outer.captured["query"] = q
                limit = int(params["limit"])
                lowered = q.lower()
                set_at = lowered.index("set j.owner")
                with_limit = re.search(r"with j limit \$limit", lowered)
                bounded = with_limit is not None and with_limit.start() < set_at
                cleared = (min(len(outer.expired_keys), limit) if bounded
                           else len(outer.expired_keys))
                return _Single({"cleared": cleared})

        return _Session()


def test_reap_query_bounds_the_set_before_returning_count():
    q = _flat(ingest_claim._REAP_QUERY)
    lowered = q.lower()
    set_at = lowered.index("set j.owner")
    limit_at = lowered.find("limit $limit")
    return_at = lowered.find("return count(j)")

    assert limit_at != -1, "the reaper lost its LIMIT entirely"
    assert return_at != -1, "the reaper lost its cleared-count aggregation"
    assert limit_at < set_at, (
        "LIMIT must bound the MATCHed rows before SET; a LIMIT after "
        "RETURN count(j) truncates only the single aggregate row, so every "
        "expired claim is cleared regardless of the caller's limit"
    )
    assert limit_at < return_at, "LIMIT must not sit after the aggregation"


def test_reap_reports_only_the_requested_number_of_claims():
    captured = {}
    driver = _BoundAwareDriver([f"old{i}" for i in range(12)], captured)

    cleared = ingest_claim.reap(driver, ttl_sec=3600, limit=3, now_ms=1_000)

    assert cleared == 3, (
        "reap(limit=3) cleared {0} claim(s); query was: {1}".format(
            cleared, captured.get("query"))
    )
