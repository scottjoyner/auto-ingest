"""The reaper must reject a TTL that would clear live claims.

Round 45 fenced `claim()` but left `list_claims()`, `expired_claims()` and
`reap()` accepting any `ttl_sec`. With a negative TTL the computed `expires`
is in the future, so `WHERE claimed_at < $expires` matches active claims and
`reap()` clears owners that still hold jobs. Non-positive limits are likewise
invalid for a bounded reaper.
"""
from __future__ import annotations

import pytest

from auto_ingest import ingest_claim


class _Driver:
    def session(self):
        raise AssertionError("no graph session should open for invalid input")


@pytest.mark.parametrize("fn", [
    ingest_claim.list_claims,
    ingest_claim.expired_claims,
    ingest_claim.reap,
])
@pytest.mark.parametrize("ttl", [0, -1, -3600])
def test_non_positive_ttl_is_refused_before_reaping(fn, ttl):
    with pytest.raises(ValueError, match="ttl"):
        fn(_Driver(), ttl_sec=ttl)


@pytest.mark.parametrize("fn", [
    ingest_claim.list_claims,
    ingest_claim.expired_claims,
    ingest_claim.reap,
])
@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_is_refused_before_reaping(fn, limit):
    with pytest.raises(ValueError, match="limit"):
        fn(_Driver(), limit=limit)
