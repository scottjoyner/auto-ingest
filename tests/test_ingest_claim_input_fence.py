"""Claim inputs must be valid before a graph query is issued.

`claim()` validated the owner but not the key or TTL. A blank key can MERGE a
shared `IngestJob(key='')` and let unrelated workers contend for one node; a
non-positive TTL makes the expiry boundary immediate (or in the future), so the
exclusive-claim guarantee silently stops holding.
"""
from __future__ import annotations

import pytest

from auto_ingest import ingest_claim


class _Driver:
    """Records whether any graph call happened at all."""

    def __init__(self):
        self.calls = 0

    def session(self):
        self.calls += 1
        raise AssertionError("no graph session should be opened for invalid input")


@pytest.mark.parametrize("key", ["", "   ", None])
def test_blank_key_is_refused_before_the_graph(key):
    with pytest.raises(ValueError, match="key"):
        ingest_claim.claim(_Driver(), key, "worker")
    with pytest.raises(ValueError, match="key"):
        ingest_claim.create_job(_Driver(), key)


@pytest.mark.parametrize("ttl", [0, -1, -3600])
def test_non_positive_ttl_is_refused_before_the_graph(ttl):
    with pytest.raises(ValueError, match="ttl"):
        ingest_claim.claim(_Driver(), "k1", "worker", ttl_sec=ttl)
