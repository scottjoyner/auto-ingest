"""Unit tests for personal link_media linkers (fake driver, no live Neo4j)."""
from __future__ import annotations

from typing import List

import auto_ingest.personal.link_media as lm


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None

    def consume(self):
        return None

    def __iter__(self):
        return iter(self._rows)


class FakeSession:
    def __init__(self, queries: List[str], canned):
        self.queries = queries
        self.canned = canned

    def run(self, cypher, **params):
        self.queries.append(cypher)
        return FakeResult(self.canned if isinstance(self.canned, list) else [])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriver:
    def __init__(self, canned_unlinked, canned_cands):
        self.queries: List[str] = []
        self.canned_unlinked = canned_unlinked
        self.canned_cands = canned_cands
        self.sessions = 0

    def session(self, database=None):
        self.sessions += 1
        # First call (per linker): _fetch_unlinked; later: candidates / merge.
        if self.sessions % 2 == 1:
            canned = self.canned_unlinked
        else:
            canned = self.canned_cands
        return FakeSession(self.queries, canned)


def test_link_to_places_uses_expected_vocabulary():
    # One unlinked media near a known place.
    media = [{"sha": "abc", "lat": 42.36, "lon": -71.06}]
    place = [{"name": "Boston", "lat": 42.3601, "lon": -71.0589}]
    drv = FakeDriver(media, place)
    n = lm.link_to_places(drv, max_radius_m=200.0, limit=10)
    assert n == 1

    joined = " ".join(drv.queries)
    assert "SummaryPlace" in joined
    assert "AT_PLACE" in joined
    # Select query must filter unlinked media.
    assert any("linked_at IS NULL" in q for q in drv.queries)
    # A candidate-fetch query for the place.
    assert any("MATCH (p:SummaryPlace)" in q for q in drv.queries)


def test_link_all_keys_and_invocation():
    media = []
    drv = FakeDriver(media, [])
    counts = lm.link_all(drv, limit=5)
    assert set(counts.keys()) == {"places", "trips", "phonelogs"}
    # link_all opens sessions for each linker + a final _fetch_unlinked pass.
    # Bounded: each candidate query contains LIMIT.
    assert any("LIMIT" in q for q in drv.queries)
    # Each of the three linkers fetched unlinked media (SELECT with LIMIT).
    select_queries = [q for q in drv.queries if "MATCH (m:MediaFile)" in q and "LIMIT" in q]
    assert len(select_queries) >= 3


def test_link_to_phonelogs_uses_loc_point_and_merges_near():
    # One unlinked media near a PhoneLog whose geo lives at pl.loc (spatial point).
    media = [{"sha": "abc", "lat": 42.3601, "lon": -71.0589}]
    # Fake session returns loc.latitude / loc.longitude shaped candidates.
    phonelog = [{"pid": "pl-1", "lat": 42.3601, "lon": -71.0589}]
    drv = FakeDriver(media, phonelog)
    n = lm.link_to_phonelogs(drv, max_radius_m=100.0, limit=10)
    assert n == 1

    joined = " ".join(drv.queries)
    # Candidate query must read the spatial point, not flat props.
    assert "pl.loc" in joined
    assert "point.distance" in joined
    # The NEAR edge MERGE must fire for the in-radius candidate.
    assert any("MERGE (m)-[:NEAR" in q for q in drv.queries)
