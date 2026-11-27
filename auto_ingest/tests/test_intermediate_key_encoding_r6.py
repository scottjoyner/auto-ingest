"""Round-6 regression: distinct media keys must never share a staged artifact.

Hermetic: only a ``tmp_path`` directory and a fake Neo4j driver are used. No
live DB, service, network, or ML import.

Round 5 covered receipt identity, canonical stage order and outbox replay, but
``ingest_write._path`` still escaped ``/`` as ``__`` and ``list_intermediates``
reversed it with ``replace("__", "/")``. That mapping is not injective: a key
with a literal ``__`` and a key with a slash in the same position mapped to the
same file. Whichever was written last silently replaced the other's staged
artifact, and discovery could never recover the literal-``__`` key, so a
receipt for one key could be satisfied by another key's replay file.
"""
from __future__ import annotations

from auto_ingest import ingest_import, ingest_write


class _FakeResult:
    def single(self):
        return None

    def data(self):
        return []


class _FakeSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._log.append((query, dict(params)))
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)


def test_slash_and_double_underscore_keys_do_not_collide(tmp_path):
    d = str(tmp_path)
    slash = "dashcam/2026-07-14/CLT"
    under = "dashcam__2026-07-14__CLT"

    p_slash = ingest_write.write_intermediate(d, slash, "metadata", [{"id": "slash"}])
    p_under = ingest_write.write_intermediate(d, under, "metadata", [{"id": "under"}])

    assert p_slash != p_under
    assert ingest_write.read_intermediate(d, slash, "metadata") == [{"id": "slash"}]
    assert ingest_write.read_intermediate(d, under, "metadata") == [{"id": "under"}]


def test_discovery_recovers_both_colliding_keys(tmp_path):
    d = str(tmp_path)
    ingest_write.write_intermediate(d, "a/b", "segments", [{"id": "1"}])
    ingest_write.write_intermediate(d, "a__b", "segments", [{"id": "2"}])

    found = ingest_write.list_intermediates(d)
    assert {m["key"] for m in found} == {"a/b", "a__b"}
    assert len(ingest_write.list_intermediates(d, key="a/b")) == 1
    assert len(ingest_write.list_intermediates(d, key="a__b")) == 1


def test_repeated_separators_round_trip_exactly(tmp_path):
    d = str(tmp_path)
    key = "a//b"
    ingest_write.write_intermediate(d, key, "metadata", [{"id": "x"}])
    assert len(ingest_write.list_intermediates(d, key=key)) == 1
    assert ingest_write.read_intermediate(d, key, "metadata") == [{"id": "x"}]


def test_percent_sequence_in_key_is_not_decoded_as_slash(tmp_path):
    d = str(tmp_path)
    ingest_write.write_intermediate(d, "a/b", "metadata", [{"id": "1"}])
    ingest_write.write_intermediate(d, "a%2Fb", "metadata", [{"id": "2"}])
    assert ingest_write.read_intermediate(d, "a/b", "metadata") == [{"id": "1"}]
    assert ingest_write.read_intermediate(d, "a%2Fb", "metadata") == [{"id": "2"}]
    assert {m["key"] for m in ingest_write.list_intermediates(d)} == {"a/b", "a%2Fb"}


def test_import_all_imports_both_colliding_keys(tmp_path):
    d = str(tmp_path)
    ingest_write.write_intermediate(d, "a/b", "transcription", [{"id": "t1"}])
    ingest_write.write_intermediate(d, "a__b", "transcription", [{"id": "t2"}])
    drv = _FakeDriver()
    summary = ingest_import.import_all(d, drv)
    assert set(summary) == {"a/b/transcription", "a__b/transcription"}
    assert len(drv.calls) == 2


def test_literal_underscore_key_still_round_trips(tmp_path):
    d = str(tmp_path)
    key = "single_underscore_key"
    ingest_write.write_intermediate(d, key, "segments", [{"id": "s1"}], fmt="csv")
    found = ingest_write.list_intermediates(d, key=key)
    assert len(found) == 1
    assert ingest_write.read_intermediate(d, key, "segments", fmt="csv") == [{"id": "s1"}]
