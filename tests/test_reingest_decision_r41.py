"""Pin the pure re-ingest admission decision ``should_reingest`` (r41).

Round-41 PARENT sweep, session 4.  This is a *test-only* addition: no production
source is touched.  ``auto_ingest.ingest.transcripts.should_reingest`` is the
last gate that decides whether a discovered media key is re-imported, yet no
test (and no prior report) pinned any of its branches.

Hermeticity: the module imports torch / transformers / neo4j at module scope and
even calls ``from_pretrained`` at import time.  All of those (plus the config
and outbox shims and the device probe) are replaced with in-memory fakes *only*
while the module is imported, then restored.  Every test below passes literal
dicts / ints / bools: no filesystem, DB, network, service or credential access.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_MISSING = object()
_saved: dict[str, object] = {}


def _stub(name: str, obj: object) -> None:
    _saved[name] = sys.modules.get(name, _MISSING)
    sys.modules[name] = obj


def _import_transcripts():
    """Import the real module with its heavy/stateful imports faked out."""
    cfg = types.ModuleType("auto_ingest_config")
    cfg.get_fileserver_path = lambda part: "/fake/" + part
    cfg.get_neo4j_env = lambda: ("", "", "", "")
    cfg.get_neo4j_password = lambda: ""
    _stub("auto_ingest_config", cfg)

    outbox = types.ModuleType("auto_ingest.outbox")
    outbox.get_outbox = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should_reingest must not touch the outbox DB")
    )
    _stub("auto_ingest.outbox", outbox)

    backend = types.ModuleType("auto_ingest.backend")
    backend.torch_device = lambda: "cpu"
    _stub("auto_ingest.backend", backend)

    embed = types.ModuleType("auto_ingest.embed")
    embed.HNSW_M, embed.HNSW_EF, embed.HNSW_QUANT = 16, 200, False
    _stub("auto_ingest.embed", embed)

    _stub("numpy", MagicMock())
    _stub("torch", MagicMock())
    tf = MagicMock()
    tf.AutoTokenizer, tf.AutoModel = MagicMock(), MagicMock()
    _stub("transformers", tf)
    neo = MagicMock()
    neo.GraphDatabase = MagicMock()
    _stub("neo4j", neo)
    exc = MagicMock()
    exc.Neo4jError = type("Neo4jError", (Exception,), {})
    _stub("neo4j.exceptions", exc)

    try:
        return importlib.import_module("auto_ingest.ingest.transcripts")
    finally:
        for name, old in _saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


transcripts = _import_transcripts()
should_reingest = transcripts.should_reingest

_SOURCES = ("json", "csv", "rttm", "media")


def _paths(**overrides):
    paths = {k: "/fake/source." + k for k in _SOURCES}
    paths.update(overrides)
    return paths


def _status(**overrides):
    status = {
        "exists": True,
        "started_at": "2026-11-27T00:00:00Z",
        "source_json": True,
        "source_csv": True,
        "source_rttm": True,
        "source_media": True,
        "seg_count": 10,
        "seg_no_emb": 0,
        "utt_count": 8,
    }
    status.update(overrides)
    return status


def test_real_module_callable_pinned():
    assert should_reingest.__module__ == "auto_ingest.ingest.transcripts"
    assert callable(should_reingest)


def test_complete_status_is_not_reingested():
    assert should_reingest(_status(), 10, True, _paths()) is False


def test_missing_record_forces_reingest():
    assert should_reingest({}, 10, True, _paths()) is True
    assert should_reingest(_status(exists=False), 10, True, _paths()) is True
    assert should_reingest(_status(exists=None), 10, True, _paths()) is True


@pytest.mark.parametrize("source", _SOURCES)
def test_recorded_source_missing_from_status_forces_reingest(source):
    status = _status()
    status.pop("source_" + source)
    assert should_reingest(status, 10, True, _paths()) is True


@pytest.mark.parametrize("source", _SOURCES)
def test_falsy_path_is_ignored(source):
    for falsy in ("", None, [], 0):
        status = _status()
        status.pop("source_" + source)
        assert should_reingest(status, 10, True, _paths(**{source: falsy})) is False


def test_unrelated_path_keys_are_ignored():
    assert should_reingest(_status(), 10, True, {"notes": "/fake/x"}) is False


def test_none_start_timestamp_forces_reingest():
    assert should_reingest(_status(started_at=None), 10, True, _paths()) is True


def test_empty_start_timestamp_does_not_force_reingest():
    # The gate is strictly `is None`; an empty (if malformed) value is accepted.
    assert should_reingest(_status(started_at=""), 10, True, _paths()) is False


def test_started_at_absent_key_forces_reingest():
    status = _status()
    status.pop("started_at")
    assert should_reingest(status, 10, True, _paths()) is True


def test_expected_segments_with_no_segments_forces_reingest():
    assert should_reingest(_status(seg_count=0), 12, True, _paths()) is True


def test_zero_expected_segments_skips_segment_check():
    assert should_reingest(_status(seg_count=0), 0, False, _paths()) is False


def test_rttm_without_utterances_forces_reingest():
    assert should_reingest(_status(utt_count=0), 10, True, _paths()) is True
    assert should_reingest(_status(utt_count=0), 10, 1, _paths()) is True


def test_no_rttm_without_utterances_is_accepted():
    assert should_reingest(_status(utt_count=0), 10, False, _paths()) is False


def test_embedding_deficit_boundary():
    assert should_reingest(_status(seg_no_emb=5), 10, True, _paths()) is False
    assert should_reingest(_status(seg_no_emb=6), 10, True, _paths()) is True


@pytest.mark.parametrize(
    ("seg_count", "ok", "bad"),
    [(1, 1, 2), (2, 1, 2), (3, 1, 2), (11, 5, 6)],
)
def test_embedding_floor_at_one(seg_count, ok, bad):
    assert should_reingest(_status(seg_count=seg_count, seg_no_emb=ok), 0, False, _paths()) is False
    assert should_reingest(_status(seg_count=seg_count, seg_no_emb=bad), 0, False, _paths()) is True


def test_numeric_strings_are_coerced():
    assert (
        should_reingest(_status(seg_count="10", seg_no_emb="6", utt_count="8"), 0, False, _paths())
        is True
    )
    assert (
        should_reingest(_status(seg_count="10", seg_no_emb="5", utt_count="8"), 0, False, _paths())
        is False
    )


def test_falsy_count_values_coerce_to_zero():
    status = _status(seg_count=None, seg_no_emb="", utt_count=None)
    assert should_reingest(status, 0, False, _paths()) is False


def test_embedding_deficit_ignored_when_no_segments():
    assert should_reingest(_status(seg_count=0, seg_no_emb=999), 0, False, _paths()) is False


def test_extra_status_keys_do_not_change_verdict():
    status = _status(owner="scott", retries=3, note="anything")
    assert should_reingest(status, 10, True, _paths()) is False
