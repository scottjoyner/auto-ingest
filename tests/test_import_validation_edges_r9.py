"""Round-9 fail-closed validation edges of the cleaned-intermediate importer.

Covers ``auto_ingest.ingest_import`` paths not exercised by
``tests/test_ingest_intermediate.py``: malformed/missing artifacts, invalid
batch sizes, missing key/parent fields, empty artifacts, and ``import_all``
selection edges (key/stage filters, unknown-stage ordering, and json-over-csv
format preference). No live Neo4j: a fake driver records every ``run`` call.

This complements (does not duplicate) R5 stage-durability, R6 write-path, R7
stage-order/backup-receipt/fail-closed, and R8 GraphOutbox lifecycle work.
"""

from __future__ import annotations

import json

import pytest

from auto_ingest import ingest_import, ingest_write


class _Call:
    __slots__ = ("query", "params")

    def __init__(self, query, params):
        self.query = query
        self.params = params


class _FakeSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._log.append(_Call(query, dict(params)))


class _FakeDriver:
    def __init__(self):
        self.calls: list[_Call] = []

    def session(self):
        return _FakeSession(self.calls)


class _NoSessionDriver:
    """Fails if a session is opened at all."""

    def session(self):
        raise AssertionError("no driver session should be opened for an empty artifact")


def _write_json(d, key, stage, rows):
    return ingest_write.write_intermediate(d, key, stage, rows, fmt="json")


# ---------------------------------------------------------------------------
# _load_rows / import_intermediate fail-closed edges
# ---------------------------------------------------------------------------


def test_missing_artifact_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="intermediate not found"):
        ingest_import.import_intermediate(_FakeDriver(), str(tmp_path / "missing.json"))


def test_json_object_instead_of_list_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "a"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list of row dicts"):
        ingest_import.import_intermediate(_FakeDriver(), str(path))


def test_non_dict_json_rows_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["scalar", 1]), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        ingest_import.import_intermediate(_FakeDriver(), str(path))


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("anything", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported intermediate extension"):
        ingest_import.import_intermediate(_FakeDriver(), str(path))


def test_row_missing_key_field_rejected(tmp_path):
    path = _write_json(tmp_path, "k", "segments", [{"value": 1}])
    with pytest.raises(KeyError, match="row missing key_field"):
        ingest_import.import_intermediate(_FakeDriver(), str(path), key_field="id")


@pytest.mark.parametrize("batch", [0, -1, -100])
def test_batch_must_be_positive(tmp_path, batch):
    path = _write_json(tmp_path, "k", "segments", [{"id": "a"}])
    with pytest.raises(ValueError, match="batch must be >= 1"):
        ingest_import.import_intermediate(_FakeDriver(), str(path), batch=batch)


def test_empty_json_artifact_opens_no_session(tmp_path):
    path = _write_json(tmp_path, "k", "segments", [])
    assert ingest_import.import_intermediate(_NoSessionDriver(), str(path)) == 0


def test_empty_csv_artifact_opens_no_session(tmp_path):
    path = ingest_write.write_intermediate(tmp_path, "k", "segments", [], fmt="csv")
    assert ingest_import.import_intermediate(_NoSessionDriver(), str(path)) == 0


def test_parent_link_requires_parent_on_every_row(tmp_path):
    path = _write_json(tmp_path, "k", "metadata", [{"id": "a", "parent": "t1"}, {"id": "b"}])
    with pytest.raises(KeyError, match="rows must carry a 'parent' field"):
        ingest_import.import_intermediate(
            _FakeDriver(),
            str(path),
            key_field="id",
            parent_label="Transcription",
            parent_field="id",
            edge_label="HAS_LOCATION",
        )


def test_partial_edge_args_do_not_require_parent(tmp_path):
    path = _write_json(tmp_path, "k", "metadata", [{"id": "a"}])
    drv = _FakeDriver()
    assert (
        ingest_import.import_intermediate(
            drv, str(path), key_field="id", parent_label="Transcription"
        )
        == 1
    )
    query = " ".join(drv.calls[0].query.split())
    assert "MATCH (p:" not in query


# ---------------------------------------------------------------------------
# import_all selection edges
# ---------------------------------------------------------------------------


def test_import_all_empty_dir_returns_empty(tmp_path):
    assert ingest_import.import_all(str(tmp_path), _FakeDriver()) == {}


def test_import_all_key_filter(tmp_path):
    _write_json(tmp_path, "k1", "metadata", [{"id": "a", "parent": "t1"}])
    _write_json(tmp_path, "k2", "metadata", [{"id": "b", "parent": "t1"}])
    drv = _FakeDriver()
    summary = ingest_import.import_all(str(tmp_path), drv, key="k1")
    assert list(summary) == ["k1/metadata"]
    assert len(drv.calls) == 1


def test_import_all_stage_filter(tmp_path):
    _write_json(tmp_path, "k", "transcription", [{"id": "t1"}])
    _write_json(tmp_path, "k", "metadata", [{"id": "m1", "parent": "t1"}])
    drv = _FakeDriver()
    summary = ingest_import.import_all(str(tmp_path), drv, stages=("metadata",))
    assert list(summary) == ["k/metadata"]


def test_import_all_unknown_stage_sorted_last(tmp_path):
    _write_json(tmp_path, "k", "zzcustom", [{"id": "x"}])
    _write_json(tmp_path, "k", "transcription", [{"id": "t1"}])
    drv = _FakeDriver()
    summary = ingest_import.import_all(str(tmp_path), drv, key="k")
    assert list(summary) == ["k/transcription", "k/zzcustom"]


def test_import_all_prefers_json_over_csv(tmp_path):
    # Write JSON first, then CSV (newer mtime); the (key, stage) must still be
    # imported from the JSON artifact, not simply the newest file on disk.
    _write_json(tmp_path, "k", "segments", [{"id": "j1", "parent": "t1"}])
    ingest_write.write_intermediate(
        tmp_path, "k", "segments", [{"id": "c1", "parent": "t1"}], fmt="csv"
    )
    drv = _FakeDriver()
    summary = ingest_import.import_all(str(tmp_path), drv, key="k")
    assert summary == {"k/segments": 1}
    assert drv.calls[0].params["rows"][0]["key"] == "j1"


def test_list_intermediates_for_import_delegates_stage_filter(tmp_path):
    _write_json(tmp_path, "k", "segments", [{"id": "a"}])
    _write_json(tmp_path, "k", "metadata", [{"id": "b"}])
    found = ingest_import.list_intermediates_for_import(str(tmp_path), key="k", stage="segments")
    assert [item["stage"] for item in found] == ["segments"]
