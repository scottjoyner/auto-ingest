"""Round-6 hermetic contracts for the cleaned-intermediate write/import path.

Focused and offline: every case uses pytest ``tmp_path`` plus a recording fake
Neo4j driver. No live graph, no live DB/WAL, no remote mount, no network, no
service. This file covers ``ingest_write`` round-trips / discovery and the
dynamic-batch behaviour of ``ingest_import.import_intermediate``. It does not
duplicate the round-5 receipt/stage-order tests (``test_receipt_stage_durability_r5``
covers ``import_all`` ordering and ``GraphOutbox.replay``) or the claim-protocol
tests (``test_ingest_claim``).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from auto_ingest import ingest_import, ingest_write


class _Result:
    """Minimal driver result: no rows are consumed by import_intermediate."""

    def single(self):
        return None

    def data(self):
        return []


class _RecordingSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        self._log.append((cypher, copy.deepcopy(params)))
        return _Result()


class _RecordingDriver:
    def __init__(self):
        self.calls: list = []

    def session(self):
        return _RecordingSession(self.calls)


# --- write / read round-trips ------------------------------------------------


def test_json_round_trip_is_deterministic_and_preserves_types(tmp_path):
    rows = [
        {"id": "b", "count": 2, "ok": True, "tags": ["x", "y"]},
        {"id": "a", "count": 1, "ok": False, "tags": []},
    ]
    path = ingest_write.write_intermediate(str(tmp_path), "k1", "metadata", rows)
    assert path.endswith("k1.metadata.json")
    assert ingest_write.read_intermediate(str(tmp_path), "k1", "metadata") == rows

    # sort_keys=True makes the artifact byte-stable across dict key orderings.
    first = Path(path).read_bytes()
    reordered = [dict(reversed(list(r.items()))) for r in rows]
    ingest_write.write_intermediate(str(tmp_path), "k1", "metadata", reordered)
    assert Path(path).read_bytes() == first


def test_json_empty_records_round_trip_as_empty_list(tmp_path):
    path = ingest_write.write_intermediate(str(tmp_path), "k1", "summary", [])
    assert os.path.getsize(path) > 0
    assert ingest_write.read_intermediate(str(tmp_path), "k1", "summary") == []


def test_csv_round_trip_unions_fields_and_emits_empty_artifact(tmp_path):
    rows = [{"id": "a", "x": "1"}, {"id": "b", "y": "2"}]
    path = ingest_write.write_intermediate(str(tmp_path), "k1", "segments", rows, fmt="csv")
    assert path.endswith("k1.segments.csv")
    assert ingest_write.read_intermediate(str(tmp_path), "k1", "segments", fmt="csv") == [
        {"id": "a", "x": "1", "y": ""},
        {"id": "b", "x": "", "y": "2"},
    ]

    empty = ingest_write.write_intermediate(str(tmp_path), "k2", "segments", [], fmt="csv")
    assert os.path.getsize(empty) == 0
    assert ingest_write.read_intermediate(str(tmp_path), "k2", "segments", fmt="csv") == []


def test_write_intermediate_overwrites_instead_of_appending(tmp_path):
    out = str(tmp_path)
    ingest_write.write_intermediate(out, "k1", "speakers", [{"id": 1}, {"id": 2}])
    ingest_write.write_intermediate(out, "k1", "speakers", [{"id": 3}])
    assert ingest_write.read_intermediate(out, "k1", "speakers") == [{"id": 3}]


def test_key_separators_are_sanitized_and_discovery_restores_them(tmp_path):
    out = str(tmp_path)
    key = "dashcam/2026-07-14/CLT_1423_F"
    path = ingest_write.write_intermediate(out, key, "metadata", [{"id": "m1"}])
    assert os.path.dirname(path) == out  # no nested directories created
    assert os.path.basename(path) == ("dashcam__2026-07-14__CLT_1423_F.metadata.json")
    found = ingest_write.list_intermediates(out, key=key, stage="metadata")
    assert len(found) == 1
    assert found[0]["key"] == key
    assert found[0]["stage"] == "metadata"
    assert found[0]["fmt"] == "json"


def test_list_intermediates_filters_and_orders_newest_first(tmp_path):
    out = str(tmp_path)
    ingest_write.write_intermediate(out, "k1", "transcription", [{"id": "t"}])
    ingest_write.write_intermediate(out, "k1", "segments", [{"id": "s"}])
    ingest_write.write_intermediate(out, "k2", "metadata", [{"id": "m"}])
    # json + csv for the same (key, stage) must both be discoverable.
    ingest_write.write_intermediate(out, "k1", "segments", [{"id": "s2"}], fmt="csv")

    base = 1_700_000_000
    for i, name in enumerate(["k1.transcription.json", "k1.segments.json", "k2.metadata.json"]):
        os.utime(os.path.join(out, name), (base + i, base + i))
    os.utime(os.path.join(out, "k1.segments.csv"), (base + 99, base + 99))

    found = ingest_write.list_intermediates(out)
    assert [f["path"].split(os.sep)[-1] for f in found] == [
        "k1.segments.csv",
        "k2.metadata.json",
        "k1.segments.json",
        "k1.transcription.json",
    ]
    assert {f["key"] for f in ingest_write.list_intermediates(out, key="k1")} == {"k1"}
    assert {f["stage"] for f in ingest_write.list_intermediates(out, stage="segments")} == {
        "segments"
    }
    assert ingest_write.list_intermediates(out, key="missing") == []


@pytest.mark.parametrize("fmt", ["xml", "JSON", "", None])
def test_unsupported_format_is_rejected(tmp_path, fmt):
    with pytest.raises(ValueError, match="unsupported format"):
        ingest_write.write_intermediate(str(tmp_path), "k1", "metadata", [], fmt=fmt)
    with pytest.raises(ValueError, match="unsupported format"):
        ingest_write.read_intermediate(str(tmp_path), "k1", "metadata", fmt=fmt)


def test_read_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_write.read_intermediate(str(tmp_path), "absent", "metadata")


# --- dynamic-batch import ----------------------------------------------------


def test_import_streams_bounded_batches_without_losing_rows(tmp_path):
    out = str(tmp_path)
    rows = [{"id": f"r{i}", "text": f"v{i}"} for i in range(5)]
    path = ingest_write.write_intermediate(out, "k1", "metadata", rows)
    driver = _RecordingDriver()

    imported = ingest_import.import_intermediate(
        driver, path, batch=2, label="LocationSample", key_field="id"
    )

    assert imported == 5
    assert len(driver.calls) == 3
    assert [len(cypher_params["rows"]) for _, cypher_params in driver.calls] == [2, 2, 1]
    cypher, params = driver.calls[0]
    assert "UNWIND $rows AS r" in cypher
    assert "MERGE (n:LocationSample {id: r.key})" in cypher
    assert "SET n += r.props" in cypher
    assert params["rows"][0] == {"key": "r0", "props": {"text": "v0"}}


def test_import_of_empty_artifact_issues_no_writes(tmp_path):
    out = str(tmp_path)
    path = ingest_write.write_intermediate(out, "k1", "summary", [])
    driver = _RecordingDriver()
    assert ingest_import.import_intermediate(driver, path) == 0
    assert driver.calls == []


def test_import_links_parent_when_edge_fields_supplied(tmp_path):
    out = str(tmp_path)
    rows = [{"id": "s1", "parent": "t1", "text": "hello"}]
    path = ingest_write.write_intermediate(out, "k1", "segments", rows)
    driver = _RecordingDriver()

    imported = ingest_import.import_intermediate(
        driver,
        path,
        batch=10,
        label="Segment",
        key_field="id",
        parent_label="Transcription",
        parent_field="id",
        edge_label="HAS_SEGMENT",
    )

    assert imported == 1
    cypher, params = driver.calls[0]
    assert "MATCH (p:Transcription {id: r.parent})" in cypher
    assert "MERGE (p)-[:HAS_SEGMENT]->(n)" in cypher
    assert params["rows"][0]["parent"] == "t1"


def test_import_requires_parent_field_for_edge_mode(tmp_path):
    out = str(tmp_path)
    path = ingest_write.write_intermediate(
        out, "k1", "segments", [{"id": "s1", "text": "no parent"}]
    )
    with pytest.raises(KeyError, match="parent"):
        ingest_import.import_intermediate(
            _RecordingDriver(),
            path,
            label="Segment",
            key_field="id",
            parent_label="Transcription",
            parent_field="id",
            edge_label="HAS_SEGMENT",
        )


@pytest.mark.parametrize("batch", [0, -1])
def test_import_rejects_nonpositive_batch(tmp_path, batch):
    out = str(tmp_path)
    path = ingest_write.write_intermediate(out, "k1", "metadata", [{"id": "r1"}])
    with pytest.raises(ValueError, match="batch must be >= 1"):
        ingest_import.import_intermediate(_RecordingDriver(), path, batch=batch)


def test_import_rejects_non_list_json_and_unknown_extension(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"id": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        ingest_import.import_intermediate(_RecordingDriver(), str(bad_json))

    bad_ext = tmp_path / "bad.parquet"
    bad_ext.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported intermediate extension"):
        ingest_import.import_intermediate(_RecordingDriver(), str(bad_ext))
