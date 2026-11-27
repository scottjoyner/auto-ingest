"""Regression guard for the embedding-coverage audit's label discovery.

``verify_coverage`` is the "full data recovery" gate: exit 0 means every
text-bearing label has vectors for every required/indexed prop. A label that is
silently dropped from the label list is never queried, so a gap on it reports as
a false clean (exit 0). These tests are hermetic: a fake Neo4j session stands in
for the driver, so no graph and no credentials are involved.
"""
from __future__ import annotations

import json
import re

import verify_coverage as vc

CANONICAL = ["Segment", "Summary", "Transcription", "Utterance"]


class _Record:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _Result:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _FakeSession:
    """Serve SHOW VECTOR INDEXES and the two count queries from fixed tables."""

    def __init__(self, indexed, counts):
        self.indexed = indexed
        self.counts = counts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **_kwargs):
        if "SHOW VECTOR INDEXES" in query:
            return _Result([_Record({"props": [p], "lbls": lbls})
                            for p, lbls in self.indexed.items()])
        label = re.search(r"MATCH \(n:`([^`]+)`\)", query).group(1)
        prop = re.search(r"n\.`([^`]+)` IS NOT NULL", query)
        # The "done" query adds `AND n.<prop> IS NOT NULL`; the total query
        # omits it, so the backticked prop is only present on the done side.
        prop_name = prop.group(1) if prop else "emb_gte_small"
        total, embedded = self.counts[(label, prop_name)]
        return _Result([_Record({"c": embedded if prop else total})])


def _patch_driver(monkeypatch, session):
    class _FakeDriver:
        def session(self):
            return session

        def close(self):
            pass

    monkeypatch.setattr(
        vc, "get_neo4j_config",
        lambda: {"uri": "bolt://fake", "user": "u", "password": "p"},
    )
    monkeypatch.setattr(vc.GraphDatabase, "driver", lambda *a, **k: _FakeDriver())


def test_canonical_text_labels_are_audited_even_when_only_one_is_indexed():
    # Only Segment currently carries an emb_gte_small index. The other three
    # canonical labels must still appear: a dropped/never-created index is the
    # most likely place for an unreported gap to hide.
    indexed = {"emb_gte_small": ["Segment"]}
    assert vc.text_bearing_labels(None, indexed) == CANONICAL


def test_newly_indexed_text_label_is_added_to_the_audit():
    indexed = {"emb_gte_small": ["Segment", "Note"]}
    assert vc.text_bearing_labels(None, indexed) == sorted(CANONICAL + ["Note"])


def test_known_non_text_labels_are_still_excluded():
    indexed = {"emb_gte_small": ["Segment"],
               "frame_embedding": ["Frame"],
               "global_speaker_embedding": ["GlobalSpeaker"],
               "appearance": ["DashcamEmbedding"]}
    assert vc.text_bearing_labels(None, indexed) == CANONICAL


def test_empty_index_falls_back_to_canonical_labels():
    assert vc.text_bearing_labels(None, {}) == CANONICAL


def test_main_exits_1_for_gap_on_an_unindexed_canonical_label(monkeypatch, capsys):
    """Full fail->pass proof: a gap on Utterance must not be reported as OK.

    Only Segment is indexed. Before the fix text_bearing_labels() returned just
    ['Segment'], so main() saw a single fully-covered row and exited 0. With the
    fix Utterance is audited, its 0/5 gap is found, and main() exits 1.
    """
    session = _FakeSession(
        indexed={"emb_gte_small": ["Segment"]},
        counts={
            ("Segment", "emb_gte_small"): (10, 10),
            ("Summary", "emb_gte_small"): (2, 2),
            ("Transcription", "emb_gte_small"): (3, 3),
            ("Utterance", "emb_gte_small"): (5, 0),
        },
    )
    _patch_driver(monkeypatch, session)

    rc = vc.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert [r["label"] for r in payload["gaps"]] == ["Utterance"]
    assert payload["rows"]
