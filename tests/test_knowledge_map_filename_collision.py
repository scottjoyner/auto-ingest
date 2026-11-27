"""Distinct Neo4j nodes must not overwrite each other in the vault export.

``auto_ingest.knowledge_map._safe_filename`` sanitises a node name into a
markdown filename. Sanitising is many-to-one: a slash, a space and an underscore
all become ``_``, and names are truncated to a fixed length. When two nodes in
one label sanitise to the same filename, ``_export_label`` writes both to the
same path and the second silently destroys the first -- a bounded, offline
correctness defect in the pure filename helper and the CLI export path.

Pure: no live DB, no network, no vault writes outside ``tmp_path``.
"""
from __future__ import annotations

import auto_ingest.knowledge_map as km


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    """Minimal stand-in for a neo4j session; paged by the kwargs _export_label uses."""

    def __init__(self, nodes):
        self._nodes = nodes

    def run(self, _query, **kwargs):
        lim = kwargs["lim"]
        skip = kwargs["skip"]
        return _FakeResult([{"n": n} for n in self._nodes[skip:skip + lim]])


def test_distinct_node_names_sanitise_to_distinct_filenames():
    # A slash, a space and an underscore are three different source names but
    # collapse to one stem on the pristine implementation.
    names = ["Entity_Alice/Bob", "Entity_Alice Bob", "Entity_Alice_Bob"]
    filenames = [km._safe_filename(n) for n in names]
    assert len(set(filenames)) == len(names), (
        f"distinct node names collided on the same filename: {filenames}"
    )


def test_long_names_differing_past_the_truncation_point_do_not_collide():
    # Truncation is another many-to-one step: two names differing only in their
    # tail must not land on the same file.
    a = "Entity_" + "a" * 130 + "_ALICE"
    b = "Entity_" + "a" * 130 + "_BOB"
    assert km._safe_filename(a) != km._safe_filename(b)


def test_export_label_writes_every_node_when_titles_collide(tmp_path):
    nodes = [
        {"name": "Alice/Bob", "text": "FIRST BODY"},
        {"name": "Alice Bob", "text": "SECOND BODY"},
    ]
    written = km._export_label(_FakeSession(nodes), "Entity", tmp_path, budget=10)
    assert written == 2
    files = sorted(tmp_path.glob("*.md"))
    assert len(files) == 2, f"expected 2 files, got {[p.name for p in files]}"
    bodies = " ".join(p.read_text(encoding="utf-8") for p in files)
    assert "FIRST BODY" in bodies, "first node was overwritten"
    assert "SECOND BODY" in bodies


def test_safe_filename_is_bounded_and_path_safe():
    # Positive control: sanitisation stays bounded and cannot escape out_root.
    name = "Entity_" + "a/b " * 200
    fname = km._safe_filename(name)
    assert fname.endswith(".md")
    assert "/" not in fname and "\\" not in fname
    assert len(fname) <= 140


def test_safe_filename_is_deterministic():
    name = "Entity_Alice/Bob"
    assert km._safe_filename(name) == km._safe_filename(name)
