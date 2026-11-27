"""Hermetic contract tests for fleet-batch item assembly (round 44).

`auto_ingest.fleet_batch.build_items` is the pure front-half of the
folder/explicit-list fan-out: it merges an explicit ``--items`` list with a
bounded ``--input`` / ``--glob`` expansion and de-duplicates while preserving
first-seen order. It has no network, DB, service, credential or Neo4j
dependency; the only I/O is a glob over a caller-supplied directory, which these
tests keep inside pytest's ``tmp_path`` (under the mandated SSD scratch).

This is regression coverage only -- no production behaviour is changed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from auto_ingest.fleet_batch import build_items


def _args(items=None, input=None, glob=None):
    """Build just the Namespace attributes build_items reads."""
    return argparse.Namespace(items=items, input=input, glob=glob)


def _seed(directory: Path, *names: str) -> list[str]:
    """Create tiny empty files and return their absolute string paths."""
    paths = []
    for name in names:
        p = directory / name
        p.touch()
        paths.append(str(p))
    return paths


def test_explicit_items_deduplicate_preserving_first_occurrence():
    assert build_items(_args(items=["x", "y", "x", "z", "y"])) == ["x", "y", "z"]


def test_glob_matches_are_sorted_and_absolute(tmp_path):
    _seed(tmp_path, "b.mp4", "a.mp4", "c.txt")
    got = build_items(_args(input=str(tmp_path), glob="*.mp4"))
    assert got == sorted([str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")])
    assert all(item.endswith(".mp4") for item in got)


def test_expansion_requires_both_input_and_glob(tmp_path):
    _seed(tmp_path, "a.mp4")
    # input without a glob pattern: no expansion happens.
    assert build_items(_args(input=str(tmp_path), glob=None)) == []
    # glob without an input root: explicit items pass through untouched.
    assert build_items(_args(items=["z"], input=None, glob="*.mp4")) == ["z"]


def test_no_glob_match_leaves_explicit_items_intact(tmp_path):
    _seed(tmp_path, "a.mp4")
    got = build_items(_args(items=["z"], input=str(tmp_path), glob="*.nope"))
    assert got == ["z"]


def test_explicit_items_precede_glob_and_dedup_is_first_wins(tmp_path):
    a, b = _seed(tmp_path, "a.mp4", "b.mp4")
    got = build_items(_args(items=[a, "z"], input=str(tmp_path), glob="*.mp4"))
    # The explicit a.mp4 keeps its leading position; the same path produced by
    # the glob is collapsed rather than appended a second time.
    assert got == [a, "z", b]


def test_empty_namespace_yields_empty_list():
    assert build_items(_args()) == []


def test_namespace_items_are_not_mutated():
    ns = _args(items=["a", "b", "a"])
    before = list(ns.items)
    build_items(ns)
    assert ns.items == before


def test_nested_glob_preserves_relative_directory_shape(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    a, b = _seed(sub, "b.mp4", "a.mp4")
    got = build_items(_args(input=str(tmp_path), glob="sub/*.mp4"))
    assert got == sorted([a, b])
    assert all("sub" in item for item in got)
