"""Round-40 hermetic contract for ``graph_communities.py``'s UnionFind.

``graph_communities.main()`` builds connected components from ANN similarity
edges with a pure-Python union-find, then assigns sequential ``community_id``
values by iterating ``uf.p`` in insertion order and mapping each node to the id
of its first-seen root.  That whole assignment depends on three small
invariants: ``find`` lazily registers unseen keys, ``union`` attaches
``root(a)`` to ``root(b)`` (there is no rank/size balancing, so call order picks
the root), and ``find`` uses halving path compression rather than full
compression.  None of these had a test; this module pins them in-process.

The import is hermetic: the Neo4j driver and the config resolver are stubbed in
``sys.modules`` while the module body executes, so no driver is opened, no
``.env``/config file is read, and no network, graph, service, credential or
live state is touched.  Every case below operates on literal dict inputs.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(scope="module")
def uf_mod():
    """Load graph_communities.py with its two heavy module-scope imports stubbed."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "graph_communities.py"
    fake_neo4j = mock.MagicMock()
    fake_cfg = mock.MagicMock()
    with (
        mock.patch.dict(
            sys.modules,
            {"neo4j": fake_neo4j, "auto_ingest_config": fake_cfg},
        ),
        mock.patch.object(logging, "basicConfig"),
    ):
        spec = importlib.util.spec_from_file_location("graph_communities_r40", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def test_new_uf_starts_empty(uf_mod):
    assert uf_mod.UnionFind().p == {}


def test_add_registers_self_and_is_idempotent(uf_mod):
    uf = uf_mod.UnionFind()
    uf.add("a")
    assert uf.p == {"a": "a"}
    uf.add("a")
    assert uf.p == {"a": "a"}


def test_add_does_not_reset_existing_parent(uf_mod):
    uf = uf_mod.UnionFind()
    uf.union("a", "b")  # p["a"] == "b"
    uf.add("a")
    assert uf.p["a"] == "b"


def test_find_autoadds_unseen_key(uf_mod):
    uf = uf_mod.UnionFind()
    assert uf.find("x") == "x"
    assert uf.p == {"x": "x"}


def test_find_compresses_path(uf_mod):
    uf = uf_mod.UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.p["a"] == "b"
    assert uf.p["b"] == "c"
    assert uf.find("a") == "c"
    assert uf.p["a"] == "c"


def test_find_compresses_one_level_per_visit(uf_mod):
    uf = uf_mod.UnionFind()
    for a, b in (("a", "b"), ("b", "c"), ("c", "d")):
        uf.union(a, b)
    assert uf.p == {"a": "b", "b": "c", "c": "d", "d": "d"}
    assert uf.find("a") == "d"
    # Halving compression: "a" jumps to its grandparent, not straight to root.
    assert uf.p["a"] == "c"
    assert uf.p["c"] == "d"
    assert uf.find("a") == "d"
    assert uf.p["a"] == "d"


def test_union_merges_into_single_component(uf_mod):
    uf = uf_mod.UnionFind()
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")
    assert uf.p == {"a": "b", "b": "b"}


def test_union_root_is_second_argument_root_not_rank_based(uf_mod):
    # Pin the order-sensitive attach so a future union-by-rank change cannot
    # silently relabel the sequential community ids assigned by main().
    uf1 = uf_mod.UnionFind()
    uf1.union("a", "b")
    assert uf1.p["a"] == "b"
    uf2 = uf_mod.UnionFind()
    uf2.union("b", "a")
    assert uf2.p["b"] == "a"


def test_union_same_set_is_noop(uf_mod):
    uf = uf_mod.UnionFind()
    uf.union("a", "b")
    before = dict(uf.p)
    uf.union("a", "b")
    uf.union("b", "a")
    assert uf.p == before


def test_union_self_is_noop(uf_mod):
    uf = uf_mod.UnionFind()
    uf.add("a")
    uf.union("a", "a")
    assert uf.p == {"a": "a"}


def test_transitive_union_all_share_root(uf_mod):
    uf = uf_mod.UnionFind()
    for a, b in (("a", "b"), ("b", "c"), ("c", "d")):
        uf.union(a, b)
    assert {uf.find(x) for x in ("a", "b", "c", "d")} == {"d"}


def test_mixed_hashable_keys(uf_mod):
    uf = uf_mod.UnionFind()
    uf.union("a", 1)
    uf.union((2, 3), 1)
    assert uf.find("a") == uf.find(1) == uf.find((2, 3)) == 1


def test_connected_components_and_sizes_mirror_main(uf_mod):
    uf = uf_mod.UnionFind()
    for a, b in (("n1", "n2"), ("n2", "n3"), ("m1", "m2"), ("solo", "solo")):
        uf.union(a, b)

    # Exact reproduction of main()'s root->cid and size computation.
    root_to_cid = {}
    nid_to_cid = {}
    for nid in uf.p:
        root = uf.find(nid)
        if root not in root_to_cid:
            root_to_cid[root] = len(root_to_cid)
        nid_to_cid[nid] = root_to_cid[root]
    sizes = {}
    for cid in nid_to_cid.values():
        sizes[cid] = sizes.get(cid, 0) + 1

    assert nid_to_cid == {
        "n1": 0,
        "n2": 0,
        "n3": 0,
        "m1": 1,
        "m2": 1,
        "solo": 2,
    }
    assert sizes == {0: 3, 1: 2, 2: 1}
