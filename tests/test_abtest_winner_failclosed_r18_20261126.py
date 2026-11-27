"""Hermetic fail-closed tests for A/B winner recording (r18, 2026-11-26).

``abtest.choose_winner`` persists a downstream-consumed A/B decision. Before
this fix it coerced whatever it was given with ``int()`` and recorded it as a
successful winner, so ``99``, ``-1``, ``True``, ``1.9`` and ``"1"`` all became
persisted winners that do not identify a real variant. These tests pin the
recording boundary: only a plain, in-range integer is accepted; everything
else raises ``ValueError`` and leaves the store byte-for-byte unchanged.

No Neo4j, cv2, moviepy, network, service or clock mocking -- just a pytest
``tmp_path`` JSONL store redirected through ``abtest.VARIANT_STORE``.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import abtest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    p = tmp_path / "ab_variants.jsonl"
    monkeypatch.setattr(abtest, "VARIANT_STORE", p)
    return p


def _plan(n: int = 2):
    return abtest.plan_variants(
        "s1", "youtube", [f"s1.v{i}.jpg" for i in range(n)],
        [f"Title {i}" for i in range(n)],
    )


def _raw(store):
    return store.read_text(encoding="utf-8") if store.exists() else None


@pytest.mark.parametrize("bad", [99, 50, -1, -5])
def test_out_of_range_winner_is_refused(store, bad):
    _plan()
    before = _raw(store)
    with pytest.raises(ValueError, match="out of range"):
        abtest.choose_winner("s1", "youtube", bad)
    assert _raw(store) == before
    assert abtest.load_variants("s1", "youtube")[0].winner is None


@pytest.mark.parametrize("bad", [True, False, 1.0, 1.9, "1", None])
def test_non_integer_winner_is_refused(store, bad):
    _plan()
    before = _raw(store)
    with pytest.raises(ValueError, match="non-bool int"):
        abtest.choose_winner("s1", "youtube", bad)
    assert _raw(store) == before
    assert abtest.load_variants("s1", "youtube")[0].winner is None


def test_valid_winner_is_recorded(store):
    _plan()
    updated = abtest.choose_winner("s1", "youtube", 1)
    assert updated is not None
    assert updated.winner == 1
    assert updated.chosen_at is not None
    reloaded = abtest.load_variants("s1", "youtube")[0]
    assert reloaded.winner == 1
    assert reloaded.chosen_at == updated.chosen_at
    assert reloaded.active_variant == 0


def test_valid_zero_winner_is_recorded(store):
    _plan()
    updated = abtest.choose_winner("s1", "youtube", 0)
    assert updated.winner == 0


def test_no_matching_plan_still_returns_none(store):
    _plan()
    before = _raw(store)
    assert abtest.choose_winner("other", "youtube", 1) is None
    assert _raw(store) == before


def test_last_valid_index_is_accepted(store):
    _plan()
    assert abtest.choose_winner("s1", "youtube", 1).winner == 1


def test_plan_with_zero_variants_refuses_every_winner(store):
    _plan(n=0)
    before = _raw(store)
    with pytest.raises(ValueError, match="out of range"):
        abtest.choose_winner("s1", "youtube", 0)
    assert _raw(store) == before


def test_refusal_does_not_disturb_other_plans(store):
    _plan()
    abtest.plan_variants("s2", "tiktok", ["a.jpg", "b.jpg"], ["A", "B"])
    before = _raw(store)
    with pytest.raises(ValueError):
        abtest.choose_winner("s1", "youtube", 99)
    assert _raw(store) == before
    assert abtest.load_variants("s2", "tiktok")[0].winner is None
