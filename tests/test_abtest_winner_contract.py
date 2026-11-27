"""Regression tests for the A/B variant selection/report contract.

`ab choose --variant N` records which variant won. An out-of-range or missing
index used to be accepted: a bogus winner was persisted (and the CLI reported
success), or a missing index crashed with a TypeError traceback. `ab list`
also silently truncated the variant table with `zip()` when the thumbnail and
title lists had different lengths.

Hermetic: the JSONL store is redirected into pytest's tmp_path; no network,
no credentials, no live short rendering.
"""
from types import SimpleNamespace

import pytest

from auto_ingest.shorts import abtest
from auto_ingest.shorts import cli as shorts_cli


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = tmp_path / "ab.jsonl"
    monkeypatch.setattr(abtest, "VARIANT_STORE", p)
    return p


def _two_variant_plan(store):
    return abtest.plan_variants(
        "s1", "youtube", ["a.jpg", "b.jpg"], ["Title A", "Title B"]
    )


# --------------------------------------------------------------------------- #
# choose_winner must validate the index before mutating the store
# --------------------------------------------------------------------------- #
def test_choose_winner_rejects_out_of_range_index_without_writing(store):
    _two_variant_plan(store)

    assert abtest.choose_winner("s1", "youtube", 999) is None
    assert abtest.choose_winner("s1", "youtube", -1) is None

    plan = abtest.load_variants(short_id="s1", platform="youtube")[0]
    assert plan.winner is None
    assert plan.chosen_at is None


def test_choose_winner_rejects_missing_index_without_raising(store):
    _two_variant_plan(store)

    # A missing --variant used to blow up in int(None) and leave a traceback.
    assert abtest.choose_winner("s1", "youtube", None) is None
    plan = abtest.load_variants(short_id="s1", platform="youtube")[0]
    assert plan.winner is None


def test_choose_winner_accepts_last_valid_index(store):
    _two_variant_plan(store)

    updated = abtest.choose_winner("s1", "youtube", 1)
    assert updated is not None
    assert updated.winner == 1
    assert abtest.load_variants(short_id="s1", platform="youtube")[0].winner == 1


# --------------------------------------------------------------------------- #
# the CLI must report a usage error instead of a traceback or a fake success
# --------------------------------------------------------------------------- #
def _choose_args(variant):
    return SimpleNamespace(key="s1", platform="youtube", variant=variant)


def test_cli_ab_choose_missing_variant_is_a_usage_error(store, capsys):
    _two_variant_plan(store)

    assert shorts_cli._cmd_ab_choose(_choose_args(None)) == 2
    assert "--variant" in capsys.readouterr().out
    assert abtest.load_variants(short_id="s1", platform="youtube")[0].winner is None


def test_cli_ab_choose_out_of_range_is_not_reported_as_success(store, capsys):
    _two_variant_plan(store)

    assert shorts_cli._cmd_ab_choose(_choose_args(999)) != 0
    assert "winner=v999" not in capsys.readouterr().out
    assert abtest.load_variants(short_id="s1", platform="youtube")[0].winner is None


# --------------------------------------------------------------------------- #
# ab list reports every variant, even when one side of the pair is missing
# --------------------------------------------------------------------------- #
def test_cli_ab_list_does_not_silently_truncate_unpaired_variants(store, capsys):
    abtest.plan_variants("s1", "youtube", ["a.jpg", "b.jpg", "c.jpg"], ["Title A"])

    assert shorts_cli._cmd_ab_list(SimpleNamespace(key="s1", platform="youtube")) == 0
    out = capsys.readouterr().out
    assert "v0:" in out and "v1:" in out and "v2:" in out
    assert "variants=3" in out
