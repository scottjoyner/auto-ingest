"""Hermetic contract: feedback with no observed views is not a measured 'loser'.

Round 27 (r27) disjoint slice.  ``content_os.core.feedback()`` classifies a run
into ``stores/winners``, ``stores/losers`` or ``stores/feedback`` from its
bookmark rate.  When ``views == 0`` there is no denominator, so the rate is a
placeholder 0.0 -- not evidence of a poor result.  Filing that placeholder into
``stores/losers`` is a vacuous classification that can drive downstream
"learn from losers" decisions off data that was never observed.

No Neo4j, LLM, network or live data: everything is an in-memory/tmp_path fixture.
"""

from pathlib import Path

from content_os import core


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "content-os"
    core.init(root)
    return root


def test_zero_views_is_not_filed_as_loser(tmp_path):
    root = _root(tmp_path)
    path = core.create_run(root, "No Views", "ORIGINAL")

    rates = core.feedback(
        root,
        path.name,
        views=0,
        bookmarks=0,
        likes=7,
        replies=3,
        reposts=1,
    )

    assert rates["bookmark_rate"] == 0.0
    rid = path.name
    assert (root / "stores" / "feedback" / f"{rid}.md").exists(), (
        "no-view feedback must land in the neutral feedback store"
    )
    assert not (root / "stores" / "losers" / f"{rid}.md").exists(), (
        "a 0.0 rate from zero views is not a measured loss"
    )
    assert not (root / "stores" / "winners" / f"{rid}.md").exists()


def test_zero_views_with_inconsistent_bookmarks_stays_neutral(tmp_path):
    root = _root(tmp_path)
    path = core.create_run(root, "Inconsistent Counts", "ORIGINAL")

    core.feedback(
        root,
        path.name,
        views=0,
        bookmarks=12,
        likes=0,
        replies=0,
        reposts=0,
    )

    rid = path.name
    assert (root / "stores" / "feedback" / f"{rid}.md").exists()
    assert not (root / "stores" / "winners" / f"{rid}.md").exists()
    assert not (root / "stores" / "losers" / f"{rid}.md").exists()


def test_measured_rates_still_classify_winners_and_losers(tmp_path):
    """Guard: the fix must not change the real, observed-data paths."""
    root = _root(tmp_path)
    winner = core.create_run(root, "Measured Winner", "ORIGINAL")
    loser = core.create_run(root, "Measured Loser", "ORIGINAL")

    core.feedback(
        root, winner.name, views=1000, bookmarks=50, likes=10, replies=5, reposts=5
    )  # 0.05 -> winners
    core.feedback(
        root, loser.name, views=1000, bookmarks=0, likes=0, replies=0, reposts=0
    )  # 0.00 -> losers

    assert (root / "stores" / "winners" / f"{winner.name}.md").exists()
    assert (root / "stores" / "losers" / f"{loser.name}.md").exists()
