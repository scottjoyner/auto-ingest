"""Read-only census does not assert destination custody from metadata matches."""
from __future__ import annotations

from auto_ingest.custody.fleet_preflight import inventory


def test_preflight_reports_unique_sequenced_mp4_and_blocked_metadata(tmp_path):
    card = tmp_path / "CARD" / "DCIM" / "Movie"
    card.mkdir(parents=True)
    for filename in ("2026_0626_034541_000001F.MP4", "2026_0626_034541_000002R.MP4",
                     "1999_1128_202128_F.MP4", "._1999_1128_202128_F.MP4"):
        (card / filename).write_bytes(b"some-bytes")
    before = sorted(p.name for p in card.iterdir())
    result = inventory([tmp_path / "CARD"])
    assert result["totals"]["mp4_named_files"] == 4
    assert result["totals"]["stageable_mp4"] == 2
    assert result["collision_destinations"] == 0
    assert result["cards"][0]["blocked"] == {
        "clock_review_pre2000": 1, "out_of_scope:other": 1,
    }
    assert result["source_release_allowed"] is False
    assert sorted(p.name for p in card.iterdir()) == before


def test_cross_card_collision_reports_both_sources_and_size_is_not_custody(tmp_path):
    cards = []
    for label in ("CARD_A", "CARD_B"):
        root = tmp_path / label
        root.mkdir()
        (root / "2026_0903_175924_F.MP4").write_bytes(b"same-size")
        cards.append(root)
    dest = tmp_path / "nas" / "2026" / "09" / "03"
    dest.mkdir(parents=True)
    (dest / "2026_0903_175924_F.MP4").write_bytes(b"DIFFERENT")
    result = inventory(cards, destination_root=tmp_path / "nas")
    assert result["collision_destinations"] == 1
    assert len(result["collision_samples"][0]["sources"]) == 2
    assert sum(c["destination_same_size_only"] for c in result["cards"]) == 2
    assert result["hashes_performed"] == 0
    assert result["destination_proof_level"] == "metadata_only_not_verified"
    assert result["source_release_allowed"] is False
