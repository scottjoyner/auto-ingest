"""Exact date-folder probing is manifest-only and release-denying."""
from __future__ import annotations

import hashlib
import json

import pytest

from auto_ingest.custody import date_archive_probe as ix


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    root.mkdir()
    rows = []
    for uuid, filename, size in (
        ("AAAA-0001", "DCIM/Movie/2026_0627_100611_F.MP4", 3),
        ("AAAA-0001", "DCIM/Movie/2026_0627_100611_R.MP4", 4),
        ("BBBB-0002", "DCIM/Movie/2026_0628_100611_F.MP4", 2),
        ("BBBB-0002", "DCIM/Movie/._1999_1128_202128_F.MP4", 1),
    ):
        rows.append({
            "card_uuid": uuid, "source_relative_path": filename,
            "bytes": size, "mp4_staging": "stageable_filename_only",
            "classification": "no_indexed_candidate_not_proven_missing",
            "current_nas_sha_verified": False,
            "source_release_allowed": False,
        })
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(x) + "\n" for x in rows))
    date1 = root / "2026" / "06" / "27"
    date1.mkdir(parents=True)
    (date1 / "2026_0627_100611_F.MP4").write_bytes(b"abc")
    (date1 / "2026_0627_100611_R.MP4").write_bytes(b"abcd")
    date2 = root / "2026" / "06" / "28"
    date2.mkdir(parents=True)
    (date2 / "2026_0628_100611_F.MP4").write_bytes(b"xyz")
    monkeypatch.setattr(ix, "_validate_root",
                        lambda path, device: path.stat().st_dev)
    return index, root, rows


def run(data, **kwargs):
    index, root, _ = data
    return ix.inspect(index, ix.sha256(index), root,
                      expected_device="//test-nas/fileserver", **kwargs)


def test_exact_date_probe_find_unique_archive_candidates(inputs):
    result = run(inputs)
    assert result["matches"] == 2
    assert result["probes"] == 3
    assert result["counts"]["different_size"] == 1
    assert result["counts"]["undated_or_appledouble"] == 1
    assert result["by_card"] == {"AAAA-0001": 2}
    assert result["source_release_allowed"] is False
    assert not result["all_destination_sha256_verified"]
    assert all(x["evidence"] == "current_path_and_size_only"
               and not x["current_nas_sha256_verified"]
               for x in result["rows"])


def test_wrong_index_sha_does_not_read_candidate(inputs):
    index, root, _ = inputs
    with pytest.raises(ValueError, match="index SHA256 mismatch"):
        ix.inspect(index, "f" * 64, root, expected_device="//test-nas/fileserver")


def test_probe_count_cap_is_enforced(inputs):
    report = run(inputs, max_probes=1)
    assert report["probes"] == 1
    assert report["counts"]["budget_deferred"] == 2
    with pytest.raises(ValueError, match="unbounded"):
        run(inputs, max_probes=ix.MAX_PROBES + 1)


def test_shared_destination_claim_across_two_cards_refused(inputs):
    index, _, rows = inputs
    fake = dict(rows[0])
    fake["card_uuid"] = "CCCC-0003"
    index.write_text(index.read_text() + json.dumps(fake) + "\n")
    with pytest.raises(ValueError, match="claimed by multiple source"):
        run(inputs)


def test_symlinked_destination_is_not_metadata_match(inputs):
    index, root, _ = inputs
    f = root / "2026/06/27/2026_0627_100611_F.MP4"
    f.unlink()
    f.symlink_to(root / "2026/06/27/2026_0627_100611_R.MP4")
    report = run(inputs)
    assert report["matches"] == 1
    assert report["counts"]["nonregular_or_unexpected_device"] == 1


def test_exclusive_report_preserves_prior_evidence(inputs, tmp_path):
    result = run(inputs)
    out = tmp_path / "receipt.jsonl"
    sha = ix.write_report(out, result)
    assert sha == hashlib.file_digest(out.open("rb"), "sha256").hexdigest()
    assert len(out.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        ix.write_report(out, result)
