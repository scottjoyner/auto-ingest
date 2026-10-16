"""Prove external NAS metadata cannot be promoted into custody release evidence."""
from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

import pytest

from auto_ingest.custody.external_inventory import (
    ExternalInventoryError,
    observe_metadata_csv,
)

FIELDS = [
    "card_uuid", "source_relative_path", "candidate_relative_path",
    "expected_bytes", "metadata_status", "observed_bytes",
    "nas_payload_sha256_verified_today", "source_release_authorized",
]


def fixture(root: Path, **updates) -> tuple[Path, str]:
    defaults = {
        "card_uuid": "CARD-1",
        "source_relative_path": "DCIM/a.MP4",
        "candidate_relative_path": "2026/10/16/a.MP4",
        "expected_bytes": "12",
        "metadata_status": "NAME_AND_SIZE_ONLY",
        "observed_bytes": "12",
        "nas_payload_sha256_verified_today": "False",
        "source_release_authorized": "False",
    }
    defaults.update(updates)
    path = root / "inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(defaults)
    return path, sha256(path.read_bytes()).hexdigest()


def observed(path: Path, digest: str):
    return observe_metadata_csv(path, expected_sha256=digest, expected_rows=1)


def test_matching_metadata_cannot_grant_custody_or_release(tmp_path):
    path, digest = fixture(tmp_path)
    before = path.read_bytes()
    result = observed(path, digest)
    assert result.to_dict()["by_status"] == {"NAME_AND_SIZE_ONLY": 1}
    assert result.destination_verified_files == 0
    assert result.source_release_allowed is False
    assert result.import_as_custody_receipt_allowed is False
    assert path.read_bytes() == before


def test_absent_metadata_cannot_grant_release(tmp_path):
    path, digest = fixture(tmp_path, metadata_status="EXACT_NAME_ABSENT", observed_bytes="")
    result = observed(path, digest)
    assert result.to_dict()["by_card_and_status"] == {"CARD-1|EXACT_NAME_ABSENT": 1}
    assert not result.source_release_allowed


@pytest.mark.parametrize("column,value", [
    ("nas_payload_sha256_verified_today", "True"),
    ("source_release_authorized", "True"),
    ("metadata_status", "VERIFIED"),
    ("observed_bytes", "11"),
    ("expected_bytes", "-1"),
])
def test_reject_release_proof_or_inconsistent_claim(tmp_path, column, value):
    path, digest = fixture(tmp_path, **{column: value})
    with pytest.raises(ExternalInventoryError):
        observed(path, digest)


def test_refuses_mutated_csv_even_if_parseable(tmp_path):
    path, digest = fixture(tmp_path)
    path.write_text(path.read_text().replace("a.MP4", "b.MP4"))
    with pytest.raises(ExternalInventoryError, match="digest mismatch"):
        observed(path, digest)


def test_refuses_wrong_row_count(tmp_path):
    path, digest = fixture(tmp_path)
    with pytest.raises(ExternalInventoryError, match="row count"):
        observe_metadata_csv(path, expected_sha256=digest, expected_rows=2)


def test_refuses_symlink_input(tmp_path):
    path, digest = fixture(tmp_path)
    (tmp_path / "alias.csv").symlink_to(path)
    with pytest.raises(ExternalInventoryError):
        observed(tmp_path / "alias.csv", digest)


def test_bounded_rows_and_pinned_sha_required(tmp_path):
    path, digest = fixture(tmp_path)
    with pytest.raises(ExternalInventoryError):
        observe_metadata_csv(path, expected_sha256=digest, expected_rows=1, max_rows=0)
    with pytest.raises(ExternalInventoryError):
        observe_metadata_csv(path, expected_sha256="a" * 63, expected_rows=1)


def test_duplicate_identity_refused(tmp_path):
    path, _ = fixture(tmp_path)
    body = path.read_text()
    header, row = body.splitlines()
    path.write_text(header + "\n" + row + "\n" + row + "\n")
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ExternalInventoryError, match="duplicate"):
        observe_metadata_csv(path, expected_sha256=digest, expected_rows=2)


def test_truncated_row_refused(tmp_path):
    path, _ = fixture(tmp_path)
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + ",".join(lines[1].split(",")[:4]) + "\n")
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ExternalInventoryError, match="malformed"):
        observed(path, digest)
