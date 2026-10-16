"""Read-only, non-authoritative observations from external NAS reconciliation.

This module never transforms path/size metadata into a custody verified,
releasable, or copied verdict. It does not write campaign bundles or ledgers.
It is deliberately separate from custody.ledger's proof-bearing rows.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REQUIRED_FIELDS = frozenset({
    "card_uuid", "source_relative_path", "candidate_relative_path",
    "expected_bytes", "metadata_status", "observed_bytes",
    "nas_payload_sha256_verified_today", "source_release_authorized",
})

METADATA_STATUSES = frozenset({
    "EXACT_NAME_ABSENT", "NAME_AND_SIZE_ONLY", "SIZE_MISMATCH",
    "NOT_REGULAR_FILE", "SYMLINK_NOT_ACCEPTED", "SYMLINK_PARENT_NOT_ACCEPTED",
    "PARENT_NOT_DIRECTORY", "IO_UNKNOWN", "PERMISSION_UNKNOWN",
    "INVALID_RELATIVE_PATH",
})


class ExternalInventoryError(ValueError):
    """A caller supplied ambiguous or contradictory observational evidence."""


@dataclass(frozen=True)
class Observation:
    rows: int
    source_sha256: str
    by_status: dict[str, int]
    by_card_and_status: dict[str, int]
    meta_only: bool = True
    destination_verified_files: int = 0
    source_release_allowed: bool = False
    import_as_custody_receipt_allowed: bool = False

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "source_sha256": self.source_sha256,
            "by_status": dict(sorted(self.by_status.items())),
            "by_card_and_status": dict(sorted(self.by_card_and_status.items())),
            "meta_only": self.meta_only,
            "destination_verified_files": self.destination_verified_files,
            "source_release_allowed": self.source_release_allowed,
            "import_as_custody_receipt_allowed": self.import_as_custody_receipt_allowed,
        }


def observe_metadata_csv(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    max_rows: int = 6000,
) -> Observation:
    """Read-only external inventory summary; never return proof-bearing records.

    A matching name/size or an externally supplied 'verified' flag cannot
    become a destination-hash receipt. The caller must pin a prior file digest
    and a row count. Missing/invalid/modified evidence is a fatal error.
    """
    if not (
        1 <= expected_rows <= max_rows <= 100_000
        and len(expected_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ExternalInventoryError("expected row count or SHA-256 pin invalid")

    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ExternalInventoryError("source inventory must be a regular, existing file")

    # Parse exactly the same immutable byte snapshot that was pinned. This
    # avoids a read-digest/reopen-parse race across two different file versions.
    if p.stat().st_size > 16 * 1024 * 1024:
        raise ExternalInventoryError("source observation CSV exceeds size limit")
    raw = p.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ExternalInventoryError("source observation CSV exceeds size limit")
    actual_sha = sha256(raw).hexdigest()
    if actual_sha != expected_sha256:
        raise ExternalInventoryError("source inventory digest mismatch")

    by_status: dict[str, int] = {}
    by_card: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    count = 0

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalInventoryError("source observation CSV is not UTF-8") from error
    with io.StringIO(text, newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or ()):
            raise ExternalInventoryError("required column missing")
        for row in reader:
            count += 1
            if count > max_rows:
                raise ExternalInventoryError("inventory too large")
            if None in row or any(value is None for value in row.values()):
                raise ExternalInventoryError("malformed CSV row")
            card = (row["card_uuid"] or "").strip()
            name = row["source_relative_path"] or ""
            if not card or not name:
                raise ExternalInventoryError("missing source identity")
            key = (card, name)
            if key in seen:
                raise ExternalInventoryError("duplicate source identity")
            seen.add(key)
            try:
                size = int(row["expected_bytes"])
                observed = int(row["observed_bytes"]) if row["observed_bytes"] else None
            except (TypeError, ValueError) as error:
                raise ExternalInventoryError("invalid size") from error
            if size < 0 or (observed is not None and observed < 0):
                raise ExternalInventoryError("negative size")
            status = row["metadata_status"]
            if status not in METADATA_STATUSES:
                raise ExternalInventoryError("unsupported metadata status")
            if status == "NAME_AND_SIZE_ONLY" and observed != size:
                raise ExternalInventoryError("name/size observation contradicts size")
            if status == "SIZE_MISMATCH" and (observed is None or observed == size):
                raise ExternalInventoryError("size-mismatch observation contradicts size")
            if row["source_release_authorized"] != "False" or row["nas_payload_sha256_verified_today"] != "False":
                raise ExternalInventoryError("this importer cannot accept release or hash proof assertions")
            by_status[status] = by_status.get(status, 0) + 1
            label = f"{card}|{status}"
            by_card[label] = by_card.get(label, 0) + 1

    if count != expected_rows:
        raise ExternalInventoryError("source inventory row count mismatch")

    return Observation(
        rows=count,
        source_sha256=actual_sha,
        by_status=by_status,
        by_card_and_status=by_card,
    )
