"""Read-only, non-authoritative observations from external NAS reconciliation.

This module never transforms path/size metadata into a custody verified,
releasable, or copied verdict. It does not write campaign bundles or ledgers.
It is deliberately separate from custody.ledger's proof-bearing rows.
"""

from __future__ import annotations

import csv
import io
import unicodedata
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
    # A case-sensitive SSD can hold both Fig1.png and fig1.png, but an SMB
    # destination may collapse them. Reject portable path aliases rather than
    # silently accepting an apparently distinct source/destination manifest.
    portable_sources: set[tuple[str, str]] = set()
    portable_destinations: set[str] = set()
    count = 0

    try:
        # NAS/Excel exports commonly prefix a UTF-8 BOM. Decode with utf-8-sig
        # so the BOM is consumed as an encoding marker instead of sticking to
        # the first field name (\\ufeffcard_uuid), which would otherwise make
        # a valid, digest-pinned manifest fail the required-columns check. The
        # pin and size checks above run on the raw bytes, so the BOM is still
        # part of the immutable snapshot being verified.
        text = raw.decode("utf-8-sig")
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
            fold_source = unicodedata.normalize(
                "NFC", name.replace("\\", "/")
            ).casefold()
            folded_key = (card, fold_source)
            if folded_key in portable_sources:
                raise ExternalInventoryError(
                    "source path casefold collision cannot be safely imported"
                )
            portable_sources.add(folded_key)

            target_path = row["candidate_relative_path"] or ""
            if not target_path:
                raise ExternalInventoryError("missing destination candidate identity")
            folded_target = unicodedata.normalize(
                "NFC", target_path.replace("\\", "/")
            ).casefold()
            if folded_target in portable_destinations:
                raise ExternalInventoryError(
                    "destination path casefold collision cannot be safely imported"
                )
            portable_destinations.add(folded_target)
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
