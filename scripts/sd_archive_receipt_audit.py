"""Read-only reconciliation of a full frozen SD inventory with NAS copy receipts.

Reports completeness, never authorizes unlink or claims snapshot retention.
Safe to run while supervised copying produces additional fsynced receipts.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

VALID_SHA = re.compile(r"[0-9a-f]{64}\Z")
VALID_UUID = re.compile(r"[0-9A-F]{4}-[0-9A-F]{4}\Z")
EXPECTED_CARDS = ("69BE-B963", "68FC-E499", "5140-6932")


def _key_valid(s: object) -> bool:
    if not isinstance(s, str) or not s or "\\" in s or "\x00" in s:
        return False
    p = PurePosixPath(s)
    return not p.is_absolute() and not any(t in ("..", ".") for t in s.split("/"))


def build_report(
    frozen: Path, expected_sha256: str, audit_root: Path,
    *, strict_receipts: bool = True, expected_files: int = 4268,
) -> dict:
    if not VALID_SHA.fullmatch(expected_sha256):
        raise ValueError("invalid source-ledger checksum pin")
    h = hashlib.sha256()
    inventory = {}
    source_totals = collections.Counter()
    source_sizes = collections.Counter()
    with frozen.open("rb") as handle:
        for lineno, raw in enumerate(handle, 1):
            h.update(raw)
            if not raw.endswith(b"\n"):
                raise ValueError(f"unterminated frozen manifest row {lineno}")
            obj = json.loads(raw)
            uuid, key, size = (
                obj.get("card_uuid"), obj.get("source_relative_path"),
                obj.get("logical_bytes"))
            if (uuid not in EXPECTED_CARDS or not _key_valid(key) or
                    type(size) is not int or size < 0):
                raise ValueError(f"invalid frozen row {lineno}")
            identity = (uuid, key)
            if identity in inventory:
                raise ValueError(f"duplicate frozen source key at row {lineno}")
            inventory[identity] = size
            source_totals[uuid] += 1
            source_sizes[uuid] += size
    if h.hexdigest() != expected_sha256:
        raise ValueError("frozen source ledger checksum differs from pinned value")
    if len(inventory) != expected_files or expected_files < 1:
        raise ValueError("unexpected source ledger inventory cardinality")

    verified = {}
    duplicate_receipts = []
    bad_receipts = []
    checked_receipts = 0
    source_audits = {}
    files = sorted(audit_root.glob("sdcopy-*.jsonl"))
    if not files:
        raise ValueError("no copy receipts found")
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for lineno, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.endswith(b"\n"):
                    raise ValueError(f"unterminated receipt in {file.name}:{lineno}")
                row = json.loads(raw)
                uuid, key = row.get("card_uuid"), row.get("key")
                checked_receipts += 1
                identity = (uuid, key)
                if uuid not in EXPECTED_CARDS or not _key_valid(key):
                    bad_receipts.append((file.name, lineno, "bad_key"))
                    continue
                if identity in verified:
                    duplicate_receipts.append((file.name, lineno, uuid, key))
                    continue
                known_size = inventory.get(identity)
                expected_path = (
                    f"/nas/recovery/dashcam-sd-ingest/"
                    f"SDCARD-20261016-{uuid}/source/{key}")
                source_sha, dest_sha = row.get("source_sha256"), row.get("nas_sha256")
                ok = (
                    known_size is not None and
                    type(row.get("bytes")) is int and row["bytes"] == known_size
                    and row.get("nas_path") == expected_path
                    and isinstance(source_sha, str) and
                    VALID_SHA.fullmatch(source_sha) is not None
                    and source_sha == dest_sha
                    and row.get("sha_match") is True
                    and row.get("source_stable") is True
                    and row.get("destination_stable") is True
                    and row.get("source_release_authorized") is False
                )
                if not ok:
                    bad_receipts.append((file.name, lineno, uuid, key))
                    continue
                verified[identity] = known_size
        source_audits[file.name] = digest.hexdigest()
    if strict_receipts and (duplicate_receipts or bad_receipts):
        raise ValueError(f"receipt integrity failure: "
                         f"duplicates={len(duplicate_receipts)} bad={len(bad_receipts)}")
    by_card = {}
    for uuid in EXPECTED_CARDS:
        completed = sum(1 for key in verified if key[0] == uuid)
        bytes_verified = sum(size for (card, _), size in verified.items()
                             if card == uuid)
        by_card[uuid] = {
            "source_files": source_totals[uuid],
            "source_bytes": source_sizes[uuid],
            "verified_receipt_files": completed,
            "verified_receipt_bytes": bytes_verified,
            "missing_receipt_files": source_totals[uuid] - completed,
            "missing_receipt_bytes": source_sizes[uuid] - bytes_verified,
            "receipt_inventory_complete": completed == source_totals[uuid],
        }
    complete = (
        len(verified) == len(inventory)
        and not duplicate_receipts and not bad_receipts
    )
    return {
        "schema": 1,
        "scope": "raw_sd_to_nas_receipt_integrity_only_not_media_reread",
        "frozen_source_sha256": h.hexdigest(),
        "expected_source_files": len(inventory),
        "expected_source_bytes": sum(source_sizes.values()),
        "verified_files": len(verified),
        "verified_bytes": sum(verified.values()),
        "missing_receipt_files": len(inventory) - len(verified),
        "total_receipt_rows": checked_receipts,
        "duplicate_receipts": duplicate_receipts[:10],
        "bad_receipts": bad_receipts[:10],
        "receipt_file_sha256": source_audits,
        "by_card": by_card,
        "copy_receipts_complete": complete,
        "destination_payload_rechecked_during_this_audit": False,
        "beelink_snapshot_retention_attested": False,
        "source_card_quiescence_attested": False,
        "source_release_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="create new OS-local report")
    args = parser.parse_args(argv)
    try:
        result = build_report(args.frozen, args.expected_sha256, args.audit_root)
        if args.out:
            output = args.out.resolve(strict=False)
            if os.stat(output.parent).st_dev != os.stat("/").st_dev:
                raise ValueError("audit output must be OS-local, not NAS or SD")
            with output.open("x", encoding="utf-8") as file:
                json.dump(result, file, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
        summary = {k: v for k, v in result.items()
                   if k != "receipt_file_sha256"}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if result["copy_receipts_complete"] else 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
