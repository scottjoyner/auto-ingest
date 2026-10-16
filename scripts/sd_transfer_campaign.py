"""Supervised, resumable full SD-card NAS copy. Never removes source files.

Runs bounded batches sequentially with UUID checks before each copy and complete
SHA-256 readbacks. Stop on any discrepancy; completion is not release authority.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.sd_verified_transfer import (
    _read_only_mount_check,
    archive_root,
    copy_one,
    make_plan,
)

CARD_MOUNTS = {
    "69BE-B963": "/media/scott/69BE-B963",
    "68FC-E499": "/media/scott/68FC-E499",
    "5140-6932": "/media/scott/UNTITLED",
}
LEGACY_PILOT = {"68FC-E499": "sdcopy-68FC-pilot-20261016.jsonl"}


def audit_files(root: Path, uuid: str) -> list[Path]:
    paths = sorted(root.glob(f"sdcopy-{uuid}-*.jsonl"))
    if uuid in LEGACY_PILOT:
        old = root / LEGACY_PILOT[uuid]
        if old.is_file():
            paths.append(old)
    return paths


def copy_campaign(
    manifest: Path, pin: str, root: Path, nas: Path, share: str,
    *, max_batches_per_card: int = 100, max_files: int = 100,
    max_bytes: int = 8_000_000_000,
    report_path: Path | None = None,
) -> dict:
    if max_batches_per_card < 1 or max_batches_per_card > 200:
        raise ValueError("invalid per-card batch cap")
    result = {"scope": "source_preservation_copy_with_sha_no_release",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": pin, "cards": {}, "source_release_allowed": False}
    for uuid, name in CARD_MOUNTS.items():
        source = Path(name)
        card_total = 0
        card_bytes = 0
        # A restart must create a NEW audit, never collide with or replace old receipts.
        prior_numbers = [int(p.stem.split("-auto-")[-1].split("-")[0])
                         for p in audit_files(root, uuid)
                         if "-auto-" in p.stem and
                         p.stem.split("-auto-")[-1].split("-")[0].isdigit()]
        first_batch_id = max(prior_numbers, default=-1) + 1
        for pass_number in range(max_batches_per_card):
            pass_idx = first_batch_id + pass_number
            prior = audit_files(root, uuid)
            chosen, metadata = make_plan(
                manifest, pin, uuid, source, nas, share, max_files, max_bytes,
                False, prior, 0)
            if not chosen:
                result["cards"][uuid] = {
                    "complete_against_frozen_manifest": True,
                    "total_files": metadata["full_census_files"],
                    "total_bytes": metadata["full_census_bytes"],
                    "verified_prior_files": metadata["prior_verified_skipped_not_rechecked"],
                    "new_batches": pass_number,
                    "newly_copied_files": card_total,
                    "newly_verified_bytes": card_bytes,
                }
                break
            stat = os.statvfs(nas)
            available = stat.f_bavail * stat.f_frsize
            if available < max_bytes + 20_000_000_000:
                raise ValueError("NAS low-space safety reserve breached")
            audit = root / f"sdcopy-{uuid}-auto-{pass_idx:03d}-20261016.jsonl"
            if audit.exists():
                raise ValueError(f"unexpected audit collision: {audit}")
            dest = archive_root(nas, uuid)
            dest.mkdir(parents=True, exist_ok=True)
            if not dest.resolve(strict=True).is_relative_to(nas.resolve(strict=True)):
                raise ValueError("NAS destination escaped expected share")
            print(json.dumps({
                "event": "batch_start", "uuid": uuid, "batch": pass_idx,
                "selected": len(chosen), "selected_bytes": metadata["selected_bytes"],
                "prior_verified": metadata["prior_verified_skipped_not_rechecked"],
            }), flush=True)
            with audit.open("x", encoding="utf-8") as log:
                for key, bytes_ in chosen:
                    _read_only_mount_check(uuid, source, nas, share)
                    rec = copy_one(source, dest, key, bytes_)
                    log.write(json.dumps({"card_uuid": uuid, **rec}, sort_keys=True) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    card_total += 1
                    card_bytes += bytes_
            print(json.dumps({
                "event": "batch_verified", "uuid": uuid, "batch": pass_idx,
                "files": len(chosen), "bytes": metadata["selected_bytes"],
                "audit": str(audit), "source_release_allowed": False,
            }), flush=True)
        else:
            result["cards"][uuid] = {
                "complete_against_frozen_manifest": False,
                "reason": "explicit max_batches_per_card limit reached",
                "newly_copied_files": card_total,
                "newly_verified_bytes": card_bytes,
            }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["all_cards_copy_complete"] = (
        len(result["cards"]) == 3 and
        all(x["complete_against_frozen_manifest"] for x in result["cards"].values()))
    if report_path:
        with report_path.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--nas", type=Path, default=Path("/nas"))
    p.add_argument("--nas-share", required=True)
    p.add_argument("--max-files", type=int, default=100)
    p.add_argument("--max-bytes", type=int, default=8_000_000_000)
    p.add_argument("--max-batches-per-card", type=int, default=100)
    p.add_argument("--report-out", type=Path, required=True)
    a = p.parse_args()
    lock_path = a.audit_root / ".sd-nas-transfer-single-writer.lock"
    if a.report_out.exists() or not a.audit_root.is_dir():
        raise ValueError("audit directory unavailable or final report already exists")
    if os.stat(a.audit_root).st_dev != os.stat("/").st_dev:
        raise ValueError("audit directory must be on independent OS-local disk")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = copy_campaign(
            a.manifest, a.expected_sha256, a.audit_root, a.nas, a.nas_share,
            max_batches_per_card=a.max_batches_per_card,
            max_files=a.max_files, max_bytes=a.max_bytes, report_path=a.report_out)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["all_cards_copy_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
