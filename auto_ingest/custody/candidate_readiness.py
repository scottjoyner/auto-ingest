"""Bounded metadata-only NAS candidate readiness report for offline SD cards.

No source file opened, no media payload bytes read, no source mount changes,
and no generated evidence can authorize SD-card source release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .mounts import observe_mount, uuid_for_device
from .store import load_campaign
from .verify_candidates import _prior_comparisons

DEFAULT_NAS = Path("/nas")


def _mount_state(root: str, uuid: str) -> dict:
    observed = observe_mount(root)
    device_uuid = uuid_for_device(observed.device) if observed.present else None
    ready = bool(observed.present and observed.is_mount_point and
                 observed.filesystem_type in ("vfat", "exfat") and
                 device_uuid == uuid)
    return {
        "source_mount": root, "source_present": observed.present,
        "source_uuid_matched": ready, "observed_uuid": device_uuid,
        "read_only": observed.read_only, "ready_for_source_reads": ready,
    }


def _nas_metadata(destination: str, expected: int, nas_root: Path) -> str:
    p = Path(destination)
    # Do not follow a link out of the scoped, authoritative share.
    if (not p.is_absolute() or not p.is_relative_to(nas_root) or
            ".." in p.parts or "\\" in destination):
        return "unsafe_destination_path"
    # CIFS path.realpath/resolve can block for seconds PER path component.
    # This metadata-only probe does not open/read a file. An lstat of the
    # named candidate suffices to distinguish source-unmounted vs NAS presence.
    # Content verification keeps its stronger full path and O_NOFOLLOW checks.
    try:
        info = p.lstat()
    except FileNotFoundError:
        return "nas_candidate_path_absent"
    except OSError:
        return "nas_candidate_unavailable"
    if stat.S_ISLNK(info.st_mode):
        return "unsafe_destination_path"
    if not stat.S_ISREG(info.st_mode):
        return "nas_candidate_not_regular"
    if info.st_size != expected:
        return "nas_candidate_size_mismatch"
    return "nas_metadata_size_match_only"


def inspect(
    index_path: Path,
    campaign_root: Path,
    prior_reports: list[Path],
    *,
    max_candidates: int = 100,
    nas_root: Path = DEFAULT_NAS,
) -> dict:
    """Review only bounded, as-yet-unhashed video candidates; no media reads."""
    if max_candidates < 1 or max_candidates > 1000:
        raise ValueError("max-candidates must be 1..1000")
    content = index_path.read_bytes()
    index_digest = hashlib.sha256(content).hexdigest()
    index_rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if len(index_rows) > 100_000:
        raise ValueError("candidate index unexpectedly large")
    if len({str(p) for p in prior_reports}) != len(prior_reports):
        raise ValueError("duplicate prior reports")
    previously = _prior_comparisons(
        index_rows, index_path, prior_reports, index_sha256=index_digest)
    remaining = [
        row for row in index_rows
        if row.get("classification") == "metadata_candidate_needs_sha"
        and row.get("video_mp4_filename_not_content_verified") is True
        and (row["card_uuid"], row["source_relative_path"]) not in previously
    ]
    remaining.sort(key=lambda row: (
        row["bytes"], row["card_uuid"], row["source_relative_path"]))
    total_remaining = len(remaining)
    remaining = remaining[:max_candidates]

    # Campaign is the authoritative mapping of UUID -> source mount.
    card_states = {}
    for uuid in sorted({row["card_uuid"] for row in remaining}):
        campaign = load_campaign(campaign_root / uuid)
        if campaign.source.card.filesystem_uuid != uuid:
            raise ValueError(f"campaign identity mismatch: {uuid}")
        if not campaign.source.mount_point:
            raise ValueError(f"unresolved campaign mount: {uuid}")
        card_states[uuid] = _mount_state(campaign.source.mount_point, uuid)
    rows = []
    totals = Counter()
    for row in remaining:
        paths = row.get("candidate_paths")
        if type(row.get("bytes")) is not int or row["bytes"] < 1 or (
            not isinstance(paths, list) or len(paths) != 1):
            raise ValueError("candidate expected-size or path ambiguity")
        destination = paths[0]
        if not isinstance(destination, str):
            raise ValueError("invalid candidate destination")
        state = _nas_metadata(destination, row["bytes"], nas_root)
        uuid = row["card_uuid"]
        source_ready = card_states[uuid]["ready_for_source_reads"]
        totals[state] += 1
        if state == "nas_metadata_size_match_only":
            totals["nas_candidate_bytes"] += row["bytes"]
        if not source_ready:
            totals["blocked_unmounted_or_uuid_mismatch"] += 1
        rows.append({
            "card_uuid": uuid, "source_relative_path": row["source_relative_path"],
            "bytes": row["bytes"], "nas_candidate": destination,
            "nas_metadata_state": state, "source_ready_for_reads": source_ready,
            "content_sha_checked_this_run": False,
            "source_release_authorized": False,
        })
    return {
        "schema": 1, "scope": "metadata_only_no_source_or_nas_payload_reads",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "index_file": str(index_path), "index_sha256": index_digest,
        "source_card_states": card_states,
        "previously_sha_matched_not_reverified": len(previously),
        "candidate_count": len(rows),
        "total_remaining_candidates": total_remaining,
        "candidates_not_inspected": total_remaining - len(rows),
        "complete_metadata_scan": total_remaining == len(rows),
        "counts": dict(sorted(totals.items())),
        "rows": rows,
        "source_release_authorized": False,
        "media_bytes_read": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--out", type=Path, help="create new OS-local report, never overwrite")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.index, args.campaign_root, args.prior_report,
                         max_candidates=args.max_candidates)
        if args.out:
            out = args.out.resolve(strict=False)
            if (out.is_relative_to(DEFAULT_NAS) or
                    os.stat(out.parent).st_dev != os.stat("/").st_dev):
                raise ValueError("report must be written to independent OS-local disk")
            with out.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            report["output_path"] = str(out)
            report["rows"] = report["rows"][:3]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
