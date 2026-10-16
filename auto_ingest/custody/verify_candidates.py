"""Bounded read-only source/NAS digest comparison, never a release receipt.

Candidate inputs come from the UUID-bound NAS evidence index. Read I/O only.
Even equal SHA digests cannot open the source-release gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from .mounts import observe_mount, uuid_for_device
from .store import load_campaign


def choose_candidates(rows: list[dict], *, max_files: int, max_read_bytes: int,
                      skip_keys: set[tuple[str, str]] | None = None) -> list[dict]:
    if max_files < 1 or max_read_bytes < 1:
        raise ValueError("positive file count and byte budget required")
    eligible = [r for r in rows
                if r.get("video_mp4_filename_not_content_verified") is True
                and r.get("classification") == "metadata_candidate_needs_sha"
                and len(r.get("candidate_paths", [])) == 1
                and (r["card_uuid"], r["source_relative_path"]) not in (skip_keys or set())]
    eligible.sort(key=lambda r: (r["bytes"], r["card_uuid"], r["source_relative_path"]))
    selected = []
    used = 0
    for row in eligible:
        size = row["bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("invalid candidate size")
        if len(selected) == max_files:
            break
        if used + size * 2 > max_read_bytes:
            continue
        used += size * 2
        selected.append(row)
    return selected


def _snapshot(fd: int) -> tuple[int, int, int, int, int, int]:
    s = os.fstat(fd)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode


def sha_stable(path: Path, expected: int) -> dict:
    """O_NOFOLLOW final-file read; verify file size/type and metadata before/after."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        pre = _snapshot(fd)
        if not stat.S_ISREG(pre[-1]) or pre[2] != expected:
            raise ValueError(f"unexpected type or size: {path}")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        post = _snapshot(fd)
        external = path.lstat()
        steady = pre == post and (
            post[0], post[1], post[2], post[3], post[4]
        ) == (external.st_dev, external.st_ino, external.st_size,
              external.st_mtime_ns, external.st_ctime_ns)
        return {"sha256": digest.hexdigest(), "stable": steady, "bytes": post[2]}
    finally:
        os.close(fd)


def check_pair(row: dict, source_root: Path, dest_root: Path = Path("/nas")) -> dict:
    key = row["source_relative_path"]
    if (not isinstance(key, str) or key.startswith("/") or
            "\\" in key or ".." in Path(key).parts):
        raise ValueError("source key is not a safe relative path")
    dests = row["candidate_paths"]
    if len(dests) != 1:
        raise ValueError("exactly one destination candidate required")
    destination = Path(dests[0])
    if (not destination.is_absolute() or
            not destination.is_relative_to(dest_root) or
            not destination.resolve(strict=True).is_relative_to(dest_root.resolve())):
        raise ValueError("destination outside authorized NAS root")
    source = source_root / key
    if source.is_symlink() or not source.resolve(strict=True).is_relative_to(source_root.resolve()):
        raise ValueError("source path escapes card root")
    expected = row["bytes"]
    first = sha_stable(source, expected)
    second = sha_stable(destination, expected)
    return {
        "card_uuid": row["card_uuid"], "source_relative_path": key,
        "destination": str(destination), "bytes": expected,
        "source_sha256_now": first["sha256"],
        "destination_sha256_now": second["sha256"],
        "sha_equal": first["sha256"] == second["sha256"],
        "source_stable": first["stable"], "destination_stable": second["stable"],
        "source_release_authorized": False,
    }


def verify_index(index_file: Path, campaign_root: Path,
                 max_files: int = 8, max_read_bytes: int = 350_000_000,
                 prior_report: Path | None = None) -> dict:
    rows = [json.loads(line) for line in index_file.open(encoding="utf-8")]
    skip_keys: set[tuple[str, str]] = set()
    if prior_report is not None:
        previous = json.loads(prior_report.read_text(encoding="utf-8"))
        if (previous.get("scope") !=
                "bounded_point_in_time_source_and_nas_sha_not_custody_release" or
                previous.get("source_index") != str(index_file)):
            raise ValueError("prior report does not match candidate index or method")
        for item in previous.get("results", []):
            if (item.get("sha_equal") is True and
                    item.get("source_stable") is True and
                    item.get("destination_stable") is True and
                    item.get("source_release_authorized") is False):
                skip_keys.add((item["card_uuid"], item["source_relative_path"]))
    chosen = choose_candidates(rows, max_files=max_files,
                               max_read_bytes=max_read_bytes, skip_keys=skip_keys)
    output = []
    for row in chosen:
        uuid = row["card_uuid"]
        campaign = load_campaign(campaign_root / uuid)
        if campaign.source.card.filesystem_uuid != uuid:
            raise ValueError("campaign UUID mismatch")
        root = campaign.source.mount_point
        if not root:
            raise ValueError("campaign source unresolved")
        mounted = observe_mount(root)
        current_uuid = uuid_for_device(mounted.device) if mounted.present else None
        if (not mounted.present or not mounted.is_mount_point or
                current_uuid != uuid or mounted.filesystem_type not in ("vfat", "exfat")):
            raise ValueError("card UUID/mount changed during verification")
        record = check_pair(row, Path(root))
        after = observe_mount(root)
        if after != mounted:
            raise ValueError("source mount changed during verification")
        record["source_observed_read_only"] = mounted.read_only
        record["eligible_for_release"] = False
        output.append(record)
    return {
        "schema": 1,
        "scope": "bounded_point_in_time_source_and_nas_sha_not_custody_release",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": str(index_file), "max_files": max_files,
        "prior_report": str(prior_report) if prior_report is not None else None,
        "previously_compared_skipped_not_reverified": len(skip_keys),
        "max_read_bytes": max_read_bytes, "source_release_authorized": False,
        "total_bytes_read": sum(row["bytes"] * 2 for row in output),
        "results": output,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=8)
    p.add_argument("--max-read-bytes", type=int, default=350_000_000)
    p.add_argument("--prior-report", type=Path,
                   help="skip matching stable pairs from one prior point-in-time report")
    p.add_argument("--out", type=Path, help="create new OS-local evidence file")
    args = p.parse_args(argv)
    try:
        report = verify_index(args.index, args.campaign_root,
                              max_files=args.max_files,
                              max_read_bytes=args.max_read_bytes,
                              prior_report=args.prior_report)
        if args.out:
            dest = args.out.resolve(strict=False)
            if dest.is_relative_to(Path("/nas")):
                raise ValueError("evidence cannot be written to NAS")
            if os.stat(dest.parent).st_dev != os.stat("/").st_dev:
                raise ValueError("evidence must be on OS local disk")
            with dest.open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            report["evidence_path"] = str(dest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
