"""Bounded, read-only date-folder archive candidate discovery.

This does filename+size metadata comparisons ONLY; never NAS payload hashing,
copy, deletion, source release or automatic custody evidence import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path

from .mounts import read_mounts

DATE_KEY = re.compile(r"^(\d{4})_(\d{2})(\d{2})_")
MAX_PROBES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_root(root: Path, expected_device: str) -> int:
    """Bind a search root to the actual CIFS share, not the autofs shim."""
    if (not root.is_absolute() or not root.is_relative_to(Path("/nas")) or
            root == Path("/nas") or not expected_device):
        raise ValueError("date archive root must be a declared /nas subtree")
    parts = root.parts
    for offset in range(3, len(parts) + 1):
        if Path(*parts[:offset]).is_symlink():
            raise ValueError("archive root contains a symlink")
    mounts = [m for m in read_mounts() if m.mount_point == "/nas" and
              m.filesystem_type == "cifs" and m.device == expected_device]
    if len(mounts) != 1:
        raise ValueError("pinned underlying NAS CIFS identity not observed")
    if not root.is_dir():
        raise ValueError("archive root missing")
    return root.stat().st_dev


def inspect(index: Path, expected_sha: str, root: Path, *,
            expected_device: str, max_probes: int = 5_000) -> dict:
    if not (1 <= max_probes <= MAX_PROBES):
        raise ValueError("unbounded metadata probe count")
    if len(expected_sha) != 64 or sha256(index) != expected_sha.lower():
        raise ValueError("source candidate index SHA256 mismatch")
    device = _validate_root(root, expected_device)
    result = {
        "schema": 1, "scope": "exact_date_folder_name_and_size_metadata_only",
        "source_index_sha256": expected_sha.lower(),
        "archive_root": str(root), "underlying_nas_device": expected_device,
        "probes": 0, "matches": 0, "by_card": {}, "counts": {},
        "all_destination_sha256_verified": False,
        "source_release_allowed": False, "rows": [],
    }
    counts: Counter[str] = Counter()
    by_card: Counter[str] = Counter()
    claimed: dict[str, tuple[str, str]] = {}
    owner_seen: set[tuple[str, str]] = set()
    with index.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed index at line {number}") from exc
            if row.get("source_release_allowed") is not False or (
                    row.get("current_nas_sha_verified") is not False):
                raise ValueError("untrusted source release or NAS verification flag")
            key = row.get("source_relative_path")
            uuid = row.get("card_uuid")
            if not isinstance(uuid, str) or not isinstance(key, str):
                raise ValueError("missing source UUID/key")
            owner = (uuid, key)
            if owner in owner_seen:
                raise ValueError("duplicate source UUID/key in immutable index")
            owner_seen.add(owner)
            if (row.get("classification") !=
                    "no_indexed_candidate_not_proven_missing" or
                    not key.lower().endswith(".mp4")):
                continue
            name = key.rsplit("/", 1)[-1]
            matched = DATE_KEY.match(name)
            if matched is None:
                counts["undated_or_appledouble"] += 1
                continue
            size = row.get("bytes")
            if type(size) is not int or size < 0:
                raise ValueError("invalid source length")
            if result["probes"] >= max_probes:
                counts["budget_deferred"] += 1
                continue
            year, month, day = matched.groups()
            relative = f"{year}/{month}/{day}/{name}"
            target = root / relative
            result["probes"] += 1
            try:
                info = target.lstat()
            except (FileNotFoundError, NotADirectoryError):
                counts["exact_path_absent"] += 1
                continue
            except OSError as exc:
                raise ValueError(f"NAS stat failed for indexed candidate: {target}") from exc
            if not stat.S_ISREG(info.st_mode) or info.st_dev != device:
                counts["nonregular_or_unexpected_device"] += 1
                continue
            if info.st_size != size:
                counts["different_size"] += 1
                continue
            destination = str(target)
            prior = claimed.setdefault(destination, owner)
            if prior != owner:
                raise ValueError("one NAS path claimed by multiple source objects")
            result["rows"].append({
                "card_uuid": uuid, "source_relative_path": key,
                "expected_bytes": size, "candidate_path": destination,
                "observed_bytes": info.st_size,
                "candidate_backing_st_dev": info.st_dev,
                "source_mp4_staging": row.get("mp4_staging"),
                "evidence": "current_path_and_size_only",
                "current_nas_sha256_verified": False,
                "source_release_allowed": False,
            })
            counts["matching_path_and_size"] += 1
            by_card[uuid] += 1
    if sha256(index) != expected_sha.lower():
        raise ValueError("source candidate index changed during scan")
    result["counts"] = dict(sorted(counts.items()))
    result["by_card"] = dict(sorted(by_card.items()))
    result["matches"] = len(result["rows"])
    return result


def write_report(path: Path, result: dict) -> str:
    target = path.absolute()
    if target.is_relative_to(Path("/nas")) or target.is_symlink():
        raise ValueError("index receipt may not be stored on NAS or symlink")
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("index receipt must live on OS filesystem")
    with target.open("x", encoding="utf-8") as out:
        for row in result["rows"]:
            out.write(json.dumps(row, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    return sha256(target)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--expected-index-sha256", required=True)
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--expected-nas-device", required=True)
    p.add_argument("--max-probes", type=int, default=5_000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = inspect(args.index, args.expected_index_sha256,
                         args.archive_root,
                         expected_device=args.expected_nas_device,
                         max_probes=args.max_probes)
        result["output_sha256"] = write_report(args.out, result)
        result["output_file"] = str(args.out)
        result.pop("rows")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
