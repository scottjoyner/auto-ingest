"""Read-only SD-card media staging census for the current fleet.

Never writes source, destination, campaign bundles or the NAS. Path/size
matches are explicitly NOT SHA-256 custody or authorization to release media.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .staging import destination_for


def inventory(cards: Iterable[str | Path], *, destination_root: str | Path | None = None) -> dict:
    """Inspect exact card roots. Optional destination lookup is metadata-only."""
    dest = Path(destination_root) if destination_root is not None else None
    reports = []
    claims: dict[str, list[str]] = defaultdict(list)
    totals = Counter()
    for root_arg in cards:
        root = Path(root_arg)
        if not root.is_dir():
            raise ValueError(f"card path is not an accessible directory: {root}")
        if root.is_symlink():
            raise ValueError(f"card root must not be a symlink: {root}")
        counts = Counter()
        blocked = Counter()
        bytes_total = 0
        candidate_bytes = 0
        errors: list[str] = []
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if not Path(directory, d).is_symlink()]
            for name in names:
                if not name.lower().endswith(".mp4"):
                    continue
                path = Path(directory, name)
                try:
                    obj_stat = path.lstat()
                except OSError as exc:
                    counts["stat_errors"] += 1
                    if len(errors) < 5:
                        errors.append(f"{path}: {exc}")
                    continue
                if not stat.S_ISREG(obj_stat.st_mode):
                    counts["nonregular_mp4_names"] += 1
                    continue
                relative = path.relative_to(root).as_posix()
                counts["mp4_named_files"] += 1
                bytes_total += obj_stat.st_size
                obj = destination_for(relative, include_sidecars=False)
                if not obj.stageable:
                    blocked[obj.reason or "unknown"] += 1
                    continue
                key = obj.destination_key
                assert key is not None
                claims[key].append(f"{root}:{relative}")
                counts["stageable_mp4"] += 1
                candidate_bytes += obj_stat.st_size
                if dest is not None:
                    target = dest / key
                    try:
                        target_stat = target.lstat()
                    except FileNotFoundError:
                        counts["destination_absent"] += 1
                    except OSError as exc:
                        counts["destination_stat_errors"] += 1
                        if len(errors) < 5:
                            errors.append(f"{target}: {exc}")
                    else:
                        if stat.S_ISREG(target_stat.st_mode) and target_stat.st_size == obj_stat.st_size:
                            counts["destination_same_size_only"] += 1
                        else:
                            counts["destination_size_or_type_mismatch"] += 1
        read_only = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
        report = {
            "root": str(root), "mounted": os.path.ismount(root),
            "observed_read_only": read_only,
            "mp4_named_files": counts["mp4_named_files"],
            "mp4_named_bytes": bytes_total,
            "stageable_mp4": counts["stageable_mp4"],
            "stageable_bytes": candidate_bytes,
            "blocked": dict(sorted(blocked.items())),
            "destination_same_size_only": counts["destination_same_size_only"],
            "destination_absent": counts["destination_absent"],
            "destination_size_or_type_mismatch": counts["destination_size_or_type_mismatch"],
            "destination_stat_errors": counts["destination_stat_errors"],
            "nonregular_mp4_names": counts["nonregular_mp4_names"],
            "source_stat_errors": counts["stat_errors"],
            "error_samples": errors,
        }
        reports.append(report)
        totals.update({"mp4_named_files": counts["mp4_named_files"],
                       "mp4_named_bytes": bytes_total,
                       "stageable_mp4": counts["stageable_mp4"],
                       "stageable_bytes": candidate_bytes})
    collisions = {key: sources for key, sources in claims.items() if len(sources) > 1}
    return {
        "schema_version": 1, "scope": "read_only_filename_size_and_staging",
        "hashes_performed": 0, "copies_performed": 0, "deletions_performed": 0,
        "source_release_allowed": False,
        "destination_root": str(dest) if dest is not None else None,
        "destination_proof_level": "metadata_only_not_verified" if dest else "not_checked",
        "cards": reports, "totals": dict(totals),
        "unique_staged_destinations": len(claims),
        "collision_destinations": len(collisions),
        "collision_samples": [{"destination": k, "sources": v} for k, v in list(sorted(collisions.items()))[:10]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", action="append", required=True,
                        help="mounted card root, repeatable; never modified")
    parser.add_argument("--destination-root", default=None,
                        help="optional per-file NAS path/size metadata probe, NOT integrity verification")
    args = parser.parse_args(argv)
    try:
        result = inventory(args.card, destination_root=args.destination_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
