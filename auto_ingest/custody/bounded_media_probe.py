"""Bounded current source-versus-NAS SHA-256 probe; NEVER a release ledger.

Reads only file payloads selected from the pinned, immutable NAS candidate
index. Explicit card UUID and kernel mount checks; rw sources remain blocked
for release even when hashes match. No copying or deletion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .mounts import observe_mount, read_mounts, uuid_for_device
from .store import load_campaign

MAX_PROBE_BYTES = 512 * 1024 * 1024
MAX_PROBE_FILES = 8
BLOCK_SIZE = 1024 * 1024


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _components(key: str) -> tuple[str, ...]:
    if not isinstance(key, str) or not key or "\\" in key:
        raise ValueError("unsafe relative path")
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in ("..", ".") for part in key.split("/")):
        raise ValueError("absolute or traversing relative path")
    if not all(part not in ("", "..", ".") for part in path.parts):
        raise ValueError("unsafe path component")
    return path.parts


def _under(root: Path, path: str) -> str:
    if not isinstance(path, str) or not path.startswith(str(root).rstrip("/") + "/"):
        raise ValueError("candidate outside declared destination")
    if ".." in path.split("/") or "\\" in path:
        raise ValueError("candidate path traversal")
    relative = path[len(str(root).rstrip("/")) + 1:]
    _components(relative)
    return relative


def _identity(fd: int) -> tuple[int, int, int, int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _hash_relative(root: Path, relative: str, expected: int) -> dict:
    """openat each component with O_NOFOLLOW; enforce one storage device."""
    parts = _components(relative)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, directory_flags)
    opened = [fd]
    try:
        source_dev = os.fstat(fd).st_dev
        for component in parts[:-1]:
            fd = os.open(component, directory_flags, dir_fd=fd)
            opened.append(fd)
            if os.fstat(fd).st_dev != source_dev:
                raise ValueError("unexpected nested mount")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(parts[-1], flags, dir_fd=fd)
        opened.append(fd)
        before = _identity(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode) or before[0] != source_dev:
            raise ValueError("not a regular file on declared storage")
        if before[2] != expected:
            raise ValueError("file size differs from immutable source index")
        h = hashlib.sha256()
        total = 0
        while total < expected:
            chunk = os.read(fd, min(BLOCK_SIZE, expected - total))
            if not chunk:
                raise ValueError("short read of expected content")
            total += len(chunk)
            h.update(chunk)
        if os.read(fd, 1):
            raise ValueError("file grew during hash")
        after = _identity(fd)
        if before != after:
            raise ValueError("file identity or metadata changed during hash")
        return {"sha256": h.hexdigest(), "bytes": total,
                "stable": True, "st_dev": before[0]}
    finally:
        for handle in reversed(opened):
            os.close(handle)


def _candidate_rows(index: Path, expected_sha: str) -> list[dict]:
    if len(expected_sha) != 64 or _digest(index) != expected_sha.lower():
        raise ValueError("pinned candidate index SHA-256 mismatch")
    rows = []
    claimed = {}
    with index.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError(f"malformed source index line {number}") from exc
            if not isinstance(row, dict) or row.get("source_release_allowed") is not False:
                raise ValueError("unexpected index authority or malformed row")
            owner = (row.get("card_uuid"), row.get("source_relative_path"))
            _components(owner[1])
            if row.get("current_nas_sha_verified") is not False:
                raise ValueError("index contains untrusted destination verification")
            for path in row.get("candidate_paths", []):
                if path in claimed and claimed[path] != owner:
                    raise ValueError("duplicate NAS path assigned to distinct sources")
                claimed[path] = owner
            if (row.get("classification") == "metadata_candidate_needs_sha"
                    and owner[1].lower().endswith(".mp4")
                    and not owner[1].rsplit("/", 1)[-1].startswith("._")
                    and row.get("mp4_staging") == "stageable_filename_only"):
                rows.append(row)
    if _digest(index) != expected_sha.lower():
        raise ValueError("index changed during reading")
    return sorted(rows, key=lambda r: (r["bytes"], r["card_uuid"],
                                       r["source_relative_path"]))


def inspect(index: Path, expected_sha: str, campaigns: Path,
            destination_root: Path, *, max_files: int = 3,
            max_read_bytes: int = 100_000_000,
            max_file_bytes: int = 64_000_000,
            allow_writable_source: bool = False,
            destination_filesystem: str = "cifs",
            expected_destination_device: str | None = None,
            exclude_report: Path | list[Path] | None = None,
            expected_exclude_sha256: str | list[str] | None = None) -> dict:
    """Bounded source+NAS read. Zero imports into campaign custody ledgers."""
    if not (1 <= max_files <= MAX_PROBE_FILES and
            1 <= max_file_bytes <= MAX_PROBE_BYTES // 2 and
            2 <= max_read_bytes <= MAX_PROBE_BYTES):
        raise ValueError("probe limits exceed safe policy bounds")
    if not destination_root.is_absolute() or destination_root.is_symlink():
        raise ValueError("destination mount must be absolute and not a symlink")
    # /nas can have two simultaneous kernel entries (systemd autofs and CIFS).
    # The generic observer returns the *first* one, which is autofs on x1.
    # Qualify the actual backing mount by exact filesystem type instead.
    observations = [m for m in read_mounts()
                    if m.mount_point == str(destination_root) and
                    m.present and m.filesystem_type == destination_filesystem]
    if len(observations) != 1:
        raise ValueError("expected destination backing mount not uniquely observed")
    dest_observed = observations[0]
    if (expected_destination_device is not None and
            dest_observed.device != expected_destination_device):
        raise ValueError("destination backing device differs from pinned share")
    exclusions: set[tuple[str, str, str]] = set()
    prior_paths = [] if exclude_report is None else (
        exclude_report if isinstance(exclude_report, list) else [exclude_report])
    prior_shas = [] if expected_exclude_sha256 is None else (
        expected_exclude_sha256 if isinstance(expected_exclude_sha256, list)
        else [expected_exclude_sha256])
    if len(prior_paths) != len(prior_shas):
        raise ValueError("prior receipt requires both path and pinned digest")
    for prior_path, prior_sha in zip(prior_paths, prior_shas):
        if (not isinstance(prior_sha, str) or len(prior_sha) != 64 or
                prior_path.stat().st_size > 2_000_000 or
                _digest(prior_path) != prior_sha.lower()):
            raise ValueError("prior research receipt SHA256 mismatch or over limit")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (not isinstance(prior, dict) or
                prior.get("scope") != "bounded_live_source_and_destination_sha256_research_only" or
                prior.get("index_sha256") != expected_sha.lower() or
                prior.get("source_release_allowed") is not False):
            raise ValueError("prior receipt is not from same research-only index")
        for record in prior.get("results", []):
            if (record.get("sha256_match") is True and
                    record.get("source_stable") is True and
                    record.get("destination_stable") is True and
                    record.get("release_source_allowed") is False):
                exclusions.add((record["uuid"], record["source_relative_path"],
                                record["candidate_path"]))
        if _digest(prior_path) != prior_sha.lower():
            raise ValueError("prior receipt changed during exclusion read")
    candidates = _candidate_rows(index, expected_sha)
    selected = []
    budget = 0
    for row in candidates:
        if (len(row["candidate_paths"]) == 1 and
                (row["card_uuid"], row["source_relative_path"],
                 row["candidate_paths"][0]) in exclusions):
            continue
        size = row["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("invalid candidate byte count")
        if size > max_file_bytes or budget + 2 * size > max_read_bytes:
            continue
        if len(row["candidate_paths"]) != 1:
            raise ValueError("ambiguous size-only NAS candidate")
        selected.append(row)
        budget += 2 * size
        if len(selected) >= max_files:
            break
    if not selected:
        raise ValueError("no candidates within safe read budget")
    roots = {}
    mount_observations = {}
    for uuid in sorted({r["card_uuid"] for r in selected}):
        campaign = load_campaign(campaigns / uuid)
        if campaign.source.card.filesystem_uuid != uuid:
            raise ValueError("campaign card UUID differs from candidate")
        source_root = Path(campaign.source.mount_point or "")
        observed = observe_mount(str(source_root))
        observed_uuid = uuid_for_device(observed.device) if observed.present else None
        if (not source_root.is_absolute() or source_root.is_symlink()
                or not observed.is_mount_point or
                observed_uuid != uuid or observed.filesystem_type not in ("vfat", "exfat")):
            raise ValueError(f"source card mount identity failed: {uuid}")
        if observed.read_only is not True and not allow_writable_source:
            raise ValueError(f"source is writable; explicit research-only opt-in needed: {uuid}")
        roots[uuid] = source_root
        mount_observations[uuid] = {"device": observed.device,
                                     "filesystem_uuid": observed_uuid,
                                     "read_only": observed.read_only}
    result = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_live_source_and_destination_sha256_research_only",
        "index_sha256": expected_sha.lower(),
        "prior_receipt_sha256s": [digest.lower() for digest in prior_shas],
        "excluded_prior_matching_file_count": len(exclusions),
        "destination_root": str(destination_root),
        "destination_filesystem": dest_observed.filesystem_type,
        "observed_destination_device": dest_observed.device,
        "destination_device_pinned": expected_destination_device is not None,
        "source_mounts": mount_observations,
        "budget_bytes": max_read_bytes, "budget_files": max_files,
        "bytes_read": 0, "source_release_allowed": False,
        "custody_evidence_imported": False,
        "source_quiescence_attested": False,
        "destination_permanence_attested": False,
        "results": [],
    }
    for row in selected:
        uuid = row["card_uuid"]
        relative = row["source_relative_path"]
        nas_relative = _under(destination_root, row["candidate_paths"][0])
        source = _hash_relative(roots[uuid], relative, row["bytes"])
        dest = _hash_relative(destination_root, nas_relative, row["bytes"])
        result["bytes_read"] += source["bytes"] + dest["bytes"]
        match = source["sha256"] == dest["sha256"]
        result["results"].append({
            "uuid": uuid, "source_relative_path": relative,
            "candidate_path": row["candidate_paths"][0],
            "bytes_per_copy": row["bytes"],
            "source_sha256_now": source["sha256"],
            "nas_sha256_now": dest["sha256"],
            "sha256_match": match,
            "source_stable": source["stable"], "destination_stable": dest["stable"],
            "source_read_only_observed": mount_observations[uuid]["read_only"],
            "release_source_allowed": False,
        })
    result["matches"] = sum(row["sha256_match"] for row in result["results"])
    return result


def _write_report(path: Path, report: dict) -> None:
    target = path.absolute()
    if target.is_symlink() or target.is_relative_to(Path("/nas")):
        raise ValueError("report may not be written to NAS")
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("report must reside on the x1 OS disk")
    with target.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--expected-index-sha256", required=True)
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--destination-root", type=Path, default=Path("/nas"))
    p.add_argument("--destination-filesystem", default="cifs")
    p.add_argument("--expected-destination-device", required=True,
                   help="kernel source for the actual backing CIFS mount, not autofs")
    p.add_argument("--max-files", type=int, default=3)
    p.add_argument("--max-read-bytes", type=int, default=100_000_000)
    p.add_argument("--max-file-bytes", type=int, default=64_000_000)
    p.add_argument("--allow-writable-source-observation", action="store_true",
                   help="research-only with release hold; never an RO-source attestation")
    p.add_argument("--exclude-report", type=Path, action="append",
                   help="repeat for each prior independently pinned research receipt")
    p.add_argument("--expected-exclude-sha256", action="append",
                   help="one matching SHA256 per --exclude-report, same order")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = inspect(
            args.index, args.expected_index_sha256, args.campaign_root,
            args.destination_root, max_files=args.max_files,
            max_read_bytes=args.max_read_bytes,
            max_file_bytes=args.max_file_bytes,
            allow_writable_source=args.allow_writable_source_observation,
            destination_filesystem=args.destination_filesystem,
            expected_destination_device=args.expected_destination_device,
            exclude_report=args.exclude_report,
            expected_exclude_sha256=args.expected_exclude_sha256,
        )
        _write_report(args.out, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
