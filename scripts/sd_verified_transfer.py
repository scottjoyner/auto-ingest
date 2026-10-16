"""Conservative SD->NAS archival copy with a second, independent SHA readback.

Read-only on source, writes under a new per-card NAS namespace, never deletes.
No standalone receipt in this script authorizes source deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from auto_ingest.custody.fleet_campaigns import card_census, frozen_inventory
from auto_ingest.custody.mounts import observe_mount, uuid_for_device
from auto_ingest.custody.verify_candidates import sha_stable

UUID = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}$")


def _read_only_mount_check(uuid: str, source: Path, nas: Path, expected_share: str) -> None:
    card = observe_mount(str(source))
    card_uuid = uuid_for_device(card.device) if card.present else None
    if (not card.present or not card.is_mount_point or
            card.filesystem_type not in ("vfat", "exfat") or
            card_uuid != uuid):
        raise ValueError("SD card mount device/UUID mismatch")
    # /nas is a systemd autofs direct mount *and* CIFS. observe_mount()
    # returns the first layer (autofs); check the real mounted CIFS too.
    if not nas.is_mount() or not nas.is_dir():
        raise ValueError("NAS root is not mounted")
    backing = [line.split() for line in Path("/proc/self/mounts").read_text().splitlines()]
    if not any(len(parts) >= 3 and parts[1] == str(nas) and
               parts[0] == expected_share and parts[2] == "cifs"
               for parts in backing):
        raise ValueError("NAS CIFS mount is not the expected authoritative share")


def make_plan(
    manifest: Path, expected_sha: str, uuid: str, source: Path, nas: Path,
    expected_share: str, max_files: int, max_bytes: int,
    with_mp4: bool, prior_audits: list[Path] | None = None,
    min_videos: int = 0,
) -> tuple[list[tuple[str, int]], dict]:
    if (not UUID.fullmatch(uuid) or max_files < 1 or max_files > 100 or
            max_bytes < 1 or min_videos < 0 or min_videos > 20 or
            min_videos > max_files):
        raise ValueError("invalid UUID or bounded transfer limits")
    _read_only_mount_check(uuid, source, nas, expected_share)
    sha, groups = frozen_inventory(manifest)
    if sha != expected_sha or uuid not in groups:
        raise ValueError("frozen ledger checksum or card identity mismatch")
    census = card_census(uuid, source, groups[uuid])
    if (not census["metadata_complete"] or
            any(census[k] for k in ("new_files", "changed_size", "unsafe_entries", "missing"))):
        raise ValueError("source census changed; transfer refused")
    previous = set()
    dest_base = archive_root(nas, uuid)
    for prior in prior_audits or []:
        for line in prior.open(encoding="utf-8"):
            old = json.loads(line)
            key = old.get("key")
            if (old.get("card_uuid") != uuid or key not in groups[uuid] or
                    old.get("bytes") != groups[uuid][key][0] or
                    old.get("nas_path") != str(dest_base / key) or
                    old.get("sha_match") is not True or
                    old.get("source_stable") is not True or
                    old.get("destination_stable") is not True or
                    old.get("source_release_authorized") is not False or
                    old.get("source_sha256") != old.get("nas_sha256")):
                raise ValueError("prior transfer receipt is inconsistent")
            final = dest_base / key
            if (not final.is_file() or final.is_symlink() or
                    final.stat().st_size != old["bytes"]):
                raise ValueError("previously copied NAS file no longer matches metadata")
            previous.add(key)
    available = sorted(((key, row) for key, row in groups[uuid].items()
                        if key not in previous), key=lambda kv: (kv[1][0], kv[0]))
    selected = []
    used = 0
    video_goal = max(min_videos, int(with_mp4))
    for name, data in available:
        if len(selected) >= video_goal:
            break
        size = data[0]
        if (name.lower().endswith(".mp4") and not Path(name).name.startswith("._")
                and size > 0 and used + size <= max_bytes):
            selected.append((name, size))
            used += size
    if len(selected) != video_goal:
        raise ValueError("not enough distinct MP4 files within source read budget")
    for key, (size, _) in available:
        if len(selected) >= max_files:
            break
        if key in {x[0] for x in selected} or used + size > max_bytes:
            continue
        selected.append((key, size))
        used += size
    return selected, {"uuid": uuid, "frozen_sha256": sha,
                      "observed_source": str(source),
                      "nas_share": expected_share, "nas_root": str(nas),
                      "full_census_files": census["matched_files"],
                      "full_census_bytes": census["matched_bytes"],
                      "selected_files": len(selected), "selected_bytes": used,
                      "prior_verified_skipped_not_rechecked": len(previous),
                      "source_release_authorized": False}


def archive_root(nas: Path, uuid: str) -> Path:
    return nas / "recovery" / "dashcam-sd-ingest" / f"SDCARD-20261016-{uuid}" / "source"


def copy_one(source: Path, dest_root: Path, key: str, size: int) -> dict:
    if (not key or key.startswith("/") or "\\" in key or
            ".." in Path(key).parts or Path(key).is_absolute()):
        raise ValueError("unsafe source relative path")
    input_path = source / key
    if (input_path.is_symlink() or not input_path.resolve(strict=True)
            .is_relative_to(source.resolve())):
        raise ValueError("source symlink/escape refused")
    final = dest_root / key
    # Reject a symlink in any destination parent as well as the final path;
    # otherwise rsync could write outside the card's unique namespace.
    if final.is_symlink():
        raise ValueError("NAS target symlink refused")
    if not final.resolve(strict=False).is_relative_to(dest_root.resolve(strict=True)):
        raise ValueError("NAS destination parent escapes selected card archive")
    had_existing = final.exists()
    if not had_existing:
        # rsync writes to a sibling temporary file and fsyncs it; if killed it
        # cannot silently turn a truncated final file into a 'verified' object.
        cmd = [
            "rsync", "--recursive", "--from0", "--files-from=-",
            "--ignore-existing", "--fsync", "--bwlimit=50000",
            "--partial-dir=.custody-partial",
            "--no-perms", "--no-owner", "--no-group",
            "--no-devices", "--no-specials",
            str(source) + "/", str(dest_root) + "/",
        ]
        run = subprocess.run(
            cmd, input=os.fsencode(key) + b"\x00", capture_output=True,
            timeout=1200, check=False,
        )
        if run.returncode != 0:
            raise ValueError(f"rsync failed for {key}: rc={run.returncode}; " +
                             run.stderr.decode(errors="replace")[-300:])
    if not final.is_file() or final.is_symlink():
        raise ValueError("destination not a regular file")
    original = sha_stable(input_path, size)
    dest = sha_stable(final, size)
    result = {
        "key": key, "bytes": size, "source_sha256": original["sha256"],
        "nas_sha256": dest["sha256"], "sha_match": original["sha256"] == dest["sha256"],
        "source_stable": original["stable"], "destination_stable": dest["stable"],
        "nas_path": str(final), "preexisting_destination": had_existing,
        "source_release_authorized": False,
    }
    if not result["sha_match"] or not result["source_stable"] or not result["destination_stable"]:
        raise ValueError(f"verified copy mismatch/stability failure: {key}")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--uuid", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--nas", type=Path, default=Path("/nas"))
    p.add_argument("--nas-share", required=True)
    p.add_argument("--max-files", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=300_000_000)
    p.add_argument("--include-one-mp4", action="store_true")
    p.add_argument("--min-videos", type=int, default=0,
                   help="select at least this many distinct MP4 source keys")
    p.add_argument("--prior-audit", type=Path, action="append", default=[],
                   help="repeat for exact, matching prior transfer receipt JSONL")
    p.add_argument("--apply", action="store_true", help="copy and re-read selected files")
    p.add_argument("--audit-out", type=Path, required=True)
    args = p.parse_args(argv)
    selected, meta = make_plan(
        args.manifest, args.expected_sha256, args.uuid, args.source, args.nas,
        args.nas_share, args.max_files, args.max_bytes, args.include_one_mp4,
        args.prior_audit, args.min_videos)
    dest = archive_root(args.nas, args.uuid)
    meta["destination"] = str(dest)
    meta["selected"] = [{"key": k, "bytes": size} for k, size in selected]
    meta["executed"] = False
    meta["verified_copied"] = 0
    if not args.apply:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    audit = args.audit_out.resolve(strict=False)
    if audit.exists() or os.stat(audit.parent).st_dev != os.stat("/").st_dev:
        raise ValueError("audit must be a new file on OS-local disk")
    if not str(dest).startswith(str(args.nas) + "/"):
        raise ValueError("bad destination namespace")
    dest.mkdir(parents=True, exist_ok=True)
    # Files must live within the intended NAS tree, and not inside a symlink.
    if not dest.resolve(strict=True).is_relative_to(args.nas.resolve(strict=True)):
        raise ValueError("destination escapes NAS share")
    records = []
    with audit.open("x", encoding="utf-8") as output:
        meta["executed"] = True
        for key, size in selected:
            _read_only_mount_check(args.uuid, args.source, args.nas, args.nas_share)
            rec = copy_one(args.source, dest, key, size)
            records.append(rec)
            output.write(json.dumps({"card_uuid": args.uuid, **rec}, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    meta["verified_copied"] = len(records)
    meta["verified_bytes"] = sum(r["bytes"] for r in records)
    meta["audit_path"] = str(audit)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
