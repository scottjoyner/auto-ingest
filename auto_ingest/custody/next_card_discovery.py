"""Read-only, UUID-bound discovery of newly inserted USB SD media.

Never mounts/unmounts cards, reads recording payloads, writes cards or NAS,
reuses previous /dev/sdX numbers, or authorizes source release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FAT_UUID = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}$")
DEVICE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_old_cards(receipt: Path, expected_sha256: str) -> set[str]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("invalid prior-handoff SHA-256")
    if receipt.stat().st_size > 2_000_000 or sha256(receipt) != expected_sha256.lower():
        raise ValueError("pinned prior-handoff SHA-256 mismatch")
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("host") != "x1-370" or (
            doc.get("source_files_deleted") != 0 or
            doc.get("nas_full_destination_verification_complete") is not False):
        raise ValueError("prior-handoff authority is inconsistent")
    cards = doc.get("cards")
    if not isinstance(cards, list) or len(cards) == 0:
        raise ValueError("prior handoff has no card identities")
    uuids = set()
    for card in cards:
        uuid = card.get("uuid")
        if (not isinstance(uuid, str) or not FAT_UUID.fullmatch(uuid)
                or uuid in uuids
                or card.get("uuid_reverified_after_unmount") is not True
                or card.get("still_kernel_mounted") is not False
                or card.get("safe_for_physical_removal_with_data_retained") is not True
                or card.get("safe_to_erase_or_reuse") is not False):
            raise ValueError("unsafe or duplicate historical card handoff")
        uuids.add(uuid)
    if sha256(receipt) != expected_sha256.lower():
        raise ValueError("prior handoff changed during read")
    return uuids


def _is_yes(value: object) -> bool:
    return value is True or value == 1 or value == "1"


def _mounted(part: dict) -> list[str]:
    mounts = part.get("mountpoints") or []
    if not isinstance(mounts, list):
        raise ValueError("unexpected lsblk mountpoint structure")
    return [x for x in mounts if isinstance(x, str) and x]


def inspect_lsblk(payload: dict, *, prior_uuids: set[str]) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("blockdevices"), list):
        raise ValueError("lsblk did not return a valid device tree")
    result = {
        "schema": 1,
        "scope": "hotplug_uuid_presence_and_mount_metadata_only",
        "host": "x1-370",
        "previous_original_cards": sorted(prior_uuids),
        "source_media_read": False,
        "source_release_allowed": False,
        "card_mounts_changed": False,
        "copies_performed": 0,
        "deletions_performed": 0,
        "new_media_candidates": [],
        "previous_original_cards_still_detected": [],
        "unqualified_removable_devices": [],
    }
    seen_uuids = set()
    for disk in payload["blockdevices"]:
        if not isinstance(disk, dict) or disk.get("type") != "disk":
            continue
        if not _is_yes(disk.get("rm")) or disk.get("tran") != "usb":
            continue
        disk_name = disk.get("name")
        if not isinstance(disk_name, str) or not DEVICE_NAME.fullmatch(disk_name):
            raise ValueError("untrusted USB device identifier")
        children = disk.get("children") or []
        if not isinstance(children, list):
            raise ValueError("untrusted USB partition list")
        if not children:
            result["unqualified_removable_devices"].append({
                "device": "/dev/" + disk_name,
                "reason": "reader_empty_or_no_partition",
            })
        for part in children:
            if not isinstance(part, dict) or part.get("type") != "part":
                raise ValueError("unrecognized USB device child")
            name, uuid = part.get("name"), part.get("uuid")
            fstype = (part.get("fstype") or "").lower()
            if (not isinstance(name, str) or not DEVICE_NAME.fullmatch(name)
                    or not isinstance(uuid, str) or not FAT_UUID.fullmatch(uuid)
                    or fstype not in {"vfat", "exfat"}):
                result["unqualified_removable_devices"].append({
                    "device": "/dev/" + str(name), "reason": "not_identified_fat_card",
                })
                continue
            if uuid in seen_uuids:
                raise ValueError("duplicate removable filesystem UUID: " + uuid)
            seen_uuids.add(uuid)
            mounts = _mounted(part)
            card = {
                "uuid": uuid,
                "current_partition": "/dev/" + name,
                "current_parent_reader": "/dev/" + disk_name,
                "filesystem": fstype,
                "current_mountpoints": mounts,
                "currently_mounted": bool(mounts),
                "observed_block_read_only": _is_yes(part.get("ro")),
                "not_allowed_to_erase_or_release": True,
                # lsblk sees a mount, but is not an independent mount/UUID witness.
                "safe_to_begin_custody_inventory": False,
                "requires_independent_mount_uuid_witness": True,
            }
            if uuid in prior_uuids:
                card["classification"] = "original_preserve_do_not_reingest_as_new"
                card["safe_to_begin_custody_inventory"] = False
                result["previous_original_cards_still_detected"].append(card)
            else:
                card["classification"] = (
                    "new_mounted_candidate_mount_witness_required_before_inventory"
                    if mounts else "new_unmounted_candidate_manual_read_only_mount_needed"
                )
                result["new_media_candidates"].append(card)
    result["new_card_count"] = len(result["new_media_candidates"])
    result["prior_card_count_still_inserted"] = len(
        result["previous_original_cards_still_detected"]
    )
    return result


def discover(handoff: Path, expected_sha: str) -> dict:
    old = verified_old_cards(handoff, expected_sha)
    output = subprocess.run(
        ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,FSTYPE,LABEL,UUID,RM,RO,TRAN,MOUNTPOINTS,MODEL"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    result = inspect_lsblk(json.loads(output.stdout), prior_uuids=old)
    result["observed_utc"] = datetime.now(timezone.utc).isoformat()
    result["prior_handoff_sha256"] = expected_sha.lower()
    return result


def write_receipt(output: Path, data: dict) -> None:
    path = output.absolute()
    if path.is_symlink() or path.is_relative_to(Path("/nas")):
        raise ValueError("read-only discovery receipt must stay off NAS")
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("discovery receipt must use OS disk")
    if path.exists():
        raise FileExistsError("refusing to overwrite discovery receipt")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-handoff", type=Path, required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--out", type=Path, help="optional exclusive local OS-disk audit JSON")
    args = parser.parse_args(argv)
    try:
        result = discover(args.prior_handoff, args.expected_handoff_sha256)
        if args.out:
            write_receipt(args.out, result)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
