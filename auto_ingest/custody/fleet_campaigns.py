"""UUID-anchored SD campaign bootstrap from frozen off-host evidence.

Dry-run by default. --apply creates only new campaign metadata on x1 local
storage, not hashes/copies/verification ledgers or files on cards/NAS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .campaign import CardIdentity
from .mounts import observe_mount, uuid_for_device
from .store import load_campaign, new_campaign

UUID_RE = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}$")


def parse_card(spec: str) -> tuple[str, Path]:
    uuid, sep, path = spec.partition("=")
    uuid = uuid.upper()
    root = Path(path)
    if not sep or not UUID_RE.fullmatch(uuid) or not root.is_absolute() or (
        ".." in root.parts or root == Path("/")
    ):
        raise ValueError("--card must be FAT-UUID=/absolute/mount with no '..'")
    return uuid, root


def safe_key(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    key = PurePosixPath(value)
    return not key.is_absolute() and all(p not in {"..", "."} for p in key.parts)


def frozen_inventory(path: Path) -> tuple[str, dict[str, dict[str, tuple[int, bool]]]]:
    """Read previous manifest; hash its bytes, but NEVER hash card media here."""
    digest = hashlib.sha256()
    by_card: dict[str, dict[str, tuple[int, bool]]] = {}
    with path.open("rb") as fh:
        for lineno, data in enumerate(fh, 1):
            digest.update(data)
            try:
                obj = json.loads(data)
            except (UnicodeError, ValueError) as exc:
                raise ValueError(f"invalid frozen JSONL line {lineno}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"non-object frozen row {lineno}")
            uuid = obj.get("card_uuid")
            key = obj.get("source_relative_path")
            size = obj.get("logical_bytes")
            if not isinstance(uuid, str) or not UUID_RE.fullmatch(uuid) or (
                not safe_key(key) or type(size) is not int or size < 0
            ):
                raise ValueError(f"invalid identity/path/size in frozen row {lineno}")
            part = by_card.setdefault(uuid, {})
            if key in part:
                raise ValueError(f"duplicate source key in {uuid}: {key}")
            # Historical off-host witness, NOT fresh source/NAS hash proof.
            qualified = any(v is True for name, v in obj.items() if (
                name in {"joyner_local_host_full_sha_verified",
                         "x1_two_nvme_full_sha_verified",
                         "xwing_local_nvme_independent_full_sha_verified"}
                or (name.startswith("xwing_tranche")
                    and name.endswith("independent_full_sha_verified"))
            ))
            part[key] = (size, qualified)
    return digest.hexdigest(), by_card


def card_census(uuid: str, root: Path,
                old: dict[str, tuple[int, bool]]) -> dict:
    mount = observe_mount(str(root))
    observed_uuid = uuid_for_device(mount.device) if mount.present else None
    identity_ok = bool(mount.present and mount.is_mount_point and
                       mount.filesystem_type in {"vfat", "exfat"} and
                       observed_uuid == uuid)
    seen: set[str] = set()
    counts = Counter()
    samples: dict[str, list[str]] = {n: [] for n in
                                     ("missing", "new", "changed", "unsafe")}
    if identity_ok:
        def note(kind: str, name: str) -> None:
            counts[kind] += 1
            if len(samples[kind]) < 5:
                samples[kind].append(name)

        def walk_error(exc: OSError) -> None:
            note("unsafe", str(exc.filename or root))

        for base, folders, files in os.walk(root, followlinks=False,
                                            onerror=walk_error):
            for folder in folders:
                if Path(base, folder).is_symlink():
                    note("unsafe", str(Path(base, folder)))
            folders[:] = [f for f in folders if not Path(base, f).is_symlink()]
            for filename in files:
                p = Path(base, filename)
                key = p.relative_to(root).as_posix()
                try:
                    meta = p.lstat()
                except OSError:
                    note("unsafe", key)
                    continue
                if not safe_key(key) or not stat.S_ISREG(meta.st_mode):
                    note("unsafe", key)
                    continue
                seen.add(key)
                witness = old.get(key)
                if witness is None:
                    note("new", key)
                elif witness[0] != meta.st_size:
                    note("changed", key)
                else:
                    counts["matching"] += 1
                    counts["matching_bytes"] += meta.st_size
                    if witness[1]:
                        counts["historical_offhost"] += 1
    missing = set(old) - seen
    counts["missing"] = len(missing)
    samples["missing"] = sorted(missing)[:5]
    complete = bool(identity_ok and not any(counts[k] for k in
                                            ("new", "changed", "unsafe", "missing")))
    return {
        "card_uuid": uuid, "source_root": str(root),
        "observed_device": mount.device, "observed_uuid": observed_uuid,
        "observed_read_only": mount.read_only, "source_identity_matched": identity_ok,
        "expected_files": len(old), "expected_bytes": sum(v[0] for v in old.values()),
        "matched_files": counts["matching"], "matched_bytes": counts["matching_bytes"],
        "historical_offhost_qualified_matches": counts["historical_offhost"],
        "new_files": counts["new"], "changed_size": counts["changed"],
        "unsafe_entries": counts["unsafe"], "missing": counts["missing"],
        "samples": samples, "metadata_complete": complete,
        "destination_sha_verified": False, "source_release_allowed": False,
    }


def plan(ledger: Path, cards: list[tuple[str, Path]],
         pinned_sha: str | None = None) -> dict:
    if len(set(u for u, _ in cards)) != len(cards):
        raise ValueError("duplicate card UUID")
    if len(set(str(m) for _, m in cards)) != len(cards):
        raise ValueError("duplicate mount path")
    sha, indexed = frozen_inventory(ledger)
    if pinned_sha is not None and pinned_sha.lower() != sha:
        raise ValueError("frozen JSONL SHA does not match independently pinned digest")
    if set(indexed) != set(uuid for uuid, _ in cards):
        raise ValueError("frozen ledger and observed card UUID sets differ")
    checks = [card_census(uuid, mount, indexed[uuid]) for uuid, mount in cards]
    return {
        "schema": 1, "scope": "frozen_manifest_and_current_name_size_only",
        "frozen_ledger": str(ledger), "frozen_sha256": sha,
        "sha_pinned": pinned_sha is not None,
        "cards": checks, "all_metadata_match": all(c["metadata_complete"] for c in checks),
        "total_files": sum(c["expected_files"] for c in checks),
        "total_bytes": sum(c["expected_bytes"] for c in checks),
        "source_release_allowed": False, "destination_sha_verified": False,
        "custody_inventory_evidence_imported": False,
    }


def create(report: dict, destination: Path, apply: bool = False) -> dict:
    """Only create new campaign.json and metadata receipts outside card and NAS."""
    path = destination.resolve(strict=False)
    # This bootstrap owns x1 OS-local receipts; NAS, USB and other mounts need
    # a separate owner-approved evidence placement decision.
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.is_symlink() or os.stat(ancestor).st_dev != os.stat("/").st_dev:
        raise ValueError("campaign records must live on the local OS filesystem")
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if any(path == root or path.is_relative_to(root) for root in
           (Path("/nas"), *(Path(c["source_root"]) for c in report["cards"]))):
        raise ValueError("campaign bundles may not live on NAS or source cards")
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("campaign output traverses a symlink")
    if apply and not report["sha_pinned"]:
        raise ValueError("--apply requires an independently pinned ledger SHA256")
    if apply and not report["all_metadata_match"]:
        raise ValueError("--apply refused: physical card differs from frozen ledger")
    if apply and path.exists() and any(path.iterdir()):
        raise ValueError("refusing to overwrite a nonempty campaign destination")
    results = []
    for card in report["cards"]:
        bundle = path / card["card_uuid"]
        if bundle.exists():
            if apply:
                raise ValueError(f"campaign exists: {bundle}")
            existing = load_campaign(bundle)
            if existing.source.card.filesystem_uuid != card["card_uuid"]:
                raise ValueError(f"existing campaign UUID differs: {bundle}")
            results.append({"uuid": card["card_uuid"], "bundle": str(bundle),
                            "campaign_id": existing.campaign_id,
                            "existing_read_only_record": True})
            continue
        args = dict(
            card_id="SD-" + card["card_uuid"],
            # Card UUID stable; /dev/sdX slot is ephemeral, included only in sidecar.
            observed=CardIdentity(filesystem_uuid=card["card_uuid"]),
            mount_point=card["source_root"],
            read_only=card["observed_read_only"] is True,
            created_at=created_at, config={}, env={}, apply=False,
        )
        proposal = new_campaign(bundle, **args)
        results.append({"uuid": card["card_uuid"], "bundle": str(bundle),
                        "campaign_id": proposal["campaign_id"]})
    if apply:
        path.mkdir(parents=True, exist_ok=True)
        for row, card in zip(results, report["cards"]):
            bundle = Path(row["bundle"])
            new_campaign(
                bundle, card_id="SD-" + card["card_uuid"],
                observed=CardIdentity(filesystem_uuid=card["card_uuid"]),
                mount_point=card["source_root"],
                read_only=card["observed_read_only"] is True,
                created_at=created_at, config={}, env={}, apply=True,
            )
            with (bundle / "frozen-source-assessment.json").open("x") as fh:
                json.dump({**card, "frozen_ledger_sha256": report["frozen_sha256"],
                           "no_payload_hash": True, "source_release_allowed": False},
                          fh, sort_keys=True, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
        with (path / "bootstrap-summary.json").open("x") as fh:
            json.dump(report, fh, sort_keys=True, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    return {**report, "applied": apply, "output_root": str(path),
            "campaigns": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-ledger", type=Path, required=True)
    parser.add_argument("--expected-sha256", help="independently pinned SHA; required to --apply")
    parser.add_argument("--card", type=parse_card, action="append", required=True,
                        help="UUID=absolute_card_mount; repeat for every frozen ledger card")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="ONLY create new OS-local campaign metadata, no media I/O")
    args = parser.parse_args(argv)
    try:
        report = create(plan(args.frozen_ledger, args.card,
                             pinned_sha=args.expected_sha256),
                        args.output_root, apply=args.apply)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
