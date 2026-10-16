"""Read-only, bounded join of old NAS candidate and hash-witness manifests.

NOT a custody destination verifier: no NAS payload reads, hash-ledger promotion,
copy, source changes or destructive operation. A historical witness is never
a current SAFE_TO_RELEASE statement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .campaign import CardIdentity
from .fleet_campaigns import frozen_inventory, safe_key
from .store import load_campaign, load_status

SHA = re.compile(r"^[0-9a-f]{64}$")

def input_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _size(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid candidate size") from exc
    if result < 0:
        raise ValueError("negative candidate size")
    return result


def _absolute_nas(path: str) -> bool:
    if not isinstance(path, str) or "\\" in path:
        return False
    p = PurePosixPath(path)
    return (p.is_absolute() and len(p.parts) > 2 and
            p.parts[1] == "nas" and all(s not in (".", "..") for s in p.parts))


def _candidate(rows: dict, card: str, key: str, size: int,
               path: str, label: str) -> None:
    if not safe_key(key) or not _absolute_nas(path):
        raise ValueError("unsafe indexed candidate path")
    obj = rows.get((card, key))
    if obj is None or obj["bytes"] != size:
        raise ValueError(f"indexed candidate does not join frozen source: {card}:{key}")
    candidates = obj["candidates"]
    previous = candidates.get(path)
    if previous and previous != label:
        raise ValueError(f"ambiguous candidate provenance for {path}")
    candidates[path] = label


def _csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("witness must be JSON object")
    return result


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(SHA.fullmatch(value.lower()))


def build_index(
    ledger: Path, expected_sha: str, campaign_root: Path,
    staging_csv: Path, alternate_csv: Path, staging_witness: Path,
    alternate_witness: Path,
) -> dict:
    """Manifest-only join. Strict about conflicts, lenient about missing candidates."""
    sha, sources = frozen_inventory(ledger)
    if not _digest(expected_sha) or sha != expected_sha.lower():
        raise ValueError("pinned frozen-ledger SHA256 mismatch")
    if not sources:
        raise ValueError("frozen inventory contains no cards")
    cards: dict[str, dict] = {}
    source_mounts: dict[str, str] = {}
    for uuid in sorted(sources):
        bundle = campaign_root / uuid
        campaign = load_campaign(bundle)
        if campaign.source.card.compare(CardIdentity(filesystem_uuid=uuid)).matched is not True:
            raise ValueError(f"campaign source identity mismatch: {uuid}")
        source_mount = campaign.source.mount_point
        if not source_mount or not Path(source_mount).is_absolute() or ".." in Path(source_mount).parts:
            raise ValueError(f"campaign source mount invalid: {uuid}")
        source_mounts[uuid] = source_mount
        status = load_status(bundle)
        if status.source_release_allowed:
            raise ValueError(f"unexpected open source-release gate: {uuid}")
        cards[uuid] = {"bundle": str(bundle), "id": campaign.campaign_id,
                       "source_mount": source_mount,
                       "status": status.derivation.state.value,
                       "source_release_allowed": False}
    rows = {
        (uuid, key): {
            "card_uuid": uuid, "source_relative_path": key, "bytes": value[0],
            "historical_offhost_source_receipt": value[1],
            "candidates": {}, "witnesses": [],
        }
        for uuid, partition in sources.items() for key, value in partition.items()
    }
    # Source mount and staging destination are facts from the UUID-bound
    # campaigns and explicitly supplied historical witness, not host defaults.
    stage_doc = _json(staging_witness)
    staging_uuid = stage_doc.get("source_card_uuid")
    staging_root = stage_doc.get("target_mount")
    if (staging_uuid not in sources or not _absolute_nas(staging_root)):
        raise ValueError("staging witness root/UUID mismatch")
    for record in _csv(staging_csv):
        if record.get("staged_exact_status") != "MATCHING_SIZE_ONLY":
            continue
        uuid = record.get("card_uuid")
        if uuid != staging_uuid:
            raise ValueError("staging candidate UUID conflicts with witness")
        key = record["source_relative_path"]
        size = _size(record["source_expected_bytes"])
        if size != _size(record["staged_observed_size"]):
            raise ValueError("staging candidate size conflicts")
        _candidate(rows, uuid, key, size,
                   staging_root.rstrip("/") + "/" + key, "staging_path_size_only")
    for record in _csv(alternate_csv):
        if record.get("metadata_status") != "PATH_AND_SIZE_MATCH":
            continue
        uuid = record["card_uuid"]
        key = record["source_relative_path"]
        size = _size(record["expected_bytes"])
        if size != _size(record["observed_bytes"]):
            raise ValueError("alternative candidate size conflicts")
        _candidate(rows, uuid, key, size, record["candidate_path"],
                   "alternate_archive_path_size_only")

    # A previous matched digest is not a live assertion about the NAS today.
    a = stage_doc
    for rec in a.get("records", []):
        uuid = staging_uuid
        key = rec["path"]
        destination = staging_root.rstrip("/") + "/" + key
        obj = rows.get((uuid, key))
        if obj is None or destination not in obj["candidates"]:
            raise ValueError("unmatched staged witness candidate")
        if (rec.get("state") != "SHA_MATCH_STABLE" or
                rec.get("stable_pre_post") is not True or
                not _digest(rec.get("source_sha256")) or
                rec.get("source_sha256") != rec.get("nas_stage_sha256") or
                _size(rec.get("bytes")) != obj["bytes"]):
            raise ValueError("inconsistent historical staged SHA witness")
        obj["witnesses"].append({"location": destination,
                                 "sha256": rec["source_sha256"],
                                 "scope": "historical_point_in_time"})
    a = _json(alternate_witness)
    for rec in a.get("cases", []):
        uuid = rec["card_uuid"]
        source = rec["source"]
        prefix = source_mounts.get(uuid)
        if not prefix or not source.startswith(prefix + "/"):
            raise ValueError("alternative source path/UUID mismatch")
        key = source[len(prefix) + 1:]
        obj = rows.get((uuid, key))
        dest = rec["alternative_nas_target"]
        if obj is None or dest not in obj["candidates"]:
            raise ValueError("unmatched alternate witness candidate")
        if (rec.get("equal_source_dest") is not True or
                rec.get("stable_pre_post") is not True or
                not _digest(rec.get("source_sha256_now")) or
                rec.get("source_sha256_now") != rec.get("destination_sha256_now") or
                _size(rec.get("source_bytes")) != obj["bytes"]):
            raise ValueError("inconsistent historical alternate SHA witness")
        obj["witnesses"].append({"location": dest,
                                 "sha256": rec["source_sha256_now"],
                                 "scope": "historical_point_in_time"})
    file_rows = []
    overall = Counter()
    per_card = {uuid: Counter() for uuid in sources}
    for (uuid, key), item in sorted(rows.items()):
        if item["witnesses"]:
            state = "prior_sha_witness_recheck_required"
        elif item["candidates"]:
            state = "metadata_candidate_needs_sha"
        else:
            state = "no_indexed_candidate_not_proven_missing"
        name = key.rsplit("/", 1)[-1]
        video_mp4 = name.lower().endswith(".mp4") and not name.startswith("._")
        entry = {
            "card_uuid": uuid, "source_relative_path": key,
            "bytes": item["bytes"], "classification": state,
            "video_mp4_filename_not_content_verified": video_mp4,
            "historical_offhost_source_receipt": item["historical_offhost_source_receipt"],
            "candidate_paths": sorted(item["candidates"]),
            "historical_sha_witnesses": item["witnesses"],
            "current_nas_sha_verified": False,
            "source_release_allowed": False,
        }
        file_rows.append(entry)
        overall[state] += 1
        overall[state + "_bytes"] += item["bytes"]
        per_card[uuid][state] += 1
        if video_mp4:
            overall["video_mp4_" + state] += 1
            overall["video_mp4_" + state + "_bytes"] += item["bytes"]
    return {
        "schema": 1, "scope": "manifest_only_no_live_destination_hash",
        "frozen_ledger_sha256": sha,
        "input_sha256": {str(p): input_sha(p) for p in
                         (staging_csv, alternate_csv, staging_witness, alternate_witness)},
        "cards": cards,
        "counts": dict(sorted(overall.items())),
        "per_card": {uuid: dict(v) for uuid, v in per_card.items()},
        "source_release_allowed": False,
        "current_nas_sha_verified_files": 0,
        "total_source_files": len(file_rows),
        "rows": file_rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-ledger", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--staging-csv", type=Path, required=True)
    p.add_argument("--alternate-csv", type=Path, required=True)
    p.add_argument("--staging-witness", type=Path, required=True)
    p.add_argument("--alternate-witness", type=Path, required=True)
    p.add_argument("--out", type=Path, help="create exclusive local JSONL report (never overwrites)")
    args = p.parse_args(argv)
    try:
        result = build_index(args.frozen_ledger, args.expected_sha256,
                             args.campaign_root, args.staging_csv,
                             args.alternate_csv, args.staging_witness,
                             args.alternate_witness)
        rows = result.pop("rows")
        if args.out:
            path = args.out.resolve(strict=False)
            if (path.is_relative_to(Path("/nas")) or
                    any(path.is_relative_to(Path(c["source_mount"]))
                        for c in result["cards"].values())):
                raise ValueError("report output must not be on NAS or source cards")
            ancestor = path.parent
            if (not ancestor.is_dir() or
                    os.stat(ancestor).st_dev != os.stat("/").st_dev):
                raise ValueError("report destination must be an existing OS-local directory")
            with path.open("x", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            result["rows_written"] = len(rows)
            result["rows_file"] = str(path)
        else:
            result["sample_rows"] = rows[:8]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
