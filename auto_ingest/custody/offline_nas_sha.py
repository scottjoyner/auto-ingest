"""Rehash NAS candidates against independently qualified historic source receipts.

No source-card access, no copying, no custody promotion, no release authority.
Only a selected, strictly budgeted subset of NAS objects is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .verify_candidates import sha_stable

SHA = re.compile(r"^[0-9a-f]{64}$")
SCOPE = "nas_only_against_qualified_historical_xwing_source_digest_not_release"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def select(
    frozen: Path,
    readiness: Path,
    *,
    expected_frozen_sha: str,
    max_files: int = 4,
    max_nas_bytes: int = 650_000_000,
    prior_reports: list[Path] | None = None,
) -> tuple[list[dict], dict]:
    """Validate provenance before selecting any candidate for NAS payload reads."""
    if max_files < 1 or max_files > 32 or max_nas_bytes < 1:
        raise ValueError("positive bounded max-files/max-nas-bytes required")
    source_sha = digest_file(frozen)
    if source_sha != expected_frozen_sha or not SHA.fullmatch(expected_frozen_sha):
        raise ValueError("frozen source ledger SHA mismatch")
    report = json.loads(readiness.read_text(encoding="utf-8"))
    if (report.get("scope") != "metadata_only_no_source_or_nas_payload_reads" or
            report.get("source_release_authorized") is not False or
            report.get("complete_metadata_scan") is not True or
            report.get("candidate_count") != len(report.get("rows", []))):
        raise ValueError("readiness report incomplete or not an offline metadata report")
    prior_reports = prior_reports or []
    seen_files = set()
    prior_sha = {}
    for path in prior_reports:
        if str(path) in seen_files:
            raise ValueError("duplicate prior evidence file")
        seen_files.add(str(path))
        obj = json.loads(path.read_text(encoding="utf-8"))
        if (obj.get("scope") != SCOPE or
                obj.get("source_ledger_sha256") != source_sha or
                obj.get("readiness_sha256") != digest_file(readiness) or
                obj.get("source_release_authorized") is not False):
            raise ValueError("prior NAS-only report provenance mismatch")
        for rec in obj.get("results", []):
            identity = (rec["card_uuid"], rec["source_relative_path"])
            if (rec.get("destination_stable") is True and
                    rec.get("sha_matches_historical_source") is True):
                if identity in prior_sha and prior_sha[identity]["nas_sha256"] != rec["nas_sha256"]:
                    raise ValueError("conflicting past NAS-only digests")
                prior_sha[identity] = rec
    qualified = {}
    with frozen.open(encoding="utf-8") as input_file:
        for line in input_file:
            obj = json.loads(line)
            key = (obj["card_uuid"], obj["source_relative_path"])
            if key in qualified:
                raise ValueError("duplicate frozen identity")
            qualified[key] = obj
    selected = []
    nbytes = 0
    nqualified = 0
    for candidate in sorted(report["rows"], key=lambda r: (
            r["bytes"], r["card_uuid"], r["source_relative_path"])):
        key = (candidate["card_uuid"], candidate["source_relative_path"])
        original = qualified.get(key)
        if original is None or original["logical_bytes"] != candidate["bytes"]:
            raise ValueError("candidate not bound to frozen source ledger")
        if (candidate.get("nas_metadata_state") != "nas_metadata_size_match_only" or
                candidate.get("content_sha_checked_this_run") is not False or
                candidate.get("source_release_authorized") is not False):
            raise ValueError("candidate metadata report cannot qualify file")
        source_digest = original.get("xwing_full_source_sha256")
        if (original.get("xwing_local_nvme_independent_full_sha_verified") is not True
                or not isinstance(source_digest, str)
                or SHA.fullmatch(source_digest) is None):
            continue
        nqualified += 1
        if key in prior_sha:
            previous = prior_sha[key]
            if (previous.get("nas_candidate") != candidate["nas_candidate"] or
                    previous.get("bytes") != candidate["bytes"] or
                    previous.get("historic_source_sha256") != source_digest or
                    previous.get("nas_sha256") != source_digest or
                    previous.get("source_card_read_this_run") is not False or
                    previous.get("source_release_authorized") is not False):
                raise ValueError("prior NAS-only receipt identity or SHA conflict")
            continue
        if len(selected) >= max_files or nbytes + candidate["bytes"] > max_nas_bytes:
            continue
        path = Path(candidate["nas_candidate"])
        if (not path.is_absolute() or not path.is_relative_to(Path("/nas"))
                or ".." in path.parts or "\\" in str(path)):
            raise ValueError("unsafe NAS destination path")
        nbytes += candidate["bytes"]
        selected.append({
            "card_uuid": key[0], "source_relative_path": key[1],
            "bytes": candidate["bytes"], "nas_candidate": str(path),
            "historic_source_sha256": source_digest,
            "source_receipt_kind": "xwing_local_nvme_independent_full_sha_verified",
        })
    return selected, {
        "source_ledger_sha256": source_sha,
        "readiness_sha256": digest_file(readiness),
        "qualified_candidates_total": nqualified,
        "previously_compared_matching": len(prior_sha),
        "nas_bytes_budget": max_nas_bytes, "selected_nas_bytes": nbytes,
    }


def verify(
    frozen: Path, readiness: Path, *, expected_frozen_sha: str,
    max_files: int = 4, max_nas_bytes: int = 650_000_000,
    prior_reports: list[Path] | None = None,
) -> dict:
    selected, audit = select(
        frozen, readiness, expected_frozen_sha=expected_frozen_sha,
        max_files=max_files, max_nas_bytes=max_nas_bytes,
        prior_reports=prior_reports)
    results = []
    nas_root = Path("/nas").resolve(strict=True)
    for item in selected:
        destination = Path(item["nas_candidate"])
        # Stronger than metadata-only readiness: inspect resolved path before
        # opening and SHA-256 reading the candidate without following final links.
        if not destination.resolve(strict=True).is_relative_to(nas_root):
            raise ValueError("NAS destination escaped authorized mount")
        actual = sha_stable(destination, item["bytes"])
        results.append({
            **item, "nas_sha256": actual["sha256"],
            "destination_stable": actual["stable"],
            "sha_matches_historical_source": actual["sha256"] == item["historic_source_sha256"],
            "source_card_read_this_run": False,
            "beelink_physical_storage_attested": False,
            "source_release_authorized": False,
        })
    return {
        "schema": 1, "scope": SCOPE,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        **audit, "max_files": max_files, "nas_bytes_read": sum(x["bytes"] for x in results),
        "source_card_bytes_read": 0, "source_release_authorized": False,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--readiness", type=Path, required=True)
    p.add_argument("--expected-frozen-sha256", required=True)
    p.add_argument("--max-files", type=int, default=4)
    p.add_argument("--max-nas-bytes", type=int, default=650_000_000)
    p.add_argument("--prior-report", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True, help="unique OS-local output path")
    a = p.parse_args(argv)
    try:
        out = a.out.resolve(strict=False)
        if (out.is_relative_to(Path("/nas")) or
                os.stat(out.parent).st_dev != os.stat("/").st_dev):
            raise ValueError("evidence output must be on independent OS-local storage")
        record = verify(a.frozen, a.readiness,
                        expected_frozen_sha=a.expected_frozen_sha256,
                        max_files=a.max_files,
                        max_nas_bytes=a.max_nas_bytes, prior_reports=a.prior_report)
        with out.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps({"out": str(out), "qualified": record["qualified_candidates_total"],
                      "prior_skipped": record["previously_compared_matching"],
                      "checked": len(record["results"]), "bytes_read": record["nas_bytes_read"],
                      "matches": sum(r["sha_matches_historical_source"] and
                                     r["destination_stable"] for r in record["results"]),
                      "source_release": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
