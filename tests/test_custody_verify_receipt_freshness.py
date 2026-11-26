"""A destination VERIFIED receipt is only valid for the digest it proved.

`verify` resumes by skipping keys it believes are already proven. That skip is
sound only while the source digest a receipt was proved against is unchanged.
This module pins the invariant that a receipt cannot vouch for a *new* source
object, and that an orphan receipt cannot inflate the custody file count.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from custody_helpers import campaign, destination, evidence, write_bundle

from auto_ingest.custody.cli import main
from auto_ingest.custody.hashing import digest_file
from auto_ingest.custody.ledger import ledger_dir
from auto_ingest.custody.verify import (
    VERIFIED,
    source_digests,
    to_evidence,
    verify_destination,
)

pytestmark = pytest.mark.usefixtures("hermetic_mounts")


def build(tmp_path, count=4, size=64):
    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(
        tmp_path / "bundle",
        campaign(dest=destination(host_path=str(dest), mounted=True)),
        evidence(),
    )
    keys = {}
    for i in range(count):
        path = src / f"c{i}.mp4"
        path.write_bytes(bytes([65 + i]) * size)
        keys[f"c{i}.mp4"] = path
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    return bundle, src, dest, keys


def copy_all(src: Path, dest: Path, keys):
    for key, path in keys.items():
        shutil.copyfile(path, dest / key)


def _hash_row(key: str, digest: str, size: int) -> str:
    # Same shape as auto_ingest.custody.hashing._record; written here so the
    # test controls the ledger directly and needs no private import.
    return json.dumps(
        {"key": key, "size": size, "digest": digest, "status": "verified",
         "algorithm": "sha256"},
        sort_keys=True, separators=(",", ":"),
    ) + "\n"


def _verified_row(key: str, digest: str, size: int) -> str:
    return json.dumps(
        {"key": key, "status": VERIFIED, "digest": digest, "size": size},
        sort_keys=True, separators=(",", ":"),
    ) + "\n"


def test_a_changed_source_digest_invalidates_the_stale_destination_receipt(tmp_path):
    """A receipt proved against an old digest must not skip re-verification."""
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    first = verify_destination(bundle, dest)
    assert first.verified == 4

    old_digest = source_digests(bundle)["c0.mp4"][0]
    # The source ledger advances for c0 (card re-hashed after bytes changed).
    # Last row wins in source_digests(); make the destination hold the new bytes
    # so a legitimate re-check would succeed cleanly.
    (dest / "c0.mp4").write_bytes(bytes([90]) * 64)
    new_digest, _ = digest_file(dest / "c0.mp4")
    assert new_digest != old_digest
    with (ledger_dir(bundle) / "hash.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(_hash_row("c0.mp4", new_digest, 64))
    assert source_digests(bundle)["c0.mp4"][0] == new_digest

    again = verify_destination(bundle, dest)

    # c0's old receipt cannot vouch for the new digest: it must be re-proved.
    assert again.verified == 1, "the stale receipt wrongly skipped c0"
    assert again.skipped_existing == 3
    rows = [json.loads(line) for line
            in (ledger_dir(bundle) / "destination.jsonl").read_text().splitlines()
            if line.strip()]
    fresh = [r for r in rows if r["key"] == "c0.mp4" and r["status"] == VERIFIED]
    assert fresh and fresh[-1]["digest"] == new_digest, (
        "custody still relies on the digest from the old source object"
    )


def test_an_orphan_receipt_does_not_inflate_the_custody_file_count(tmp_path):
    """A VERIFIED row for a key the source never hashed is not custody."""
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest)

    ledger = ledger_dir(bundle) / "destination.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(_verified_row("ghost.mp4", "f" * 64, 10))

    again = verify_destination(bundle, dest)
    # Only the four real source keys are skipped; the ghost row is not custody.
    assert again.skipped_existing == 4
    assert to_evidence(again)["destination"]["verified_files"] == 4
