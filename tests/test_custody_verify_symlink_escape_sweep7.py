"""Focused fail-closed tests for custody verify: symlink/escape handling.

Hermetic: uses tmp_path only, no live DB/network/credentials.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from custody_helpers import campaign, destination, evidence, write_bundle

from auto_ingest.custody.cli import main
from auto_ingest.custody.verify import (
    FAILED,
    MISSING,
    VERIFIED,
    verify_destination,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"

pytestmark = pytest.mark.usefixtures("hermetic_mounts")


def build(tmp_path, count=2, size=64):
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


def copy_all(src, dest, keys):
    for key, path in keys.items():
        shutil.copyfile(path, dest / key)


def test_traversing_key_is_failed_not_verified(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    # Inject a hostile key into a separate bundle by hashing an escape path is not
    # possible via CLI; instead exercise verify with a staged map pointing out.
    # Simpler: call verify on a bundle whose hash ledger contains a hostile key.
    from auto_ingest.custody.ledger import ledger_dir, HASH_LEDGER
    import json

    ledger = ledger_dir(bundle) / HASH_LEDGER
    # append a hostile verified row (digest arbitrary but well-formed)
    with ledger.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"key": "../escape.mp4", "status": "verified",
                            "digest": "a" * 64, "size": 8}) + "\n")
    result = verify_destination(bundle, dest)
    assert result.failed >= 1
    assert any("escape" in e or "escapes" in e for e in result.errors)
    # the hostile key must not be counted as verified
    assert result.verified == len(keys)


def test_absolute_key_is_failed(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    from auto_ingest.custody.ledger import ledger_dir, HASH_LEDGER
    import json

    ledger = ledger_dir(bundle) / HASH_LEDGER
    with ledger.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"key": "/etc/passwd", "status": "verified",
                            "digest": "b" * 64, "size": 8}) + "\n")
    result = verify_destination(bundle, dest)
    assert result.failed >= 1
    assert result.verified == len(keys)


def test_symlinked_destination_is_not_followed(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    # replace one real destination file with a symlink to an outside target
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"NOT THE CARD" * 16)
    target = dest / "c0.mp4"
    target.unlink()
    target.symlink_to(outside)
    result = verify_destination(bundle, dest)
    # the symlinked object must not be verified (not followed)
    assert result.verified <= len(keys) - 1
    assert result.failed >= 1
    assert any("symlink" in e for e in result.errors)
    # outside target untouched
    assert outside.exists()


def test_staged_destination_escape_is_failed(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    mapping = {k: k for k in keys}
    # one key's staged destination escapes the root
    mapping[list(keys)[0]] = "../escape.mp4"
    result = verify_destination(bundle, dest, staged_destinations=mapping)
    assert result.failed >= 1
    assert any("escape" in e for e in result.errors)
