"""Staged destination keys must not escape the destination root during verify.

Hermetic and offline: every fixture is a handful of bytes under pytest's
``tmp_path`` (redirected to the sweep scratch tree via ``TMPDIR``). No card,
mount, graph, network, model, service or live state is touched.

The copy side already refuses a staged key that resolves outside the destination
root (``executor._safe_join``). The verify side did not: a staged.jsonl row could
point at any file on the host, which would then be hashed and recorded as
``verified_at_destination`` for that campaign key.
"""

from __future__ import annotations

import json

from auto_ingest.custody.hashing import digest_file
from auto_ingest.custody.ledger import DESTINATION_LEDGER, HASH_LEDGER, ledger_dir
from auto_ingest.custody.verify import verify_destination

KEY = "clip.mp4"
PAYLOAD = b"custody staged-key containment regression fixture\n"
VERIFIED = "verified_at_destination"


def _bundle(tmp_path, key: str = KEY):
    """A bundle whose hash ledger proves one object whose payload is known."""
    bundle = tmp_path / "bundle"
    dest = tmp_path / "dest"
    dest.mkdir()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(PAYLOAD)
    digest, size = digest_file(payload)
    ledgers = ledger_dir(bundle)
    ledgers.mkdir(parents=True, exist_ok=True)
    (ledgers / HASH_LEDGER).write_text(
        json.dumps({"key": key, "status": "verified",
                    "digest": digest, "size": size}) + "\n",
        encoding="utf-8",
    )
    return bundle, dest


def _rows(bundle):
    path = ledger_dir(bundle) / DESTINATION_LEDGER
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _outside_file(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    target = outside / KEY
    target.write_bytes(PAYLOAD)
    return target


# --- fail-before / pass-after: a staged key may not leave the root --------

def test_staged_parent_component_is_not_verified(tmp_path):
    bundle, dest = _bundle(tmp_path)
    _outside_file(tmp_path)

    result = verify_destination(
        bundle, dest, staged_destinations={KEY: f"../outside/{KEY}"})

    assert result.verified == 0
    assert result.failed == 1
    assert any("key_escapes_destination" in err for err in result.errors)
    assert not any(row.get("status") == VERIFIED for row in _rows(bundle))


def test_staged_absolute_path_is_not_verified(tmp_path):
    bundle, dest = _bundle(tmp_path)
    outside = _outside_file(tmp_path)

    result = verify_destination(
        bundle, dest, staged_destinations={KEY: str(outside)})

    assert result.verified == 0
    assert result.failed == 1
    assert not any(row.get("status") == VERIFIED for row in _rows(bundle))


def test_source_key_with_parent_component_is_not_verified(tmp_path):
    key = "../escape.mp4"
    bundle, dest = _bundle(tmp_path, key=key)
    (tmp_path / "escape.mp4").write_bytes(PAYLOAD)

    result = verify_destination(bundle, dest)

    assert result.verified == 0
    assert result.failed == 1
    assert not any(row.get("status") == VERIFIED for row in _rows(bundle))


# --- controls: ordinary verification is unchanged -------------------------

def test_normal_staged_destination_still_verifies(tmp_path):
    bundle, dest = _bundle(tmp_path)
    landed = dest / "2026" / "08" / KEY
    landed.parent.mkdir(parents=True)
    landed.write_bytes(PAYLOAD)

    result = verify_destination(
        bundle, dest, staged_destinations={KEY: "2026/08/clip.mp4"})

    assert result.verified == 1
    assert result.failed == 0
    assert any(row.get("status") == VERIFIED for row in _rows(bundle))


def test_flat_destination_still_verifies(tmp_path):
    bundle, dest = _bundle(tmp_path)
    (dest / KEY).write_bytes(PAYLOAD)

    result = verify_destination(bundle, dest)

    assert result.verified == 1
    assert result.missing == 0


def test_absent_destination_still_reports_missing(tmp_path):
    bundle, dest = _bundle(tmp_path)

    result = verify_destination(bundle, dest)

    assert result.verified == 0
    assert result.missing == 1
    assert not any(row.get("status") == VERIFIED for row in _rows(bundle))
