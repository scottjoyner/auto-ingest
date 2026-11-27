"""Resume must not reuse hash evidence for a file whose size has changed.

`hash_source` skipped every key already present in the hash ledger, ignoring the
size recorded with that digest. A source object that changed between passes was
therefore treated as already verified, and the stale digest was carried into
the resumed campaign.
"""
from __future__ import annotations

import hashlib
import json

from auto_ingest.custody.hashing import hash_source


def _record(key: str, size: int, digest: str) -> str:
    return json.dumps({
        "algorithm": "sha256", "digest": digest, "key": key,
        "size": size, "status": "verified",
    }, sort_keys=True, separators=(",", ":")) + "\n"


def _bundle(tmp_path, key: str, size: int, recorded_size: int):
    bundle = tmp_path / "bundle"
    ledger_dir = bundle / "ledgers"
    ledger_dir.mkdir(parents=True)
    source = tmp_path / key
    source.write_bytes(b"x" * size)
    (ledger_dir / "hash.jsonl").write_text(
        _record(key, recorded_size, hashlib.sha256(b"old").hexdigest()),
        encoding="utf-8",
    )
    return bundle, {key: source}


def test_matching_size_still_resumes_without_rehashing(tmp_path):
    bundle, keys = _bundle(tmp_path, "clip.mp4", 16, 16)
    progress = hash_source(bundle, keys)
    assert progress.hashed == 0
    assert progress.skipped_existing == 1


def test_size_drift_forces_a_fresh_hash(tmp_path):
    bundle, keys = _bundle(tmp_path, "clip.mp4", 16, 8)
    progress = hash_source(bundle, keys)
    assert progress.hashed == 1
    assert progress.skipped_existing == 0
    rows = [json.loads(line) for line in
            (bundle / "ledgers" / "hash.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["size"] == 16
    assert rows[-1]["digest"] == hashlib.sha256(b"x" * 16).hexdigest()
