"""A foreign-object walk that cannot read a subtree must not report a count.

`scan_foreign_objects` feeds `strict_destination_scope`. `os.walk` silently
ignores directories it cannot open, so an unreadable subtree produced a partial
count that was reported as the whole answer — a second campaign could sit in
that subtree and the release gate would call the destination clean.
"""
from __future__ import annotations

import os

import pytest

from auto_ingest.custody.ledger import scan_foreign_objects


def test_readable_destination_still_counts_foreign_objects(tmp_path):
    dest = tmp_path / "dest"
    (dest / "DCIM").mkdir(parents=True)
    (dest / "DCIM" / "clip.mp4").write_bytes(b"x")
    count, samples = scan_foreign_objects(dest, tmp_path / "absent-ledger.jsonl")
    assert count == 1
    assert samples == ("DCIM/clip.mp4",)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory mode bits")
def test_unreadable_subtree_fails_closed(tmp_path):
    dest = tmp_path / "dest"
    hidden = dest / "other-campaign"
    hidden.mkdir(parents=True)
    (hidden / "secret.mp4").write_bytes(b"x")
    visible = dest / "DCIM"
    visible.mkdir()
    (visible / "clip.mp4").write_bytes(b"x")

    hidden.chmod(0o000)
    try:
        with pytest.raises(OSError):
            scan_foreign_objects(dest, tmp_path / "absent-ledger.jsonl")
    finally:
        hidden.chmod(0o755)
