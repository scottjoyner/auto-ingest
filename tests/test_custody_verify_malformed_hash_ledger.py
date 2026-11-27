"""Verification must refuse an incoherent source hash ledger.

``read_records()`` skips unparseable rows on purpose (a producer may have died
mid-append), but ``verify_destination()`` used to build its list of objects to
prove from those very rows. A malformed or torn source row therefore vanished
from the pass while ``verification_complete`` stayed true - a verifier that
silently skipped work and then declared success. These tests pin the refusal at
both the pure-logic and the CLI layer, using only ``tmp_path`` fixtures.
"""
from __future__ import annotations

import hashlib
import json

import pytest
from custody_helpers import campaign, destination, evidence, write_bundle

from auto_ingest.custody.cli import EXIT_GATE_CLOSED, main
from auto_ingest.custody.ledger import DESTINATION_LEDGER, HASH_LEDGER, ledger_dir
from auto_ingest.custody.verify import verify_destination

pytestmark = pytest.mark.usefixtures("hermetic_mounts")


def _bundle(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    bundle = write_bundle(
        tmp_path / "bundle",
        campaign(dest=destination(host_path=str(dest), mounted=True)),
        evidence(),
    )
    return bundle, dest


def _write_hash_ledger(bundle, rows, *, terminated: bool = True) -> None:
    ledger = ledger_dir(bundle) / HASH_LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows) + ("\n" if terminated else "")
    ledger.write_text(text, encoding="utf-8")


def _row(key: str, data: bytes) -> str:
    return json.dumps({
        "key": key,
        "status": "verified",
        "digest": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    })


def test_malformed_hash_row_refuses_the_whole_pass(tmp_path):
    bundle, dest = _bundle(tmp_path)
    a, b = b"A-data", b"B-data"
    (dest / "a.bin").write_bytes(a)
    (dest / "b.bin").write_bytes(b)
    # One sound row, one row truncated mid-object by an interrupted producer.
    _write_hash_ledger(bundle, [
        _row("a.bin", a),
        '{"key":"b.bin","status":"verified","dig',
    ])

    with pytest.raises(Exception, match="not coherent.*malformed_lines=1") as caught:
        verify_destination(bundle, dest)
    assert type(caught.value).__name__ == "VerifyUnavailable"

    # Refusal is whole-run: no destination receipt is written for the subset
    # that happened to be readable.
    assert not (ledger_dir(bundle) / DESTINATION_LEDGER).exists()


def test_torn_hash_row_refuses_the_whole_pass(tmp_path):
    bundle, dest = _bundle(tmp_path)
    a = b"A-data"
    (dest / "a.bin").write_bytes(a)
    _write_hash_ledger(bundle, [_row("a.bin", a)], terminated=False)

    with pytest.raises(Exception, match="not coherent.*truncated=True"):
        verify_destination(bundle, dest)


def test_coherent_hash_ledger_still_verifies(tmp_path):
    """Control: the guard must not disturb a healthy verification pass."""
    bundle, dest = _bundle(tmp_path)
    a, b = b"A-data", b"B-data"
    (dest / "a.bin").write_bytes(a)
    (dest / "b.bin").write_bytes(b)
    _write_hash_ledger(bundle, [_row("a.bin", a), _row("b.bin", b)])

    result = verify_destination(bundle, dest)
    assert result.verified == 2
    assert (result.missing, result.mismatched, result.failed) == (0, 0, 0)
    assert result.complete is True


def test_cli_verify_refuses_malformed_hash_ledger_and_applies_nothing(tmp_path, capsys):
    bundle, dest = _bundle(tmp_path)
    a = b"A-data"
    (dest / "a.bin").write_bytes(a)
    _write_hash_ledger(bundle, [_row("a.bin", a), '{"key":"b.bin"'])
    before = (bundle / "evidence.json").read_text(encoding="utf-8")

    code = main(["verify", "--bundle", str(bundle), "--destination", str(dest),
                 "--apply", "--json"])

    assert code == EXIT_GATE_CLOSED
    assert "not coherent" in capsys.readouterr().err
    assert (bundle / "evidence.json").read_text(encoding="utf-8") == before
