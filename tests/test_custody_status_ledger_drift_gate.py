"""--require-release must not pass while the status reports ledger drift.

`cmd_status` exited 0 whenever `source_release_allowed` was true, even when
`status_text` printed a `ledger drift` line saying a ledger was incoherent. A CI
caller reading only the exit code would treat a corrupt bundle as releasable.
"""
from __future__ import annotations

from custody_helpers import fully_copied_campaign, write_bundle
from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main


def _run(capsys, bundle):
    capsys.readouterr()
    code = main(["status", "--bundle", str(bundle), "--require-release"])
    out, _ = capsys.readouterr()
    return code, out


def test_clean_fully_custodied_bundle_still_passes(tmp_path, capsys):
    camp, ev = fully_copied_campaign()
    code, out = _run(capsys, write_bundle(tmp_path / "clean", camp, ev))
    assert code == EXIT_OK
    assert "SAFE_TO_RELEASE" in out


def test_incoherent_ledger_closes_the_require_release_gate(tmp_path, capsys):
    camp, ev = fully_copied_campaign()
    bundle = write_bundle(tmp_path / "drift", camp, ev)
    ledgers = bundle / "ledgers"
    ledgers.mkdir()
    # Unterminated final line: summarize_ledger marks the ledger incoherent.
    (ledgers / "destination.jsonl").write_bytes(
        b'{"key":"a","status":"verified","size":1,"digest":"abc"}'
    )
    code, out = _run(capsys, bundle)
    assert "ledger drift" in out
    assert code == EXIT_GATE_CLOSED
