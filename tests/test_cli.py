from pathlib import Path

from content_os.cli import main


def test_cli_init_capture_and_status(tmp_path: Path, capsys):
    root = tmp_path / "content-os"

    assert main(["init", "--root", str(root)]) == 0
    assert (
        main(
            [
                "--root",
                str(root),
                "capture",
                "--title",
                "CLI note",
                "--text",
                "raw",
                "--no-run",
            ]
        )
        == 0
    )
    assert main(["--root", str(root), "status"]) == 0

    out = capsys.readouterr().out
    assert "Initialized Content OS" in out
    assert "captured_to_inbox" in out
    assert "id" in out and "next_action" in out
    assert "CLI note" in (root / "stores" / "inbox.md").read_text(encoding="utf-8")


def test_cli_ingest_source_creates_proof_and_run(tmp_path: Path, capsys):
    root = tmp_path / "content-os"
    source = tmp_path / "segments.json"
    source.write_text(
        '{"title":"CLI Segments","segments":[{"start":0,"text":"CLI proof"}]}',
        encoding="utf-8",
    )

    assert main(["init", "--root", str(root)]) == 0
    assert (
        main(
            ["--root", str(root), "ingest-source", str(source), "--format", "linkedin"]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "source_type=transcript_json" in out
    assert "run=2026-05-cli-segments" in out
    assert list((root / "stores" / "proof").glob("*cli-segments.md"))


def test_cli_failed_approval_without_force_returns_error(tmp_path: Path, capsys):
    root = tmp_path / "content-os"
    assert main(["init", "--root", str(root)]) == 0
    assert (
        main(
            [
                "--root",
                str(root),
                "new-run",
                "--title",
                "Needs work",
                "--route",
                "ORIGINAL",
            ]
        )
        == 0
    )
    run_id = "2026-05-needs-work"
    draft = root / "runs" / "active" / run_id / "draft-package.md"
    draft.write_text("TODO groundbreaking", encoding="utf-8")
    assert main(["--root", str(root), "verify", run_id]) == 0

    assert main(["--root", str(root), "approve", run_id]) == 2
    err = capsys.readouterr().err
    assert "Verification failed" in err


def test_cli_scan_inputs_writes_report(tmp_path: Path):
    root = tmp_path / "content-os"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "note.md").write_text("# Note\n\nProof", encoding="utf-8")
    report = tmp_path / "scan.md"

    assert main(["init", "--root", str(root)]) == 0
    assert (
        main(["--root", str(root), "scan-inputs", str(inputs), "--out", str(report)])
        == 0
    )

    text = report.read_text(encoding="utf-8")
    assert "Input Scan Report" in text
    assert "note.md" in text


def test_cli_list_adapters_includes_custom_rules(tmp_path: Path, capsys):
    root = tmp_path / "content-os"
    assert main(["init", "--root", str(root)]) == 0

    assert main(["--root", str(root), "list-adapters"]) == 0

    out = capsys.readouterr().out
    assert "rule\tautoingest-cypher-note" in out
    assert "built-in\tjson" in out
