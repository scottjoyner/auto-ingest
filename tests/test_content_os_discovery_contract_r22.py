"""Bounded, deterministic local source discovery (r22).

Pins the contract of ``content_os.adapters.iter_supported_files`` /
``scan_path`` / ``scan_report``: directory scans stay inside their file budget,
ignored parts never spend that budget, ordering is stable, and the generated
scan report is byte-deterministic with digest-bound source IDs.

Hermetic: only tmp_path files and in-memory objects; no network, graph,
credential, mount, service or live database access.
"""

import hashlib
from pathlib import Path

from content_os.adapters import (
    extract_source,
    iter_supported_files,
    scan_path,
    scan_report,
)


def _write_note(directory: Path, name: str, text: str = "note") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_iter_supported_files_honors_the_file_budget(tmp_path: Path):
    names = [f"{i:02d}.txt" for i in range(5)]
    for name in reversed(names):
        _write_note(tmp_path, name)

    selected = iter_supported_files(tmp_path, limit=2)
    assert [path.name for path in selected] == names[:2]
    assert iter_supported_files(tmp_path, limit=0) == []
    assert len(iter_supported_files(tmp_path, limit=100)) == 5


def test_ignored_parts_do_not_consume_the_scan_budget(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()
    _write_note(tmp_path / ".git", "config.txt", "not a source")
    _write_note(tmp_path / "__pycache__", "cached.txt", "not a source")
    wanted = [_write_note(tmp_path, "a.txt"), _write_note(tmp_path, "b.txt")]

    selected = iter_supported_files(tmp_path, limit=2)

    assert selected == wanted
    assert not any(
        part in {".git", "__pycache__"}
        for path in selected
        for part in path.parts
    )


def test_iter_supported_files_order_is_stable_across_calls(tmp_path: Path):
    for name in ["z.txt", "a.txt", "m.txt", "b.txt"]:
        _write_note(tmp_path, name)

    first = iter_supported_files(tmp_path)
    second = iter_supported_files(tmp_path)

    assert first == second
    assert [path.name for path in first] == ["a.txt", "b.txt", "m.txt", "z.txt"]


def test_single_file_base_is_returned_even_with_zero_budget(tmp_path: Path):
    note = _write_note(tmp_path, "one.txt")

    assert iter_supported_files(note, limit=0) == [note]


def test_scan_path_propagates_limit_and_keeps_sorted_order(tmp_path: Path):
    for name in ["05.txt", "02.txt", "04.txt", "01.txt", "03.txt"]:
        _write_note(tmp_path, name, text=f"body of {name}")

    docs = scan_path(tmp_path, limit=3)

    assert [doc.path.name for doc in docs] == ["01.txt", "02.txt", "03.txt"]
    assert [doc.text for doc in docs] == [
        "body of 01.txt",
        "body of 02.txt",
        "body of 03.txt",
    ]
    assert all(doc.source_type == "text_note" for doc in docs)
    assert all(doc.sha256 for doc in docs)


def test_unreadable_source_stays_in_budget_without_stopping_scan(tmp_path: Path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    _write_note(tmp_path, "good.txt", "usable note")

    docs = scan_path(tmp_path, limit=2)
    by_name = {doc.path.name: doc for doc in docs}

    assert len(docs) == 2
    assert by_name["bad.json"].source_type == "unreadable"
    assert by_name["bad.json"].text == ""
    assert by_name["bad.json"].sha256 == ""
    assert any("Could not read source" in w for w in by_name["bad.json"].warnings)
    assert by_name["good.txt"].source_type == "text_note"
    assert by_name["good.txt"].text == "usable note"


def test_scan_report_is_byte_deterministic(tmp_path: Path):
    a = _write_note(tmp_path, "a.txt", "alpha")
    b = _write_note(tmp_path, "b.txt", "beta")
    docs = [extract_source(a), extract_source(b)]

    first = scan_report(tmp_path, docs)
    second = scan_report(tmp_path, docs)

    assert first == second
    assert first.index(str(a)) < first.index(str(b))
    assert first.index(docs[0].source_id) < first.index(docs[1].source_id)
    assert "# Input Scan Report" in first
    assert "No external platforms were scraped." in first


def test_scan_report_source_id_is_bound_to_exact_bytes(tmp_path: Path):
    original = _write_note(tmp_path, "first.txt", "identical bytes")
    copy = _write_note(tmp_path, "second.txt", "identical bytes")
    changed = _write_note(tmp_path, "third.txt", "identical byteZ")

    first = extract_source(original)
    copy_doc = extract_source(copy)
    changed_doc = extract_source(changed)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()

    assert first.sha256 == digest
    assert first.source_id == f"text_note:{digest[:12]}"
    assert copy_doc.source_id == first.source_id
    assert changed_doc.source_id != first.source_id
    report = scan_report(tmp_path, [first, copy_doc, changed_doc])
    assert report.count(first.source_id) == 2
    assert changed_doc.source_id in report


def test_empty_scan_report_is_stable(tmp_path: Path):
    report = scan_report(tmp_path, [])

    assert report == scan_report(tmp_path, [])
    assert "Scanned: `" in report
    assert "| Path | Source ID |" in report
    assert "No external platforms were scraped." in report
