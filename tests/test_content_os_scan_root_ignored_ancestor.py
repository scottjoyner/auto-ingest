"""Regression: `scan-inputs` must not silently omit files because an *ancestor*
of the requested scan root happens to share a name with IGNORED_PARTS.

`iter_supported_files` prunes ignored directories inside the scan tree. It used
to apply that check to `path.parts` (the whole absolute path), so a root such as
``/work/build/notes`` made every file invisible while scanning that same file
directly still worked. The generated scan report then came back empty with no
explanation — a silent omission.
"""
from pathlib import Path

from content_os import adapters, core


def _root_with_ignored_ancestor(tmp_path: Path) -> Path:
    # "build" is in content_os.adapters.IGNORED_PARTS. It is an ancestor of the
    # scan root, not a directory inside it, so it must not prune the scan.
    scan_root = tmp_path / "build" / "notes"
    scan_root.mkdir(parents=True)
    (scan_root / "important.md").write_text("# Important\n\nProof.", encoding="utf-8")
    return scan_root


def test_iter_supported_files_keeps_files_under_ignored_ancestor(tmp_path: Path):
    scan_root = _root_with_ignored_ancestor(tmp_path)
    expected = scan_root / "important.md"

    assert adapters.iter_supported_files(scan_root) == [expected]


def test_scan_inputs_report_includes_file_under_ignored_ancestor(tmp_path: Path):
    scan_root = _root_with_ignored_ancestor(tmp_path)
    content_root = tmp_path / "content-os"
    report_path = tmp_path / "scan.md"

    docs, written = core.scan_inputs(content_root, scan_root, out=report_path)

    assert [doc.path for doc in docs] == [scan_root / "important.md"]
    assert written == report_path
    report = report_path.read_text(encoding="utf-8")
    assert "important.md" in report
    assert "markdown_note" in report


def test_ignored_directories_inside_scan_root_are_still_pruned(tmp_path: Path):
    """The fix must narrow the check, not remove it: nested ignored dirs stay out."""
    scan_root = tmp_path / "notes"
    scan_root.mkdir()
    (scan_root / "keep.md").write_text("kept", encoding="utf-8")
    for ignored in ("__pycache__", "build", ".git"):
        nested = scan_root / ignored
        nested.mkdir()
        (nested / "hidden.md").write_text("hidden", encoding="utf-8")

    found = adapters.iter_supported_files(scan_root)

    assert found == [scan_root / "keep.md"]
