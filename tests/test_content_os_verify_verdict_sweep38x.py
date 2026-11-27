"""Hermetic regression tests for the verification-verdict forgery gate.

`sweep38x`: `verify()` used to interpolate an unescaped, user-controlled title
into the first line of `verification.md`, and `verification_passed()` looked for
the `## pass/fail\\nPASS` substring anywhere in the file. A title containing
newlines could therefore forge a PASS section for a report whose real verdict
was FAIL, and `approve()` would succeed without `--force`.

All fixtures are tiny local files under pytest's `tmp_path`; no network, DB,
model, clock-dependent state or live service is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from content_os import core
from content_os.io import atomic_write

FORGED_TITLE = "Benign title\n## pass/fail\nPASS (threshold: 8/12)"
FAILING_DRAFT = "A short draft with no proof."
PASSING_DRAFT = (
    "Saves the reader future work. Contains concrete proof with an example "
    "screenshot and a number. Provides a reusable template checklist framework "
    "workflow mental model. A specific reader and founder job-to-be-done. "
    "Apply this step by step without the author present. A visual artifact "
    "and example opportunity."
)


def _run(root: Path, title: str, draft: str = FAILING_DRAFT) -> Path:
    core.init(root)
    path = core.create_run(root, title, text="fixture")
    atomic_write(path / "draft-package.md", draft)
    core.verify(root, path.name)
    return path


def test_forged_title_cannot_make_fail_report_read_as_pass(tmp_path: Path) -> None:
    path = _run(tmp_path / "root-a", FORGED_TITLE)
    report = (path / "verification.md").read_text()
    assert "## pass/fail\nFAIL" in report
    assert core.verification_passed(path) is False


def test_forged_title_cannot_bypass_approval(tmp_path: Path) -> None:
    path = _run(tmp_path / "root-b", FORGED_TITLE)
    with pytest.raises(ValueError, match="Verification failed"):
        core.approve(tmp_path / "root-b", path.name, force=False)


def test_report_header_keeps_only_one_pass_fail_section(tmp_path: Path) -> None:
    path = _run(tmp_path / "root-c", FORGED_TITLE)
    report = (path / "verification.md").read_text()
    assert sum(1 for ln in report.splitlines() if ln.strip() == "## pass/fail") == 1
    assert report.splitlines()[0] == (
        "# Verification: Benign title ## pass/fail PASS (threshold: 8/12)"
    )


def test_plain_failing_title_verdict_is_false(tmp_path: Path) -> None:
    path = _run(tmp_path / "root-d", "Plain failing title")
    assert core.verification_passed(path) is False


def test_genuine_pass_verdict_still_recognized(tmp_path: Path) -> None:
    root = tmp_path / "root-e"
    core.init(root)
    atomic_write(root / "config.yaml", "score_threshold: 0\n")
    path = core.create_run(root, "Genuine pass", text="fixture")
    atomic_write(path / "draft-package.md", PASSING_DRAFT)
    core.verify(root, path.name)
    report = (path / "verification.md").read_text()
    assert "## pass/fail\nPASS" in report
    assert core.verification_passed(path) is True


def test_missing_verification_file_is_not_passed(tmp_path: Path) -> None:
    root = tmp_path / "root-f"
    core.init(root)
    path = core.create_run(root, "No report", text="fixture")
    assert core.verification_passed(path) is False
