"""Regression for case-insensitive unresolved-placeholder detection in verify().

The verification gate is the last deterministic check before human approval.
It must reject a draft that still contains an unfinished ``TODO`` marker in any
case: ``TODO``, ``Todo`` or ``todo`` are equally unfinished. Before the fix the
gate used a case-sensitive ``"TODO" in text`` check, so a lowercase ``todo``
placeholder was marked PASS and ``approve()`` (without ``--force``) succeeded.

Hermetic: only tmp_path files and pure Python. No LLM, network, credentials,
Neo4j, service, clock or platform API is touched.
"""
from __future__ import annotations

from pathlib import Path

from content_os import core


def _make_run(tmp_path: Path, draft: str, rid: str = "r1") -> tuple[Path, Path]:
    root = tmp_path / "content-os"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("score_threshold: 0\n", encoding="utf-8")
    rundir = root / "runs" / "active" / rid
    rundir.mkdir(parents=True)
    (rundir / "content-object.yaml").write_text(
        "id: r1\ntitle: T\nslug: t\n", encoding="utf-8"
    )
    (rundir / "draft-package.md").write_text(draft, encoding="utf-8")
    return root, rundir


def _report(rundir: Path) -> str:
    return (rundir / "verification.md").read_text(encoding="utf-8")


def test_case_insensitive_placeholder_helper(tmp_path) -> None:
    """The helper exists and treats every case variant as a placeholder."""
    helper = getattr(core, "has_unresolved_placeholder", None)
    assert helper is not None, "case-insensitive placeholder helper is missing"
    assert helper("TODO: fill this in") is True
    assert helper("Todo: fill this in") is True
    assert helper("todo: fill this in") is True
    assert helper("a draft with no placeholder") is False


def test_lowercase_todo_placeholder_fails_verify_gate(tmp_path) -> None:
    """A lowercase ``todo`` must hold the gate (the original defect)."""
    root, rundir = _make_run(
        tmp_path, "A short draft with a todo for the author.\n"
    )
    core.verify(root, "r1")
    assert core.verification_passed(rundir) is False
    assert "## pass/fail\nFAIL" in _report(rundir)
    assert "- Draft still contains TODO placeholders." in _report(rundir)


def test_mixed_case_todo_placeholder_fails_verify_gate(tmp_path) -> None:
    """A mixed-case ``Todo`` must hold the gate too."""
    root, rundir = _make_run(
        tmp_path, "A short draft with a Todo for the author.\n"
    )
    core.verify(root, "r1")
    assert core.verification_passed(rundir) is False
    assert "- Draft still contains TODO placeholders." in _report(rundir)


def test_uppercase_todo_still_fails_verify_gate(tmp_path) -> None:
    """Control: the original uppercase marker was already rejected."""
    root, rundir = _make_run(
        tmp_path, "A short draft with a TODO for the author.\n"
    )
    core.verify(root, "r1")
    assert core.verification_passed(rundir) is False


def test_clean_draft_still_passes_verify_gate(tmp_path) -> None:
    """Control: the case-insensitive check must not reject a finished draft."""
    root, rundir = _make_run(
        tmp_path, "A short draft that is complete and ready.\n"
    )
    core.verify(root, "r1")
    assert core.verification_passed(rundir) is True
    assert "## pass/fail\nPASS" in _report(rundir)
    assert "- Draft still contains TODO placeholders." not in _report(rundir)
