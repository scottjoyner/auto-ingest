"""Hermetic tests for tools/branch_dedup_report.py.

The tests build a tiny throwaway git repo under pytest's ``tmp_path`` (pointed at
the SSD scratch area with TMPDIR when run here). Nothing touches the real fleet
repos, and no branch is ever fetched, merged, or pushed.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "branch_dedup_report.py"
PREFIX = "refs/heads/agent/"


@pytest.fixture(scope="module")
def tool():
    loader = SourceFileLoader("branch_dedup_report", str(TOOL_PATH))
    spec = importlib.util.spec_from_loader("branch_dedup_report", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["branch_dedup_report"] = module  # dataclasses resolves annotations here
    loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    """Run a git command in the synthetic repo with an isolated identity."""
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def commit_file(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text + "\n", encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-qm", f"add {name}")


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """Repo with one safe, one rebase-needed, one divergent, one contained ref."""
    repo = tmp_path / "dedup-repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit_file(repo, "base.txt", "base")
    base = git(repo, "rev-parse", "HEAD")

    # A ref still pointing at the old base is already contained once HEAD moves.
    git(repo, "branch", "agent/contained")

    # Advance HEAD so branches off the old base are "behind".
    commit_file(repo, "main.txt", "main-1")
    head = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "-qb", "agent/needs-rebase", base)
    commit_file(repo, "rebase.txt", "rebase-1")

    git(repo, "checkout", "-qb", "agent/diverged", base)
    commit_file(repo, "diverge1.txt", "diverge-1")
    commit_file(repo, "diverge2.txt", "diverge-2")

    # This one is based on the current HEAD: exactly one commit ahead, none behind.
    git(repo, "checkout", "-qb", "agent/keep-safe", head)
    commit_file(repo, "safe.txt", "safe-1")
    git(repo, "checkout", "-q", "main")
    return repo


def test_classifies_bands_and_excludes_contained_refs(tool, synthetic_repo: Path) -> None:
    report = tool.collect(synthetic_repo, "HEAD", PREFIX, 40)

    assert (report.enumerated, report.contained, report.unmerged) == (4, 1, 3)
    assert [(b.name, b.behind, b.ahead) for b in report.bands["A"]] == [
        (f"{PREFIX}keep-safe", 0, 1)
    ]
    assert [(b.name, b.behind, b.ahead) for b in report.bands["B"]] == [
        (f"{PREFIX}needs-rebase", 1, 1)
    ]
    assert [(b.name, b.behind, b.ahead) for b in report.bands["C"]] == [
        (f"{PREFIX}diverged", 1, 2)
    ]

    rendered = report.render()
    assert f"`{PREFIX}keep-safe`" in rendered
    assert f"{PREFIX}contained" not in rendered
    assert "| 0 | 1 |" in rendered
    assert "## Band B - rebase needed" in rendered
    assert "## Band C - divergent older lineage" in rendered


def test_limit_bounds_each_band(tool, synthetic_repo: Path) -> None:
    # Two more one-commit branches on HEAD -> Band A now has three members.
    head = git(synthetic_repo, "rev-parse", "HEAD")
    for suffix in ("extra-one", "extra-two"):
        git(synthetic_repo, "checkout", "-qb", f"agent/{suffix}", head)
        commit_file(synthetic_repo, f"{suffix}.txt", suffix)
    git(synthetic_repo, "checkout", "-q", "main")

    report = tool.collect(synthetic_repo, "HEAD", PREFIX, 2)
    assert len(report.bands["A"]) == 3
    assert report.omitted("A") == 1

    rendered = report.render()
    shown = [line for line in rendered.splitlines() if line.startswith("| ") and "`refs/" in line]
    safe_rows = [line for line in shown if "safe" in line or "extra-" in line]
    assert len(safe_rows) == 2
    assert f"{PREFIX}keep-safe" not in rendered  # ranked third by name, therefore omitted
    assert "Omitted from display: 1 (of 3" in rendered


def test_report_is_read_only(tool, synthetic_repo: Path, capsys) -> None:
    before = git(synthetic_repo, "for-each-ref", "--format=%(refname) %(objectname)")
    fetch_head = synthetic_repo / ".git" / "FETCH_HEAD"

    assert tool.main(["--repo", str(synthetic_repo), "--limit", "40"]) == 0
    captured = capsys.readouterr()

    after = git(synthetic_repo, "for-each-ref", "--format=%(refname) %(objectname)")
    assert after == before
    assert not fetch_head.exists()
    assert "# Unmerged agent branch dedup report" in captured.out
    assert captured.err == ""
