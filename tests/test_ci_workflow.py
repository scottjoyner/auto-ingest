"""The CI workflow is code too, and it has broken in ways pytest never saw.

Two real incidents motivated this file:

  1. A `#` comment placed after a line-continuation backslash. In shell, a
     comment inside a continued command swallows the rest of the *command*, so
     the tokens after it were executed as their own command - the workflow tried
     to run `pydantic-settings` and died with exit 127. It cost a CI cycle
     because YAML parsed fine and every test passed locally.

  2. Missing/incorrect pins, which only surface on a machine with different
     peripherals than the author's.

These assertions are cheap and run in the same suite as everything else, so a
malformed workflow fails in the same place as a malformed test.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None,
                                reason="needs bash to parse the workflow's run blocks")


def run_blocks() -> list[tuple[str, str]]:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    out = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            block = step.get("run")
            if block:
                out.append((f"{job_name}: {step.get('name', '(unnamed)')}", block))
    assert out, "workflow has no run blocks - did the file get restructured?"
    return out


def test_the_workflow_is_valid_yaml():
    """Cheap, and the failure mode nobody reads the diff for otherwise."""
    assert isinstance(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")), dict)


@pytest.mark.parametrize("name, block", run_blocks(),
                         ids=[n for n, _ in run_blocks()])
def test_every_run_block_is_valid_shell(name, block):
    """Each step's script must parse as shell."""
    proc = subprocess.run(["bash", "-n"], input=block, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, f"{name}: {proc.stderr}"


@pytest.mark.parametrize("name, block", run_blocks(),
                         ids=[n for n, _ in run_blocks()])
def test_no_comment_swallows_a_continued_command(name, block):
    """A `#` after a `\\` continuation silently truncates the command.

    The remaining tokens get executed as separate commands, which is how the
    workflow ended up trying to run a package name as a program.
    """
    for i, line in enumerate(block.splitlines(), 1):
        if line.rstrip().endswith("\\"):
            nxt = block.splitlines()[i] if i < len(block.splitlines()) else ""
            assert not nxt.lstrip().startswith("#"), (
                f"{name} line {i + 1}: comment inside a continued command - "
                f"move it above the command")


@pytest.mark.parametrize("name, block", run_blocks(),
                         ids=[n for n, _ in run_blocks()])
def test_each_pip_install_is_a_single_command(name, block):
    """Catch continuation damage that bash -n alone would not flag."""
    script = "\n".join(
        re.sub(r"^(\s*)(\.venv/bin/)?pip install", r"\1echo ARGV:", line)
        for line in block.splitlines()
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=60)
    argv_lines = [line for line in proc.stdout.splitlines() if line.startswith("ARGV:")]
    if "pip install" not in block:
        assert not argv_lines, f"{name}: unexpected install parsing"
        return
    installs = [line for line in argv_lines if "ARGV: --upgrade" not in line]
    assert len(installs) >= 1, f"{name}: pip install produced no single command"
    for line in installs:
        pkgs = line[len("ARGV:"):].split()
        assert pkgs, f"{name}: empty package list"
        for pkg in pkgs:
            assert not pkg.startswith("#"), (
                f"{name}: a comment became a package argument: {pkg}")


def test_moviepy_is_pinned_to_the_version_the_code_expects():
    """The shorts code imports moviepy.editor, removed in moviepy 2.x.

    requirements.txt pins 1.0.3; CI previously said ">=1.0" and resolved to 2.x,
    so CI tested a major version the code does not support while the real venv
    passed. Keep the two in step.
    """
    requirements = (WORKFLOW.parents[2] / "requirements.txt").read_text(encoding="utf-8")
    pinned = re.search(r"^moviepy==([0-9.]+)", requirements, re.M)
    assert pinned, "requirements.txt no longer pins moviepy exactly"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert f'moviepy=={pinned.group(1)}' in workflow, (
        f"CI must install moviepy=={pinned.group(1)} to match requirements.txt")


def test_ci_installs_pillow():
    """shorts/abtest.py imports PIL at module scope, unguarded by its test.

    Without Pillow, collection dies before a single test runs - exit 2 in under
    two seconds, which reads like a broken runner rather than a missing dep.
    Not pulled in by opencv-python-headless, so it must be named explicitly.
    """
    workflow = WORKFLOW.read_text(encoding="utf-8")
    install_lines = [line for line in workflow.splitlines() if "pip install" in line]
    # every install that runs the suite needs Pillow, including the moviepy venv
    assert len(install_lines) >= 2
    blocks = [b for _, b in run_blocks() if "pip install" in b and "moviepy" not in b]
    for block in blocks:
        assert "Pillow" in block, "a CI environment running pytest lacks Pillow"
