"""Hermetic regression: the Scott anchor health guard holds under ``python -O``.

``python -O`` (equivalently ``PYTHONOPTIMIZE=1``) makes CPython delete
``assert`` statements at compile time. A guard written as a bare ``assert``
therefore fails open: ``check_anchor_health`` returns a fragmented count with
no error and ``scripts/check_anchor_health.py`` prints ``"ok": true`` and exits
0 even when four ``is_me`` GlobalSpeakers exist -- exactly the failure the
check exists to catch.

Every case spawns a tiny ``python -O`` child with an in-process stub for the
count helper. No live Neo4j, network, service, credential or file IO.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PROBE = textwrap.dedent("""\
    import sys
    root = sys.argv[1]
    sys.path.insert(0, root)
    import auto_ingest.speaker_health as sh
    sh.count_is_me_global_speakers = lambda: int(sys.argv[2])
    try:
        got = sh.check_anchor_health(expected=1)
    except AssertionError as ex:
        print("RAISED:" + str(ex))
    else:
        print("NO_RAISE:" + repr(got))
""")

_CLI_PROBE = textwrap.dedent("""\
    import sys, importlib.util
    root = sys.argv[1]
    sys.path.insert(0, root)
    import auto_ingest.speaker_health as sh
    sh.count_is_me_global_speakers = lambda: 4
    spec = importlib.util.spec_from_file_location(
        "anchor_cli", root + "/scripts/check_anchor_health.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sys.argv = ["check_anchor_health.py", "--expected", "1", "--json"]
    print("CLI_RC:" + str(cli.main()))
""")


def _child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(script: str, *args: object, optimize: bool = False):
    cmd = [sys.executable]
    if optimize:
        cmd.append("-O")
    cmd += ["-c", script, str(REPO_ROOT)] + [str(a) for a in args]
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=_child_env(),
        capture_output=True, text=True, timeout=30,
    )


def test_guard_raises_in_normal_mode():
    """Control: documented behaviour is unchanged without -O."""
    out = _run(_PROBE, 4)
    assert out.returncode == 0, out.stderr
    assert "RAISED:" in out.stdout, out.stdout


def test_guard_raises_under_optimize():
    """Core: -O must not disable the guard (FAIL pre-fix, PASS post-fix)."""
    out = _run(_PROBE, 4, optimize=True)
    assert out.returncode == 0, out.stderr
    assert "RAISED:" in out.stdout, out.stdout
    assert "found 4." in out.stdout, out.stdout


def test_healthy_count_passes_under_optimize():
    """Control: a genuinely healthy anchor is not over-blocked by the fix."""
    out = _run(_PROBE, 1, optimize=True)
    assert out.returncode == 0, out.stderr
    assert "NO_RAISE:1" in out.stdout, out.stdout


def test_cli_fail_closed_under_optimize():
    """End-to-end: the CLI must not report ok / exit 0 on a 4-node anchor."""
    out = _run(_CLI_PROBE, optimize=True)
    assert out.returncode == 0, out.stderr
    assert '"ok": false' in out.stdout, out.stdout
    assert "CLI_RC:1" in out.stdout, out.stdout
