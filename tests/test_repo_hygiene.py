"""Repo hygiene guards: importing code must not walk the filesystem, and running
the suite must not rewrite files that live in the repo.

Both of these were invisible to the suite. `yolo_vehicle_detction.py` called
`list_directories()` at module scope, so *importing* it walked the dashcam share
-- a no-op only because /nas/fileserver/dashcam happens to be unmounted on the
machine that runs the tests, and a full recursive walk (plus a YOLO load per clip)
on the machine that owns the share. `scripts/kg_health_watchdog.py` had no way to
redirect its state file, so any test that exercised `main()` wrote into
`scripts/.kg_health_state.json`, and every run of the suite left a tracked file
dirty.

The invariant these tests pin down: a module body may declare constants and
functions and nothing else, and a test that exercises a writing code path must be
able to aim that write at `tmp_path`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest.mock as mock
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
KG_STATE = REPO / "scripts" / ".kg_health_state.json"
KG_WATCHDOG = REPO / "scripts" / "kg_health_watchdog.py"
YOLO_VEHICLE = REPO / "yolo_vehicle_detction.py"

# yolo_vehicle_detction.py imports the ML stack at module scope. Same stubs the
# sibling suites use (auto_ingest/tests/test_ml_pure.py) so the module loads
# without torch / ultralytics / moviepy. No inference ever runs here either.
_ML_STUBS = {
    "torch": mock.MagicMock(),
    "ultralytics": mock.MagicMock(),
    "ultralytics.utils": mock.MagicMock(),
    "ultralytics.utils.plotting": mock.MagicMock(),
    "cv2": mock.MagicMock(),
    "moviepy": mock.MagicMock(),
    "moviepy.editor": mock.MagicMock(),
    "PIL": mock.MagicMock(),
}


def _load_from_path(path: Path, name: str):
    """Execute `path` under `name`, bypassing the finder entirely.

    Going through spec_from_file_location means the import machinery never has to
    list a directory to locate the module, which is what lets the caller assert
    that the module body enumerated nothing.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _forbid(*names: str):
    """Build an os-function replacement that fails loudly, naming what called it."""

    def make(name: str):
        def forbidden(*args, **kwargs):
            raise AssertionError(
                f"importing the module called os.{name}(); module bodies must not "
                f"touch the filesystem"
            )

        return forbidden

    return {name: make(name) for name in names}


# ---------------------------------------------------------------------------
# yolo_vehicle_detction.py: importing it must not enumerate any directory
# ---------------------------------------------------------------------------
@pytest.fixture
def yolo_real_deps():
    """Import the module's genuine dependencies while enumeration still works.

    os.listdir/os.scandir are how importlib's FileFinder populates its directory
    cache, so they have to be replaced only after the real dependencies are in
    sys.modules and the module under test is located by exact path.
    """
    import auto_ingest.backend  # noqa: F401
    import auto_ingest_config  # noqa: F401


def test_importing_yolo_vehicle_detction_enumerates_no_directory(
    monkeypatch, yolo_real_deps
):
    """Regression: the module body used to call list_directories() at import.

    Before the fix this raised os.walk()/os.listdir() from inside exec_module,
    because the last two lines of the file ran the walk unconditionally. That
    never showed up as a failure on the test host -- os.walk over an unmounted
    share yields nothing -- so the only way to pin the invariant down is to make
    enumeration itself fatal and assert the import still completes.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for name, forbidden in _forbid("walk", "listdir", "scandir").items():
        monkeypatch.setattr(os, name, forbidden)

    with mock.patch.dict(sys.modules, _ML_STUBS):
        mod = _load_from_path(YOLO_VEHICLE, "yolo_vehicle_detction_hygiene")

    # The walk now lives behind main(), and the guard is what keeps it there.
    assert callable(mod.main)
    assert callable(mod.list_directories)


def test_yolo_vehicle_detction_entry_point_is_guarded():
    """The script contract is `python3 yolo_vehicle_detction.py` (runall.sh)."""
    source = YOLO_VEHICLE.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    # and the walk must be reachable only from main()
    assert source.index("def main():") < source.index("list_directories(base_directory)")


def test_yolo_vehicle_detction_is_not_imported_for_its_side_effect():
    """No module may rely on the old import-time walk. Only script callers exist."""
    callers = []
    for candidate in sorted(REPO.glob("*.py")) + sorted((REPO / "auto_ingest").rglob("*.py")):
        if candidate == YOLO_VEHICLE:
            continue
        if "import yolo_vehicle_detection" in candidate.read_text(encoding="utf-8", errors="ignore"):
            callers.append(candidate.relative_to(REPO).as_posix())
    # Note: the module name is misspelled on disk ("detction"), so nothing can
    # import it under the corrected spelling anyway.
    assert callers == []


# ---------------------------------------------------------------------------
# scripts/kg_health_watchdog.py: the writing path must be redirectable
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FakeIngestedAt:
    """Stands in for a neo4j DateTime -- the watchdog calls .to_native()."""

    seconds: float

    def to_native(self) -> datetime:
        return datetime.fromtimestamp(self.seconds, tz=timezone.utc)


class _FakeSession:
    """Answers the watchdog's four single-column queries, in any order."""

    def __init__(self, now: float):
        self._answers = (
            ("max(p.ingested_at)", {"c": _FakeIngestedAt(now)}),
            ("p.embedding_768", {"c": 7}),
            ("method='vector'", {"c": 3}),
            ("SignalMessage", {"c": 0}),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def run(self, cypher):
        for needle, row in self._answers:
            if needle in cypher:
                return SimpleNamespace(single=lambda: row)
        raise AssertionError(f"unrecognized watchdog query: {cypher}")


class _FakeDriver:
    def __init__(self, now: float):
        self._now = now
        self.closed = False

    def session(self, database=None):
        return _FakeSession(self._now)

    def close(self):
        self.closed = True


def _read_kg_state() -> bytes | None:
    return KG_STATE.read_bytes() if KG_STATE.exists() else None


def _load_watchdog(env_state_file: Path | None, monkeypatch):
    """Load a fresh copy of the watchdog with its state file aimed elsewhere.

    Takes the monkeypatch fixture so the env change is undone even if the module
    raises during import. A bare os.environ assignment leaks into every later
    test, and because STATE_FILE is read at IMPORT time, a leak silently changes
    a subsequent test's subject rather than failing loudly.
    """
    monkeypatch.delenv("KG_HEALTH_STATE_FILE", raising=False)
    if env_state_file is not None:
        monkeypatch.setenv("KG_HEALTH_STATE_FILE", str(env_state_file))
    return _load_from_path(KG_WATCHDOG, "kg_health_watchdog_hygiene")


def test_kg_health_watchdog_run_leaves_the_repo_state_file_byte_identical(
    tmp_path, monkeypatch, capsys
):
    """Regression: main() wrote into scripts/.kg_health_state.json.

    The watchdog only has one writer, so a test that reaches the write is a test
    that must be able to aim it somewhere. Compare bytes rather than shelling out
    to git -- the point is that the file on disk did not change, and that is true
    whether or not the path happens to be tracked.
    """
    redirected = tmp_path / "kg_health_state.json"
    before = _read_kg_state()

    monkeypatch.setattr(sys, "argv", ["kg_health_watchdog.py"])
    kg = _load_watchdog(redirected, monkeypatch)
    assert kg.STATE_FILE == redirected, "the override did not take effect"

    driver = _FakeDriver(now=datetime.now(tz=timezone.utc).timestamp())
    monkeypatch.setattr(kg, "get_neo4j", lambda: (driver, "neo4j"))
    kg.main()
    capsys.readouterr()

    # The real code path ran: the state was written, atomically, to the tmp path.
    assert json.loads(redirected.read_text())["emb768_count"] == 7
    assert driver.closed, "main() must still close the driver"
    # And the repo's own copy is untouched.
    assert _read_kg_state() == before


def test_kg_health_state_file_default_is_unchanged(tmp_path, monkeypatch):
    """The operator's baseline must keep landing where cron expects it."""
    monkeypatch.delenv("KG_HEALTH_STATE_FILE", raising=False)
    kg = _load_watchdog(None, monkeypatch)
    assert kg.STATE_FILE == REPO / "scripts" / ".kg_health_state.json"


def test_kg_health_save_state_is_atomic_and_leaves_no_residue(tmp_path, monkeypatch):
    """A failed write must not truncate the state file or strand a temp file.

    save_state() used Path.write_text(), which truncates the target in place: a
    kill between truncate and write leaves invalid JSON. (load_state() swallows
    that and returns {}, so the watchdog silently forgot its baseline.) The fix
    writes a uniquely-named temp file and os.replace()s it into place, so the
    file on disk is always the previous complete state or the new complete state.
    """
    state = tmp_path / "state.json"
    kg = _load_watchdog(state, monkeypatch)

    kg.save_state({"emb768_count": 7, "emb768_time": 1.0})
    assert json.loads(state.read_text()) == {"emb768_count": 7, "emb768_time": 1.0}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    good = state.read_bytes()
    with mock.patch.object(kg.os, "fsync", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            kg.save_state({"emb768_count": 99})
    assert state.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file stranded"


def test_kg_health_save_state_uses_a_unique_temp_name_per_write(tmp_path, monkeypatch):
    """A fixed ".tmp" suffix is the bug _write_json_atomic exists to avoid.

    scripts/neo4j_watchdog.py still has it (see _save_last_action). If the
    unique-per-write name regresses here too, two watchdogs on one box can
    silently lose one update.
    """
    state = tmp_path / "state.json"
    kg = _load_watchdog(state, monkeypatch)

    seen = []
    real_replace = os.replace

    def spy(src, dst, *args, **kwargs):
        seen.append(Path(src).name)
        return real_replace(src, dst, *args, **kwargs)

    with mock.patch.object(kg.os, "replace", side_effect=spy):
        kg.save_state({"n": 1})
        kg.save_state({"n": 2})

    assert len(set(seen)) == 2, f"temp name reused across writes: {seen}"
    assert all(name.startswith("state.json.") for name in seen), seen


def test_kg_health_state_file_is_gitignored_as_runtime_state():
    """The file is a cached observation, so git must never pick it up again."""
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "scripts/.kg_health_state.json" in ignore, (
        "scripts/.kg_health_state.json is rewritten in full on every watchdog run; "
        "untracked-but-unignored it reappears in git status as an untracked file"
    )


def test_runtime_cursors_are_not_tracked():
    """Runtime state must not be tracked, or every run dirties the tree.

    `.kg_health_state.json` and `.arxiv_kg_cursor.json` are both caches rewritten
    in full on every run, and nothing but their own watchdog/bridge reads them.
    Tracked once by accident, they make `git add -A` a way to commit a wall clock.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for name in ("scripts/.kg_health_state.json", "scripts/.arxiv_kg_cursor.json"):
        proc = subprocess.run(["git", "ls-files", "--error-unmatch", name],
                              cwd=repo, capture_output=True, timeout=30)
        assert proc.returncode != 0, f"{name} is tracked; it is runtime state"
        proc = subprocess.run(["git", "check-ignore", "-q", name],
                              cwd=repo, capture_output=True, timeout=30)
        assert proc.returncode == 0, f"{name} is not gitignored"


def test_the_shared_atomic_helper_is_reachable_as_a_subpackage():
    """`auto_ingest/util` must be importable and advertised.

    The scripts import `auto_ingest.util.atomic`, but `auto_ingest.__all__`
    omitted `util`, so the one dependency-free subpackage in the repo was the one
    a reader could not discover from the package root.
    """
    import auto_ingest

    assert "util" in auto_ingest.__all__, (
        "auto_ingest.__all__ should list util")
    from auto_ingest.util.atomic import write_json_atomic
    assert callable(write_json_atomic)
