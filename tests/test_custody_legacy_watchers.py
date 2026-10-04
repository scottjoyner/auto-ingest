"""Legacy watcher protection.

The custody state machine must stand entirely on its own evidence contract: no
legacy autonomous SD/NAS watcher, no historical host mount, no scheduler, no
network, no database. These tests are the executable form of the disposition
recorded in docs/sd-card-campaign-custody.md.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from custody_helpers import CARD_01_BUNDLE, campaign, evidence, strict_policy

import auto_ingest.custody as custody_pkg
from auto_ingest.custody import derive_state

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTODY_DIR = Path(custody_pkg.__file__).parent
DISPOSITION_DOC = REPO_ROOT / "docs" / "sd-card-campaign-custody.md"

#: Autonomous / legacy ingest machinery that this slice must not depend on.
LEGACY_WATCHER_LITERALS = (
    "sdcard-nas-ingest-watcher",
    "untitled-sd-ingest",
    "dashcam_copy.sh",
    "bodycam_copy.sh",
    "audio_copy.sh",
    "ingest_supervisor.py",
    "sync_cache_to_nas5.sh",
    "runall.sh",
)

#: Historical destinations that must never be embedded in core custody logic.
HISTORICAL_DESTINATION_LITERALS = (
    "NAS3",
    "NAS5",
    "SSD_4TB",
    "/media/scott/",
    "/mnt/8TB",
)

BANNED_IMPORTS = {
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "neo4j",
    "sqlite3",
    "paramiko",
    "inotify",
    "watchdog",
    "pyudev",
    "schedule",
    "apscheduler",
}

PURE_MODULES = ("states.py", "policy.py", "campaign.py", "destination.py",
                "evidence.py", "machine.py", "release.py", "planner.py", "report.py",
                "ledger.py")


def _module_paths():
    return sorted(CUSTODY_DIR.glob("*.py"))


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# code-level independence
# ---------------------------------------------------------------------------
def test_custody_package_never_names_a_legacy_watcher():
    for path in _module_paths():
        text = _source(path)
        for literal in LEGACY_WATCHER_LITERALS:
            assert literal not in text, f"{path.name} references legacy {literal}"


def test_custody_package_never_embeds_a_historical_destination():
    for path in _module_paths():
        text = _source(path)
        for literal in HISTORICAL_DESTINATION_LITERALS:
            assert literal not in text, f"{path.name} hardcodes historical {literal}"


def test_custody_package_imports_nothing_that_could_autonomously_run():
    for path in _module_paths():
        tree = ast.parse(_source(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in BANNED_IMPORTS, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in BANNED_IMPORTS, f"{path.name} imports {node.module}"


@pytest.mark.parametrize("name", PURE_MODULES)
def test_pure_modules_touch_no_mutating_syscall(name):
    """`ledger.py` joins them: it is a pure reader, so it must never write."""
    path = CUSTODY_DIR / name
    tree = ast.parse(_source(path))
    # `replace` is excluded: in these modules it is always dataclasses.replace.
    banned_calls = {"remove", "unlink", "rmtree", "copyfile", "copy2", "copytree",
                    "move", "system", "popen", "check_call", "check_output",
                    "mkdir", "makedirs", "rmdir", "rename",
                    "write_text", "write_bytes", "truncate", "chmod", "chown", "utime"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert called not in banned_calls, f"{name} calls {called}()"
    text = _source(path)
    for banned in ("os.replace(", "os.rename(", "os.remove(", "shutil.", "subprocess."):
        assert banned not in text, f"{name} references {banned}"


def test_ledger_module_only_ever_opens_for_reading():
    text = _source(CUSTODY_DIR / "ledger.py")
    write_modes = ['"w"', '"a"', '"x"', '"+"', "'w'", "'a'", "'x'", "'+'"]
    for mode in write_modes:
        assert f", {mode}" not in text, f"ledger.py opens in {mode} mode"
    assert "Path.open" not in text and ".write_text" not in text


# ---------------------------------------------------------------------------
# state does not depend on any watcher
# ---------------------------------------------------------------------------
def test_state_is_derived_without_any_watcher_input():
    """A brand new bundle derives its state with no external actor involved."""
    derivation = derive_state(campaign(), evidence(), strict_policy())
    assert derivation.state.value == "DISCOVERED"
    assert derivation.reasons == ("no_inventory_evidence",)


def test_card01_derives_without_a_watcher_and_without_live_storage():
    status = __import__("auto_ingest.custody.store", fromlist=["load_status"]).load_status(
        CARD_01_BUNDLE
    )
    assert status.state.value == "RECONCILE_REQUIRED"
    assert status.source_release_allowed is False


def test_no_udev_or_watcher_triggers_are_added_by_this_slice():
    """No unit/rule files added: the custody machine cannot self-start."""
    forbidden_suffixes = (".rules", ".path", ".timer", ".mount")
    for path in CUSTODY_DIR.rglob("*"):
        assert path.suffix not in forbidden_suffixes, path


def test_exactly_four_modules_write_and_nothing_else():
    """store (evidence), executor (bytes), hashing + verify (their ledgers)."""
    writers = {
        path.name for path in _module_paths()
        if "os.replace(" in _source(path) or '"w", encoding' in _source(path)
        or '"a", encoding' in _source(path) or '"xb"' in _source(path)
    }
    assert writers == {"store.py", "executor.py", "hashing.py", "verify.py"}, writers

    # hashing and verify append only inside their own bundle's ledgers dir
    for name in ("hashing.py", "verify.py"):
        text = _source(CUSTODY_DIR / name)
        assert "ledger_dir(" in text, name
        for banned in ("os.remove", "os.rmdir", "shutil", "subprocess"):
            assert banned not in text, f"{name} references {banned}"


def test_the_executor_has_no_bulk_delete_primitive():
    """It may clean up its own temp file; it must have no way to remove a tree."""
    executor = _source(CUSTODY_DIR / "executor.py")
    for banned in ("shutil", "os.rmdir", "os.removedirs", "rmtree",
                   "os.chmod", "os.truncate"):
        assert banned not in executor, banned
    assert 'source.open("rb")' in executor
    assert "TEMP_DIRNAME" in executor

    source = _source(CUSTODY_DIR / "store.py")
    # every write goes through the single atomic helper
    assert source.count("def _write_json_atomic") == 1
    assert source.count("_write_json_atomic(") == 3  # def + import-free: 2 call sites
    for fn in ("new_campaign", "import_evidence"):
        assert f"def {fn}(" in source
    # and both are gated
    assert "apply: bool = False" in source


def test_writing_commands_are_the_only_ones_taking_a_clock():
    """`status`/`plan`/`verify` must stay clock-free or they stop being idempotent."""
    for name in ("machine.py", "planner.py", "release.py", "report.py", "evidence.py"):
        text = _source(CUSTODY_DIR / name)
        for banned in ("datetime.now", "time.time", "utcnow", "uuid4", "random"):
            assert banned not in text, f"{name} reads the clock/randomness ({banned})"


# ---------------------------------------------------------------------------
# the disposition is documented, not just asserted
# ---------------------------------------------------------------------------
def test_disposition_doc_exists_and_classifies_each_legacy_watcher():
    text = DISPOSITION_DOC.read_text(encoding="utf-8")
    for literal in ("sdcard-nas-ingest-watcher", "untitled-sd-ingest"):
        assert literal in text, f"{literal} disposition is undocumented"
    assert "DEAD_REFERENCE" in text or "DEAD / UNTRACKED" in text
    assert "MANUAL_ONLY" in text


def test_disposition_doc_states_the_operator_loop():
    text = DISPOSITION_DOC.read_text(encoding="utf-8")
    for step in ("custody status", "custody plan", "custody verify"):
        assert step in text
