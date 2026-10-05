"""Legacy watcher protection.

The custody state machine must stand entirely on its own evidence contract: no
legacy autonomous SD/NAS watcher, no historical host mount, no scheduler, no
network, no database. These tests are the executable form of the disposition
recorded in docs/sd-card-campaign-custody.md.
"""
from __future__ import annotations

import ast
import io
import tokenize
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

#: The single custody module permitted to unlink a source object. Every other
#: module in the package is held to a total ban; see
#: test_release_source_is_the_only_module_that_may_unlink_anything.
RELEASE_SOURCE_MODULE = "release_source.py"

#: Call names that remove something. `unlink` is the one narrow form any custody
#: module may ever call, and only in RELEASE_SOURCE_MODULE - everything else in
#: this set is a bulk or in-place destruction and appears nowhere in the package.
DELETION_CALL_NAMES = frozenset({
    "unlink", "remove", "removefile", "removedirs", "rmdir", "rmtree", "truncate",
})

#: Substrings that would reintroduce a tree-delete, in ANY custody module. Checked
#: against code with comments and docstrings removed, so a module is still allowed
#: to *say* what it refuses to do.
TREE_DELETE_LITERALS = (
    "shutil",
    "rmtree",
    "os.rmdir",
    "os.removedirs",
    "os.remove",
    "os.unlink",
    "os.chmod",
    "os.truncate",
)

#: Every unlink receiver in the package, pinned exactly. Four of these predate
#: source release and are all module-owned transient artefacts, so none of them
#: can be reached by a caller-supplied or ledger-derived path. The fifth - and only
#: the fifth - is a path resolved from a ledger key under a campaign source root.
PERMITTED_UNLINK_SITES = {
    "executor.py": ["tmp", "tmp"],
    "lock.py": ["active_marker(root)"],
    "release_source.py": ["Path(obj.path)"],
    "store.py": ["tmp"],
}

#: Receivers a module may unlink without asking the release gate, because the
#: module created them itself in the same operation. Anything else is a new
#: capability.
MODULE_OWNED_ARTEFACTS = frozenset({"tmp", "active_marker(root)"})

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


def _code_only(source: str) -> str:
    """The module with comments and string literals removed.

    A prose mention of ``shutil.rmtree`` in a docstring is documentation, not a
    capability. Checking raw text would force this package to be unable to explain
    what it refuses to do - and an unexplainable refusal is one nobody trusts.
    """
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def _deletion_calls(tree: ast.AST) -> list[tuple[str, str]]:
    """Every call in ``tree`` whose function name removes something.

    Returned as ``(callee, receiver_source)``, sorted; the receiver is ``""`` for a
    bare name. Resolved through the AST rather than grepped, so ``shutil.rmtree``,
    ``rmtree``, ``os.unlink(p)`` and ``p.unlink()`` are all seen - and a bare
    ``p.unlink()`` cannot hide behind a substring that happens to be spelled
    differently.
    """
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name, receiver = func.attr, ast.unparse(func.value)
        elif isinstance(func, ast.Name):
            name, receiver = func.id, ""
        else:
            continue
        if name in DELETION_CALL_NAMES:
            found.append((name, receiver))
    return sorted(found)


def _receivers(path: Path) -> list[str]:
    return sorted(receiver for _, receiver in _deletion_calls(ast.parse(_source(path))))


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
    """store (evidence), executor (bytes + where they land), hashing + verify
    (their ledgers).

    `executor` grew a second ledger when the staged layout became a recorded
    decision that `execute` obeys. It is deliberately not a fifth module, and
    `ledger.py` stays read-only: that module owns formats, not writes. The
    layout record went where the bytes go, which is where a reader looking for
    "where was this object told to land" will look first.
    """
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


def test_the_planner_writes_nothing():
    """Staging is the module that decides where every byte goes. It must stay a
    pure function of the source tree, or "stage is read-only" is a claim."""
    staging = _source(CUSTODY_DIR / "staging.py")
    for banned in ('open(', "os.replace(", '"w", encoding', '"a", encoding'):
        assert banned not in staging, f"staging.py references {banned}"

    # hashing and verify append only inside their own bundle's ledgers dir
    for name in ("hashing.py", "verify.py"):
        text = _source(CUSTODY_DIR / name)
        assert "ledger_dir(" in text, name
        for banned in ("os.remove", "os.rmdir", "shutil", "subprocess"):
            assert banned not in text, f"{name} references {banned}"


def test_the_executor_has_no_bulk_delete_primitive():
    """Unchanged intent: the byte-mover has no way to remove a tree.

    This assertion was NOT weakened when source deletion became possible
    elsewhere. The executor may still clean up its own temp file and nothing else:
    it copies bytes, and a copy step that could also delete would make an
    interrupted run unrecoverable.
    """
    executor = _source(CUSTODY_DIR / "executor.py")
    for banned in ("shutil", "os.rmdir", "os.removedirs", "rmtree",
                   "os.chmod", "os.truncate"):
        assert banned not in executor, banned
    assert 'source.open("rb")' in executor
    assert "TEMP_DIRNAME" in executor

    # And through the AST, so the same ban holds for a call this test has never
    # heard of: executor.py may call `unlink` on a path, never `remove`/`rmdir`/
    # `rmtree`, and never through `os` or `shutil`.
    for callee, receiver in _deletion_calls(ast.parse(executor)):
        assert callee == "unlink", f"executor.py calls {callee}()"
        assert "os" not in receiver and "shutil" not in receiver, receiver

    source = _source(CUSTODY_DIR / "store.py")
    # every write goes through the single atomic helper
    assert source.count("def _write_json_atomic") == 1
    assert source.count("_write_json_atomic(") == 3  # def + import-free: 2 call sites
    for fn in ("new_campaign", "import_evidence"):
        assert f"def {fn}(" in source
    # and both are gated
    assert "apply: bool = False" in source


def test_release_source_is_the_only_module_that_may_unlink_a_source_object():
    """Every unlink call site in the package, pinned exactly. One is new.

    Before source release existed the package contained four unlink sites and all
    four were module-owned transient artefacts: an executor's `.partial`, the
    atomic writer's `.tmp`, the legacy-sync marker. None of them can be reached by
    a caller-supplied or ledger-derived path. `release_source.py` adds the fifth,
    and it is the only one whose target is resolved from a ledger key under a
    campaign source root.

    Pinning the whole map - rather than "executor has none" - is what makes this
    the *narrowing* it is meant to be: a sixth unlink anywhere, in any module, in
    any spelling the AST resolves, fails here.
    """
    actual = {
        path.name: _receivers(path)
        for path in _module_paths()
        if _receivers(path)
    }
    assert actual == PERMITTED_UNLINK_SITES
    # and the new entry is the only one built from a resolved path
    assert actual[RELEASE_SOURCE_MODULE] == ["Path(obj.path)"]
    for name, receivers in actual.items():
        if name == RELEASE_SOURCE_MODULE:
            continue
        for receiver in receivers:
            assert receiver in MODULE_OWNED_ARTEFACTS, f"{name} unlinks {receiver}"


def test_release_source_permits_only_file_unlink_and_never_a_tree_delete():
    """`os.unlink` / `Path.unlink` only. Never `shutil`, never `rmtree`.

    Two independent reasons this is pinned by AST rather than by grep:

    * a recursive delete is the one primitive that turns a bounded, reversible
      mistake into an unbounded, irreversible one - it has no per-object gate and
      no per-object audit record;
    * exactly ONE unlink call site is permitted. A second one would be a new
      capability and has to be argued for in review, not slipped in.
    """
    source = _source(CUSTODY_DIR / RELEASE_SOURCE_MODULE)
    code = _code_only(source)
    for banned in TREE_DELETE_LITERALS:
        assert banned not in code, f"{RELEASE_SOURCE_MODULE} references {banned}"

    calls = _deletion_calls(ast.parse(source))
    assert len(calls) == 1, f"exactly one unlink site is permitted, found {calls}"
    assert calls[0] == ("unlink", "Path(obj.path)")


def test_release_source_writes_only_its_append_only_audit_ledger():
    """The audit is the only file it creates, and it never truncates one.

    An audit ledger that could be rewritten would erase the only record that bytes
    are gone, so the *mode* is pinned rather than trusted. This also covers the
    write this module does, which the substring scan in
    test_exactly_four_modules_write_and_nothing_else deliberately does not see -
    `release_source` opens its audit through the `AUDIT_MODE` constant rather than
    an inline literal, so a new truncating mode has nowhere to hide.
    """
    from auto_ingest.custody import release_source as module

    assert module.AUDIT_MODE == "a"
    source = _source(CUSTODY_DIR / RELEASE_SOURCE_MODULE)
    tree = ast.parse(source)
    modes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        callee = node.func.attr
        if callee in {"write_text", "write_bytes", "rename", "replace", "truncate"}:
            raise AssertionError(f"{RELEASE_SOURCE_MODULE} calls {callee}()")
        if callee != "open":
            continue
        assert node.args, "open() with no positional mode cannot be audited"
        modes.add(_mode_of(node.args[0]))
    assert modes == {"r", "rb", "AUDIT_MODE"}, sorted(modes)


def _mode_of(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)   # a computed mode: named so it can be pinned


def test_no_custody_module_reads_a_clock_or_randomness_for_its_deletion_set():
    """`release_source` is the one writer, so it is the one that must stay pure.

    The plan it returns is the input to an irreversible operation. If its output
    depended on a clock or on randomness then "propose, then execute" could
    legitimately describe a different set than the one that gets removed, and the
    operator's approval would be meaningless.
    """
    text = _code_only(_source(CUSTODY_DIR / RELEASE_SOURCE_MODULE))
    for banned in ("datetime.now", "time.time", "utcnow", "uuid4", "random"):
        assert banned not in text, f"release_source.py reads the clock/randomness ({banned})"
    # The instant is supplied by the caller, exactly as `custody new --created-at`
    # works, and lands in the audit as a field rather than being invented.
    assert '"at": decided_at' in _source(CUSTODY_DIR / RELEASE_SOURCE_MODULE)


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
