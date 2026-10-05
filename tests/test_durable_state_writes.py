"""Durable state writes: one shared helper, and the two scripts that were not using it.

Three scripts keep JSON state next to themselves. ``kg_health_watchdog.py`` got
this right first and named the trap; the other two did not, in two different
ways, and both failures were invisible because the reader swallowed the damage.

* ``scripts/neo4j_watchdog.py`` wrote through a **fixed** ``.tmp`` name.
  ``os.replace`` is atomic per call, so the target was never half-written - but
  the watchdog loop and an operator (or a second container) both opened that one
  path, and therefore the same inode. The winner's rename publishes that inode
  as the target, and the loser, still holding a descriptor, writes its own
  payload over the front: the state file ends up a blend of the two that no
  longer parses, one writer got an ENOENT, and neither can say whose heal
  cooldown it is holding.

* ``scripts/arxiv_kg_bridge.py`` wrote its cursor with ``Path.write_text``,
  which **truncates in place**. A kill between the truncate and the last byte
  left invalid JSON - and ``load_seen`` caught the parse error and returned an
  empty set, so a corrupted cursor looked exactly like a first run and every
  already-ingested paper was re-ingested, silently.

Both scripts now use ``auto_ingest.util.atomic.write_json_atomic``. Each test
below pins a *behaviour* - interleaved real writers, a real reader, a real
raised failure - rather than grepping the source for a shape, because the source
shape is not what was broken. The two pre-fix bodies are kept verbatim above, and
the tests that use them are the proof that these tests have teeth.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from auto_ingest.util.atomic import write_json_atomic

REPO = Path(__file__).resolve().parent.parent

#: The temp name format: <target name>.<pid>.<per-call seq>.tmp. Both the pid and
#: the sequence are load-bearing - see the module docstring.
TEMP_NAME_RE = re.compile(r"^(?P<stem>.+)\.(?P<pid>\d+)\.(?P<seq>\d+)\.tmp$")

#: Deliberately different lengths, so that a writer still holding a file
#: descriptor on the inode another writer just installed leaves a *torn* file
#: rather than an exact copy - which is what actually happens on Linux.
PAYLOAD_A = {"writer": "A", "round": 1}
PAYLOAD_B = {"writer": "B", "round": 22}


# ---------------------------------------------------------------------------
# references to the pre-fix behaviour, kept verbatim so the tests can be shown
# to have teeth: each of these loses data under the interleaving below
# ---------------------------------------------------------------------------
def write_json_fixed_tmp(target, payload):
    """The old ``neo4j_watchdog._save_last_action`` body, unchanged."""
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, target)


def write_json_truncating(target, payload):
    """The old ``arxiv_kg_bridge.save_seen`` shape: truncate in place, then stream.

    ``Path.write_text`` is ``open(mode="w")`` followed by one ``write``; spelling
    it out this way is the same operation with a point a test can interleave at.
    """
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


# ---------------------------------------------------------------------------
# harnesses
# ---------------------------------------------------------------------------
def park_one_dump(monkeypatch, thread_name):
    """Return ``(entered, release)``: that thread parks inside ``json.dump``.

    ``json.dump`` is the first thing to touch the handle after ``open``, so a
    writer parked there sits in exactly the window the defect lives in: the file
    it opened already exists and is empty, and none of its bytes are down yet.
    Only the named thread parks, so pytest's own serialisation is unaffected.
    """
    entered, released = threading.Event(), threading.Event()
    real_dump = json.dump

    def dump(obj, fp, **kwargs):
        if threading.current_thread().name == thread_name and not entered.is_set():
            entered.set()
            if not released.wait(30):
                raise AssertionError("parked writer was never released")
        return real_dump(obj, fp, **kwargs)

    monkeypatch.setattr(json, "dump", dump)
    return entered, released


def record_installs(monkeypatch):
    """Record the bytes each successful ``os.replace`` moved onto the target.

    Returns a list of ``(temp_name, content)``. A writer whose payload never
    reaches the target is a writer that silently lost, which is precisely what a
    shared temp name causes.
    """
    installed: list[tuple[str, bytes]] = []
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        name, content = Path(src).name, Path(src).read_bytes()
        real_replace(src, dst, *args, **kwargs)
        installed.append((name, content))

    monkeypatch.setattr(os, "replace", replace)
    return installed


def spy_temp_names(monkeypatch):
    """Record the temp name of every ``os.replace``, without changing behaviour."""
    names: list[str] = []
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        names.append(Path(src).name)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    return names


def interleave(target, writer, monkeypatch):
    """Run two writers so that A parks mid-write while B completes end to end.

    Returns ``(errors, installed)``: the exception each writer raised (None if it
    returned), and the bytes that were actually installed on the target.
    """
    entered, released = park_one_dump(monkeypatch, "writer-A")
    installed = record_installs(monkeypatch)
    errors: dict[str, BaseException | None] = {}

    def run(tag, payload):
        try:
            writer(target, payload)
            errors[tag] = None
        except BaseException as exc:  # noqa: BLE001 - the harness reports, never hides
            errors[tag] = exc

    a = threading.Thread(target=run, args=("A", PAYLOAD_A), name="writer-A")
    b = threading.Thread(target=run, args=("B", PAYLOAD_B), name="writer-B")
    a.start()
    assert entered.wait(30), "writer A never reached the write window"
    b.start()
    b.join(30)
    released.set()
    a.join(30)
    assert not a.is_alive() and not b.is_alive(), "a writer thread hung"
    return errors, installed


def serialised(payload):
    """The exact bytes ``write_json_atomic`` produces for ``payload``."""
    return (json.dumps(payload, sort_keys=True, default=str) + "\n").encode()


def legacy_serialised(payload):
    """The exact bytes the old ``json.dump(payload, fh)`` produced."""
    return json.dumps(payload).encode()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the shared helper
# ---------------------------------------------------------------------------
def test_two_concurrent_writers_both_updates_survive(tmp_path, monkeypatch):
    """Two writers, one target, forced to overlap: neither update is lost.

    "Both survive" means both halves. Each writer returned cleanly *and* each
    writer's exact bytes reached the target through a rename at some point -
    recorded by ``record_installs``, which sees what actually landed rather than
    what a writer believed it wrote. A shared temp name fails both halves.
    """
    target = tmp_path / "state.json"
    errors, installed = interleave(target, write_json_atomic, monkeypatch)

    assert errors == {"A": None, "B": None}, f"a writer failed: {errors}"
    assert len(installed) == 2, f"only {len(installed)} of 2 payloads were installed: {installed}"
    assert {content for _, content in installed} == {serialised(PAYLOAD_A), serialised(PAYLOAD_B)}
    # Whatever won, the target is one writer's complete document, never a blend.
    assert target.read_bytes() in {serialised(PAYLOAD_A), serialised(PAYLOAD_B)}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file stranded"


def test_a_fixed_temp_name_loses_an_update_under_the_same_interleaving(tmp_path, monkeypatch):
    """The test above, run against the old body - it must fail, and this says how.

    Kept in the suite so the guard above cannot quietly stop guarding: if the
    helper ever regresses to a fixed suffix, this is the failure it regressed
    from, spelled out.

    It is worse than one update going missing. Both writers opened the *same*
    inode, so the winner's rename published that inode as the target - and the
    loser, still holding its descriptor, then wrote its own shorter payload over
    the front of it. The target ends up as the loser's bytes plus a tail of the
    winner's: a torn file that no longer parses, produced by a write that
    reported success on one side and ENOENT on the other.
    """
    target = tmp_path / "state.json"
    errors, installed = interleave(target, write_json_fixed_tmp, monkeypatch)

    assert isinstance(errors["A"], FileNotFoundError), errors
    assert len(installed) == 1, f"expected exactly one surviving update, got {installed}"
    assert installed[0][1] == legacy_serialised(PAYLOAD_B), (
        "the surviving bytes are B's, not A's: A's update was destroyed by the "
        "shared temp name and A itself died on the ENOENT")
    torn = target.read_bytes()
    assert torn != legacy_serialised(PAYLOAD_A), "expected the target to be torn, not A's"
    with pytest.raises(json.JSONDecodeError):
        json.loads(torn)


def test_a_reader_never_observes_a_partial_file(tmp_path, monkeypatch):
    """Park a writer mid-write; the file a reader sees must still be complete.

    This is the invariant ``truncate-then-write`` cannot hold: while the writer
    sits between its truncate and its last byte, the previous complete document
    has already been destroyed.
    """
    target = tmp_path / "state.json"
    small, large = {"n": 1}, {"blob": "x" * 200_000, "n": 2}
    write_json_atomic(target, small)
    entered, released = park_one_dump(monkeypatch, "writer-A")

    thread = threading.Thread(
        target=lambda: write_json_atomic(target, large), name="writer-A")
    thread.start()
    try:
        assert entered.wait(30), "writer never reached the write window"
        assert json.loads(target.read_text(encoding="utf-8")) == small
    finally:
        released.set()
        thread.join(30)

    assert not thread.is_alive(), "writer thread hung"
    assert json.loads(target.read_text(encoding="utf-8")) == large
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file stranded"


def test_a_truncating_writer_is_observable_mid_write(tmp_path, monkeypatch):
    """The same interleaving against the old shape: the reader sees an empty file.

    Proof that the test above is not vacuous - if a truncating writer also passed
    it, the test would not be measuring anything.
    """
    target = tmp_path / "state.json"
    write_json_atomic(target, {"n": 1})
    entered, released = park_one_dump(monkeypatch, "writer-A")

    thread = threading.Thread(
        target=lambda: write_json_truncating(target, {"blob": "y" * 200_000}),
        name="writer-A")
    thread.start()
    try:
        assert entered.wait(30), "writer never reached the write window"
        with pytest.raises(json.JSONDecodeError):
            json.loads(target.read_text(encoding="utf-8"))
    finally:
        released.set()
        thread.join(30)


def test_a_failed_write_leaves_no_temp_file_and_no_damage(tmp_path, monkeypatch):
    """A write that dies partway must not truncate the target or strand a temp."""
    target = tmp_path / "state.json"
    write_json_atomic(target, {"n": 1})
    good = target.read_bytes()

    monkeypatch.setattr(os, "fsync", lambda *a, **k: (_ for _ in ()).throw(
        OSError("disk full")))
    with pytest.raises(OSError):
        write_json_atomic(target, {"n": 2})

    assert target.read_bytes() == good, "a failed write damaged the previous state"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file stranded"


def test_two_sequential_writes_in_one_process_do_not_collide(tmp_path, monkeypatch):
    """Same pid, same target, two calls - the temp names must differ.

    This is the whole reason the name carries a per-call counter: a pid alone
    would hand both writes the same name, which is the fixed-name bug wearing a
    pid as a disguise.
    """
    target = tmp_path / "state.json"
    names = spy_temp_names(monkeypatch)

    write_json_atomic(target, {"n": 1})
    write_json_atomic(target, {"n": 2})

    assert len(names) == 2
    assert len(set(names)) == 2, f"temp name reused within one process: {names}"
    assert all(TEMP_NAME_RE.match(name) for name in names), names
    assert json.loads(target.read_text(encoding="utf-8")) == {"n": 2}


def test_the_temp_name_is_unique_per_process_too(tmp_path, monkeypatch):
    """The pid is the other half of the name, and it is what stops two hosts."""
    target = tmp_path / "state.json"
    names = spy_temp_names(monkeypatch)

    monkeypatch.setattr(os, "getpid", lambda: 4242)
    write_json_atomic(target, {"n": 1})
    monkeypatch.setattr(os, "getpid", lambda: 4243)
    write_json_atomic(target, {"n": 2})

    pids = {TEMP_NAME_RE.match(name)["pid"] for name in names}
    assert pids == {"4242", "4243"}, names


#: Four writers, twenty rounds each, in four separate interpreters. Against a
#: fixed temp name this is where the ENOENTs come from; against a per-process
#: name, every process completes and every read of the target parses.
_CHILD = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(REPO)!r})\n"
    "from auto_ingest.util.atomic import write_json_atomic\n"
    "target, tag, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
    "errors = 0\n"
    "for i in range(rounds):\n"
    "    try:\n"
    "        write_json_atomic(target, {'writer': tag, 'round': i})\n"
    "    except OSError:\n"
    "        errors += 1\n"
    "json.loads(open(target, encoding='utf-8').read())\n"
    "print(json.dumps({'errors': errors}))\n"
)


def test_separate_processes_sharing_one_target_never_collide(tmp_path):
    """Real processes, not threads: the pid component does the work here."""
    target = tmp_path / "state.json"
    procs = [
        subprocess.run(
            [sys.executable, "-c", _CHILD, str(target), tag, "20"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        for tag in ("w1", "w2", "w3", "w4")
    ]
    for tag, proc in zip(("w1", "w2", "w3", "w4"), procs, strict=True):
        assert json.loads(proc.stdout)["errors"] == 0, f"{tag}: {proc.stdout}{proc.stderr}"
    assert json.loads(target.read_text(encoding="utf-8"))["writer"] in {"w1", "w2", "w3", "w4"}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file stranded"


def test_serialisation_shape_is_the_callers_choice(tmp_path):
    """Compact by default, sorted, newline-terminated, and never raising on a Path."""
    target = tmp_path / "state.json"

    write_json_atomic(target, {"b": 1, "a": 2})
    assert target.read_text(encoding="utf-8") == '{"a": 2, "b": 1}\n'

    write_json_atomic(target, {"b": 1, "a": 2}, indent=2)
    assert target.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'

    write_json_atomic(target, {"p": tmp_path})
    assert json.loads(target.read_text(encoding="utf-8")) == {"p": str(tmp_path)}


# ---------------------------------------------------------------------------
# scripts/neo4j_watchdog.py
# ---------------------------------------------------------------------------
def _watchdog(tmp_path, monkeypatch, name="neo4j_watchdog_durable"):
    wd = _load(REPO / "scripts" / "neo4j_watchdog.py", name)
    monkeypatch.setattr(wd, "STATE_FILE", tmp_path / "last_action.json")
    return wd


def test_watchdog_save_uses_a_unique_temp_name_per_call(tmp_path, monkeypatch):
    """The heal state used a fixed ``.tmp``; two saves must not share one name."""
    wd = _watchdog(tmp_path, monkeypatch)
    names = spy_temp_names(monkeypatch)

    wd._save_last_action("heal", "oom:systemctl restart neo4j")
    wd._save_last_action("failed", "down:systemctl restart neo4j")

    assert len(names) == 2
    assert len(set(names)) == 2, f"temp name reused across saves: {names}"
    assert all(TEMP_NAME_RE.match(name) for name in names), names
    state = json.loads((tmp_path / "last_action.json").read_text(encoding="utf-8"))
    assert state["action"] == "failed"
    assert isinstance(state["ts"], int)
    assert [p.name for p in tmp_path.iterdir()] == ["last_action.json"]


def test_watchdog_save_keeps_the_previous_state_when_the_write_fails(tmp_path, monkeypatch):
    """It promises never to raise - and it must not damage the state either.

    The cooldown is read from this file; losing it to a failed write would let a
    flapping Neo4j be restarted in a tight loop, which is the thing the state
    file exists to prevent.
    """
    wd = _watchdog(tmp_path, monkeypatch)
    state = tmp_path / "last_action.json"
    wd._save_last_action("heal", "oom:first")
    good = state.read_bytes()

    monkeypatch.setattr(os, "fsync", lambda *a, **k: (_ for _ in ()).throw(
        OSError("read-only file system")))
    wd._save_last_action("failed", "down:second")  # documented never to raise

    assert state.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["last_action.json"], "temp file stranded"


# ---------------------------------------------------------------------------
# scripts/arxiv_kg_bridge.py
# ---------------------------------------------------------------------------
def _bridge(tmp_path, monkeypatch, name="arxiv_kg_bridge_durable"):
    bridge = _load(REPO / "scripts" / "arxiv_kg_bridge.py", name)
    monkeypatch.setattr(bridge, "CURSOR_FILE", tmp_path / ".arxiv_kg_cursor.json")
    return bridge


def test_cursor_round_trips(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    bridge.save_seen({"2401.00002", "2401.00001"})

    assert json.loads(bridge.CURSOR_FILE.read_text(encoding="utf-8")) == {
        "seen": ["2401.00001", "2401.00002"]}
    assert bridge.load_seen() == {"2401.00001", "2401.00002"}


def test_a_crashed_cursor_save_leaves_the_previous_cursor_intact(tmp_path, monkeypatch):
    """The cursor is the only record of what was ingested; a lost one re-does work.

    ``save_seen`` used ``write_text``, which truncates first, so a kill mid-write
    left invalid JSON behind - and ``load_seen`` turned that into an empty set.
    """
    bridge = _bridge(tmp_path, monkeypatch)
    bridge.save_seen({"a", "b"})
    good = bridge.CURSOR_FILE.read_bytes()

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(
        OSError("no space left on device")))
    with pytest.raises(OSError):
        bridge.save_seen({"a", "b", "c"})

    assert bridge.CURSOR_FILE.read_bytes() == good
    assert bridge.load_seen() == {"a", "b"}
    assert [p.name for p in tmp_path.iterdir()] == [".arxiv_kg_cursor.json"]


def test_an_absent_cursor_is_silent(tmp_path, monkeypatch, capsys):
    """First run: no cursor, no complaint. This is normal and must stay quiet."""
    bridge = _bridge(tmp_path, monkeypatch)
    assert not bridge.CURSOR_FILE.exists()

    assert bridge.load_seen() == set()
    assert capsys.readouterr().err == "", "a first run should not cry wolf"


def test_a_corrupt_cursor_is_reported_rather_than_swallowed(tmp_path, monkeypatch, capsys):
    """Corrupt is not the same as absent, and only one of them was reported.

    Both return an empty set - re-ingesting is wasteful, not wrong, and the
    bridge must keep running so an operator can fix it - but the corrupt case
    now names the file on stderr instead of quietly re-doing every paper.
    """
    bridge = _bridge(tmp_path, monkeypatch)
    bridge.CURSOR_FILE.write_text('{"seen": ["a","b","c"', encoding="utf-8")
    capsys.readouterr()

    assert bridge.load_seen() == set()
    err = capsys.readouterr().err
    assert "WARNING" in err, err
    assert str(bridge.CURSOR_FILE) in err, err
    assert "JSONDecodeError" in err, err


def test_a_cursor_that_is_valid_json_but_the_wrong_shape_is_reported(tmp_path, monkeypatch,
                                                                   capsys):
    """Not just syntax: a cursor whose ``seen`` is not a list is also a reset."""
    bridge = _bridge(tmp_path, monkeypatch)
    bridge.CURSOR_FILE.write_text('{"seen": 7}', encoding="utf-8")
    capsys.readouterr()

    assert bridge.load_seen() == set()
    assert "WARNING" in capsys.readouterr().err


def test_atomic_write_temps_are_gitignored():
    """A crash mid-write leaves a temp behind; it must not appear in git status.

    The helper writes `<name>.<pid>.<n>.tmp` then os.replace's it, so a stranded
    temp is untracked and unignored unless .gitignore says otherwise - which
    would reintroduce exactly the noise the ignore rules exist to prevent.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for pattern in ("scripts/.kg_health_state.json.*.tmp",
                    "scripts/.arxiv_kg_cursor.json.*.tmp"):
        proc = subprocess.run(["git", "check-ignore", "-q", pattern.replace("*", "1")],
                              cwd=repo, capture_output=True, timeout=30)
        assert proc.returncode == 0, f"{pattern} is not gitignored"


def test_kg_health_watchdog_uses_the_shared_helper(monkeypatch):
    """No fourth copy of the atomic-write routine.

    It had one, written before `auto_ingest/util/` existed. Now that the helper
    is shared by the arxiv bridge and the neo4j watchdog, a divergent copy here
    would be three implementations to keep in step - and this one is the file
    whose reader maps a parse failure to {}, so a torn write silently resets the
    stall-detector baseline.
    """
    import importlib.util
    import tempfile

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "kg_health_watchdog.py"
    source = script.read_text(encoding="utf-8")

    # Behaviour, not just a grep: point the state file at a tmp dir and drive it.
    # monkeypatch.setenv, not os.environ[...] - a bare assignment leaks into every
    # later test if this one raises before its cleanup, and the module reads the
    # variable at IMPORT time, so the leak changes another test's subject.
    with tempfile.TemporaryDirectory() as tmp, monkeypatch.context() as mp:
        mp.setenv("KG_HEALTH_STATE_FILE", str(Path(tmp) / "state.json"))
        spec = importlib.util.spec_from_file_location("kg_health_watchdog_probe", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.save_state({"emb768_count": 1, "emb768_time": 2.0})
        module.save_state({"emb768_count": 2, "emb768_time": 3.0})
        written = json.loads(Path(module.STATE_FILE).read_text())
        assert written == {"emb768_count": 2, "emb768_time": 3.0}
        leftovers = [p.name for p in Path(module.STATE_FILE).parent.iterdir()
                     if p.name != "state.json"]
        assert leftovers == [], f"stranded temp files: {leftovers}"

    # The private copy is gone; the shared one is what gets called.
    assert "def save_state" in source
    assert "write_json_atomic(STATE_FILE" in source
    assert "os.replace" not in source, "the local copy should be gone, not merged"
