"""Atomic JSON writes for the small state files the scripts keep beside themselves.

Two failure modes are being designed out here, and they are different:

1. A reader can see a half-written file. ``open(path, "w")`` and
   ``Path.write_text()`` both *truncate first*; anything that dies between the
   truncate and the last byte leaves a truncated or syntactically invalid file
   on disk. Readers that swallow the parse error then treat the damage as
   "no state" - which is not a safe default, it is a silent reset.

2. Two writers can destroy each other even when the write itself is atomic.
   ``os.replace`` is atomic *per call*, so the target is never half-written -
   but a *fixed* temp name shared by two writers is not. Both open the same
   inode; the winner's rename then publishes that inode as the target, and the
   loser - still holding a descriptor to it - writes its own payload over the
   front. The file left behind is a blend of the two that no longer parses, one
   writer got an ENOENT, and neither can say whose update it holds. A temp name
   unique per process *and per call* makes that impossible rather than merely
   unlikely.

The name is ``<name>.<pid>.<n>.tmp``: the pid separates processes, and ``n``
separates the calls of one process. The counter is taken under a lock so that
two threads in one process cannot draw the same number - ``itertools.count``
happens to be atomic on CPython, but that is an implementation detail rather
than a guarantee, and the lock is released before any I/O, so it cannot
deadlock.

``auto_ingest.custody.store`` carries a private copy of this helper and must
keep it: that package is required to stand on its own, and its tests pin the
exact set of modules permitted to write. Two implementations of the same
invariant in two packages is the price of that independence, and it is cheaper
than coupling them.
"""
from __future__ import annotations

import json
import os
import threading
from itertools import count
from pathlib import Path
from typing import Any

__all__ = ["write_json_atomic"]

_WRITE_SEQ = count(1)
_SEQ_LOCK = threading.Lock()


def _next_seq() -> int:
    """Return a per-process serial number, unique even across threads."""
    with _SEQ_LOCK:
        return next(_WRITE_SEQ)


def write_json_atomic(path: Path | str, payload: Any, *,
                      indent: int | None = None, sort_keys: bool = True) -> None:
    """Serialise ``payload`` to ``path`` as JSON, atomically.

    The bytes land via a uniquely-named temp file that is flushed, fsynced, and
    then ``os.replace``d onto the target, so a concurrent reader sees either the
    previous complete file or the new complete file and never a mixture.

    ``indent=None`` (the default) keeps the compact single-line form the scripts
    have always written; pass ``indent=2`` for a file a human reads. Keys are
    sorted so that an unchanged payload produces unchanged bytes. Values that
    json cannot represent (a ``Path``, a ``datetime``) are stringified rather
    than raising, matching the behaviour the call sites already relied on.

    Any failure - including ``KeyboardInterrupt`` - unlinks the temp file before
    re-raising, so an aborted write leaves neither a partial target nor a
    stranded temp. The parent directory must already exist; this helper will not
    create it, because silently materialising a directory would turn a
    misconfigured path into a plausible-looking file in the wrong place.

    The fsync covers the file's contents, not the directory entry, so this
    survives a process death but not a power cut between the rename and the
    metadata flush. Closing that needs a directory fsync, which
    ``auto_ingest.custody.store`` deliberately does not do either; do it here
    first if it is ever needed.
    """
    target = Path(path)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{_next_seq()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
