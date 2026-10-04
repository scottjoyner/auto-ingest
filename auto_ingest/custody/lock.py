"""auto_ingest.custody.lock - a campaign-scoped exclusive lock.

The investigation found a live race that has nothing to do with custody but
would corrupt a campaign if left alone:

* ``sync-service`` (``docker-compose.yml:85``, every 10 min) and
  ``deploy/cron/ingest.crontab:5`` (every 5 min) both run
  ``deploy/sync_from_legacy_drop.sh``, which rsyncs ``--archive --ignore-existing``
  into ``$AUDIO_ROOT`` / ``$DASHCAM_ROOT`` / ``$BODYCAM_ROOT`` - the same roots a
  custody campaign writes to. It has **no lock and no campaign awareness**, so it
  will silently skip whatever a campaign produced, and can populate those roots
  independently.
* ``ingest-worker`` (``docker-compose.yml:49``, every 30 s) claims and executes
  arbitrary ``.job`` files from ``$DROP_ROOT``, also uncoordinated.

This module is the primitive that resolves it. It deliberately does **not** edit
those scripts - changing a running service's behaviour belongs in its own change
with its own review. What it provides is:

* an advisory lock a campaign takes for its duration;
* a probe a *pre-existing* writer can consult to stand down, with no custody
  import required (it reads one lock file);
* a second, shared lock for coordinating *between* campaigns.

Locks are advisory by design: they coordinate software that agrees to take them.
The legacy writers do not, yet - so :func:`competing_activity` reports that fact
honestly rather than pretending the lock makes them safe.
"""

from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Where locks live. Overridable so tests never touch a shared location.
DEFAULT_LOCK_ROOT = "/tmp/auto_ingest_custody"

def lock_root(root: Optional[str | Path] = None) -> Path:
    return Path(root or os.environ.get("CUSTODY_LOCK_ROOT") or DEFAULT_LOCK_ROOT)


def _ensure(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o777)  # multiple operators, one shared campaign host


@dataclass
class CampaignLock:
    """An advisory exclusive lock held for the duration of a campaign.

    Context-manager so the lock releases on an exception too - a crashed
    executor must not leave the destination permanently blocked.
    """

    path: Path
    _fd: Optional[int] = None

    def acquire(self) -> bool:
        _ensure(self.path.parent)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        self._fd = fd
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder_pid(self) -> Optional[int]:
        """PID of the holder, for an operator-facing error message."""
        if self._fd is None:
            return None
        try:
            # The write left the offset at the end; rewind before reading.
            os.lseek(self._fd, 0, os.SEEK_SET)
            raw = os.read(self._fd, 32).decode("ascii", "replace").strip()
            return int(raw) or None
        except (OSError, ValueError):  # pragma: no cover
            return None

    def __enter__(self) -> "CampaignLock":
        if not self.acquire():
            raise LockUnavailable(f"another campaign holds {self.path}")
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


class LockUnavailable(RuntimeError):
    """Another campaign already holds this destination."""


#: Markers a pre-existing writer can leave to make itself visible to custody.
#:
#: The per-destination lock is keyed on the campaign's *logical* destination
#: ("primary:fileserver/dashcam"), which a shell script that only knows
#: $DASHCAM_ROOT cannot derive. So the handshake is deliberately cruder and
#: host-path independent: while any campaign is running it drops one marker file,
#: and a writer checks for that file. One `test -e`, no import, no arguments.
ACTIVE_MARKER = "campaign-active"

#: Marker string the patched writer must contain. `competing_activity` greps for
#: it rather than trusting an operator's word that the script was updated, so
#: reverting the script makes preflight fail closed again.
WRITER_PROBE_MARKER = "custody-campaign-lock"


def active_marker(root: Optional[str | Path] = None) -> Path:
    """Path of the "a custody campaign is running" marker."""
    return lock_root(root) / ACTIVE_MARKER


def is_campaign_active(root: Optional[str | Path] = None) -> bool:
    """Whether any campaign currently holds a destination.

    Deliberately a plain existence test, so a shell script can do the same with
    ``test -e`` and no dependency on this package.
    """
    return active_marker(root).exists()


def set_active(root: Optional[str | Path] = None) -> Path:
    """Drop the marker. Called when a campaign takes a destination."""
    path = active_marker(root)
    _ensure(path.parent)
    path.write_text("custody campaign in progress\n", encoding="utf-8")
    return path


def clear_active(root: Optional[str | Path] = None) -> None:
    """Remove the marker. Best-effort: a stale marker only causes writers to
    stand down, which is the safe direction."""
    try:
        active_marker(root).unlink()
    except OSError:
        pass


def campaign_lock(destination_key: str, root: Optional[str | Path] = None) -> CampaignLock:
    """The exclusive lock for one canonical destination."""
    return CampaignLock(lock_root(root) / f"campaign-{_safe(destination_key)}.lock")


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)


def is_locked(destination_key: str, root: Optional[str | Path] = None) -> bool:
    """Whether some process holds the campaign lock. Does not take it.

    Safe to call from a legacy writer that knows nothing about custody: it stats
    and attempts a non-blocking flock on one path.
    """
    path = lock_root(root) / f"campaign-{_safe(destination_key)}.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return True
        return False
    finally:
        os.close(fd)


@dataclass(frozen=True)
class CompetingActivity:
    """Something that may write the destination without taking the campaign lock."""

    kind: str
    detail: str
    honours_lock: bool
    remedy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detail": self.detail,
            "honours_lock": self.honours_lock,
            "kind": self.kind,
            "remedy": self.remedy,
        }


def writer_consults_lock(script: Optional[str | Path]) -> bool:
    """Whether a writer script actually checks the campaign marker.

    Verified by reading the script rather than trusting a declaration, so
    reverting the patch makes preflight fail closed again instead of silently
    claiming a coordination that no longer exists.
    """
    if not script:
        return False
    path = Path(script)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return WRITER_PROBE_MARKER in text and ACTIVE_MARKER in text


def competing_activity(
    *,
    job_dir: Optional[str | Path] = None,
    drop_root: Optional[str | Path] = None,
    legacy_sync_script: Optional[str | Path] = None,
) -> Tuple[CompetingActivity, ...]:
    """Report writers that could touch the destination uncoordinated.

    Deliberately *reports* rather than blocks. Three of these are real and none
    currently takes a campaign lock, so claiming "safe to execute" on their
    absence would be a lie - the honest statement is that they exist, that they
    do not participate, and what an operator should do about it.
    """
    found: List[CompetingActivity] = []

    if drop_root:
        drop = Path(drop_root)
        queued = []
        if drop.is_dir():
            queued = sorted(p.name for p in drop.glob("*.job"))
        found.append(CompetingActivity(
            kind="ingest_worker_queue",
            detail=f"queued_jobs={len(queued)}" + (f" {queued[:5]}" if queued else ""),
            honours_lock=False,
            remedy=("let the worker drain, or stop ingest-worker for the campaign "
                    "(docker-compose.yml:49 claims .job files every 30s)"),
        ))

    if job_dir:
        base = Path(job_dir)
        pending = []
        if base.is_dir():
            pending = sorted(p.name for p in base.glob("*.job"))
        if pending:
            found.append(CompetingActivity(
                kind="queued_jobs",
                detail=f"{len(pending)} {pending[:5]}",
                honours_lock=False,
                remedy="drain the queue before copying",
            ))

    # The legacy drop sync is the one writer that matters: it runs every 5 minutes
    # from two places and rsyncs --ignore-existing into the campaign's roots.
    # Whether it participates is *verified by reading the script*, not asserted.
    respects = writer_consults_lock(legacy_sync_script)
    found.append(CompetingActivity(
        kind="legacy_drop_sync",
        detail=("deploy/sync_from_legacy_drop.sh stands down while a custody "
                "campaign is active"
                if respects else
                "deploy/sync_from_legacy_drop.sh rsyncs --ignore-existing into "
                "$DASHCAM_ROOT/$AUDIO_ROOT/$BODYCAM_ROOT with no lock check"),
        honours_lock=respects,
        remedy=("" if respects else
                "sync-service (docker-compose.yml:85, every 10m) and "
                "ingest.crontab:5 (every 5m) write the same roots without a lock; "
                f"add the '{WRITER_PROBE_MARKER}' stand-down probe to that script"),
    ))
    return tuple(found)


def uncoordinated_writers(activity: Tuple[CompetingActivity, ...]) -> Tuple[str, ...]:
    """Kinds of competing writer that do not take the campaign lock."""
    return tuple(a.kind for a in activity if not a.honours_lock)


__all__ = [
    "ACTIVE_MARKER",
    "WRITER_PROBE_MARKER",
    "CampaignLock",
    "CompetingActivity",
    "DEFAULT_LOCK_ROOT",
    "LockUnavailable",
    "active_marker",
    "campaign_lock",
    "clear_active",
    "competing_activity",
    "is_campaign_active",
    "is_locked",
    "lock_root",
    "set_active",
    "uncoordinated_writers",
]
