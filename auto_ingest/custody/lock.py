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


def competing_activity(
    *,
    job_dir: Optional[str | Path] = None,
    drop_root: Optional[str | Path] = None,
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

    # Known uncoordinated writers, named explicitly. This is a standing hazard,
    # not a transient state, so it is always reported.
    found.append(CompetingActivity(
        kind="legacy_drop_sync",
        detail="deploy/sync_from_legacy_drop.sh rsyncs --ignore-existing into "
               "$DASHCAM_ROOT/$AUDIO_ROOT/$BODYCAM_ROOT",
        honours_lock=False,
        remedy=("sync-service (docker-compose.yml:85, every 10m) and "
                "ingest.crontab:5 (every 5m) write the same roots without a lock; "
                "either stop them for the campaign or add a lock check to that "
                "script as a separate change"),
    ))
    return tuple(found)


def uncoordinated_writers(activity: Tuple[CompetingActivity, ...]) -> Tuple[str, ...]:
    """Kinds of competing writer that do not take the campaign lock."""
    return tuple(a.kind for a in activity if not a.honours_lock)


__all__ = [
    "CampaignLock",
    "CompetingActivity",
    "DEFAULT_LOCK_ROOT",
    "LockUnavailable",
    "campaign_lock",
    "competing_activity",
    "is_locked",
    "lock_root",
    "uncoordinated_writers",
]
