"""Safe local/NAS fallback queue for ingest work.

Queue items are typed JSON documents that name a known orchestration profile.
Arbitrary shell commands are intentionally unsupported. Claiming is an atomic
rename, while execution still uses Neo4j fenced orchestration for distributed
correctness and retry/quarantine state.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from auto_ingest.orchestration import PROFILES, run_profile

QUEUE_VERSION = 1
ALLOWED_KEYS = {"version", "job_id", "profile", "created_at", "metadata"}


class QueueJobError(ValueError):
    pass


@dataclass(frozen=True)
class QueueJob:
    version: int
    job_id: str
    profile: str
    created_at: int
    metadata: dict

    @classmethod
    def from_dict(cls, value: dict) -> "QueueJob":
        unknown = set(value) - ALLOWED_KEYS
        if unknown:
            raise QueueJobError(f"unknown queue job fields: {sorted(unknown)}")
        if value.get("version") != QUEUE_VERSION:
            raise QueueJobError(f"unsupported queue version: {value.get('version')!r}")
        job_id = value.get("job_id")
        profile = value.get("profile")
        if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
            raise QueueJobError("job_id must be a non-empty string <= 128 chars")
        if profile not in PROFILES:
            raise QueueJobError(f"unknown profile: {profile!r}")
        created_at = value.get("created_at")
        if not isinstance(created_at, int) or created_at <= 0:
            raise QueueJobError("created_at must be a positive epoch integer")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise QueueJobError("metadata must be an object")
        encoded = json.dumps(metadata, sort_keys=True, default=str)
        if len(encoded.encode("utf-8")) > 16_384:
            raise QueueJobError("metadata exceeds 16 KiB")
        return cls(QUEUE_VERSION, job_id, profile, created_at, metadata)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "job_id": self.job_id,
            "profile": self.profile,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


def queue_dirs(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "root": base,
        "claimed": base / "claimed",
        "done": base / "done",
        "failed": base / "failed",
        "rejected": base / "rejected",
    }


def ensure_dirs(root: str | Path) -> dict[str, Path]:
    dirs = queue_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def enqueue(
    root: str | Path,
    profile: str,
    *,
    metadata: dict | None = None,
    job_id: str | None = None,
    now: int | None = None,
) -> Path:
    if profile not in PROFILES:
        raise QueueJobError(f"unknown profile: {profile!r}")
    dirs = ensure_dirs(root)
    jid = job_id or uuid.uuid4().hex
    job = QueueJob(
        version=QUEUE_VERSION,
        job_id=jid,
        profile=profile,
        created_at=int(time.time() if now is None else now),
        metadata=dict(metadata or {}),
    )
    QueueJob.from_dict(job.to_dict())
    final = dirs["root"] / f"{jid}.job.json"
    fd, temp_name = tempfile.mkstemp(prefix=f".{jid}.", suffix=".tmp", dir=dirs["root"])
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(job.to_dict(), fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, final)
        return final
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def load_job(path: str | Path) -> QueueJob:
    """Load a typed queue payload before or after its atomic claim rename."""
    p = Path(path)
    if p.suffix != ".json":
        raise QueueJobError("only JSON typed queue items are accepted")
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueJobError(f"invalid queue JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueJobError("queue job must be a JSON object")
    return QueueJob.from_dict(value)


def claim_one(root: str | Path, *, owner: str | None = None) -> Path | None:
    dirs = ensure_dirs(root)
    worker = owner or f"{socket.gethostname()}-{os.getpid()}"
    for legacy in sorted(dirs["root"].glob("*.job")):
        target = dirs["rejected"] / f"{legacy.name}.legacy-shell-rejected"
        try:
            os.replace(legacy, target)
        except FileNotFoundError:
            pass
    for path in sorted(dirs["root"].glob("*.job.json")):
        target = dirs["claimed"] / f"{path.stem}.{worker}.json"
        try:
            os.replace(path, target)
            return target
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def process_claimed(
    driver,
    path: str | Path,
    *,
    root: str | Path,
    owner: str | None = None,
    runner: Callable[..., int] = run_profile,
) -> int:
    dirs = ensure_dirs(root)
    claimed = Path(path)
    worker = owner or f"queue:{socket.gethostname()}:{os.getpid()}"
    try:
        job = load_job(claimed)
    except QueueJobError:
        target = dirs["rejected"] / claimed.name
        os.replace(claimed, target)
        return 4

    rc = runner(
        driver,
        job.profile,
        job_key=f"filequeue:{job.job_id}",
        owner=worker,
    )
    if rc == 0:
        target = dirs["done"] / f"{job.job_id}.done.json"
    elif rc == 3:
        target = dirs["failed"] / f"{job.job_id}.quarantined.json"
    else:
        target = dirs["root"] / f"{job.job_id}.job.json"
    os.replace(claimed, target)
    return rc


def work_once(
    driver,
    root: str | Path,
    *,
    owner: str | None = None,
    runner: Callable[..., int] = run_profile,
) -> int:
    claimed = claim_one(root, owner=owner)
    if claimed is None:
        return 0
    return process_claimed(driver, claimed, root=root, owner=owner, runner=runner)


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.file_queue")
    sub = parser.add_subparsers(dest="command", required=True)
    enq = sub.add_parser("enqueue")
    enq.add_argument("--root", default=os.environ.get("DROP_ROOT", "/nas/drop"))
    enq.add_argument("--profile", choices=sorted(PROFILES), required=True)
    enq.add_argument("--job-id", default=None)
    work = sub.add_parser("work")
    work.add_argument("--root", default=os.environ.get("DROP_ROOT", "/nas/drop"))
    work.add_argument("--owner", default=None)
    work.add_argument("--once", action="store_true")
    work.add_argument("--sleep-sec", type=int, default=30)
    args = parser.parse_args(argv)

    if args.command == "enqueue":
        print(enqueue(args.root, args.profile, job_id=args.job_id))
        return 0

    driver = _driver()
    try:
        while True:
            rc = work_once(driver, args.root, owner=args.owner)
            if args.once or rc not in (0, 2):
                return rc
            time.sleep(max(args.sleep_sec, 1))
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
