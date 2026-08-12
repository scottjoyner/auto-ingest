"""Canonical fenced stage execution with crash-recoverable artifact commits.

The executor coordinates a lease generation, deterministic artifact identity,
atomic filesystem publication, a durable journal, and fenced Neo4j artifact
registration. A stale worker cannot register an artifact or mark a stage
complete after lease takeover.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from auto_ingest import ingest_claim
from auto_ingest.artifacts import ArtifactIdentity, artifact_relative_path, build_identity, sha256_bytes
from auto_ingest.commit_protocol import atomic_commit_bytes, verify_artifact

STATE_PREPARING = "PREPARING"
STATE_ARTIFACT_COMMITTED = "ARTIFACT_COMMITTED"
STATE_COMMITTED = "COMMITTED"
STATE_QUARANTINED = "QUARANTINED"


class LeaseLost(RuntimeError):
    pass


@dataclass(frozen=True)
class StageCommit:
    job_key: str
    owner: str
    stage: str
    artifact_id: str
    artifact_path: str
    artifact_sha256: str
    fence_token: int
    reused: bool


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _journal_path(root: Path, artifact_id: str) -> Path:
    return root / ".commit-journal" / artifact_id[:2] / f"{artifact_id}.json"


def read_journal(root: str | Path, artifact_id: str) -> dict | None:
    path = _journal_path(Path(root), artifact_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def register_artifact_fenced(driver, commit: StageCommit, *, stage_version: str) -> bool:
    """Fence-check the lease and register artifact provenance in one transaction."""
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$job_key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$fence_token
            MERGE (a:IngestArtifact {artifact_id:$artifact_id})
            ON CREATE SET a.created_at=timestamp()
            SET a.stage=$stage,
                a.stage_version=$stage_version,
                a.path=$path,
                a.sha256=$sha256,
                a.fence_token=$fence_token,
                a.updated_at=timestamp()
            MERGE (j)-[:PRODUCED]->(a)
            RETURN a.artifact_id AS artifact_id
            """,
            job_key=commit.job_key,
            owner=commit.owner,
            fence_token=commit.fence_token,
            artifact_id=commit.artifact_id,
            stage=commit.stage,
            stage_version=stage_version,
            path=commit.artifact_path,
            sha256=commit.artifact_sha256,
        ).single()
    return rec is not None


def execute_stage(
    driver,
    *,
    job_key: str,
    owner: str,
    stage: str,
    stage_version: str,
    source_hash: str,
    artifact_root: str | Path,
    artifact_bytes: bytes,
    artifact_suffix: str = "",
    config: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    ttl_sec: int = 3600,
    graph_commit: Callable[[Any, StageCommit], None] | None = None,
    fault: Callable[[str], None] | None = None,
) -> StageCommit:
    """Execute one artifact-producing stage under a fenced lease.

    Fault hook checkpoints: ``after_claim``, ``after_prepare``,
    ``after_artifact``, ``after_graph``, ``after_stage_state``.
    """
    token = ingest_claim.claim_fenced(driver, job_key, owner, ttl_sec=ttl_sec)
    if token is None:
        raise LeaseLost(f"unable to acquire lease for {job_key}")
    if fault:
        fault("after_claim")

    identity: ArtifactIdentity = build_identity(
        source_hash=source_hash,
        stage=stage,
        stage_version=stage_version,
        config=config,
        model=model,
    )
    artifact_id = identity.artifact_id
    root = Path(artifact_root)
    final_path = root / artifact_relative_path(identity, artifact_suffix)
    journal = _journal_path(root, artifact_id)
    digest = sha256_bytes(artifact_bytes)

    base_journal = {
        "state": STATE_PREPARING,
        "job_key": job_key,
        "owner": owner,
        "fence_token": token,
        "identity": asdict(identity),
        "artifact_id": artifact_id,
        "artifact_path": str(final_path),
        "artifact_sha256": digest,
    }
    _write_json_atomic(journal, base_journal)
    if fault:
        fault("after_prepare")

    committed = atomic_commit_bytes(final_path, artifact_bytes, expected_sha256=digest)
    _write_json_atomic(journal, {**base_journal, "state": STATE_ARTIFACT_COMMITTED})
    if fault:
        fault("after_artifact")

    result = StageCommit(
        job_key=job_key,
        owner=owner,
        stage=stage,
        artifact_id=artifact_id,
        artifact_path=str(final_path),
        artifact_sha256=digest,
        fence_token=token,
        reused=committed.reused,
    )
    if not register_artifact_fenced(driver, result, stage_version=stage_version):
        raise LeaseLost(f"lease generation lost before artifact registration for {job_key}")

    if graph_commit is not None:
        graph_commit(driver, result)
    if fault:
        fault("after_graph")

    updated = ingest_claim.update_stage_fenced(
        driver, job_key, stage, owner=owner, fence_token=token
    )
    if updated is None:
        raise LeaseLost(f"lease generation lost before stage completion for {job_key}")
    if fault:
        fault("after_stage_state")

    _write_json_atomic(journal, {**base_journal, "state": STATE_COMMITTED})
    return result


def recover_artifact(root: str | Path, artifact_id: str) -> str:
    """Classify an interrupted commit journal without mutating production state."""
    record = read_journal(root, artifact_id)
    if record is None:
        return "MISSING"
    state = record.get("state")
    if state == STATE_COMMITTED:
        return STATE_COMMITTED
    path = record.get("artifact_path")
    digest = record.get("artifact_sha256")
    if path and digest and verify_artifact(path, digest):
        return STATE_ARTIFACT_COMMITTED
    return STATE_PREPARING
