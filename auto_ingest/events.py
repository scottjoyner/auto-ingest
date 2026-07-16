"""auto_ingest.events - local shim mirroring the shared fleet event envelope.

This is a STOPGAP until the swarm-contracts package (``src/assistx/contracts``)
is wired in as a git submodule / path dependency. It mirrors the canonical
``EventEnvelope`` shape locally so auto-ingest emits events with the required
``correlation_id`` (UUID) field without taking a cross-repo import dependency.

When the shared contract is available, migrate callers to:
    from assistx.contracts.event_envelope import EventEnvelope, AuthState

See LLD §1 / §3.4 (auto-ingest emits ``ingest.evidence.linked`` etc).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventEnvelope:
    """Local mirror of the canonical envelope (correlation_id required)."""

    def __init__(
        self,
        schema_version: str,
        source_repo: str,
        event_type: str,
        correlation_id: Optional[str] = None,
        actor: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        links: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        # Validate it is a UUID (mirrors contract requirement, LLD §2 G1).
        try:
            uuid.UUID(correlation_id)
        except (ValueError, AttributeError, TypeError):
            raise ValueError("correlation_id must be a valid UUID")
        self.schema_version = schema_version
        self.source_repo = source_repo
        self.event_type = event_type
        self.correlation_id = correlation_id
        self.actor = actor or {}
        self.ts = _now_iso()
        self.payload = payload or {}
        self.links = links or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_repo": self.source_repo,
            "event_type": self.event_type,
            "correlation_id": self.correlation_id,
            "actor": self.actor,
            "ts": self.ts,
            "payload": self.payload,
            "links": self.links,
        }


SCHEMA_VERSION = "2026-06-08.v1"
SOURCE_REPO = "auto-ingest"


def emit(event_type: str, payload: Dict[str, Any], correlation_id: Optional[str] = None,
         links: Optional[List[Dict[str, Any]]] = None) -> EventEnvelope:
    """Build (and conceptually dispatch) a local event envelope.

    In this shim the envelope is only constructed + returned. The durable
    delivery side is the outbox (``auto_ingest.outbox``): callers should stage
    the op there so a downstream (Neo4j/AssistX) outage mid-ingest cannot lose
    the work.
    """
    return EventEnvelope(
        schema_version=SCHEMA_VERSION,
        source_repo=SOURCE_REPO,
        event_type=event_type,
        correlation_id=correlation_id,
        payload=payload,
        links=links,
    )
