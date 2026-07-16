"""auto_ingest.events - shim re-exporting the shared fleet event envelope.

When the canonical package is importable (auto-assist on sys.path as a
submodule/path) this module re-exports the real ``assistx.contracts``
``EventEnvelope`` so auto-ingest emits the single source-of-truth envelope.
Otherwise it falls back to the local mirror class below so standalone ingest
runs keep working without the canonical package.

Canonical source of truth:
  /media/scott/SSD_4TB/hermes-home/home_scott_git_auto-assist/src/assistx/contracts/
See docs/LLD_UNIFIED_FLEET.md §1 and §3.4 (auto-ingest emits
``ingest.evidence.linked`` etc).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


try:
    from assistx.contracts.event_envelope import (
        Actor,
        AuthState,
        EventEnvelope,
        EventLink,
        TraceEvent,
        TraceGroup,
    )
    from assistx.contracts.version import SCHEMA_VERSION

    _USING_CANONICAL = True
except ImportError:
    from enum import Enum

    class AuthState(str, Enum):  # pragma: no cover - fallback only
        AUTHENTICATED_SCOTT = "authenticated_scott"
        UNKNOWN_SPEAKER = "unknown_speaker"
        REGISTERED_USER_UNVERIFIED = "registered_user_unverified"
        ADMIN_VOICE_OVERRIDE = "admin_voice_override"
        REJECTED = "rejected"

    class Actor:  # pragma: no cover - fallback only
        pass

    class EventLink:  # pragma: no cover - fallback only
        pass

    class TraceEvent:  # pragma: no cover - fallback only
        pass

    class TraceGroup:  # pragma: no cover - fallback only
        pass

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
            self.ts = datetime.now(timezone.utc).isoformat()
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

    _USING_CANONICAL = False


SOURCE_REPO = "auto-ingest"


def emit(event_type: str, payload: Dict[str, Any], correlation_id: Optional[str] = None,
         links: Optional[List[Dict[str, Any]]] = None) -> EventEnvelope:
    """Build (and conceptually dispatch) a local event envelope.

    In this shim the envelope is only constructed + returned. The durable
    delivery side is the outbox (``auto_ingest.outbox``): callers should stage
    the op there so a downstream (Neo4j/AssistX) outage mid-ingest cannot lose
    the work. ``correlation_id`` is REQUIRED (UUID) and is auto-generated when
    not supplied (contract enforcement, LLD §2).
    """
    return EventEnvelope(
        schema_version=SCHEMA_VERSION,
        source_repo=SOURCE_REPO,
        event_type=event_type,
        correlation_id=correlation_id,
        payload=payload,
        links=links,
    )


__all__ = [
    "AuthState",
    "Actor",
    "EventEnvelope",
    "EventLink",
    "TraceEvent",
    "TraceGroup",
    "SCHEMA_VERSION",
    "SOURCE_REPO",
    "emit",
    "_USING_CANONICAL",
]
