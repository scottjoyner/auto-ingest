from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if (
    _PYDANTIC
):  # Prefer pydantic when installed; keep CLI local-first without network/deps.
    from pydantic import BaseModel, Field
else:
    BaseModel = object  # type: ignore

    def Field(default_factory=None, default=None):  # type: ignore
        return (
            field(default_factory=default_factory)
            if default_factory
            else field(default=default)
        )


class Route(str, Enum):
    auto = "auto"
    ORIGINAL = "ORIGINAL"
    REPURPOSE = "REPURPOSE"
    REWRITE = "REWRITE"
    RESEARCH_IDEATE = "RESEARCH_IDEATE"


class Format(str, Enum):
    x_post = "x_post"
    x_thread = "x_thread"
    linkedin = "linkedin"
    blog = "blog"
    newsletter = "newsletter"
    custom = "custom"


class State(str, Enum):
    captured = "captured"
    idea_review = "idea_review"
    brief_ready = "brief_ready"
    drafting = "drafting"
    verification = "verification"
    human_review = "human_review"
    approved = "approved"
    scheduler_ready = "scheduler_ready"
    scheduled = "scheduled"
    published = "published"
    feedback_24h = "feedback_24h"
    feedback_72h = "feedback_72h"
    learned = "learned"
    archived = "archived"


STATE_ORDER = [s.value for s in State]
ALLOWED_TRANSITIONS = dict(zip(STATE_ORDER, STATE_ORDER[1:]))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if _PYDANTIC:

    class ContentObject(BaseModel):
        id: str
        title: str
        slug: str
        route: str = Route.auto.value
        format: str = Format.x_post.value
        state: str = State.idea_review.value
        created_at: str = Field(default_factory=now_iso)
        updated_at: str = Field(default_factory=now_iso)
        source: str | None = None
        platform_targets: list[str] = Field(default_factory=list)
        score: int | None = None
        next_action: str = "Review idea and confirm route."
        human_approved: bool = False
        assumptions: list[str] = Field(default_factory=list)
        attribution: dict[str, Any] = Field(default_factory=dict)
        files: dict[str, str] = Field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            # Compatibility shim for pydantic v1/v2 without forcing a dependency pin.
            if hasattr(super(), "model_dump"):
                return super().model_dump()  # type: ignore[misc]
            return self.dict()

        def transition(self, new_state: str, *, force: bool = False) -> bool:
            return _transition(self, new_state, force=force)

else:

    @dataclass
    class ContentObject:  # type: ignore[no-redef]
        id: str
        title: str
        slug: str
        route: str = Route.auto.value
        format: str = Format.x_post.value
        state: str = State.idea_review.value
        created_at: str = field(default_factory=now_iso)
        updated_at: str = field(default_factory=now_iso)
        source: str | None = None
        platform_targets: list[str] = field(default_factory=list)
        score: int | None = None
        next_action: str = "Review idea and confirm route."
        human_approved: bool = False
        assumptions: list[str] = field(default_factory=list)
        attribution: dict[str, Any] = field(default_factory=dict)
        files: dict[str, str] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

        def transition(self, new_state: str, *, force: bool = False) -> bool:
            return _transition(self, new_state, force=force)


def _transition(obj: ContentObject, new_state: str, *, force: bool = False) -> bool:
    old = obj.state
    if old == new_state:
        return False
    expected = ALLOWED_TRANSITIONS.get(old)
    if expected == new_state or force:
        obj.state = new_state
        obj.updated_at = now_iso()
        return True
    raise ValueError(
        f"Invalid state transition: {old} -> {new_state}; expected {expected!r} or force=True"
    )
