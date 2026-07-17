"""Data models for the research-scripted shorts package.

A :class:`Brief` is the curated research content for one Topic. A :class:`Plan`
is a batch of :class:`PlannedShort` items plus iteration metadata, persisted as
JSON so planning is reproducible and refinable.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SourceRef:
    """A research source (paper / vault doc / concept) backing a short."""

    kind: str  # "paper" | "vault" | "concept"
    ref_id: str
    title: str
    url: Optional[str] = None
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceRef":
        return cls(**d)


@dataclass
class Brief:
    """Curated research content for a Topic, ready to be turned into shorts."""

    topic: str
    title: str
    hook: str
    points: List[str] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    topic_type: Optional[str] = None  # paper | concept | debate | opinion | utterance
    series_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "hook": self.hook,
            "points": list(self.points),
            "sources": [s.to_dict() for s in self.sources],
            "tags": list(self.tags),
            "topic_type": self.topic_type,
            "series_id": self.series_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Brief":
        return cls(
            topic=d["topic"],
            title=d.get("title", d["topic"]),
            hook=d.get("hook", ""),
            points=list(d.get("points", [])),
            sources=[SourceRef.from_dict(s) for s in d.get("sources", [])],
            tags=list(d.get("tags", [])),
            topic_type=d.get("topic_type"),
            series_id=d.get("series_id"),
        )


@dataclass
class Shot:
    """One highway-footage clip used as B-roll for part of a short."""

    clip_key: str
    fr_path: str
    t_sec: float
    dur: float
    mph: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Shot":
        return cls(**d)


@dataclass
class Cue:
    """A timed caption line (the research script) overlaid on the B-roll."""

    start: float
    end: float
    text: str
    kind: str = "line"  # "hook" | "point" | "source" | "line"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cue":
        return cls(**d)


@dataclass
class PlannedShort:
    """One planned short: a script + a highway B-roll shot list."""

    id: str
    brief_topic: str
    title: str
    cues: List[Cue] = field(default_factory=list)
    shots: List[Shot] = field(default_factory=list)
    notes: str = ""
    status: str = "planned"  # planned | rendered | rejected
    out_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    persona: Optional[str] = None  # stylized | photo | video (on-screen personality)

    def duration(self) -> float:
        if not self.cues:
            return sum(s.dur for s in self.shots)
        return max(c.end for c in self.cues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "brief_topic": self.brief_topic,
            "title": self.title,
            "cues": [c.to_dict() for c in self.cues],
            "shots": [s.to_dict() for s in self.shots],
            "notes": self.notes,
            "status": self.status,
            "out_path": self.out_path,
            "thumbnail_path": self.thumbnail_path,
            "persona": self.persona,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlannedShort":
        return cls(
            id=d["id"],
            brief_topic=d["brief_topic"],
            title=d["title"],
            cues=[Cue.from_dict(c) for c in d.get("cues", [])],
            shots=[Shot.from_dict(s) for s in d.get("shots", [])],
            notes=d.get("notes", ""),
            status=d.get("status", "planned"),
            out_path=d.get("out_path"),
            thumbnail_path=d.get("thumbnail_path"),
            persona=d.get("persona"),
        )


@dataclass
class Plan:
    """A batch of planned shorts for one Topic (the iteration artifact)."""

    topic: str
    brief: Brief
    shorts: List[PlannedShort] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    iteration: int = 1
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "topic": self.topic,
            "iteration": self.iteration,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "brief": self.brief.to_dict(),
            "shorts": [s.to_dict() for s in self.shorts],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            plan_id=d.get("plan_id", uuid.uuid4().hex[:12]),
            topic=d["topic"],
            brief=Brief.from_dict(d["brief"]),
            shorts=[PlannedShort.from_dict(s) for s in d.get("shorts", [])],
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            iteration=d.get("iteration", 1),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Plan":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
