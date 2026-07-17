"""Local analytics store + ingestion stub for research-scripted shorts.

Records real post performance (views, retention, CTR) and compares it against the
pre-publish virality prediction from :mod:`auto_ingest.shorts.virality`.

No network calls. Metrics are ingested from local platform exports (CSV/JSON)
the user supplies later, and persisted as JSONL under ``~/.config/auto-ingest/``.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_ingest.shorts.models import PlannedShort

log = logging.getLogger("shorts.metrics")

DEFAULT_STORE = Path(
    os.environ.get("AUTO_INGEST_METRICS",
                   str(Path.home() / ".config" / "auto-ingest" / "metrics.jsonl"))
)


@dataclass
class MetricsRecord:
    short_id: str
    platform: str
    topic: str
    published_at: str
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    watch_time_s: Optional[float] = None
    avg_view_pct: Optional[float] = None
    ctr: Optional[float] = None
    fetched_at: str = ""
    pred_virality: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsRecord":
        return cls(
            short_id=d["short_id"],
            platform=d["platform"],
            topic=d["topic"],
            published_at=d.get("published_at", ""),
            views=d.get("views"),
            likes=d.get("likes"),
            comments=d.get("comments"),
            shares=d.get("shares"),
            watch_time_s=d.get("watch_time_s"),
            avg_view_pct=d.get("avg_view_pct"),
            ctr=d.get("ctr"),
            fetched_at=d.get("fetched_at", ""),
            pred_virality=d.get("pred_virality"),
        )


def _store_path(path: Optional[Path] = None) -> Path:
    return Path(path) if path is not None else DEFAULT_STORE


def record_metric(rec: MetricsRecord, *, path: Optional[Path] = None) -> None:
    """Append a metrics record to the JSONL store."""
    sp = _store_path(path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec.to_dict()) + "\n")


def load_metrics(*, path: Optional[Path] = None) -> List[MetricsRecord]:
    """Load all metrics records from the JSONL store."""
    sp = _store_path(path)
    if not sp.exists():
        return []
    records: List[MetricsRecord] = []
    for line in sp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(MetricsRecord.from_dict(json.loads(line)))
        except Exception as e:
            log.warning("Skipping malformed metrics line: %s", e)
    return records


def upsert_metric(rec: MetricsRecord, *, path: Optional[Path] = None) -> None:
    """Store a record, replacing any existing one with the same (short_id, platform)."""
    sp = _store_path(path)
    existing = load_metrics(path=sp)
    key = (rec.short_id, rec.platform)
    merged = [r for r in existing if (r.short_id, r.platform) != key]
    merged.append(rec)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r.to_dict()) + "\n")


def publish_prediction(short_id: str, platform: str, pred: float,
                       *, topic: str = "", published_at: str = "",
                       path: Optional[Path] = None) -> None:
    """Store / update the virality prediction for a short (compared later vs actuals)."""
    sp = _store_path(path)
    existing = {r.short_id: r for r in load_metrics(path=sp)
                if r.platform == platform}
    if short_id in existing:
        rec = existing[short_id]
        rec.pred_virality = pred
    else:
        rec = MetricsRecord(short_id=short_id, platform=platform, topic=topic,
                            published_at=published_at, pred_virality=pred)
    upsert_metric(rec, path=sp)


_COLUMN_ALIASES = {
    "short_id": ("short_id", "id", "video_id", "key", "external_id"),
    "views": ("views", "view_count", "video_views", "plays", "impressions"),
    "likes": ("likes", "like_count", "favorites"),
    "comments": ("comments", "comment_count", "comments_count"),
    "shares": ("shares", "share_count", "reposts"),
    "watch_time_s": ("watch_time_s", "watch_time", "watch_time_seconds",
                     "total_watch_time"),
    "avg_view_pct": ("avg_view_pct", "average_view_percentage", "avg_view_percentage",
                     "avg_view_duration_pct", "retention", "view_rate"),
    "ctr": ("ctr", "click_through_rate", "ctr_%", "impression_ctr"),
    "topic": ("topic", "title", "video_title"),
    "published_at": ("published_at", "publish_date", "publish_time", "created_at"),
    "fetched_at": ("fetched_at", "exported_at", "report_date"),
}


def _pick(d: Dict[str, Any], names: tuple) -> Any:
    lk = {k.lower(): k for k in d.keys()}
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
        if n.lower() in lk:
            v = d[lk[n.lower()]]
            if v not in (None, ""):
                return v
    return None


def _coerce_num(v: Any, as_int: bool = True) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v) if as_int else float(v)
    s = str(v).strip().replace(",", "").replace("%", "")
    if s == "":
        return None
    try:
        return int(float(s)) if as_int else float(s)
    except ValueError:
        return None


def _coerce_str(v: Any) -> str:
    return "" if v is None else str(v)


def ingest_dicts(platform: str, rows: List[Dict[str, Any]], *,
                 short_id_field: str = "short_id",
                 path: Optional[Path] = None) -> int:
    """Convert a list of platform-export dicts into MetricsRecords (best-effort).

    Missing fields are left as ``None``. Returns the number of records written.
    """
    written = 0
    for row in rows:
        if not row:
            continue
        rec = MetricsRecord(
            short_id=_coerce_str(row.get(short_id_field) or _pick(row, _COLUMN_ALIASES["short_id"])),
            platform=platform,
            topic=_coerce_str(_pick(row, _COLUMN_ALIASES["topic"])),
            published_at=_coerce_str(_pick(row, _COLUMN_ALIASES["published_at"])),
            views=_coerce_num(_pick(row, _COLUMN_ALIASES["views"]), as_int=True),
            likes=_coerce_num(_pick(row, _COLUMN_ALIASES["likes"]), as_int=True),
            comments=_coerce_num(_pick(row, _COLUMN_ALIASES["comments"]), as_int=True),
            shares=_coerce_num(_pick(row, _COLUMN_ALIASES["shares"]), as_int=True),
            watch_time_s=_coerce_num(_pick(row, _COLUMN_ALIASES["watch_time_s"]), as_int=False),
            avg_view_pct=_coerce_num(_pick(row, _COLUMN_ALIASES["avg_view_pct"]), as_int=False),
            ctr=_coerce_num(_pick(row, _COLUMN_ALIASES["ctr"]), as_int=False),
            fetched_at=_coerce_str(_pick(row, _COLUMN_ALIASES["fetched_at"]))
            or _coerce_str(row.get("fetched_at")),
        )
        if not rec.short_id:
            log.warning("Skipping row with no short_id: %s", row)
            continue
        upsert_metric(rec, path=path)
        written += 1
    return written


def ingest_platform_csv(platform: str, csv_path: str, *,
                        short_id_field: str = "short_id",
                        path: Optional[Path] = None) -> int:
    """Read a platform analytics CSV export and ingest it as MetricsRecords."""
    p = Path(csv_path)
    if not p.exists():
        log.error("Metrics CSV not found: %s", csv_path)
        return 0
    rows: List[Dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(dict(r))
    return ingest_dicts(platform, rows, short_id_field=short_id_field, path=path)


def store_prediction_for_short(item: PlannedShort, platform: str, *,
                               topic: str = "", published_at: str = "",
                               path: Optional[Path] = None) -> None:
    """Convenience: score a PlannedShort via virality and persist the prediction."""
    from auto_ingest.shorts import virality

    pred = virality.score_short(item).total
    publish_prediction(item.id, platform, pred, topic=topic or item.brief_topic,
                       published_at=published_at, path=path)
