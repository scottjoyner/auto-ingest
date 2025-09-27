"""Business logic for searching and inspecting transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Result

from ..config import settings
from ..embeddings import embed_texts
from ..neo4j_client import get_session


@dataclass
class Speaker:
    id: str
    label: Optional[str] = None


@dataclass
class SegmentSummary:
    id: Optional[str]
    idx: Optional[int]
    start: Optional[float]
    end: Optional[float]
    absolute_start: Optional[float]
    absolute_end: Optional[float]


@dataclass
class UtteranceSummary:
    id: Optional[str]
    text: Optional[str]
    start: Optional[float]
    end: Optional[float]
    absolute_start: Optional[float]
    absolute_end: Optional[float]
    speakers: List[Speaker]
    score: Optional[float]
    segment: Optional[SegmentSummary]


@dataclass
class LocationMetadata:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    label: Optional[str] = None
    source: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass
class TranscriptionHit:
    id: Optional[str]
    key: Optional[str]
    started_at: Optional[str]
    media_path: Optional[str]
    rttm_path: Optional[str]
    score: Optional[float]
    best_utterance: Optional[UtteranceSummary]
    location: Optional[LocationMetadata]


@dataclass
class TranscriptUtterance:
    id: Optional[str]
    text: Optional[str]
    start: Optional[float]
    end: Optional[float]
    absolute_start: Optional[float]
    absolute_end: Optional[float]
    speakers: List[Speaker]
    segment: Optional[SegmentSummary]


@dataclass
class TranscriptDetail:
    id: str
    key: Optional[str]
    started_at: Optional[str]
    media_path: Optional[str]
    rttm_path: Optional[str]
    utterances: List[TranscriptUtterance]
    location: Optional[LocationMetadata]


class TranscriptService:
    """High level entry point for transcript operations."""

    _SEARCH_CYPHER = """
        CALL {
            WITH $qvec AS q, $k AS k, $idx AS idx
            CALL db.index.vector.queryNodes(idx, k, q)
            YIELD node AS t, score AS t_score
            OPTIONAL MATCH (t)-[:RECORDED_AT]->(loc:Location)
            RETURN collect({t: t, t_score: t_score, loc: loc}) AS t_hits
        }

        CALL {
            WITH $qvec AS q, $m AS m, $utt_idx AS uidx
            CALL db.index.vector.queryNodes(uidx, m, q)
            YIELD node AS u, score AS u_score
            MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u)
            OPTIONAL MATCH (u)-[:OF_SEGMENT]->(s:Segment)
            OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)
            WITH t, u, u_score, s, collect(DISTINCT {id: sp.id, label: sp.label}) AS sp_list
            RETURN collect({
                t_id: t.id, t_key: t.key,
                u_id: u.id, u_text: u.text, u_start: u.start, u_end: u.end,
                u_abs_start: u.absolute_start, u_abs_end: u.absolute_end,
                s_id: s.id, s_idx: s.idx, s_start: s.start, s_end: s.end,
                s_abs_start: s.absolute_start, s_abs_end: s.absolute_end,
                speakers: sp_list,
                u_score: u_score
            }) AS utt_hits
        }

        WITH t_hits, utt_hits
        UNWIND t_hits AS th
        WITH th, [uh IN utt_hits WHERE uh.t_id = th.t.id] AS utt_for_t
        UNWIND CASE WHEN size(utt_for_t) = 0 THEN [NULL] ELSE utt_for_t END AS uh
        WITH th, uh
        ORDER BY uh.u_score DESC
        WITH th, collect(uh)[0] AS best
        RETURN
            th.t.id                    AS t_id,
            th.t.key                   AS t_key,
            th.t.started_at            AS t_started_at,
            th.t.source_media          AS t_media,
            th.t.source_rttm           AS t_rttm,
            th.t_score                 AS t_score,
            best.u_id                  AS u_id,
            best.u_text                AS u_text,
            best.u_start               AS u_start,
            best.u_end                 AS u_end,
            best.u_abs_start           AS u_abs_start,
            best.u_abs_end             AS u_abs_end,
            best.speakers              AS speakers,
            best.s_id                  AS s_id,
            best.s_idx                 AS s_idx,
            best.s_start               AS s_start,
            best.s_end                 AS s_end,
            best.s_abs_start           AS s_abs_start,
            best.s_abs_end             AS s_abs_end,
            best.u_score               AS u_score,
            th.loc.latitude            AS loc_latitude,
            th.loc.longitude           AS loc_longitude,
            th.loc.label               AS loc_label,
            th.loc.source              AS loc_source
        ORDER BY t_score DESC
        LIMIT $k;
    """

    _TRANSCRIPT_DETAIL_CYPHER = """
        MATCH (t:Transcription {id: $transcription_id})
        OPTIONAL MATCH (t)-[:RECORDED_AT]->(loc:Location)
        WITH t, loc
        OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u:Utterance)
        OPTIONAL MATCH (u)-[:OF_SEGMENT]->(s:Segment)
        OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)
        WITH t, loc, u, s, collect(DISTINCT {id: sp.id, label: sp.label}) AS sp_list
        ORDER BY u.absolute_start
        RETURN t, loc, collect({
            id: u.id,
            text: u.text,
            start: u.start,
            end: u.end,
            absolute_start: u.absolute_start,
            absolute_end: u.absolute_end,
            speakers: sp_list,
            segment: {
                id: s.id,
                idx: s.idx,
                start: s.start,
                end: s.end,
                absolute_start: s.absolute_start,
                absolute_end: s.absolute_end
            }
        }) AS utterances
    """

    def _format_location(self, record: Dict[str, Any]) -> Optional[LocationMetadata]:
        latitude = record.get("loc_latitude") if record else None
        longitude = record.get("loc_longitude") if record else None
        label = record.get("loc_label") if record else None
        source = record.get("loc_source") if record else None
        if latitude is None and longitude is None and label is None:
            return None
        try:
            lat_val = float(latitude) if latitude is not None else None
            lon_val = float(longitude) if longitude is not None else None
        except (TypeError, ValueError):
            lat_val = None
            lon_val = None
        return LocationMetadata(latitude=lat_val, longitude=lon_val, label=label, source=source)

    def search(
        self,
        query: str,
        *,
        mode: str = "both",
        top_k: int | None = None,
        candidate_pool: int | None = None,
    ) -> Dict[str, List[TranscriptionHit]]:
        """Search transcripts returning hits per embedding variant."""
        query = (query or "").strip()
        if not query:
            return {"v1": [], "v2": []}

        vectors = embed_texts([query])
        if not vectors:
            return {"v1": [], "v2": []}

        qvec = vectors[0]
        top_k = top_k or settings.ui.default_top_k
        candidate_pool = candidate_pool or settings.ui.default_pool_size

        response: Dict[str, List[TranscriptionHit]] = {"v1": [], "v2": []}
        with get_session() as session:
            if mode in {"both", "v1"}:
                result = session.run(
                    self._SEARCH_CYPHER,
                    qvec=qvec,
                    k=top_k,
                    m=candidate_pool,
                    idx=settings.neo4j.transcription_index_v1,
                    utt_idx=settings.neo4j.utterance_index,
                )
                response["v1"] = self._parse_search_results(result)
            if mode in {"both", "v2"}:
                result = session.run(
                    self._SEARCH_CYPHER,
                    qvec=qvec,
                    k=top_k,
                    m=candidate_pool,
                    idx=settings.neo4j.transcription_index_v2,
                    utt_idx=settings.neo4j.utterance_index,
                )
                response["v2"] = self._parse_search_results(result)
        return response

    def _parse_search_results(self, result: Result) -> List[TranscriptionHit]:
        hits: List[TranscriptionHit] = []
        for record in result:
            location = self._format_location(record.data())
            speakers = [Speaker(**sp) for sp in record.get("speakers") or []]
            segment_values = {
                "id": record.get("s_id"),
                "idx": record.get("s_idx"),
                "start": record.get("s_start"),
                "end": record.get("s_end"),
                "absolute_start": record.get("s_abs_start"),
                "absolute_end": record.get("s_abs_end"),
            }
            has_segment = any(value is not None for value in segment_values.values())
            segment = SegmentSummary(**segment_values) if has_segment else None
            utterance = UtteranceSummary(
                id=record.get("u_id"),
                text=record.get("u_text"),
                start=record.get("u_start"),
                end=record.get("u_end"),
                absolute_start=record.get("u_abs_start"),
                absolute_end=record.get("u_abs_end"),
                speakers=speakers,
                score=record.get("u_score"),
                segment=segment,
            ) if record.get("u_id") else None
            hits.append(
                TranscriptionHit(
                    id=record.get("t_id"),
                    key=record.get("t_key"),
                    started_at=record.get("t_started_at"),
                    media_path=record.get("t_media"),
                    rttm_path=record.get("t_rttm"),
                    score=record.get("t_score"),
                    best_utterance=utterance,
                    location=location,
                )
            )
        return hits

    def get_transcript(self, transcription_id: str) -> Optional[TranscriptDetail]:
        if not transcription_id:
            return None
        with get_session() as session:
            record = session.run(
                self._TRANSCRIPT_DETAIL_CYPHER, transcription_id=transcription_id
            ).single()
        if not record:
            return None

        t_node = record["t"]
        loc_node = record["loc"]
        utterances_data = record.get("utterances") or []
        location = None
        if loc_node is not None:
            location = LocationMetadata(
                latitude=_safe_float(loc_node.get("latitude")),
                longitude=_safe_float(loc_node.get("longitude")),
                label=loc_node.get("label"),
                source=loc_node.get("source"),
            )
        utterances: List[TranscriptUtterance] = []
        for item in utterances_data:
            speakers = [Speaker(**sp) for sp in item.get("speakers") or []]
            segment_data = item.get("segment") or {}
            has_segment = any(
                value is not None for value in segment_data.values()
            )
            segment = (
                SegmentSummary(
                    id=segment_data.get("id"),
                    idx=segment_data.get("idx"),
                    start=segment_data.get("start"),
                    end=segment_data.get("end"),
                    absolute_start=segment_data.get("absolute_start"),
                    absolute_end=segment_data.get("absolute_end"),
                )
                if has_segment
                else None
            )
            utterances.append(
                TranscriptUtterance(
                    id=item.get("id"),
                    text=item.get("text"),
                    start=item.get("start"),
                    end=item.get("end"),
                    absolute_start=item.get("absolute_start"),
                    absolute_end=item.get("absolute_end"),
                    speakers=speakers,
                    segment=segment,
                )
            )
        return TranscriptDetail(
            id=t_node.get("id"),
            key=t_node.get("key"),
            started_at=t_node.get("started_at"),
            media_path=t_node.get("source_media"),
            rttm_path=t_node.get("source_rttm"),
            utterances=utterances,
            location=location,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

