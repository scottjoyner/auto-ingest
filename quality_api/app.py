"""Flask application factory for the quality toolkit."""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict

from flask import (
    Blueprint,
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from .config import settings
from .neo4j_client import close_driver
from .services.transcript_service import TranscriptService


def create_app() -> Flask:
    app = Flask(__name__)
    service = TranscriptService()

    bp = Blueprint(
        "quality",
        __name__,
        template_folder="ui/templates",
        static_folder="ui/static",
    )

    @bp.route("/")
    def index():
        query = request.args.get("q", "")
        mode = (request.args.get("mode") or settings.ui.default_mode).lower()
        top_k = request.args.get("k", type=int) or settings.ui.default_top_k
        pool = request.args.get("m", type=int) or settings.ui.default_pool_size
        results = None
        if query:
            results = service.search(query, mode=mode, top_k=top_k, candidate_pool=pool)
        return render_template(
            "search.html",
            initial_query=query,
            initial_mode=mode,
            initial_k=top_k,
            initial_m=pool,
            initial_results=results,
            defaults=settings.ui,
        )

    @bp.route("/search", methods=["POST"])
    def search():
        if request.is_json:
            payload = request.get_json(force=True)
        else:
            payload = request.form.to_dict()
        query = (payload.get("query") or payload.get("q") or "").strip()
        mode = (payload.get("mode") or settings.ui.default_mode).lower()
        top_k = int(payload.get("k") or settings.ui.default_top_k)
        pool = int(payload.get("m") or settings.ui.default_pool_size)
        results = service.search(query, mode=mode, top_k=top_k, candidate_pool=pool)
        if request.is_json:
            return jsonify({"query": query, "mode": mode, "results": _serialize_hits(results)})
        return render_template(
            "search.html",
            initial_query=query,
            initial_mode=mode,
            initial_k=top_k,
            initial_m=pool,
            initial_results=results,
            defaults=settings.ui,
        )

    @bp.route("/api/search", methods=["POST"])
    def api_search():
        payload = request.get_json(force=True)
        query = (payload.get("query") or "").strip()
        mode = (payload.get("mode") or settings.ui.default_mode).lower()
        top_k = int(payload.get("k") or settings.ui.default_top_k)
        pool = int(payload.get("m") or settings.ui.default_pool_size)
        results = service.search(query, mode=mode, top_k=top_k, candidate_pool=pool)
        return jsonify({"query": query, "mode": mode, "results": _serialize_hits(results)})

    @bp.route("/transcripts/<transcription_id>")
    def transcript_detail(transcription_id: str):
        transcript = service.get_transcript(transcription_id)
        if not transcript:
            abort(404)
        return render_template("transcript_detail.html", transcript=transcript)

    @bp.route("/api/transcripts/<transcription_id>")
    def transcript_detail_api(transcription_id: str):
        transcript = service.get_transcript(transcription_id)
        if not transcript:
            abort(404)
        return jsonify(_serialize_transcript(transcript))

    @bp.route("/media")
    def media():
        path = request.args.get("path", "")
        if not _is_safe_media(path):
            abort(404)
        guessed = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return send_file(path, mimetype=guessed, as_attachment=False, conditional=True)

    @bp.route("/speaker/rename", methods=["POST"])
    def speaker_rename():
        payload = request.get_json(force=True) if request.is_json else request.form
        speaker_id = (payload.get("speaker_id") or "").strip()
        new_label = (payload.get("new_label") or "").strip()
        if not speaker_id or not new_label:
            return jsonify({"ok": False, "error": "speaker_id and new_label required"}), 400

        cypher = """
            MATCH (sp:Speaker {id:$id})
            SET sp.label = $label, sp.updated_at = datetime()
            WITH sp
            MATCH (s:Segment) WHERE s.speaker_id = $id
            SET s.speaker_label = $label
            RETURN sp.id AS id, sp.label AS label
        """
        from .neo4j_client import get_session  # lazy import to avoid cycles

        with get_session() as session:
            record = session.run(cypher, id=speaker_id, label=new_label).single()
        if not record:
            return jsonify({"ok": False, "error": "Speaker not found"}), 404
        return jsonify({"ok": True, "id": record["id"], "label": record["label"]})

    app.register_blueprint(bp)

    @app.teardown_appcontext  # pragma: no cover - teardown hook
    def _shutdown(exception: Exception | None) -> None:  # noqa: ARG001
        close_driver()

    return app


def _is_safe_media(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        base = os.path.realpath(settings.media.base_path)
        return real.startswith(base) and os.path.isfile(real)
    except Exception:
        return False


def _serialize_hits(results: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for variant, hits in results.items():
        payload[variant] = [
            {
                "id": hit.id,
                "key": hit.key,
                "started_at": hit.started_at,
                "media_path": hit.media_path,
                "rttm_path": hit.rttm_path,
                "score": hit.score,
                "best_utterance": _serialize_utterance(hit.best_utterance),
                "location": _serialize_location(hit.location),
            }
            for hit in hits
        ]
    return payload


def _serialize_utterance(utterance):
    if not utterance:
        return None
    return {
        "id": utterance.id,
        "text": utterance.text,
        "start": utterance.start,
        "end": utterance.end,
        "absolute_start": utterance.absolute_start,
        "absolute_end": utterance.absolute_end,
        "score": getattr(utterance, "score", None),
        "speakers": [
            {"id": speaker.id, "label": speaker.label}
            for speaker in getattr(utterance, "speakers", [])
        ],
        "segment": _serialize_segment(getattr(utterance, "segment", None)),
    }


def _serialize_segment(segment):
    if not segment:
        return None
    return {
        "id": segment.id,
        "idx": segment.idx,
        "start": segment.start,
        "end": segment.end,
        "absolute_start": segment.absolute_start,
        "absolute_end": segment.absolute_end,
    }


def _serialize_location(location):
    if not location:
        return None
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "label": location.label,
        "source": location.source,
    }


def _serialize_transcript(transcript):
    return {
        "id": transcript.id,
        "key": transcript.key,
        "started_at": transcript.started_at,
        "media_path": transcript.media_path,
        "rttm_path": transcript.rttm_path,
        "location": _serialize_location(transcript.location),
        "utterances": [
            {
                "id": u.id,
                "text": u.text,
                "start": u.start,
                "end": u.end,
                "absolute_start": u.absolute_start,
                "absolute_end": u.absolute_end,
                "speakers": [{"id": sp.id, "label": sp.label} for sp in u.speakers],
                "segment": _serialize_segment(u.segment),
            }
            for u in transcript.utterances
        ],
    }

