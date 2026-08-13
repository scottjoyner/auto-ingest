#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build transcript summary and task JSON sidecars using an Ollama endpoint.

The implementation is deliberately sequential. It supports transcript text,
CSV, JSON, and WebVTT inputs, robustly extracts JSON from model responses,
normalizes summary/task schemas, and can derive a tasks sidecar from an existing
summary without another full-transcript model call.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summaries")


class OllamaClient:
    """Minimal synchronous Ollama generate client with bounded retries."""

    def __init__(self, base_url: str):
        parsed = urlparse(base_url)
        self._is_https = parsed.scheme == "https"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if self._is_https else 11434)
        self._base_path = parsed.path.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        retries: int = 3,
        timeout: float = 120.0,
        options: Dict[str, Any] | None = None,
    ) -> str:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        body = json.dumps(payload).encode("utf-8")
        path = f"{self._base_path}/api/generate" if self._base_path else "/api/generate"

        for attempt in range(1, retries + 1):
            try:
                conn_cls = (
                    http.client.HTTPSConnection
                    if self._is_https
                    else http.client.HTTPConnection
                )
                conn = conn_cls(self._host, self._port, timeout=timeout)
                conn.request(
                    "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
                doc = json.loads(data.decode("utf-8", errors="replace"))
                out = doc.get("response") or doc.get("text")
                if not out:
                    raise RuntimeError("No 'response' in Ollama output")
                return str(out)
            except Exception as exc:
                log.warning("Ollama call failed (%d/%d): %s", attempt, retries, exc)
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Ollama call failed after retries")


TRANSCRIPT_PATTERNS = [
    "{stem}_large-v3_transcription.txt",
    "{stem}_BC_medium_transcription.txt",
    "{stem}_medium_transcription.txt",
    "{stem}_BC_transcription.csv",
    "{stem}_transcription.txt",
    "{stem}_transcription.csv",
    "{stem}.json",
    "{stem}.txt",
    "{stem}.vtt",
]

SUMMARY_SCHEMA_HINT = {
    "version": "1.0",
    "language": "<ISO 639-1 code>",
    "summary": "<5-9 sentence summary>",
    "key_points": [],
    "topics": [],
    "people": [],
    "organizations": [],
    "places": [],
    "quality_notes": "",
    "tasks": [
        {
            "title": "<concise action>",
            "description": "<what needs to be done>",
            "labels": [],
            "priority": "medium",
            "owner_hint": "<team/role/person if obvious>",
        }
    ],
}

PROMPT_SCHEMA = (
    "You are a precise summarization model. Output STRICT JSON only matching this schema: "
    + json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False)
    + "\nRules:\n"
    "- Output ONLY JSON, no markdown, no preface.\n"
    "- If unknown, use empty array or empty string.\n"
    "- Ensure 'tasks' is an array of task objects with 'title' and 'description'.\n"
    "\nTranscript begins:\n\n"
)
PROMPT_END = "END."

TASKS_SCHEMA_HINT = {
    "tasks": [
        {
            "title": "<concise action>",
            "description": "<what needs to be done>",
            "labels": [],
            "priority": "medium",
            "owner_hint": "<team/role/person if obvious>",
            "agent": {
                "name": "<suggested agent>",
                "confidence": 0.6,
                "rationale": "<why this agent>",
            },
            "plan": [
                {
                    "step": 1,
                    "action": "<what to do>",
                    "tool": "<system/tool name>",
                    "operation": "<endpoint or verb>",
                    "inputs": {"key": "value"},
                    "expected_output": "<artifact/confirmation>",
                }
            ],
        }
    ]
}

TASKS_PROMPT_PREFIX = (
    "Extract actionable tasks from the transcript below. Output STRICT JSON ONLY matching this schema: "
    + json.dumps(TASKS_SCHEMA_HINT, ensure_ascii=False)
    + "Rules:"
    "- Provide 0..N tasks."
    "- Keep titles imperative and <= 12 words."
    "- Include an 'agent' suggestion with confidence (0..1) and short rationale."
    "- Include a 'plan' with 1..6 ordered steps; keep tools generic."
    "- Prefer agent-neutral owner_hint like 'DevOps', 'Finance', 'Legal', 'Personal'."
    "- Use labels for routing."
    "Transcript begins:"
)

STEM_RE = re.compile(r"^(\d{4}_\d{4}_\d{6}|\d{12,})(?:_[0-9]{6})?", re.ASCII)


def _detect_stem(path: Path) -> Optional[str]:
    match = STEM_RE.match(path.stem)
    return match.group(0) if match else None


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        log.warning("Failed reading %s: %s", path, exc)
        return None


def _read_csv_concat_text(path: Path) -> Optional[str]:
    try:
        rows: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return None
            preferred = next(
                (
                    name
                    for name in reader.fieldnames
                    if name
                    and name.lower() in ("text", "utterance", "transcript", "content")
                ),
                None,
            )
            for row in reader:
                if preferred and row.get(preferred):
                    rows.append(str(row[preferred]))
                else:
                    values = [
                        str(value)
                        for value in row.values()
                        if isinstance(value, str) and value.strip()
                    ]
                    if values:
                        rows.append(" ".join(values))
        return "\n".join(rows).strip() or None
    except Exception as exc:
        log.warning("Failed reading CSV %s: %s", path, exc)
        return None


def _extract_first_json_object(value: str) -> Optional[str]:
    text = value.strip()
    if text.startswith("```"):
        text = text.strip("`\n ")
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n")

    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : index + 1]
    return None


def _normalize_payload(any_payload: Any) -> Dict[str, Any]:
    if isinstance(any_payload, list):
        any_payload = next(
            (element for element in any_payload if isinstance(element, dict)),
            any_payload,
        )
    if isinstance(any_payload, str):
        any_payload = {"summary": any_payload}
    if not isinstance(any_payload, dict):
        any_payload = {}

    any_payload.setdefault("version", "1.0")
    any_payload.setdefault("language", "")
    any_payload.setdefault("summary", "")
    any_payload.setdefault("key_points", [])
    any_payload.setdefault("topics", [])
    any_payload.setdefault("people", [])
    any_payload.setdefault("organizations", [])
    any_payload.setdefault("places", [])
    any_payload.setdefault("quality_notes", "")
    tasks = any_payload.get("tasks")
    any_payload["tasks"] = _normalize_tasks(tasks) if isinstance(tasks, list) else []
    return any_payload


def _normalize_tasks(tasks_any: Any) -> List[Dict[str, Any]]:
    """Normalize every valid task; never silently drop later tasks."""
    if not isinstance(tasks_any, list):
        return []

    out: List[Dict[str, Any]] = []
    for task in tasks_any:
        if not isinstance(task, dict):
            continue
        title = str(task.get("title") or "").strip()
        description = str(task.get("description") or "").strip()
        if not title and not description:
            continue

        labels_raw = task.get("labels") if isinstance(task.get("labels"), list) else []
        labels = [str(item).strip() for item in labels_raw if str(item).strip()]
        priority = str(task.get("priority") or "medium").strip().lower()
        if priority not in ("low", "medium", "high", "urgent"):
            priority = "medium"
        owner_hint = str(task.get("owner_hint") or "").strip()

        agent_raw = task.get("agent") if isinstance(task.get("agent"), dict) else {}
        agent_name = str(agent_raw.get("name") or "").strip()
        try:
            confidence = float(agent_raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(agent_raw.get("rationale") or "").strip()

        plan_raw = task.get("plan") if isinstance(task.get("plan"), list) else []
        plan: List[Dict[str, Any]] = []
        for default_step, step in enumerate(plan_raw[:6], 1):
            if not isinstance(step, dict):
                continue
            try:
                step_number = int(step.get("step", default_step))
            except (TypeError, ValueError):
                step_number = default_step
            plan.append(
                {
                    "step": step_number,
                    "action": str(step.get("action") or "").strip()[:200],
                    "tool": str(step.get("tool") or "").strip()[:50],
                    "operation": str(step.get("operation") or "").strip()[:80],
                    "inputs": step.get("inputs")
                    if isinstance(step.get("inputs"), dict)
                    else {},
                    "expected_output": str(step.get("expected_output") or "").strip()[:200],
                }
            )

        out.append(
            {
                "title": title[:140],
                "description": description[:1000],
                "labels": list(dict.fromkeys(labels))[:10],
                "priority": priority,
                "owner_hint": owner_hint[:100],
                "agent": {
                    "name": agent_name[:60],
                    "confidence": confidence,
                    "rationale": rationale[:200],
                },
                "plan": plan,
            }
        )
    return out


def _choose_best_path(
    dirpath: Path, stem: str, candidates: List[Path]
) -> Optional[Path]:
    names = {path.name for path in candidates}
    for pattern in TRANSCRIPT_PATTERNS:
        expected = pattern.format(stem=stem)
        if expected in names:
            return dirpath / expected
    for pattern in TRANSCRIPT_PATTERNS:
        candidate = dirpath / pattern.format(stem=stem)
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def _discover_stems(roots: List[Path]) -> List[Tuple[Path, str, List[Path]]]:
    buckets: Dict[Tuple[str, str], List[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            directory = Path(dirpath)
            for name in files:
                path = directory / name
                stem = _detect_stem(path)
                if stem:
                    buckets.setdefault((str(directory), stem), []).append(path)
    return [
        (Path(directory), stem, paths)
        for (directory, stem), paths in buckets.items()
    ]


def _auto_sleep(state: Dict[str, Any], success: bool, last_duration: float) -> None:
    if success:
        state["consec_fail"] = 0
    else:
        state["consec_fail"] = int(state.get("consec_fail", 0)) + 1
    base = 0.25 * max(0.0, float(last_duration))
    penalty = 0.4 * float(state.get("consec_fail", 0))
    sleep_min = float(state.get("sleep_min", 0.1))
    sleep_max = float(state.get("sleep_max", 2.0))
    sleep_sec = min(sleep_max, max(sleep_min, base + penalty))
    if state.get("auto_throttle", False):
        time.sleep(sleep_sec)


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        log.warning("Failed reading JSON %s: %s", path, exc)
        return None


def _read_vtt_to_text(path: Path) -> Optional[str]:
    try:
        lines: List[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                value = line.rstrip("\n")
                if not value or value.upper().startswith("WEBVTT"):
                    continue
                if re.fullmatch(r"\d+", value):
                    continue
                timestamp = re.match(
                    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}",
                    value,
                ) or re.match(
                    r"^\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}\.\d{3}",
                    value,
                )
                if timestamp:
                    continue
                lines.append(value)
        return "\n".join(lines).strip() or None
    except Exception as exc:
        log.warning("Failed reading VTT %s: %s", path, exc)
        return None


_AGENT_MAP = {
    "DevOps": {
        "name": "DevOpsAgent",
        "labels": ["DevOps", "Kubernetes", "deploy", "cluster", "ingress"],
        "plan": [
            {
                "action": "Open incident or ticket",
                "tool": "ticketing",
                "operation": "create",
                "expected_output": "INC- id",
            },
            {
                "action": "Prepare rollout plan",
                "tool": "docs",
                "operation": "create",
                "expected_output": "doc link",
            },
        ],
    },
    "Finance": {
        "name": "FinanceAgent",
        "labels": ["invoice", "payment", "budget", "expense"],
        "plan": [
            {
                "action": "Log expense",
                "tool": "finance",
                "operation": "record_expense",
                "expected_output": "entry id",
            }
        ],
    },
    "Legal": {
        "name": "LegalAgent",
        "labels": ["nda", "contract", "agreement"],
        "plan": [
            {
                "action": "Draft document",
                "tool": "docs",
                "operation": "create",
                "expected_output": "draft link",
            }
        ],
    },
    "Personal": {
        "name": "PersonalAssistant",
        "labels": ["Errand", "Shopping", "Family", "Health"],
        "plan": [
            {
                "action": "Add calendar reminder",
                "tool": "calendar",
                "operation": "create_event",
                "expected_output": "event link",
            }
        ],
    },
}

_GENERIC_PLAN = [
    {
        "action": "Create task",
        "tool": "ticketing",
        "operation": "create",
        "expected_output": "task id",
    },
    {
        "action": "Notify stakeholders",
        "tool": "email",
        "operation": "send",
        "expected_output": "sent confirmation",
    },
]


def _enrich_tasks_with_agent_plan(
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for task in tasks:
        labels = [str(label).lower() for label in task.get("labels", [])]
        owner = str(task.get("owner_hint") or "").strip()
        agent = task.get("agent") or {}
        if not agent.get("name"):
            best_score: int | None = None
            best_key: str | None = None
            haystack = " ".join(
                [
                    str(task.get("title") or ""),
                    str(task.get("description") or ""),
                ]
            ).lower()
            for key, metadata in _AGENT_MAP.items():
                score = sum(
                    1
                    for keyword in metadata.get("labels", [])
                    if str(keyword).lower() in labels
                    or str(keyword).lower() in haystack
                )
                if owner and key.lower() in owner.lower():
                    score += 1
                if best_score is None or score > best_score:
                    best_score = score
                    best_key = key
            if best_key:
                metadata = _AGENT_MAP[best_key]
                task["agent"] = {
                    "name": metadata.get("name", "Agent"),
                    "confidence": min(1.0, 0.5 + 0.1 * float(best_score or 0)),
                    "rationale": f"Matched domain '{best_key}' via labels/keywords",
                }
            else:
                task["agent"] = {
                    "name": owner or "GeneralAgent",
                    "confidence": 0.4,
                    "rationale": "Fallback to owner_hint or general",
                }
        if not task.get("plan"):
            domain = next(
                (
                    key
                    for key, metadata in _AGENT_MAP.items()
                    if task.get("agent", {}).get("name", "") == metadata["name"]
                ),
                None,
            )
            steps = _AGENT_MAP.get(domain, {}).get("plan", _GENERIC_PLAN)
            task["plan"] = [
                {
                    "step": index,
                    "action": step.get("action", ""),
                    "tool": step.get("tool", ""),
                    "operation": step.get("operation", ""),
                    "inputs": {},
                    "expected_output": step.get("expected_output", ""),
                }
                for index, step in enumerate(steps, 1)
            ]
    return tasks


def _load_transcript_text(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_concat_text(path)
    if suffix == ".vtt":
        return _read_vtt_to_text(path)
    if suffix == ".json":
        payload = _read_json_file(path)
        if isinstance(payload, dict):
            if payload.get("text"):
                return str(payload["text"]).strip()
            if payload.get("transcript"):
                return str(payload["transcript"]).strip()
            segments = payload.get("segments") or []
            if isinstance(segments, list) and segments:
                parts = [
                    str(segment.get("text") or "").strip()
                    for segment in segments
                    if isinstance(segment, dict)
                ]
                text = "".join(part for part in parts if part)
                if text:
                    return text
        return _read_text_file(path)
    return _read_text_file(path)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _tasks_document(
    tasks: List[Dict[str, Any]],
    path: Path,
    model: str,
    derived_from: Path | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_transcript": str(path),
        "generated_by_model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "tasks.sidecar.v1",
    }
    if derived_from is not None:
        metadata["derived_from"] = str(derived_from)
    return {"tasks": tasks, "_meta": metadata}


def process_transcript(
    path: Path,
    client: OllamaClient,
    model: str,
    opts: Dict[str, Any],
    state: Dict[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    stem = _detect_stem(path)
    if not stem:
        return
    out_summary = path.parent / f"{stem}_summary.json"
    out_tasks = path.parent / f"{stem}_tasks.json"

    if out_summary.exists() and out_tasks.exists() and not overwrite:
        _auto_sleep(state, True, 0.02)
        return

    if out_summary.exists() and not overwrite:
        try:
            summary_doc = json.loads(out_summary.read_text(encoding="utf-8"))
        except Exception:
            summary_doc = None

        existing_tasks = (
            _normalize_tasks(summary_doc.get("tasks"))
            if isinstance(summary_doc, dict)
            else []
        )
        if existing_tasks:
            existing_tasks = _enrich_tasks_with_agent_plan(existing_tasks)
            _atomic_json(
                out_tasks,
                _tasks_document(existing_tasks, path, model, out_summary),
            )
            log.info("Wrote tasks %s (from existing summary; no model call)", out_tasks)
            _auto_sleep(state, True, 0.02)
            return

        if out_tasks.exists():
            _auto_sleep(state, True, 0.02)
            return

        parts: List[str] = []
        if isinstance(summary_doc, dict):
            if summary_doc.get("summary"):
                parts.append(str(summary_doc["summary"]))
            for key in ("key_points", "topics", "people", "organizations", "places"):
                values = summary_doc.get(key)
                if isinstance(values, list) and values:
                    parts.append(f"{key}: " + ", ".join(str(value) for value in values))
        context = "".join(parts).strip() or (
            f"Summary present but minimal: {out_summary.name}"
        )
        prompt = TASKS_PROMPT_PREFIX + context + PROMPT_END
        if dry_run:
            log.info("DRY-RUN would create tasks for %s from summary context", path)
            return

        started = time.time()
        try:
            raw = client.generate(model, prompt, options=opts)
            state["consec_fail"] = 0
        except Exception as exc:
            state["consec_fail"] += 1
            log.error("Task extraction (from summary) failed for %s: %s", stem, exc)
            return
        finally:
            state["last_duration"] = max(0.0, time.time() - started)

        object_text = _extract_first_json_object(raw) or raw
        try:
            payload = json.loads(object_text)
        except Exception:
            bad = Path(str(out_tasks) + ".bad.txt")
            bad.write_text(raw, encoding="utf-8")
            log.error("Tasks JSON parse error for %s, wrote raw to %s", stem, bad)
            return
        tasks = (
            _normalize_tasks(payload.get("tasks"))
            if isinstance(payload, dict)
            else []
        )
        tasks = _enrich_tasks_with_agent_plan(tasks)
        _atomic_json(out_tasks, _tasks_document(tasks, path, model, out_summary))
        log.info("Wrote tasks %s (from summary context)", out_tasks)
        _auto_sleep(state, True, state["last_duration"])
        return

    transcript_text = _load_transcript_text(path)
    if not transcript_text:
        return
    if len(transcript_text) > state["prompt_chars"]:
        head = int(state["prompt_chars"] * 0.7)
        tail = int(state["prompt_chars"] * 0.25)
        transcript_text = transcript_text[:head] + transcript_text[-tail:]
    prompt = PROMPT_SCHEMA + transcript_text + PROMPT_END

    if dry_run:
        log.info("DRY-RUN would summarize & extract tasks for %s", path)
        return

    started = time.time()
    try:
        raw = client.generate(model, prompt, options=opts)
        state["consec_fail"] = 0
    except Exception as exc:
        state["consec_fail"] += 1
        log.error("Generation failed for %s: %s", stem, exc)
        return
    finally:
        state["last_duration"] = max(0.0, time.time() - started)

    object_text = _extract_first_json_object(raw)
    payload: Any = None
    if object_text:
        try:
            payload = json.loads(object_text)
        except Exception:
            payload = None
    if payload is None:
        try:
            payload = json.loads(raw)
        except Exception:
            bad = Path(str(out_summary) + ".bad.txt")
            bad.write_text(raw, encoding="utf-8")
            log.error("JSON parse error for %s, wrote raw to %s", stem, bad)
            return

    payload = _normalize_payload(payload)
    payload["_meta"] = {
        "source_transcript": str(path),
        "generated_by_model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "summary.sidecar.v1",
    }
    _atomic_json(out_summary, payload)
    log.info("Wrote summary %s", out_summary)

    tasks = _enrich_tasks_with_agent_plan(_normalize_tasks(payload.get("tasks")))
    _atomic_json(out_tasks, _tasks_document(tasks, path, model))
    log.info("Wrote tasks %s", out_tasks)

    base = 0.25 * state["last_duration"]
    penalty = 0.4 * state["consec_fail"]
    sleep_sec = (
        min(state["sleep_max"], max(state["sleep_min"], base + penalty))
        if state["auto_throttle"]
        else state["sleep_min"]
    )
    if sleep_sec > 0:
        time.sleep(sleep_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots", nargs="+", required=True, help="Root directories to scan (recursive)"
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-throttle", action="store_true")
    parser.add_argument("--sleep-min", type=float, default=0.1)
    parser.add_argument("--sleep-max", type=float, default=2.0)
    parser.add_argument("--ctx", type=int, default=1536)
    parser.add_argument("--predict", type=int, default=512)
    parser.add_argument("--prompt-chars", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = OllamaClient(args.ollama_host)
    options = {"num_ctx": args.ctx, "num_predict": args.predict, "temperature": 0.2}
    state = {
        "auto_throttle": args.auto_throttle,
        "sleep_min": args.sleep_min,
        "sleep_max": args.sleep_max,
        "prompt_chars": args.prompt_chars,
        "last_duration": 0.0,
        "consec_fail": 0,
    }

    roots = [Path(root).expanduser().resolve() for root in args.roots]
    roots = [root for root in roots if root.exists()]
    if not roots:
        log.error("No valid roots provided.")
        return

    stems = _discover_stems(roots)
    log.info("Found %d stem groups", len(stems))
    total = 0
    processed = 0
    for directory, stem, files_for_stem in stems:
        if args.limit and total >= args.limit:
            break
        total += 1
        best = _choose_best_path(directory, stem, files_for_stem)
        if not best:
            continue
        process_transcript(
            best,
            client,
            args.model,
            options,
            state,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        processed += 1
    log.info("Completed sequentially. stems=%d processed=%d", total, processed)


if __name__ == "__main__":
    main()
