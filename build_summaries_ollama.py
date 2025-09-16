#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSON summaries (sidecars) for transcripts using Ollama (Gemma3:4b or other model).

Enhancements:
- Sequential processing only (never parallel)
- Adaptive auto-throttle between requests (based on call duration and failures)
- Robust JSON parsing with fallback extraction
- Normalizes payload to required schema
- Writes `.bad.txt` on parse failure for inspection
- Supports multiple transcript naming patterns (including JSON-in-.txt sidecars)
- Creates tasks JSON sidecars: extracts or generates task descriptions from summaries
"""

import os, re, json, time, logging, argparse, csv
from pathlib import Path
from typing import Optional, Dict, Any, List
import http.client
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summaries")

# ----------------------------
# Ollama client (single instance)
# ----------------------------
class OllamaClient:
    def __init__(self, base_url: str):
        u = urlparse(base_url)
        self._is_https = (u.scheme == "https")
        self._host = u.hostname or "127.0.0.1"
        self._port = u.port or (443 if self._is_https else 11434)
        self._base_path = u.path.rstrip("/")

    def generate(self, model: str, prompt: str, retries: int = 3, timeout: float = 120.0, options: Dict[str, Any] | None = None) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        body = json.dumps(payload).encode("utf-8")
        path = f"{self._base_path}/api/generate" if self._base_path else "/api/generate"
        for attempt in range(1, retries+1):
            try:
                conn_cls = http.client.HTTPSConnection if self._is_https else http.client.HTTPConnection
                conn = conn_cls(self._host, self._port, timeout=timeout)
                conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
                doc = json.loads(data.decode("utf-8", errors="replace"))
                out = doc.get("response") or doc.get("text")
                if not out:
                    raise RuntimeError("No 'response' in Ollama output")
                return out
            except Exception as e:
                log.warning("Ollama call failed (%d/%d): %s", attempt, retries, e)
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Ollama call failed after retries")

# ----------------------------
# Transcript patterns
# ----------------------------
TRANSCRIPT_PATTERNS = [
    "{stem}_large-v3_transcription.txt",   # JSON format
    "{stem}_BC_medium_transcription.txt", # JSON format
    "{stem}_medium_transcription.txt",    # JSON format
    "{stem}_BC_transcription.csv",
    "{stem}_transcription.txt",           # JSON format
    "{stem}_transcription.csv",
    "{stem}.json",                        # JSON format
    "{stem}.txt",
    "{stem}.vtt",
]

# ----------------------------
# Prompt and schema
# ----------------------------
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
            "owner_hint": "<team/role/person if obvious>"
        }
    ]
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

# -- Tasks-only extraction (used when summary already exists) --
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
                "rationale": "<why this agent>"
            },
            "plan": [
                {
                    "step": 1,
                    "action": "<what to do>",
                    "tool": "<system/tool name>",
                    "operation": "<endpoint or verb>",
                    "inputs": {"key": "value"},
                    "expected_output": "<artifact/confirmation>"
                }
            ]
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
      "- Include a 'plan' with 1..6 ordered steps; keep tools generic (e.g., 'calendar', 'email', 'neo4j', 'filesystem', 'ticketing', 'browser')."
      "- Prefer agent-neutral owner_hint like 'Developer', 'Personal', 'Legal', 'Personal', etc."
      "- Use labels for routing (e.g., ['Developer','Urgent'])."
      "Transcript begins:"
)

# ----------------------------
# Helpers
# ----------------------------
STEM_RE = re.compile(r"^(\d{4}_\d{4}_\d{6}|\d{12,})(?:_[0-9]{6})?", re.ASCII)

def _detect_stem(p: Path) -> Optional[str]:
    m = STEM_RE.match(p.stem)
    return m.group(0) if m else None

def _read_text_file(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        log.warning("Failed reading %s: %s", p, e)
        return None

def _read_csv_concat_text(p: Path) -> Optional[str]:
    try:
        rows: List[str] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            if rdr.fieldnames is None:
                return None
            prefer = None
            for name in rdr.fieldnames:
                if name and name.lower() in ("text", "utterance", "transcript", "content"):
                    prefer = name
                    break
            for row in rdr:
                if prefer and row.get(prefer):
                    rows.append(str(row[prefer]))
                else:
                    vals = [str(v) for v in row.values() if isinstance(v, str) and v.strip()]
                    if vals:
                        rows.append(" ".join(vals))
        return "\n".join(rows).strip() or None
    except Exception as e:
        log.warning("Failed reading CSV %s: %s", p, e)
        return None

def _extract_first_json_object(s: str) -> Optional[str]:
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`\n ")
        if t.lower().startswith("json"):
            t = t[4:].lstrip("\n")
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return t[start:i+1]
    return None

def _normalize_payload(any_payload: Any) -> Dict[str, Any]:
    if isinstance(any_payload, list):
        for el in any_payload:
            if isinstance(el, dict):
                any_payload = el
                break
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
    # tasks optional
    tasks = any_payload.get("tasks")
    if not isinstance(tasks, list):
        any_payload["tasks"] = []
    else:
        any_payload["tasks"] = _normalize_tasks(tasks)
    return any_payload


def _normalize_tasks(tasks_any: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(tasks_any, list):
        return out
    for idx, t in enumerate(tasks_any, 1):
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "").strip()
        desc = str(t.get("description") or "").strip()
        labels = t.get("labels") if isinstance(t.get("labels"), list) else []
        labels = [str(x).strip() for x in labels if str(x).strip()]
        priority = str(t.get("priority") or "medium").strip().lower()
        if priority not in ("low","medium","high","urgent"):
            priority = "medium"
        owner_hint = str(t.get("owner_hint") or "").strip()

        agent = t.get("agent") if isinstance(t.get("agent"), dict) else {}
        agent_name = str(agent.get("name") or "").strip()
        try:
            agent_conf = float(agent.get("confidence", 0.0))
        except Exception:
            agent_conf = 0.0
        agent_conf = max(0.0, min(1.0, agent_conf))
        agent_rat = str(agent.get("rationale") or "").strip()

        plan = t.get("plan") if isinstance(t.get("plan"), list) else []
        norm_plan = []
        step_no = 1
        for s in plan[:6]:
            if not isinstance(s, dict):
                continue
            norm_plan.append({
                "step": int(s.get("step", step_no)),
                "action": str(s.get("action") or "").strip()[:200],
                "tool": str(s.get("tool") or "").strip()[:50],
                "operation": str(s.get("operation") or "").strip()[:80],
                "inputs": s.get("inputs") if isinstance(s.get("inputs"), dict) else {},
                "expected_output": str(s.get("expected_output") or "").strip()[:200],
            })
            step_no += 1

        if not title and not desc:
            continue
        out.append({
            "title": title[:140],
            "description": desc[:1000],
            "labels": list(dict.fromkeys(labels))[:10],
            "priority": priority,
            "owner_hint": owner_hint[:100],
            "agent": {"name": agent_name[:60], "confidence": agent_conf, "rationale": agent_rat[:200]},
            "plan": norm_plan
        })
    return out
    for t in tasks_any:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "").strip()
        desc = str(t.get("description") or "").strip()
        labels = t.get("labels") if isinstance(t.get("labels"), list) else []
        labels = [str(x).strip() for x in labels if str(x).strip()]
        priority = str(t.get("priority") or "medium").strip().lower()
        if priority not in ("low","medium","high","urgent"):
            priority = "medium"
        owner_hint = str(t.get("owner_hint") or "").strip()
        if not title and not desc:
            continue
        out.append({
            "title": title[:140],
            "description": desc[:1000],
            "labels": list(dict.fromkeys(labels))[:10],
            "priority": priority,
            "owner_hint": owner_hint[:100]
        })
    return out

# ----------------------------
# Lightweight heuristics for agent routing & plan (fallback if model omits)
# ----------------------------

_AGENT_MAP = {
    "DevOps": {"name": "DevOpsAgent", "labels": ["DevOps","Kubernetes","deploy","cluster","ingress"],
                "plan": [
                    {"action":"Open incident or ticket","tool":"ticketing","operation":"create","expected_output":"INC- id"},
                    {"action":"Prepare rollout plan","tool":"docs","operation":"create","expected_output":"doc link"}
                ]},
    "Finance": {"name": "FinanceAgent", "labels": ["invoice","payment","budget","expense"],
                 "plan": [
                    {"action":"Log expense","tool":"finance","operation":"record_expense","expected_output":"entry id"}
                 ]},
    "Legal": {"name": "LegalAgent", "labels": ["nda","contract","agreement"],
               "plan": [
                   {"action":"Draft document","tool":"docs","operation":"create","expected_output":"draft link"}
               ]},
    "Personal": {"name": "PersonalAssistant", "labels": ["Errand","Shopping","Family","Health"],
                  "plan": [
                      {"action":"Add calendar reminder","tool":"calendar","operation":"create_event","expected_output":"event link"}
                  ]},
}

_GENERIC_PLAN = [
    {"action":"Create task","tool":"ticketing","operation":"create","expected_output":"task id"},
    {"action":"Notify stakeholders","tool":"email","operation":"send","expected_output":"sent confirmation"}
]

def _enrich_tasks_with_agent_plan(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for t in tasks:
        labels = [l.lower() for l in t.get("labels", [])]
        owner = (t.get("owner_hint") or "").strip()
        agent = t.get("agent") or {}
        if not agent.get("name"):
            best = None
            best_key = None
            for key, meta in _AGENT_MAP.items():
                score = 0
                for kw in meta.get("labels", []):
                    if kw.lower() in labels or kw.lower() in t.get("title"," ").lower() or kw.lower() in t.get("description"," ").lower():
                        score += 1
                if owner and key.lower() in owner.lower():
                    score += 1
                if best is None or score > best:
                    best = score; best_key = key
            if best_key:
                meta = _AGENT_MAP[best_key]
                t["agent"] = {
                    "name": meta.get("name","Agent"),
                    "confidence": min(1.0, 0.5 + 0.1 * float(best or 0)),
                    "rationale": f"Matched domain '{best_key}' via labels/keywords"
                }
            else:
                t["agent"] = {"name": owner or "GeneralAgent", "confidence": 0.4, "rationale": "Fallback to owner_hint or general"}
        if not t.get("plan"):
            key = None
            for k, meta in _AGENT_MAP.items():
                if t.get("agent",{}).get("name","") == meta["name"]:
                    key = k; break
            steps = _AGENT_MAP.get(key, {}).get("plan", _GENERIC_PLAN)
            norm = []
            for i, s in enumerate(steps, 1):
                norm.append({
                    "step": i,
                    "action": s.get("action",""),
                    "tool": s.get("tool",""),
                    "operation": s.get("operation",""),
                    "inputs": {},
                    "expected_output": s.get("expected_output",""),
                })
            t["plan"] = norm
    return tasks

# ----------------------------
# Processing
# ----------------------------
def process_transcript(path: Path, client: OllamaClient, model: str, opts: Dict[str, Any], state: Dict[str, Any], overwrite=False, dry_run=False):
    stem = _detect_stem(path)
    if not stem:
        return
    out_summary = path.parent / f"{stem}_summary.json"
    out_tasks = path.parent / f"{stem}_tasks.json"

    # Helper to load text by suffix
    def _load_text(p: Path) -> Optional[str]:
        if p.suffix.lower() == ".csv":
            return _read_csv_concat_text(p)
        elif p.suffix.lower() == ".vtt":
            return _read_vtt_to_text(p)
        elif p.suffix.lower() == ".json":
            j = _read_json_file(p)
            if isinstance(j, dict):
                if j.get("text"):
                    return str(j.get("text")).strip()
                if j.get("transcript"):
                    return str(j.get("transcript")).strip()
                segs = j.get("segments") or []
                if isinstance(segs, list) and segs:
                    parts = [str(s.get("text") or "").strip() for s in segs if isinstance(s, dict)]
                    txt = "".join([t for t in parts if t])
                    if txt:
                        return txt
            return _read_text_file(p)
        else:
            return _read_text_file(p)

    # If summary exists and not overwriting: generate only tasks (if missing)
    if out_summary.exists() and not overwrite:
        if out_tasks.exists():
            return
        transcript_text = _load_text(path)
        if not transcript_text:
            return
        if len(transcript_text) > state["prompt_chars"]:
            head = int(state["prompt_chars"] * 0.7)
            tail = int(state["prompt_chars"] * 0.25)
            transcript_text = transcript_text[:head] + "" + transcript_text[-tail:]
        prompt = TASKS_PROMPT_PREFIX + transcript_text + PROMPT_END
        if dry_run:
            log.info("DRY-RUN would create tasks for %s", path)
            return
        start = time.time()
        try:
            raw = client.generate(model, prompt, options=opts)
            state["consec_fail"] = 0
        except Exception as e:
            state["consec_fail"] += 1
            log.error("Task extraction failed for %s: %s", stem, e)
            return
        finally:
            state["last_duration"] = max(0.0, time.time() - start)
        obj_text = _extract_first_json_object(raw) or raw
        try:
            payload = json.loads(obj_text)
        except Exception:
            bad = Path(str(out_tasks) + ".bad.txt")
            bad.write_text(raw, encoding="utf-8")
            log.error("Tasks JSON parse error for %s, wrote raw to %s", stem, bad)
            return
        tasks = _normalize_tasks(payload.get("tasks")) if isinstance(payload, dict) else []
        tasks = _enrich_tasks_with_agent_plan(tasks)
        task_doc = {
            "tasks": tasks,
            "_meta": {
                "source_transcript": str(path),
                "generated_by_model": model,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "schema": "tasks.sidecar.v1",
            }
        }
        tmp = Path(str(out_tasks) + ".tmp")
        tmp.write_text(json.dumps(task_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_tasks)
        log.info("Wrote tasks %s (summary existed)", out_tasks)

        # auto-throttle
        base = 0.25 * state["last_duration"]
        penalty = 0.4 * state["consec_fail"]
        sleep_sec = min(state["sleep_max"], max(state["sleep_min"], base + penalty)) if state["auto_throttle"] else state["sleep_min"]
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return

    # Build transcript text for new summary + tasks
    transcript_text = _load_text(path)
    if not transcript_text:
        return
    if len(transcript_text) > state["prompt_chars"]:
        head = int(state["prompt_chars"] * 0.7)
        tail = int(state["prompt_chars"] * 0.25)
        transcript_text = transcript_text[:head] + "" + transcript_text[-tail:]

    prompt = PROMPT_SCHEMA + transcript_text + PROMPT_END

    if dry_run:
        log.info("DRY-RUN would summarize & extract tasks for %s", path)
        return

    start = time.time()
    try:
        raw = client.generate(model, prompt, options=opts)
        state["consec_fail"] = 0
    except Exception as e:
        state["consec_fail"] += 1
        log.error("Generation failed for %s: %s", stem, e)
        return
    finally:
        state["last_duration"] = max(0.0, time.time() - start)

    obj_text = _extract_first_json_object(raw)
    payload: Any = None
    if obj_text:
        try:
            payload = json.loads(obj_text)
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

    # Write summary
    tmp = Path(str(out_summary) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_summary)
    log.info("Wrote summary %s", out_summary)

    # Derive tasks sidecar (from payload.tasks)
    tasks = payload.get("tasks") if isinstance(payload, dict) else []
    tasks = _normalize_tasks(tasks)
    tasks = _enrich_tasks_with_agent_plan(tasks)
    task_doc = {
        "tasks": tasks,
        "_meta": {
            "source_transcript": str(path),
            "generated_by_model": model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema": "tasks.sidecar.v1",
        }
    }
    tmp2 = Path(str(out_tasks) + ".tmp")
    tmp2.write_text(json.dumps(task_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp2.replace(out_tasks)
    log.info("Wrote tasks %s", out_tasks)

    # auto-throttle
    base = 0.25 * state["last_duration"]
    penalty = 0.4 * state["consec_fail"]
    sleep_sec = min(state["sleep_max"], max(state["sleep_min"], base + penalty)) if state["auto_throttle"] else state["sleep_min"]
    if sleep_sec > 0:
        time.sleep(sleep_sec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    ap.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-throttle", action="store_true", help="Adapt sleep based on generation time and failures")
    ap.add_argument("--sleep-min", type=float, default=0.1)
    ap.add_argument("--sleep-max", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=1536, help="Ollama num_ctx")
    ap.add_argument("--predict", type=int, default=512, help="Ollama num_predict")
    ap.add_argument("--prompt-chars", type=int, default=100_000, help="Max transcript characters to send")
    args = ap.parse_args()

    client = OllamaClient(args.ollama_host)

    opts = {"num_ctx": args.ctx, "num_predict": args.predict, "temperature": 0.2}
    state = {
        "auto_throttle": args.auto_throttle,
        "sleep_min": args.sleep_min,
        "sleep_max": args.sleep_max,
        "prompt_chars": args.prompt_chars,
        "last_duration": 0.0,
        "consec_fail": 0,
    }

    total = 0
    for dirpath, _, files in os.walk(args.root):
        for f in files:
            stem = _detect_stem(Path(f))
            if not stem:
                continue
            for pat in TRANSCRIPT_PATTERNS:
                cand = Path(dirpath) / pat.format(stem=stem)
                if cand.exists():
                    process_transcript(cand, client, args.model, opts, state, overwrite=args.overwrite, dry_run=args.dry_run)
                    total += 1
                    break
    log.info("Completed sequentially. Files checked: %d", total)

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    ap.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-throttle", action="store_true", help="Adapt sleep based on generation time and failures")
    ap.add_argument("--sleep-min", type=float, default=0.1)
    ap.add_argument("--sleep-max", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=1536, help="Ollama num_ctx")
    ap.add_argument("--predict", type=int, default=512, help="Ollama num_predict")
    ap.add_argument("--prompt-chars", type=int, default=100_000, help="Max transcript characters to send")
    args = ap.parse_args()

    client = OllamaClient(args.ollama_host)

    opts = {"num_ctx": args.ctx, "num_predict": args.predict, "temperature": 0.2}
    state = {
        "auto_throttle": args.auto_throttle,
        "sleep_min": args.sleep_min,
        "sleep_max": args.sleep_max,
        "prompt_chars": args.prompt_chars,
        "last_duration": 0.0,
        "consec_fail": 0,
    }

    total = 0
    for dirpath, _, files in os.walk(args.root):
        for f in files:
            stem = _detect_stem(Path(f))
            if not stem:
                continue
            for pat in TRANSCRIPT_PATTERNS:
                cand = Path(dirpath) / pat.format(stem=stem)
                if cand.exists():
                    process_transcript(cand, client, args.model, opts, state, overwrite=args.overwrite, dry_run=args.dry_run)
                    total += 1
                    break
    log.info("Completed sequentially. Files checked: %d", total)

if __name__ == "__main__":
    main()
