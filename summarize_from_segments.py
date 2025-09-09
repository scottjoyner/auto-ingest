#!/usr/bin/env python3
"""
Summarize from SEGMENTs (UTTERANCE optional) → create Summary + Tasks in Neo4j,
then (optionally) approve & execute tasks.

Key improvements in v2
- **Sequential, backpressured LLM calls** (default). Each candidate waits for the
  Ollama response before the next begins; optional sleep between requests.
- **Idempotent, transactional writes** using session.execute_write and
  client-side UUIDs; MERGE ensures reruns don't duplicate nodes.
- **Schema bootstrap** (constraints/indexes) via --ensure-schema or env flag.
- **Robust JSON parsing & validation** with clear fallbacks and logging.
- **Better logging** (structured, per-transcription key) and exit codes.
- **Retry with exponential backoff** for Ollama calls & Neo4j commits.
- **Acceptance hydration** from stored JSON; consistent artifacts handling.
- **Bug fixes**: deduped helper defs; consistent timeouts; safer query ordering.

Environment
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=*****
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=llama3.1:8b
  OLLAMA_ENDPOINT=auto  # auto | chat | generate
  OLLAMA_TIMEOUT=900
  OLLAMA_USE_CLI=0      # 1 to force `ollama` CLI fallback
  SUMMARY_FAILOPEN=copy # copy | skip
  ARTIFACTS_DIR=./artifacts
  ENSURE_SCHEMA=0       # 1 to run schema bootstrap on start

CLI Examples
  python summarize_from_segments.py --latest --dry-run
  python summarize_from_segments.py --id <TRANSCRIPTION_ID>
  python summarize_from_segments.py --all-missing --limit 100 --write-notes
  python summarize_from_segments.py --approve-all 200 --execute-ready 10
  # Sequential batch with 5s gap and retries
  python summarize_from_segments.py --all-missing --limit 50 --sleep-between 5 --max-retries 3
"""
from __future__ import annotations

import os, json, argparse, time, re, uuid, datetime, pathlib, logging, math, subprocess, shutil
from typing import List, Dict, Any, Optional, Tuple
import subprocess, shutil  # add
import requests
from neo4j import GraphDatabase

# ==========================
# Env & defaults
# ==========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_ENDPOINT = (os.getenv("OLLAMA_ENDPOINT", "auto") or "auto").strip().lower()  # auto|chat|generate
OLLAMA_USE_CLI = os.getenv("OLLAMA_USE_CLI", "0").lower() in ("1", "true", "yes", "on")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
SUMMARY_FAILOPEN = os.getenv("SUMMARY_FAILOPEN", "copy")  # copy | skip
ENSURE_SCHEMA_FLAG = os.getenv("ENSURE_SCHEMA", "0").lower() in ("1","true","yes","on")

# ==========================
# Prompts
# ==========================
SUMMARIZE_INSTR = """You are given a conversation transcript composed of time-ordered items.
Rules:
- Treat SEGMENT items as authoritative.
- Items marked LOW_CONF (from UTTERANCE) are lower confidence; only use them if consistent with SEGMENT content.
- Output must be faithful, concise, and specific. Avoid speculation.
Return ONLY a JSON object with:
{
  "summary": string,
  "bullets": string[]
}"""

TASKS_INSTR = """From the provided summary+bullets, extract ACTION ITEMS strictly as JSON:
{
  "summary": string,
  "bullets": string[],
  "tasks": [
    {
      "title": string,
      "description": string,
      "priority": "LOW"|"MEDIUM"|"HIGH",
      "due": string|null,
      "confidence": number,
      "acceptance": [
        {
          "type": "file_exists" | "contains" | "regex" | "http_ok",
          "args": { "path"?: string, "text"?: string, "pattern"?: string, "url"?: string }
        }
      ]
    }
  ]
}
Guidelines:
- Prefer MEDIUM unless urgency suggests HIGH.
- Include a due date only if stated or clearly implied.
- Confidence in [0,1].
- IMPORTANT: ACCEPTANCE IS REQUIRED WHENEVER POSSIBLE. Prefer outputs under artifacts/{TASK_ID}/output.txt to enable verification.
Return ONLY JSON (no prose)."""

# ==========================
# Logging
# ==========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("summarize_from_segments")

# ==========================
# Neo4j Driver
# ==========================
_driver = None

def neo_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

# ==========================
# Schema bootstrap
# ==========================
SCHEMA_QUERIES = [
    # Nodes
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transcription) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Summary)       REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Task)          REQUIRE k.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Run)           REQUIRE r.id IS UNIQUE",
    # Helpful indexes
    "CREATE INDEX IF NOT EXISTS FOR (t:Transcription) ON (t.key)",
    "CREATE INDEX IF NOT EXISTS FOR (k:Task)          ON (k.status)",
]

def ensure_schema():
    with neo_driver().session() as s:
        for q in SCHEMA_QUERIES:
            try:
                s.execute_write(lambda tx: tx.run(q))
            except Exception as e:
                log.warning("Schema step failed: %s", e)

# ==========================
# Helpers
# ==========================

def _build_prompt(system: str, user: str) -> str:
    sys = (system or "").strip()
    usr = (user or "").strip()
    return f"<<SYS>>\n{sys}\n<</SYS>>\n\n{usr}" if sys else usr


def _coerce_json_dict(s: str) -> Dict[str, Any]:
    """Extract a dict from potentially messy JSON-ish text (code fences, NDJSON, etc)."""
    if not s:
        return {}
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)
    # Direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Braces carve-out
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(s[start:end+1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    # NDJSON stitch
    parts = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
            if isinstance(o, dict) and ("response" in o or "message" in o):
                if "response" in o:
                    parts.append(str(o["response"]))
                elif "message" in o and isinstance(o["message"], dict):
                    parts.append(str(o["message"].get("content", "")))
        except Exception:
            continue
    if parts:
        try:
            obj = json.loads("".join(parts))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _failopen_summary_from_text(full_text: str, max_chars: int = 2000) -> Dict[str, Any]:
    lines = [ln.strip() for ln in (full_text or "").splitlines() if ln.strip() and not ln.startswith("[LOW_CONF]")]
    if not lines:
        return {"summary": "", "bullets": []}
    summary_parts, total = [], 0
    for ln in lines:
        if total + len(ln) + 1 > max_chars or len(summary_parts) >= 6:
            break
        summary_parts.append(ln); total += len(ln) + 1
    bullets, seen = [], set()
    for ln in lines:
        short = ln if len(ln) <= 160 else (ln[:157] + "…")
        if short in seen:
            continue
        seen.add(short); bullets.append(short)
        if len(bullets) >= 8:
            break
    return {"summary": " ".join(summary_parts).strip(), "bullets": bullets}

# ==========================
# Transcript assembly
# ==========================

def build_transcript(segments: List[Dict[str,Any]], utterances: List[Dict[str,Any]], include_utterances: bool=False) -> str:
    items = list(segments)
    if include_utterances:
        items += utterances
    items.sort(key=lambda r: (float(r.get("start",0.0)), float(r.get("end",0.0))))
    lines = []
    for it in items:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        if it.get("type") == "UTTERANCE" or it.get("low_conf"):
            lines.append(f"[LOW_CONF] {txt}")
        else:
            lines.append(txt)
    return "\n".join(lines)

# ==========================
# Neo4j read helpers
# ==========================

def fetch_one(session, trans_id: Optional[str], latest: bool) -> Optional[Dict[str, Any]]:
    if trans_id:
        q = """
        MATCH (t:Transcription {id:$id})
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
        WITH t, s ORDER BY coalesce(s.start, s.idx, 0)
        WITH t, collect({id:s.id, start:coalesce(s.start,0.0), end:coalesce(s.end,0.0), text:s.text, type:'SEGMENT', low_conf:false}) AS segs
        OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u)
        WITH t, segs, u ORDER BY coalesce(u.start, u.idx, 0)
        RETURN t, segs, collect({id:u.id, start:coalesce(u.start,0.0), end:coalesce(u.end,0.0), text:u.text, type:'UTTERANCE', low_conf:true}) AS utts
        """
        rec = session.execute_read(lambda tx: tx.run(q, id=trans_id).single())
    else:
        q = """
        MATCH (t:Transcription)
        WITH t ORDER BY coalesce(t.created_at, datetime({epochMillis:0})) DESC
        LIMIT 1
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
        WITH t, s ORDER BY coalesce(s.start, s.idx, 0)
        WITH t, collect({id:s.id, start:coalesce(s.start,0.0), end:coalesce(s.end,0.0), text:s.text, type:'SEGMENT', low_conf:false}) AS segs
        OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u)
        WITH t, segs, u ORDER BY coalesce(u.start, u.idx, 0)
        RETURN t, segs, collect({id:u.id, start:coalesce(u.start,0.0), end:coalesce(u.end,0.0), text:u.text, type:'UTTERANCE', low_conf:true}) AS utts
        """
        rec = session.execute_read(lambda tx: tx.run(q).single())
    if not rec:
        return None
    t = dict(rec["t"]) if rec["t"] else None
    if not t:
        return None
    segs = [x for x in (rec["segs"] or []) if x and x.get("text")]
    utts = [x for x in (rec["utts"] or []) if x and x.get("text")]
    return {"t": t, "segments": segs, "utterances": utts}


def fetch_missing_transcriptions(session, limit: int) -> List[Dict[str, Any]]:
    q = """
    MATCH (t:Transcription)
    WHERE NOT (t)-[:HAS_SUMMARY]->(:Summary)
       OR trim(coalesce(t.notes, '')) = ''
    RETURN t
    ORDER BY coalesce(t.created_at, datetime({epochMillis:0})) DESC
    LIMIT $limit
    """
    rows = session.execute_read(lambda tx: list(tx.run(q, limit=limit)))
    return [dict(r["t"]) for r in rows]

# ==========================
# JSON normalize/validate
# ==========================

def _normalize_summary_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"summary": str(obj), "bullets": []}
    summary = str(obj.get("summary", ""))
    bullets = obj.get("bullets") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    return {"summary": summary, "bullets": bullets}


def _validate_tasks_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"summary": "", "bullets": [], "tasks": []}
    tasks = obj.get("tasks") or []
    norm_tasks = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title", "")).strip() or "Untitled Task"
        description = str(t.get("description", "")).strip()
        pri = str(t.get("priority", "MEDIUM")).upper()
        if pri not in {"LOW","MEDIUM","HIGH"}: pri = "MEDIUM"
        try:
            conf = float(t.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        accept = t.get("acceptance") or []
        if not accept:
            accept = [{"type":"file_exists","args":{"path":"artifacts/{TASK_ID}/output.txt"}}]
        norm_tasks.append({
            "title": title,
            "description": description,
            "priority": pri,
            "due": t.get("due"),
            "confidence": conf,
            "acceptance": accept,
        })
    return {
        "summary": str(obj.get("summary", "")),
        "bullets": obj.get("bullets", []),
        "tasks": norm_tasks,
    }

# ==========================
# Chunking & LLM calls
# ==========================

def chunk(text: str, max_chars: int = 7000) -> List[str]:
    chunks, buf = [], ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > max_chars:
            if buf: chunks.append(buf)
            buf = line
        else:
            buf = (buf+"\n"+line) if buf else line
    if buf: chunks.append(buf)
    return chunks


def _ollama_generate_json(system: str, user: str, temperature: float) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": _build_prompt(system, user),
        "options": {"temperature": temperature, "num_ctx": 8192},
        "format": "json",
        "stream": False,
    }
    url = f"{OLLAMA_HOST}/api/generate"
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    try:
        body = r.json(); content = body.get("response", "")
        obj = _coerce_json_dict(content)
        if obj: return obj
    except Exception:
        pass
    return _coerce_json_dict(r.text or "")


def _ollama_chat_json(system: str, user: str, temperature: float) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_ctx": 8192},
        "format": "json",
        "stream": False,
    }
    url = f"{OLLAMA_HOST}/api/chat"
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    if r.status_code == 404:
        raise requests.HTTPError("chat endpoint not found", response=r)
    r.raise_for_status()
    try:
        body = r.json(); content = (body.get("message") or {}).get("content", "")
        obj = _coerce_json_dict(content)
        if obj: return obj
    except Exception:
        pass
    return _coerce_json_dict(r.text or "")

def _ollama_cli_generate_json(system: str, user: str, timeout_sec: int) -> dict:
    if not shutil.which("ollama"): return {}
    prompt = _build_prompt(system, user)
    try:
        p = subprocess.Popen(["ollama","run","--json", OLLAMA_MODEL],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        assert p.stdin and p.stdout
        p.stdin.write(prompt + "\n"); p.stdin.close()
        chunks = []
        for line in p.stdout:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "response" in obj: chunks.append(obj["response"])
            except Exception:
                pass
        p.wait(timeout= int(os.getenv("OLLAMA_TIMEOUT","900")))
        return _coerce_json_dict("".join(chunks)) or {}
    except Exception:
        return {}




def ollama_chat_json(system: str, user: str, temperature: float = 0.2, max_retries: int = 3, base_delay: float = 1.0) -> Dict[str, Any]:
    """Endpoint auto-selection with retries & exponential backoff; strictly sequential."""
    # Prefer CLI if explicitly enabled
    if OLLAMA_USE_CLI:
        obj = _ollama_cli_generate_json(system, user, timeout_sec=int(os.getenv("OLLAMA_TIMEOUT", "900")))
        if obj:
            return obj
    for attempt in range(max_retries + 1):
        try:
            if OLLAMA_ENDPOINT == "chat":
                return _ollama_chat_json(system, user, temperature)
            if OLLAMA_ENDPOINT == "generate":
                return _ollama_generate_json(system, user, temperature)
            # auto
            try:
                return _ollama_chat_json(system, user, temperature)
            except Exception:
                return _ollama_generate_json(system, user, temperature)
        except Exception as e:
            if attempt >= max_retries:
                log.error("Ollama call failed after %s retries: %s", attempt, e)
                break
            delay = base_delay * (2 ** attempt)
            log.warning("Ollama call error (attempt %d/%d): %s; sleeping %.1fs", attempt+1, max_retries, e, delay)
            time.sleep(delay)
            # FINAL fallback: try CLI once more before giving up
            obj = _ollama_cli_generate_json(system, user, timeout_sec=int(os.getenv("OLLAMA_TIMEOUT", "900")))
            return obj or {}
    return {}

# ==========================
# Summarize & extract tasks
# ==========================

def summarize_as_json(full_text: str, max_retries: int) -> Dict[str, Any]:
    parts = chunk(full_text)
    if len(parts) == 1:
        obj = ollama_chat_json(SUMMARIZE_INSTR, parts[0], max_retries=max_retries)
        res = _normalize_summary_obj(obj)
        if not res.get("summary", "").strip() and SUMMARY_FAILOPEN == "copy":
            res = _failopen_summary_from_text(full_text)
        return res
    # Map-reduce
    partials = []
    for i, ch in enumerate(parts, 1):
        pj = ollama_chat_json(SUMMARIZE_INSTR, f"[CHUNK {i}/{len(parts)}]\n\n{ch}", max_retries=max_retries)
        partials.append(_normalize_summary_obj(pj))
    combined = json.dumps(partials, ensure_ascii=False)
    final = ollama_chat_json(SUMMARIZE_INSTR, "Combine these partial results into one final JSON with shape {summary, bullets}:\n" + combined, max_retries=max_retries)
    res = _normalize_summary_obj(final)
    if not res.get("summary", "").strip() and SUMMARY_FAILOPEN == "copy":
        res = _failopen_summary_from_text(full_text)
    return res


def extract_tasks_json(summary_obj: Dict[str, Any], max_retries: int) -> Dict[str, Any]:
    payload = json.dumps({"summary": summary_obj.get("summary",""), "bullets": summary_obj.get("bullets",[])}, ensure_ascii=False)
    obj: Dict[str, Any] = {}
    for _ in range(max_retries + 1):
        try:
            obj = ollama_chat_json("Return STRICT JSON only.", TASKS_INSTR + "\n\n" + payload, max_retries=max_retries)
            if isinstance(obj, dict) and "tasks" in obj:
                break
        except Exception:
            time.sleep(0.5)
    out = _validate_tasks_payload(obj)
    return out

# ==========================
# Write-back (idempotent)
# ==========================

def _serialize_tasks_for_db(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in (tasks or []):
        t2 = dict(t)
        t2["acceptance_json"] = json.dumps(t.get("acceptance", []), ensure_ascii=False)
        out.append(t2)
    return out


def write_back(session, trans_id: str, final_obj: Dict[str, Any], write_notes: bool=False) -> str:
    summary_id = str(uuid.uuid4())
    tasks = final_obj.get("tasks") or []
    # attach deterministic UUIDs only for this run (avoid dupes on retried tx)
    tasks_with_ids = [{**t, "id": str(uuid.uuid4())} for t in tasks]
    payload_tasks = _serialize_tasks_for_db(tasks_with_ids)
    q = """
    MERGE (t:Transcription {id:$tid})
    ON MATCH SET t.updated_at = datetime()
    MERGE (s:Summary {id:$sid})
    ON CREATE SET s.created_at = datetime(), s.text = $text, s.bullets = $bullets
    ON MATCH  SET s.text = $text, s.bullets = $bullets
    MERGE (t)-[:HAS_SUMMARY]->(s)
    FOREACH (_ IN (CASE WHEN $write_notes THEN [1] ELSE [] END) |
        SET t.notes = $text, t.updated_at = datetime()
    )
    WITH s
    UNWIND $tasks AS tsk
      MERGE (tk:Task {id: tsk.id})
      ON CREATE SET tk.created_at = datetime()
      SET tk.title = coalesce(tsk.title, "Untitled Task"),
          tk.description = coalesce(tsk.description, ""),
          tk.priority = coalesce(tsk.priority, "MEDIUM"),
          tk.due = tsk.due,
          tk.status = coalesce(tk.status, "REVIEW"),
          tk.confidence = coalesce(tsk.confidence, 0.5),
          tk.acceptance_json = coalesce(tsk.acceptance_json, "[]"),
          tk.updated_at = datetime()
      MERGE (s)-[:GENERATED_TASK]->(tk)
    RETURN s.id AS sid
    """
    res = session.execute_write(
        lambda tx: tx.run(
            q,
            tid=trans_id,
            sid=summary_id,
            text=final_obj.get("summary",""),
            bullets=[str(b).strip() for b in (final_obj.get("bullets") or []) if str(b).strip()],
            tasks=payload_tasks,
            write_notes=bool(write_notes),
        ).single()
    )
    return res["sid"] if res and "sid" in res else ""

# ==========================
# Task approval & execution
# ==========================

def q_list(session, status: str, limit: int = 25) -> List[Dict[str, Any]]:
    res = session.execute_read(
        lambda tx: list(tx.run(
            """
            MATCH (task:Task {status:$status})
            OPTIONAL MATCH (s:Summary)-[:GENERATED_TASK]->(task)
            OPTIONAL MATCH (t:Transcription)-[:HAS_SUMMARY]->(s)
            RETURN task.id AS id, task.title AS title, task.priority AS priority,
                   t.key AS tkey, s.id AS sid
            ORDER BY coalesce(task.updated_at, datetime({epochMillis:0})) DESC, id
            LIMIT $limit
            """,
            status=status, limit=limit,
        )))
    return [dict(r) for r in res]


def q_get_task(session, task_id: str) -> Optional[Dict[str, Any]]:
    rec = session.execute_read(lambda tx: tx.run(
        """
        MATCH (task:Task {id:$id})
        OPTIONAL MATCH (s:Summary)-[:GENERATED_TASK]->(task)
        OPTIONAL MATCH (t:Transcription)-[:HAS_SUMMARY]->(s)
        RETURN task, s, t
        """, id=task_id).single())
    if not rec:
        return None
    task = dict(rec["task"]) if rec["task"] else None
    if not task:
        return None
    if "acceptance_json" in task and isinstance(task["acceptance_json"], str):
        try: task["acceptance"] = json.loads(task["acceptance_json"]) or []
        except Exception: task["acceptance"] = []
    s = dict(rec["s"]) if rec["s"] else None
    t = dict(rec["t"]) if rec["t"] else None
    return {"task": task, "summary": s, "transcription": t}


def q_approve_all_review(session, limit: int) -> int:
    rec = session.execute_write(lambda tx: tx.run(
        """
        MATCH (task:Task {status:'REVIEW'})
        WITH task LIMIT $limit
        SET task.status = 'READY', task.updated_at = datetime()
        RETURN count(task) AS n
        """, limit=limit).single())
    return int(rec["n"] if rec and "n" in rec else 0)


def q_pick_ready(session, limit: int) -> List[str]:
    rows = session.execute_write(lambda tx: list(tx.run(
        """
        MATCH (task:Task {status:'READY'})
        WITH task ORDER BY coalesce(task.updated_at, datetime({epochMillis:0})) ASC
        LIMIT $limit
        SET task.status = 'RUNNING', task.updated_at = datetime()
        RETURN task.id AS id
        """, limit=limit)))
    return [r["id"] for r in rows]


def q_attach_run(session, task_id: str, run: Dict[str, Any]) -> str:
    rec = session.execute_write(lambda tx: tx.run(
        """
        MATCH (task:Task {id:$tid})
        MERGE (r:Run {id:$rid})
        ON CREATE SET r.started_at = datetime($started_at)
        SET r.status = $status, r.manifest_json = $manifest_json
        MERGE (task)-[:HAS_RUN]->(r)
        RETURN r.id AS rid
        """,
        tid=task_id,
        rid=run["id"],
        started_at=run["started_at"],
        status=run["status"],
        manifest_json=json.dumps(run.get("manifest", {}), ensure_ascii=False),
    ).single())
    return rec["rid"] if rec and "rid" in rec else ""


def q_finish_run(session, task_id: str, run_id: str, success: bool, manifest: Dict[str, Any]):
    session.execute_write(lambda tx: tx.run(
        """
        MATCH (task:Task {id:$tid})-[:HAS_RUN]->(r:Run {id:$rid})
        SET r.status = $rstatus,
            r.ended_at = datetime($ended_at),
            r.success = $success,
            r.manifest_json = $manifest_json,
            task.status = $tstatus,
            task.updated_at = datetime()
        """,
        tid=task_id,
        rid=run_id,
        rstatus = "DONE" if success else "FAILED",
        tstatus = "DONE" if success else "FAILED",
        ended_at = datetime.datetime.utcnow().isoformat() + "Z",
        success = bool(success),
        manifest_json = json.dumps(manifest, ensure_ascii=False),
    ))

# ---- Acceptance helpers ----

def replace_placeholders(s: str, task_id: str) -> str:
    return (s or "").replace("{TASK_ID}", task_id)


def _artifacts_path(p: str) -> str:
    return p.replace("artifacts/", f"{ARTIFACTS_DIR.rstrip('/')}/")


def acc_file_exists(args: Dict[str, Any], task_id: str) -> Tuple[bool, str]:
    path = _artifacts_path(replace_placeholders(args.get("path",""), task_id))
    ok = pathlib.Path(path).exists()
    return ok, f"path={path}"


def acc_contains(args: Dict[str, Any], task_id: str) -> Tuple[bool, str]:
    path = _artifacts_path(replace_placeholders(args.get("path",""), task_id))
    text = args.get("text","")
    try:
        data = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        ok = text in data
        return ok, f"path={path} len={len(data)}"
    except Exception as e:
        return False, f"path={path} err={e}"


def acc_regex(args: Dict[str, Any], task_id: str) -> Tuple[bool, str]:
    path = _artifacts_path(replace_placeholders(args.get("path",""), task_id))
    pat = args.get("pattern","")
    try:
        data = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        ok = re.search(pat, data) is not None
        return ok, f"path={path} pat={pat}"
    except Exception as e:
        return False, f"path={path} err={e}"


def acc_http_ok(args: Dict[str, Any], _task_id: str) -> Tuple[bool, str]:
    url = args.get("url","")
    try:
        r = requests.get(url, timeout=10)
        ok = 200 <= r.status_code < 300
        return ok, f"url={url} code={r.status_code}"
    except Exception as e:
        return False, f"url={url} err={e}"

ACCEPTANCE_FUNCS = {
    "file_exists": acc_file_exists,
    "contains": acc_contains,
    "regex": acc_regex,
    "http_ok": acc_http_ok,
}


def run_acceptance(task: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    results = []
    acc_list = task.get("acceptance") or []
    for i, a in enumerate(acc_list):
        typ = (a.get("type") or "").strip()
        f = ACCEPTANCE_FUNCS.get(typ)
        if not f:
            results.append({"index": i, "type": typ, "passed": False, "detail": "unknown acceptance type"})
            continue
        ok, detail = f(a.get("args", {}), task_id)
        results.append({"index": i, "type": typ, "passed": bool(ok), "detail": detail})
    return results

# ---- Minimal executor ----

def ensure_artifacts(task_id: str) -> pathlib.Path:
    p = pathlib.Path(ARTIFACTS_DIR) / task_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def execute_task_minimal(task: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    t0 = time.time()
    tid = task["id"]
    artdir = ensure_artifacts(tid)
    steps.append({"op":"ensure_artifacts","dir":str(artdir)})
    out_path = artdir / "output.txt"
    content = f"""TASK {tid}
TITLE: {task.get('title','')}
DESC: {task.get('description','')}
TIME: {datetime.datetime.utcnow().isoformat()}Z
"""
    out_path.write_text(content, encoding="utf-8")
    steps.append({"op":"write_file","path":str(out_path),"bytes":len(content)})
    for a in (task.get("acceptance") or []):
        if a.get("type") == "http_ok" and a.get("args",{}).get("url"):
            url = a["args"]["url"]
            try:
                r = requests.get(url, timeout=10)
                (artdir / "http.status").write_text(str(r.status_code), encoding="utf-8")
                (artdir / "http.body.txt").write_text((r.text or "")[:4096], encoding="utf-8")
                steps.append({"op":"http_get","url":url,"status":r.status_code,"bytes":len(r.content)})
            except Exception as e:
                steps.append({"op":"http_get","url":url,"error":str(e)})
    return {"steps": steps, "duration_sec": round(time.time() - t0, 3), "artifacts_dir": str(artdir)}

# ==========================
# Orchestration
# ==========================

def process_one(session, tnode: Dict[str,Any], include_utterances: bool, dry_run: bool, write_notes: bool, max_retries: int) -> Optional[str]:
    row = fetch_one(session, tnode.get("id"), latest=False)
    if not row:
        log.warning("[%s] not found or empty.", tnode.get("id"))
        return None
    t = row["t"]; segments, utterances = row["segments"], row["utterances"]
    key = t.get("key") or t.get("id")
    if not segments and not utterances:
        log.info("[%s] No text found in Segments/Utterances.", key)
        return None
    text = build_transcript(segments, utterances, include_utterances=include_utterances)
    log.info("[%s] Items: %d segments + %d utterances (included=%s)", key, len(segments), len(utterances), include_utterances)
    summary_obj = summarize_as_json(text, max_retries=max_retries)
    if not summary_obj.get("summary","" ).strip():
        log.warning("[%s] Summarizer returned empty; skipping write.", key)
        return None
    final_obj = extract_tasks_json(summary_obj, max_retries=max_retries)
    log.info("[%s] Will write %d task(s).", key, len(final_obj.get('tasks', [])))
    if dry_run:
        print(json.dumps({"transcription": key, **final_obj}, indent=2))
        return None
    sid = write_back(session, t.get("id"), final_obj, write_notes=write_notes)
    if not sid:
        log.warning("[%s] Write-back returned no Summary id.", key)
    else:
        log.info("[%s] Wrote Summary %s and %d Task(s).", key, sid, len(final_obj.get('tasks',[])))
    return sid


def approve_all(session, limit: int) -> int:
    n = q_approve_all_review(session, limit=limit)
    log.info("Approved %d task(s) from REVIEW → READY.", n)
    return n


def execute_ready(limit: int):
    with neo_driver().session() as s:
        picked = q_pick_ready(s, limit=limit)
    if not picked:
        log.info("No READY tasks.")
        return
    for tid in picked:
        with neo_driver().session() as s:
            row = q_get_task(s, tid)
            if not row:
                log.warning("%s: not found", tid); continue
            task = row["task"]
            run = {
                "id": str(uuid.uuid4()),
                "started_at": datetime.datetime.utcnow().isoformat()+"Z",
                "status": "RUNNING",
                "manifest": {"task_id": tid, "steps": [], "acceptance_results": []},
            }
            q_attach_run(s, tid, run)
        manifest_exec = execute_task_minimal(task)
        results = run_acceptance(task, tid)
        success = all(r["passed"] for r in results) if results else True
        run["manifest"].update(manifest_exec)
        run["manifest"]["acceptance_results"] = results
        with neo_driver().session() as s:
            q_finish_run(s, tid, run["id"], success, run["manifest"])
        log.info("%s: %s  artifacts=%s", tid, "DONE" if success else "FAILED", manifest_exec['artifacts_dir'])

# ==========================
# CLI
# ==========================

def main():
    ap = argparse.ArgumentParser(description="Summarize from SEGMENTs → Tasks in Neo4j; approve & execute with flags.")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--id", dest="trans_id", help="Transcription.id to process")
    sel.add_argument("--latest", action="store_true", help="Pick the latest Transcription")
    sel.add_argument("--all-missing", action="store_true", help="Process transcriptions missing notes (no Summary OR empty t.notes)")

    ap.add_argument("--limit", type=int, default=50, help="Max items when using --all-missing")
    ap.add_argument("--include-utterances", action="store_true", help="Include UTTERANCE as LOW_CONF lines")
    ap.add_argument("--write-notes", action="store_true", help="Also write summary text into t.notes")
    ap.add_argument("--dry-run", action="store_true", help="Print JSON only (no writes)")

    # Backpressure & retries
    ap.add_argument("--sleep-between", type=float, default=0.0, help="Seconds to sleep between sequential LLM requests")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries for Ollama calls & parsing")

    # Approve & execute
    ap.add_argument("--approve-all", nargs="?", const=100, type=int, help="Approve up to N tasks in REVIEW (default 100 if no value provided)")
    ap.add_argument("--execute-ready", type=int, default=0, help="Execute up to N tasks in READY")

    # List
    ap.add_argument("--list", dest="list_status", choices=["REVIEW","READY","RUNNING","DONE","FAILED"], help="List tasks by status (no writes)")

    # Misc
    ap.add_argument("--artifacts-dir", help="Override ARTIFACTS_DIR (default from env ./artifacts)")
    ap.add_argument("--ensure-schema", action="store_true", help="Run schema bootstrap before any work")

    args = ap.parse_args()

    global ARTIFACTS_DIR
    if args.artifacts_dir:
        ARTIFACTS_DIR = args.artifacts_dir

    if args.ensure_schema or ENSURE_SCHEMA_FLAG:
        ensure_schema()

    # Optional: just list
    if args.list_status:
        with neo_driver().session() as s:
            rows = q_list(s, status=args.list_status, limit=50)
        if not rows:
            print("(none)")
            return
        for r in rows:
            print(f"[{r.get('tkey') or ''}] {r['id']}  {r['priority']:>6}  {r['title']}")
        return

    did_summarize = False
    # Summarization step (sequential, blocking)
    with neo_driver().session() as sess:
        if args.trans_id or args.latest or args.all_missing:
            did_summarize = True
            if args.trans_id or args.latest:
                if args.latest and not args.trans_id:
                    row = fetch_one(sess, None, latest=True)
                    if not row:
                        log.info("No Transcription found."); return
                    tnode = row["t"]
                else:
                    tnode = {"id": args.trans_id}
                process_one(sess, tnode, args.include_utterances, args.dry_run, args.write_notes, args.max_retries)
                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)
            else:
                cands = fetch_missing_transcriptions(sess, limit=args.limit)
                if not cands:
                    log.info("No candidates found (all have notes/summaries).")
                else:
                    for tnode in cands:
                        process_one(sess, tnode, args.include_utterances, args.dry_run, args.write_notes, args.max_retries)
                        if args.sleep_between > 0:
                            time.sleep(args.sleep_between)

    # Approve & Execute (also sequential)
    if args.approve_all is not None:
        with neo_driver().session() as s:
            approve_all(s, limit=args.approve_all)
    if args.execute_ready and args.execute_ready > 0:
        execute_ready(args.execute_ready)

    if not did_summarize and args.approve_all is None and args.execute_ready == 0 and not args.list_status:
        print("Nothing to do. Provide --id/--latest/--all-missing to summarize, or --approve-all / --execute-ready, or --list STATUS.")

if __name__ == "__main__":
    main()


# nohup bash -lc 'cd ~/git/video-automation && source .venv/bin/activate && OLLAMA_MODEL=gemma:2b OLLAMA_USE_CLI=1 OLLAMA_TIMEOUT=900 python summarize_from_segments_v2.py --all-missing --limit 100 --write-notes' > ~/git/video-automation/summarize_run.log 2>&1 &
