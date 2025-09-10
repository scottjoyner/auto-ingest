#!/usr/bin/env python3
"""
Summarize from SEGMENTs (UTTERANCE optional) → create Summary + Tasks in Neo4j,
then (optionally) approve & execute tasks in one go.

Features
- SEGMENTs are canonical; UTTERANCEs are low-confidence (optional).
- Robust Ollama JSON handling: stream=false + fallback for NDJSON / code fences.
- Batch mode over transcriptions "missing notes" (no Summary OR empty t.notes).
- Optional: write final summary into t.notes (--write-notes).
- Approve tasks in REVIEW:      --approve-all [LIMIT]
- Execute READY tasks:          --execute-ready N
- List tasks by status:         --list STATUS

Env
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=*****
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=llama3.1:8b
  ARTIFACTS_DIR=./artifacts   # where execution writes outputs

Examples
  # Dry-run latest transcription (no writes)
  python summarize_from_segments.py --latest --dry-run

  # One by id (write Summary+Tasks)
  python summarize_from_segments.py --id <TRANSCRIPTION_ID>

  # Batch: all "missing notes" (no Summary or empty t.notes), up to 100
  python summarize_from_segments.py --all-missing --limit 100 --write-notes

  # Only approve & execute (no summarization step)
  python summarize_from_segments.py --approve-all 200 --execute-ready 10

  # Summarize EVERYTHING missing, then approve all, then execute 5
  python summarize_from_segments.py --all-missing --limit 100 --approve-all 1000 --execute-ready 5
"""
import os, json, argparse, time, re, uuid, datetime, pathlib
from typing import List, Dict, Any, Optional
import requests
from neo4j import GraphDatabase
import subprocess, shutil

# ---- env ----
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "auto")  # auto | chat | generate
OLLAMA_USE_CLI  = os.getenv("OLLAMA_USE_CLI", "0").lower() in ("1","true","yes","on")
# ---- env ----
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))  # seconds; was 180
OLLAMA_USE_CLI  = os.getenv("OLLAMA_USE_CLI", "0").lower() in ("1","true","yes","on")
SUMMARY_FAILOPEN = os.getenv("SUMMARY_FAILOPEN", "copy")  # copy | skip

# ---- prompts ----
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
# Helpers
# ==========================



# =========================
# Neo4j helpers (summarize)
# =========================
def neo_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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
        rec = session.run(q, id=trans_id).single()
    else:
        q = """
        MATCH (t:Transcription)
        WITH t ORDER BY coalesce(t.created_at, datetime()) DESC
        LIMIT 1
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
        WITH t, s ORDER BY coalesce(s.start, s.idx, 0)
        WITH t, collect({id:s.id, start:coalesce(s.start,0.0), end:coalesce(s.end,0.0), text:s.text, type:'SEGMENT', low_conf:false}) AS segs
        OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u)
        WITH t, segs, u ORDER BY coalesce(u.start, u.idx, 0)
        RETURN t, segs, collect({id:u.id, start:coalesce(u.start,0.0), end:coalesce(u.end,0.0), text:u.text, type:'UTTERANCE', low_conf:true}) AS utts
        """
        rec = session.run(q).single()

    if not rec:
        return None
    t = dict(rec["t"])
    segs = [x for x in (rec["segs"] or []) if x and x.get("text")]
    utts = [x for x in (rec["utts"] or []) if x and x.get("text")]
    return {"t": t, "segments": segs, "utterances": utts}

def _serialize_tasks_for_db(tasks: list[dict]) -> list[dict]:
    out = []
    for t in (tasks or []):
        t2 = dict(t)
        # store JSON string; keep original in-memory copy for printing
        t2["acceptance_json"] = json.dumps(t.get("acceptance", []), ensure_ascii=False)
        out.append(t2)
    return out

def fetch_missing_transcriptions(session, limit: int) -> List[Dict[str, Any]]:
    """
    Candidates interpreted as "missing notes":
      - Transcriptions with NO Summary OR empty t.notes
    """
    q = """
    MATCH (t:Transcription)
    WHERE NOT (t)-[:HAS_SUMMARY]->(:Summary)
       OR trim(coalesce(t.notes, '')) = ''
    RETURN t
    ORDER BY coalesce(t.created_at, datetime()) DESC
    LIMIT $limit
    """
    return [dict(r["t"]) for r in session.run(q, limit=limit)]

# ==================
# Transcript assembly
# ==================
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
        if it["type"] == "UTTERANCE":
            lines.append(f"[LOW_CONF] {txt}")
        else:
            lines.append(txt)
    return "\n".join(lines)

# ==========================
# Ollama chat (robust JSON)
# ==========================
def _build_prompt(system: str, user: str) -> str:
    """
    Build a single-prompt string for /api/generate when /api/chat is unavailable.
    We keep it simple and robust for JSON tasks.
    """
    sys = (system or "").strip()
    usr = (user or "").strip()
    if sys:
        return f"<<SYS>>\n{sys}\n<</SYS>>\n\n{usr}"
    return usr


def _ollama_generate_json(system: str, user: str, temperature: float) -> Dict[str, Any]:
    """
    Fallback to /api/generate for Ollama servers without /api/chat.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": _build_prompt(system, user),
        "options": {"temperature": temperature, "num_ctx": 8192},
        # Many Ollama builds accept "format":"json" on /api/generate; if not, we'll still parse text.
        "format": "json",
        "stream": False,
    }
    url = f"{OLLAMA_HOST}/api/generate"
    r = requests.post(url, json=payload, timeout=180)
    # Some older servers return 200 with a textual body even on errors; raise_for_status is still fine.
    r.raise_for_status()

    # Try the normal (non-streaming) shape first
    try:
        body = r.json()
        # Newer builds: {"response":"{...json...}", ...}
        content = body.get("response", "")
        obj = _coerce_json_dict(content)
        if obj:
            return obj
    except Exception:
        pass

    # NDJSON fallback (or odd servers that ignore stream:false)
    text = r.text or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pieces = []
    ndjson_ok = True
    for ln in lines:
        try:
            o = json.loads(ln)
            # Streaming emits {"response":"..."} chunks
            if "response" in o:
                pieces.append(o["response"])
        except Exception:
            ndjson_ok = False
            break
    if ndjson_ok and pieces:
        stitched = "".join(pieces)
        obj = _coerce_json_dict(stitched)
        if obj:
            return obj

    # Last resort: scrape outer { ... }
    return _coerce_json_dict(text)

def _build_prompt(system: str, user: str) -> str:
    sys = (system or "").strip()
    usr = (user or "").strip()
    return f"<<SYS>>\n{sys}\n<</SYS>>\n\n{usr}" if sys else usr

def _ollama_cli_generate_json(system: str, user: str) -> Dict[str, Any]:
    """
    Stream-aware CLI fallback. Uses `ollama run --json` and stitches the "response" tokens.
    Returns a dict if JSON parse succeeds, else {}.
    """
    if not shutil.which("ollama"):
        return {}
    prompt = _build_prompt(system, user)
    try:
        # --json streams objects like {"response":"...","done":false} ... {"done":true}
        proc = subprocess.Popen(
            ["ollama", "run", "--json", OLLAMA_MODEL, "-p", prompt],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        pieces = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "response" in obj:
                    pieces.append(obj["response"])
            except Exception:
                # If a line isn't JSON, ignore it; stderr will capture errors
                pass
        proc.wait(timeout=OLLAMA_TIMEOUT)
        raw = "".join(pieces)
        return _coerce_json_dict(raw) or {}
    except Exception:
        return {}



def ollama_chat_json(system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    def _try_chat() -> Dict[str, Any]:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
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

        # Non-streaming happy path
        try:
            body = r.json()
            content = body.get("message", {}).get("content", "")
            obj = _coerce_json_dict(content)
            if obj:
                return obj
        except Exception:
            pass

        # NDJSON fallback (servers that ignore stream:false)
        text = r.text or ""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        pieces, ndjson_ok = [], True
        for ln in lines:
            try:
                o = json.loads(ln)
                if "message" in o and "content" in o["message"]:
                    pieces.append(o["message"]["content"])
                elif "response" in o:
                    pieces.append(o["response"])
            except Exception:
                ndjson_ok = False
                break
        if ndjson_ok and pieces:
            stitched = "".join(pieces)
            obj = _coerce_json_dict(stitched)
            if obj:
                return obj

        return _coerce_json_dict(r.text or "")

    def _try_generate() -> Dict[str, Any]:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": _build_prompt(system, user),
            "options": {"temperature": temperature, "num_ctx": 8192},
            "format": "json",
            "stream": False,
        }
        url = f"{OLLAMA_HOST}/api/generate"
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 404:
            raise requests.HTTPError("generate endpoint not found", response=r)
        r.raise_for_status()
        try:
            body = r.json()
            content = body.get("response", "")
            obj = _coerce_json_dict(content)
            if obj:
                return obj
        except Exception:
            pass

        # NDJSON gather (if server streamed anyway)
        text = r.text or ""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        pieces, ndjson_ok = [], True
        for ln in lines:
            try:
                o = json.loads(ln)
                if "response" in o:
                    pieces.append(o["response"])
            except Exception:
                ndjson_ok = False
                break
        if ndjson_ok and pieces:
            stitched = "".join(pieces)
            obj = _coerce_json_dict(stitched)
            if obj:
                return obj

        return _coerce_json_dict(text)

    endpoint = (os.getenv("OLLAMA_ENDPOINT", "auto") or "auto").strip().lower()
    if endpoint == "chat":
        try:
            return _try_chat()
        except Exception:
            # fallback cascade
            try: return _try_generate()
            except Exception: pass
            if OLLAMA_USE_CLI: return _ollama_cli_generate_json(system, user)
            return {}
    if endpoint == "generate":
        try:
            return _try_generate()
        except Exception:
            try: return _try_chat()
            except Exception: pass
            if OLLAMA_USE_CLI: return _ollama_cli_generate_json(system, user)
            return {}

    # auto
    try:
        return _try_chat()
    except Exception:
        pass
    try:
        return _try_generate()
    except Exception:
        pass
    if OLLAMA_USE_CLI:
        return _ollama_cli_generate_json(system, user)
    return {}

def _failopen_summary_from_text(full_text: str, max_chars: int = 2000) -> Dict[str, Any]:
    """
    If the LLM doesn't return usable JSON, build a concise, deterministic summary
    from the SEGMENT text so notes aren't blank.
    """
    lines = [ln.strip() for ln in (full_text or "").splitlines() if ln.strip() and not ln.startswith("[LOW_CONF]")]
    if not lines:
        return {"summary": "", "bullets": []}
    # Summary: first ~4-6 sentences/lines up to max_chars
    summary_parts = []
    total = 0
    for ln in lines:
        if total + len(ln) + 1 > max_chars:
            break
        summary_parts.append(ln)
        total += len(ln) + 1
        if len(summary_parts) >= 6:
            break
    summary = " ".join(summary_parts).strip()

    # Bullets: up to 8 distinct non-duplicate lines (shortened)
    bullets, seen = [], set()
    for ln in lines:
        short = ln if len(ln) <= 160 else (ln[:157] + "…")
        if short in seen:
            continue
        seen.add(short)
        bullets.append(short)
        if len(bullets) >= 8:
            break

    return {"summary": summary, "bullets": bullets}

def _coerce_json_dict(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = re.sub(r"^\s*```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.IGNORECASE)

    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return o
    except Exception:
        pass

    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            o = json.loads(s[start:end+1])
            if isinstance(o, dict):
                return o
        except Exception:
            pass
    return {}

# ==================
# Map-reduce summary
# ==================
def chunk(text: str, max_chars: int = 7000) -> List[str]:
    chunks, buf = [], ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > max_chars:
            if buf: chunks.append(buf); buf = line
        else:
            buf = (buf+"\n"+line) if buf else line
    if buf: chunks.append(buf)
    return chunks

def summarize_as_json(full_text: str) -> Dict[str, Any]:
    parts = chunk(full_text)
    if len(parts) == 1:
        obj = ollama_chat_json(SUMMARIZE_INSTR, parts[0])
        res = _normalize_summary_obj(obj)
        if not res["summary"].strip() and SUMMARY_FAILOPEN == "copy":
            res = _failopen_summary_from_text(full_text)
        return res

    partials = []
    for i, ch in enumerate(parts, 1):
        pj = ollama_chat_json(SUMMARIZE_INSTR, f"[CHUNK {i}/{len(parts)}]\n\n{ch}")
        partials.append(_normalize_summary_obj(pj))

    combined = json.dumps(partials, ensure_ascii=False)
    final = ollama_chat_json(
        SUMMARIZE_INSTR,
        "Combine these partial results into one final JSON with shape {summary, bullets}:\n" + combined
    )
    res = _normalize_summary_obj(final)
    if not res["summary"].strip() and SUMMARY_FAILOPEN == "copy":
        res = _failopen_summary_from_text(full_text)
    return res


def _normalize_summary_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"summary": str(obj), "bullets": []}
    summary = str(obj.get("summary",""))
    bullets = obj.get("bullets") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    return {"summary": summary, "bullets": bullets}

def extract_tasks_json(summary_obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = None
    payload = json.dumps(
        {"summary": summary_obj.get("summary",""), "bullets": summary_obj.get("bullets",[])},
        ensure_ascii=False
    )
    for _ in range(3):
        try:
            obj = ollama_chat_json("Return STRICT JSON only.", TASKS_INSTR + "\n\n" + payload)
            if isinstance(obj, dict) and "tasks" in obj:
                break
        except Exception:
            time.sleep(0.5)
    if not isinstance(obj, dict):
        obj = {"summary": summary_obj.get("summary",""), "bullets": summary_obj.get("bullets",[]), "tasks": []}

    for t in obj.get("tasks", []):
        if not t.get("acceptance"):
            t["acceptance"] = [{"type":"file_exists","args":{"path":"artifacts/{TASK_ID}/output.txt"}}]
        t["title"] = str(t.get("title","")).strip() or "Untitled Task"
        t["description"] = str(t.get("description","")).strip()
        pri = str(t.get("priority","MEDIUM")).upper()
        if pri not in {"LOW","MEDIUM","HIGH"}: pri = "MEDIUM"
        t["priority"] = pri
        try:
            t["confidence"] = float(t.get("confidence", 0.5))
        except Exception:
            t["confidence"] = 0.5
    return obj

# ===============
# Write-back
# ===============
def write_back(session, trans_id: str, final_obj: Dict[str, Any], write_notes: bool=False) -> str:
    q = """
    MATCH (t:Transcription {id:$tid})
    CREATE (s:Summary {id: randomUUID(), text:$text, bullets:$bullets, created_at: datetime()})
    MERGE (t)-[:HAS_SUMMARY]->(s)
    FOREACH (_ IN (CASE WHEN $write_notes THEN [1] ELSE [] END) |
        SET t.notes = $text, t.updated_at = datetime()
    )
    FOREACH (tsk IN $tasks |
        CREATE (tk:Task {
          id: randomUUID(),
          title: coalesce(tsk.title,"Untitled Task"),
          description: coalesce(tsk.description,""),
          priority: coalesce(tsk.priority,"MEDIUM"),
          due: tsk.due,
          status: "REVIEW",
          confidence: coalesce(tsk.confidence,0.5),
          acceptance_json: coalesce(tsk.acceptance_json, "[]")
        })
        MERGE (s)-[:GENERATED_TASK]->(tk)
    )
    RETURN s.id AS sid
    """
    bullets = [str(b).strip() for b in (final_obj.get("bullets") or []) if str(b).strip()]
    tasks_serialized = _serialize_tasks_for_db(final_obj.get("tasks") or [])
    res = session.run(
        q,
        tid=trans_id,
        text=final_obj.get("summary",""),
        bullets=bullets,
        tasks=tasks_serialized,
        write_notes=bool(write_notes),
    )
    row = res.single()
    return row["sid"] if row and "sid" in row else ""


def process_one(session, tnode: Dict[str,Any], include_utterances: bool, dry_run: bool, write_notes: bool) -> Optional[str]:
    row = fetch_one(session, tnode.get("id"), latest=False)
    if not row:
        print(f"[{tnode.get('key') or tnode.get('id')}] not found or empty.")
        return None
    t = row["t"]; segments, utterances = row["segments"], row["utterances"]
    key = t.get("key") or t.get("id")
    if not segments and not utterances:
        print(f"[{key}] No text found in Segments/Utterances.")
        return None

    text = build_transcript(segments, utterances, include_utterances=include_utterances)
    print(f"[{key}] Items: {len(segments)} segments + {len(utterances)} utterances (included={include_utterances})")

    summary_obj = summarize_as_json(text)

    if not summary_obj.get("summary","").strip():
        print(f"[{key}] ⚠️  Summarizer returned empty; skipping write (set SUMMARY_FAILOPEN=copy to auto-fill).")
        return None

    final_obj = extract_tasks_json(summary_obj)
    print(f"[{key}] Will write {len(final_obj.get('tasks', []))} task(s).")

    if dry_run:
        print(json.dumps({"transcription": key, **final_obj}, indent=2))
        return None

    sid = write_back(session, t.get("id"), final_obj, write_notes=write_notes)
    if not sid:
        print(f"[{key}] ⚠️  Write-back returned no Summary id.")
    else:
        print(f"[{key}] Wrote Summary {sid} and {len(final_obj.get('tasks',[]))} Task(s).")
    return sid

# ====================================
# Approve & Execute (integrated flags)
# ====================================
def q_list(session, status: str, limit: int = 25) -> List[Dict[str, Any]]:
    res = session.run(
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
    )
    return res.data()

def q_get_task(session, task_id: str) -> Optional[Dict[str, Any]]:
    rec = session.run(
        """
        MATCH (task:Task {id:$id})
        OPTIONAL MATCH (s:Summary)-[:GENERATED_TASK]->(task)
        OPTIONAL MATCH (t:Transcription)-[:HAS_SUMMARY]->(s)
        RETURN task, s, t
        """, id=task_id
    ).single()
    if not rec:
        return None
    task = dict(rec["task"])
    # 🔎 hydrate acceptance from JSON text if present
    if "acceptance_json" in task and isinstance(task["acceptance_json"], str):
        try:
            task["acceptance"] = json.loads(task["acceptance_json"])
        except Exception:
            task["acceptance"] = []
    s = dict(rec["s"]) if rec["s"] else None
    t = dict(rec["t"]) if rec["t"] else None
    return {"task": task, "summary": s, "transcription": t}


def q_approve_all_review(session, limit: int) -> int:
    res = session.run(
        """
        MATCH (task:Task {status:'REVIEW'})
        WITH task LIMIT $limit
        SET task.status = 'READY', task.updated_at = datetime()
        RETURN count(task) AS n
        """, limit=limit
    ).single()
    return res["n"]

def q_pick_ready(session, limit: int) -> List[str]:
    res = session.run(
        """
        MATCH (task:Task {status:'READY'})
        WITH task ORDER BY coalesce(task.updated_at, datetime()) ASC
        LIMIT $limit
        SET task.status = 'RUNNING', task.updated_at = datetime()
        RETURN task.id AS id
        """, limit=limit
    )
    return [r["id"] for r in res]

def q_attach_run(session, task_id: str, run: Dict[str, Any]):
    res = session.run(
        """
        MATCH (task:Task {id:$tid})
        CREATE (r:Run {
          id: $rid,
          started_at: datetime($started_at),
          status: $status,
          manifest_json: $manifest_json
        })
        MERGE (task)-[:HAS_RUN]->(r)
        RETURN r.id AS rid
        """,
        tid=task_id,
        rid=run["id"],
        started_at=run["started_at"],
        status=run["status"],
        manifest_json=json.dumps(run.get("manifest", {}), ensure_ascii=False),
    ).single()
    return res["rid"]

def q_finish_run(session, task_id: str, run_id: str, success: bool, manifest: Dict[str, Any]):
    session.run(
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
        success = success,
        manifest_json = json.dumps(manifest, ensure_ascii=False),
    )

# ---- Acceptance helpers ----
def replace_placeholders(s: str, task_id: str) -> str:
    return (s or "").replace("{TASK_ID}", task_id)

def acc_file_exists(args: Dict[str, Any], task_id: str) -> (bool, str):
    path = replace_placeholders(args.get("path",""), task_id)
    path = path.replace("artifacts/", f"{ARTIFACTS_DIR.rstrip('/')}/")
    ok = pathlib.Path(path).exists()
    return ok, f"path={path}"

def acc_contains(args: Dict[str, Any], task_id: str) -> (bool, str):
    path = replace_placeholders(args.get("path",""), task_id)
    path = path.replace("artifacts/", f"{ARTIFACTS_DIR.rstrip('/')}/")
    text = args.get("text","")
    try:
        data = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        ok = text in data
        return ok, f"path={path} len={len(data)}"
    except Exception as e:
        return False, f"path={path} err={e}"

def acc_regex(args: Dict[str, Any], task_id: str) -> (bool, str):
    path = replace_placeholders(args.get("path",""), task_id)
    path = path.replace("artifacts/", f"{ARTIFACTS_DIR.rstrip('/')}/")
    pat = args.get("pattern","")
    try:
        data = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        ok = re.search(pat, data) is not None
        return ok, f"path={path} pat={pat}"
    except Exception as e:
        return False, f"path={path} err={e}"

def acc_http_ok(args: Dict[str, Any], _task_id: str) -> (bool, str):
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
    """
    Minimal, safe "execution":
      - make artifacts/{TASK_ID}
      - write output.txt with task info + timestamp
      - if acceptance includes http_ok, fetch and store short response for transparency
    """
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

    return {
        "steps": steps,
        "duration_sec": round(time.time() - t0, 3),
        "artifacts_dir": str(artdir),
    }

def approve_all(session, limit: int) -> int:
    n = q_approve_all_review(session, limit=limit)
    print(f"Approved {n} task(s) from REVIEW → READY.")
    return n

def execute_ready(limit: int):
    picked = []
    with neo_driver().session() as s:
        picked = q_pick_ready(s, limit=limit)
    if not picked:
        print("No READY tasks.")
        return

    for tid in picked:
        with neo_driver().session() as s:
            row = q_get_task(s, tid)
            if not row:
                print(f"{tid}: not found"); continue
            task = row["task"]

            run = {
                "id": str(uuid.uuid4()),
                "started_at": datetime.datetime.utcnow().isoformat()+"Z",
                "status": "RUNNING",
                "manifest": {"task_id": tid, "steps": [], "acceptance_results": []},
            }
            q_attach_run(s, tid, run)

        # Execute out of tx
        manifest_exec = execute_task_minimal(task)
        results = run_acceptance(task, tid)
        success = all(r["passed"] for r in results) if results else True

        run["manifest"].update(manifest_exec)
        run["manifest"]["acceptance_results"] = results

        with neo_driver().session() as s:
            q_finish_run(s, tid, run["id"], success, run["manifest"])

        print(f"{tid}: {'DONE' if success else 'FAILED'}  artifacts={manifest_exec['artifacts_dir']}")

# ===========
# CLI driver
# ===========
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

    # New: approve & execute flags
    ap.add_argument("--approve-all", nargs="?", const=100, type=int,
                    help="Approve up to N tasks in REVIEW (default 100 if no value provided)")
    ap.add_argument("--execute-ready", type=int, default=0,
                    help="Execute up to N tasks in READY")

    # New: list by status
    ap.add_argument("--list", dest="list_status", choices=["REVIEW","READY","RUNNING","DONE","FAILED"],
                    help="List tasks by status (no writes)")

    # Optional override for artifacts dir
    ap.add_argument("--artifacts-dir", help="Override ARTIFACTS_DIR (default from env ./artifacts)")

    args = ap.parse_args()
    global ARTIFACTS_DIR
    if args.artifacts_dir:
        ARTIFACTS_DIR = args.artifacts_dir

    # Optional: just list and exit
    if args.list_status:
        with neo_driver().session() as s:
            rows = q_list(s, status=args.list_status, limit=50)
        if not rows:
            print("(none)")
            return
        for r in rows:
            print(f"[{r.get('tkey') or ''}] {r['id']}  {r['priority']:>6}  {r['title']}")
        return

    # Summarization step (only if one of the selectors is present)
    did_summarize = False
    with neo_driver().session() as sess:
        if args.trans_id or args.latest or args.all_missing:
            did_summarize = True
            if args.trans_id or args.latest:
                if args.latest and not args.trans_id:
                    row = fetch_one(sess, None, latest=True)
                    if not row:
                        print("No Transcription found."); return
                    tnode = row["t"]
                else:
                    tnode = {"id": args.trans_id}
                process_one(sess, tnode, args.include_utterances, args.dry_run, args.write_notes)
            else:
                cands = fetch_missing_transcriptions(sess, limit=args.limit)
                if not cands:
                    print("No candidates found (all have notes/summaries).")
                else:
                    for tnode in cands:
                        process_one(sess, tnode, args.include_utterances, args.dry_run, args.write_notes)

    # Approve & Execute steps (run whether or not we summarized this invocation)
    if args.approve_all is not None:
        with neo_driver().session() as s:
            approve_all(s, limit=args.approve_all)
    if args.execute_ready and args.execute_ready > 0:
        execute_ready(args.execute_ready)

    # Guidance if user ran without any action
    if not did_summarize and args.approve_all is None and args.execute_ready == 0 and not args.list_status:
        print("Nothing to do. Provide --id/--latest/--all-missing to summarize, or --approve-all / --execute-ready, or --list STATUS.")

if __name__ == "__main__":
    main()





# cd ~/git/video-automation
# source .venv/bin/activate

# export OLLAMA_MODEL="llama3:latest"   # or: gemma2:2b
# export OLLAMA_USE_CLI=1               # force CLI, avoids HTTP 404s

# nohup bash -lc '
# while true; do
#   OUT=$(python summarize_from_segments.py --all-missing --limit 200 --write-notes 2>&1 || true)
#   echo "$OUT" | tee -a summarize_run.log
#   echo "$OUT" | grep -q "No candidates found" && break
#   sleep 5
# done
# echo "[DONE] $(date -Is)" | tee -a summarize_run.log
# ' >/dev/null 2>&1 & disown
