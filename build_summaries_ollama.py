#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSON summaries (sidecars) for transcripts using Ollama (Gemma3:4b or other model).

Key improvements:
- Pattern-aware best-source selection per stem (uses your sidecars order)
- Sequential only (safe on mlx / MacBook Air)
- Auto-throttle based on last call time & consecutive failures
- Robust JSON extraction (balanced-brace) + schema normalization
- Writes <stem>_summary.bad.txt on parse failure

Usage
-----
python build_summaries_ollama.py \
  --roots /Volumes/Untitled/audio \
  --model gemma3:4b \
  --auto-throttle \
  --ctx 1536 --predict 512 --prompt-chars 100000
"""

import os, re, json, time, logging, argparse, csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import http.client
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summaries")

# ----------------------------
# Transcript patterns (priority order)
# ----------------------------
TRANSCRIPT_PATTERNS = [
    "{stem}_large-v3_transcription.txt",  # JSON formatted (.txt)
    "{stem}_BC_medium_transcription.txt", # JSON formatted (.txt)
    "{stem}_medium_transcription.txt",    # JSON formatted (.txt)
    "{stem}_BC_transcription.csv",        # CSV
    "{stem}_transcription.txt",           # JSON formatted (.txt)
    "{stem}_transcription.csv",           # CSV
    "{stem}.json",                        # JSON
    "{stem}.txt",                         # JSON formatted or plain text
    "{stem}.vtt",                         # WebVTT
]

# Accepts: 2024_1229_194226 | 20250827061344 | 20250405212855_000029
STEM_RE = re.compile(r"^(\d{4}_\d{4}_\d{6}|\d{12,})(?:_[0-9]{6})?", re.ASCII)

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

    def generate(self, model: str, prompt: str, retries: int = 3, timeout: float = 120.0,
                 options: Dict[str, Any] | None = None) -> str:
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
}

PROMPT_SCHEMA = (
    "You are a precise summarization model. Output STRICT JSON only matching this schema: "
    + json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False)
    + "\nRules:\n"
      "- Output ONLY JSON, no markdown, no preface.\n"
      "- If unknown, use empty array or empty string.\n"
      "\nTranscript begins:\n\n"
)
PROMPT_END = "\n\nEND."

# ----------------------------
# Helpers (readers & parsing)
# ----------------------------
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

def _read_json_file(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed reading JSON %s: %s", p, e)
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
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return t[start:i+1]
    return None

def _read_json_from_txt_or_raw_txt(p: Path) -> Optional[str]:
    """Some .txt files are actually JSON. Try JSON first; fallback to raw text."""
    txt = _read_text_file(p)
    if txt is None:
        return None
    j_blob = _extract_first_json_object(txt)
    if j_blob:
        try:
            j = json.loads(j_blob)
            if isinstance(j, dict):
                if j.get("text"):
                    return str(j.get("text")).strip()
                if j.get("transcript"):
                    return str(j.get("transcript")).strip()
                segs = j.get("segments") or []
                if isinstance(segs, list) and segs:
                    parts = [str(s.get("text") or "").strip() for s in segs if isinstance(s, dict)]
                    return "\n".join([p for p in parts if p]) or txt
            if isinstance(j, str):
                return j.strip()
        except Exception:
            pass
    return txt

def _read_vtt_to_text(p: Path) -> Optional[str]:
    try:
        lines = []
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip("\n")
                # Skip WEBVTT header, timestamp lines, cue numbers
                if not s:
                    continue
                if s.upper().startswith("WEBVTT"):
                    continue
                if re.match(r"^\d+$", s):
                    continue
                if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> ", s) or re.match(r"^\d{2}:\d{2}\.\d{3} --> ", s):
                    continue
                lines.append(s)
        text = "\n".join(lines).strip()
        return text or None
    except Exception as e:
        log.warning("Failed reading VTT %s: %s", p, e)
        return None

def _normalize_payload(any_payload: Any) -> Dict[str, Any]:
    if isinstance(any_payload, list):
        for el in any_payload:
            if isinstance(el, dict):
                any_payload = el
                break
        else:
            any_payload = {"summary": " ".join(str(x) for x in any_payload)}
    elif isinstance(any_payload, str):
        any_payload = {"summary": any_payload}
    if not isinstance(any_payload, dict):
        any_payload = {}
    any_payload.setdefault("version", "1.0")
    any_payload.setdefault("language", "")
    any_payload.setdefault("summary", "")
    for k in ("key_points","topics","people","organizations","places"):
        v = any_payload.get(k)
        if not isinstance(v, list):
            any_payload[k] = []
        else:
            # de-dupe while preserving order
            any_payload[k] = list(dict.fromkeys([str(x).strip() for x in v if str(x).strip()]))
    any_payload.setdefault("quality_notes", "")
    return any_payload

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.70)
    tail = int(limit * 0.25)
    return text[:head] + "\n...\n" + text[-tail:]

# ----------------------------
# Discovery: group files by stem per directory and pick best by pattern order
# ----------------------------
def _discover_stems(roots: List[Path]) -> List[Tuple[Path, str, List[Path]]]:
    """Return list of (dirpath, stem, files_for_stem)"""
    buckets: Dict[Tuple[str,str], List[Path]] = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            dp = Path(dirpath)
            for name in files:
                p = dp / name
                st = _detect_stem(p)
                if not st:
                    continue
                buckets.setdefault((str(dp), st), []).append(p)
    out: List[Tuple[Path,str,List[Path]]] = []
    for (dp_str, st), paths in buckets.items():
        out.append((Path(dp_str), st, paths))
    return out

def _choose_best_path(dirpath: Path, stem: str, candidates: List[Path]) -> Optional[Path]:
    names = {p.name for p in candidates}
    for pat in TRANSCRIPT_PATTERNS:
        expected = pat.format(stem=stem)
        if expected in names:
            return dirpath / expected
        cand = dirpath / expected
        try:
            if cand.exists():
                return cand
        except Exception:
            pass
    return None

# ----------------------------
# Processing
# ----------------------------
def _load_transcript_text(p: Path) -> Optional[str]:
    suf = p.suffix.lower()
    if suf == ".csv":
        return _read_csv_concat_text(p)
    if suf == ".json":
        j = _read_json_file(p)
        if isinstance(j, dict):
            if j.get("text"):
                return str(j.get("text")).strip()
            if j.get("transcript"):
                return str(j.get("transcript")).strip()
            segs = j.get("segments") or []
            if isinstance(segs, list) and segs:
                parts = [str(s.get("text") or "").strip() for s in segs if isinstance(s, dict)]
                txt = "\n".join([t for t in parts if t])
                if txt:
                    return txt
        return _read_text_file(p)
    if suf == ".vtt":
        return _read_vtt_to_text(p)
    # .txt or others -> try JSON-in-.txt, else raw text
    return _read_json_from_txt_or_raw_txt(p)

def process_one(dirpath: Path, stem: str, files_for_stem: List[Path],
                client: OllamaClient, model: str, opts: Dict[str, Any], state: Dict[str, Any],
                overwrite=False, dry_run=False) -> bool:
    best = _choose_best_path(dirpath, stem, files_for_stem)
    if not best:
        return False
    out_path = best.parent / f"{stem}_summary.json"
    bad_path = best.parent / f"{stem}_summary.bad.txt"

    if out_path.exists() and not overwrite:
        _auto_sleep(state, True, 0.02)
        return True

    text = _load_transcript_text(best)
    if not text:
        _auto_sleep(state, True, 0.01)
        return False

    text = _truncate(text.strip(), state["prompt_chars"])
    prompt = PROMPT_SCHEMA + text + PROMPT_END

    if dry_run:
        log.info("DRY-RUN would summarize %s -> %s", best.name, out_path.name)
        _auto_sleep(state, True, 0.05)
        return True

    start = time.time()
    try:
        raw = client.generate(model, prompt, options=opts)
        state["consec_fail"] = 0
    except Exception as e:
        state["consec_fail"] += 1
        log.error("Generation failed for %s: %s", stem, e)
        _auto_sleep(state, False, time.time() - start)
        return False
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
            bad_path.write_text(raw, encoding="utf-8")
            log.error("JSON parse error for %s, wrote raw to %s", stem, bad_path)
            _auto_sleep(state, False, state["last_duration"])
            return False

    payload = _normalize_payload(payload)
    payload["_meta"] = {
        "source_transcript": str(best),
        "generated_by_model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "summary.sidecar.v1",
    }

    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    if bad_path.exists():
        try: bad_path.unlink()
        except Exception: pass
    log.info("Wrote summary %s (from %s)", out_path, best.name)

    _auto_sleep(state, True, state["last_duration"])
    return True

def _auto_sleep(state: Dict[str, Any], success: bool, last_duration: float) -> None:
    if success:
        state["consec_fail"] = 0
    else:
        state["consec_fail"] += 1
    base = 0.25 * max(0.0, last_duration)
    penalty = 0.4 * state["consec_fail"]
    sleep_sec = min(state["sleep_max"], max(state["sleep_min"], base + penalty))
    if state["auto_throttle"]:
        time.sleep(sleep_sec)

# ----------------------------
# Main (sequential only)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Root directories to scan (recursive)")
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
    ap.add_argument("--limit", type=int, default=0, help="Process at most N stems (0 = no limit)")
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

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    roots = [r for r in roots if r.exists()]
    if not roots:
        log.error("No valid roots provided.")
        return

    stems = _discover_stems(roots)
    log.info("Found %d stem groups", len(stems))

    total = 0
    ok = 0
    for dirpath, stem, files_for_stem in stems:
        if args.limit and total >= args.limit:
            break
        total += 1
        if process_one(dirpath, stem, files_for_stem, client, args.model, opts, state,
                       overwrite=args.overwrite, dry_run=args.dry_run):
            ok += 1

    log.info("Completed sequentially. stems=%d ok=%d errors=%d", total, ok, total - ok)

if __name__ == "__main__":
    main()
