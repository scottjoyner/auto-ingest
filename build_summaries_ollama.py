#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSON summary sidecars for transcripts using a single local Ollama instance.

Key features for low-memory stability (mlx / MacBook Air):
- SEQUENTIAL ONLY (one request at a time)
- AUTO-THROTTLE (pacing derived from last gen duration + failures)
- Memory-friendly defaults for small models (e.g., gemma3:4b)
- Robust JSON extraction and normalization (no more `"version"` KeyError)
- Idempotent: skips existing valid <stem>_summary.json unless --overwrite
- Prefers higher-quality transcript files:
    1) *_medium_transcription.txt
    2) *_transcription.txt
    3) *_transcription.csv (concats text columns)

Outputs (alongside transcript):
  <stem>_summary.json
  <stem>_summary.bad.txt   (if model output wasn’t valid JSON)

Usage example:
  python build_summaries_ollama.py \
    --roots /Volumes/Untitled/audio \
    --model gemma3:4b \
    --auto-throttle \
    --ctx 1536 --predict 512 --prompt-chars 100000
"""

import os, re, csv, json, time, logging, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import http.client
from urllib.parse import urlparse

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summaries")

# ----------------------------
# Patterns & Limits
# ----------------------------
# Accepted stems:
#   2024_1229_194226
#   20250827061344
#   20250405212855_000029  (suffix allowed)
STEM_RE = re.compile(r"^(\d{4}_\d{4}_\d{6}|\d{12,})(?:_[0-9]{6})?", re.ASCII)

TRANSCRIPT_PATTERNS = [
    "{stem}_large-v3_transcription.txt",
    "{stem}_BC_medium_transcription.txt",
    "{stem}_medium_transcription.txt",
    "{stem}_BC_transcription.csv",
    "{stem}_transcription.txt",
    "{stem}_transcription.csv",
    "{stem}.json",
    "{stem}.txt",
    "{stem}.vtt"
]
# ----------------------------
# Ollama client (SEQUENTIAL)
# ----------------------------
class OllamaClient:
    """Minimal HTTP client for Ollama /api/generate; single instance / sequential calls."""
    def __init__(self, base_url: str):
        u = urlparse(base_url)
        if not u.scheme.startswith("http"):
            raise ValueError("OLLAMA_HOST must be http(s) URL, e.g. http://127.0.0.1:11434")
        self._is_https = (u.scheme == "https")
        self._host = u.hostname or "127.0.0.1"
        self._port = u.port or (443 if self._is_https else 11434)
        self._base_path = u.path.rstrip("/")

    def generate(self, model: str, prompt: str,
                 retries: int = 3, timeout: float = 180.0,
                 options: Dict[str, Any] | None = None) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        body = json.dumps(payload).encode("utf-8")
        path = f"{self._base_path}/api/generate" if self._base_path else "/api/generate"

        for attempt in range(1, retries + 1):
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
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Ollama call failed after retries")

# ----------------------------
# Prompt (NO str.format on schema-containing string)
# ----------------------------
SUMMARY_SCHEMA_HINT = {
    "version": "1.0",
    "language": "<ISO 639-1, e.g., en>",
    "summary": "<5-9 sentence summary>",
    "key_points": ["<bullet 1>", "<bullet 2>"],
    "topics": ["<topic-1>", "<topic-2>"],
    "people": ["<name>", "..."],
    "organizations": ["<org>", "..."],
    "places": ["<place>", "..."],
    "quality_notes": "<noticed noise/mishears/language mix/etc.>",
}

PROMPT_SCHEMA = (
    "You are a precise summarizer. Provided raw data files of arbitrary content, transcriptions, diraitizations. OUTPUT STRICT JSON ONLY (no markdown/code fences/prose), "
    "matching this schema exactly: " + json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False) + "\n"
    "Rules:\n"
    "- Output ONLY JSON (no preface/trailing text).\n"
    "- Use empty arrays/strings when unknown.\n"
    "- Deduplicate arrays.\n"
    "- Keep it concise and factual.\n\n"
    "- No mentions the file strucutre, format in the summary\n"
    "Transcript begins:\n\n"
)
PROMPT_END = "\n\nEND."

# ----------------------------
# Types & Helpers
# ----------------------------
@dataclass
class Transcript:
    stem: str
    text: str
    source_path: Path

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
            if not rdr.fieldnames:
                return None
            prefer = None
            for name in rdr.fieldnames:
                if name and name.lower() in ("text", "utterance", "transcript", "content"):
                    prefer = name
                    break
            for row in rdr:
                if prefer and (prefer in row) and row[prefer]:
                    rows.append(str(row[prefer]))
                else:
                    vals = [str(v) for v in row.values() if isinstance(v, str) and v.strip()]
                    if vals:
                        rows.append(" ".join(vals))
        text = "\n".join(rows).strip()
        return text or None
    except Exception as e:
        log.warning("Failed reading CSV %s: %s", p, e)
        return None

def _choose_best_transcript(dirpath: Path, stem: str) -> Optional[Transcript]:
    for pat in TRANSCRIPT_PATTERNS:
        cand = dirpath / pat.format(stem=stem)
        try:
            if not cand.exists():
                continue
        except Exception:
            continue
        if cand.suffix.lower() == ".csv":
            txt = _read_csv_concat_text(cand)
        else:
            txt = _read_text_file(cand)
        if txt:
            return Transcript(stem=stem, text=txt, source_path=cand)
    return None

def _iter_unique_stems(roots: List[Path]) -> List[Tuple[str, Path]]:
    """Scan roots for *_transcription.(txt|csv), return unique (stem, dirpath) pairs (per directory)."""
    found: List[Tuple[str, Path]] = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            dp = Path(dirpath)
            for name in files:
                if name.endswith("_transcription.txt") or name.endswith("_transcription.csv"):
                    stem = _detect_stem(Path(name))
                    if stem:
                        found.append((stem, dp))
    # dedupe while preserving order
    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for stem, dp in found:
        key = (stem, str(dp))
        if key not in seen:
            seen.add(key)
            uniq.append((stem, dp))
    return uniq

def _extract_first_json_object(s: str) -> Optional[str]:
    """Extract the first top-level balanced {...} block, tolerating pre/post junk or code fences."""
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
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return t[start:i+1]
    return None

def _as_list_of_strings(x: Any) -> List[str]:
    out: List[str] = []
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str):
                v2 = v.strip()
                if v2:
                    out.append(v2)
    # dedupe preserving order
    return list(dict.fromkeys(out))

def _normalize_payload(any_payload: Any) -> Dict[str, Any]:
    """Coerce arbitrary model output into our dict schema with sane defaults."""
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
    any_payload["language"] = str(any_payload.get("language") or "")
    any_payload["summary"] = str(any_payload.get("summary") or "")
    any_payload["key_points"] = _as_list_of_strings(any_payload.get("key_points"))
    any_payload["topics"] = _as_list_of_strings(any_payload.get("topics"))
    any_payload["people"] = _as_list_of_strings(any_payload.get("people"))
    any_payload["organizations"] = _as_list_of_strings(any_payload.get("organizations"))
    any_payload["places"] = _as_list_of_strings(any_payload.get("places"))
    any_payload["quality_notes"] = str(any_payload.get("quality_notes") or "")
    return any_payload

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.70)
    tail = int(limit * 0.25)
    return text[:head] + "\n...\n" + text[-tail:]

# ----------------------------
# Core processing (sequential + auto-throttle)
# ----------------------------
def summarize_one(transcript: Transcript, client: OllamaClient, model: str,
                  overwrite: bool,
                  options: Dict[str, Any],
                  prompt_chars: int,
                  throttle_state: Dict[str, Any]) -> bool:
    stem = transcript.stem
    src = transcript.source_path
    out_path = src.parent / f"{stem}_summary.json"
    bad_path = src.parent / f"{stem}_summary.bad.txt"

    # Skip if valid summary exists unless overwrite
    if out_path.exists() and not overwrite:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and "version" in existing:
                log.debug("Skip existing valid summary: %s", out_path)
                # Still apply pacing to be gentle when skipping many
                _apply_sleep(throttle_state, success=True, last_duration=0.02)
                return True
        except Exception:
            pass  # will rewrite if corrupt

    text = _truncate(transcript.text.strip(), prompt_chars)
    # Build prompt WITHOUT str.format on schema-containing string
    prompt = PROMPT_SCHEMA + text + PROMPT_END

    # Sequential single call + timing
    start = time.time()
    try:
        raw = client.generate(model=model, prompt=prompt, timeout=180.0, options=options)
        cleaned = raw.strip()

        # Parse JSON
        obj_text = _extract_first_json_object(cleaned)
        if obj_text is None:
            try:
                payload = json.loads(cleaned)
            except Exception:
                bad_path.write_text(cleaned, encoding="utf-8")
                log.error("JSON not found for %s. Saved raw to %s", stem, bad_path)
                _apply_sleep(throttle_state, success=False, last_duration=time.time() - start)
                return False
        else:
            try:
                payload = json.loads(obj_text)
            except Exception:
                try:
                    payload = json.loads(cleaned)
                except Exception:
                    bad_path.write_text(cleaned, encoding="utf-8")
                    log.error("JSON parse error for %s. Saved raw to %s", stem, bad_path)
                    _apply_sleep(throttle_state, success=False, last_duration=time.time() - start)
                    return False

        payload = _normalize_payload(payload)
        payload["_meta"] = {
            "source_transcript": str(src),
            "generated_by_model": model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema": "summary.sidecar.v1",
        }

        tmp = Path(str(out_path) + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        log.info("Wrote summary %s", out_path)

        _apply_sleep(throttle_state, success=True, last_duration=time.time() - start)
        return True

    except Exception as e:
        log.error("Ollama error on %s: %s", stem, e)
        _apply_sleep(throttle_state, success=False, last_duration=time.time() - start)
        return False

def _apply_sleep(throttle_state: Dict[str, Any], success: bool, last_duration: float) -> None:
    """
    Auto-throttle: pace based on last wall-clock duration + consecutive failures.
    sleep = clamp( max(min_sleep, 0.25 * last_duration + 0.4 * consec_fail), max_sleep )
    """
    if success:
        throttle_state["consec_fail"] = 0
    else:
        throttle_state["consec_fail"] += 1

    base = 0.25 * max(0.0, last_duration)
    penalty = 0.4 * throttle_state["consec_fail"]
    sleep_sec = max(throttle_state["sleep_min"], base + penalty)
    sleep_sec = min(sleep_sec, throttle_state["sleep_max"])
    if throttle_state["auto_throttle"] or throttle_state["force_sleep"]:
        time.sleep(sleep_sec)

# ----------------------------
# CLI / Entrypoint
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build JSON summaries (sidecars) for transcripts using local Ollama.")
    ap.add_argument("--roots", nargs="+", help="One or more root directories to scan (recursive)")
    ap.add_argument("--root", help="(deprecated) single root directory; kept for compatibility")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"), help="Ollama model name")
    ap.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"), help="Ollama base URL")
    ap.add_argument("--overwrite", action="store_true", help="Re-generate even if _summary.json exists and is valid")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; do not call model or write files")

    # Auto-throttle & memory knobs
    ap.add_argument("--auto-throttle", action="store_true", help="Adapt pacing from last gen time and failures")
    ap.add_argument("--sleep-min", type=float, default=0.1, help="Minimum sleep between calls")
    ap.add_argument("--sleep-max", type=float, default=2.0, help="Maximum sleep between calls")
    ap.add_argument("--force-sleep", action="store_true", help="Always sleep using the computed schedule (even on skips)")

    ap.add_argument("--ctx", type=int, default=1536, help="Ollama num_ctx (prompt context)")
    ap.add_argument("--predict", type=int, default=512, help="Ollama num_predict (max new tokens)")
    ap.add_argument("--temperature", type=float, default=0.2, help="Ollama temperature")
    ap.add_argument("--prompt-chars", type=int, default=120_000, help="Max transcript chars to send in prompt")

    ap.add_argument("--limit", type=int, default=0, help="Process at most N items (0 = no limit)")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    # Resolve roots
    roots: List[Path] = []
    if args.roots:
        roots = [Path(r).expanduser().resolve() for r in args.roots]
    elif args.root:
        roots = [Path(args.root).expanduser().resolve()]
    else:
        raise SystemExit("Provide --roots <dir...> or --root <dir>")

    roots = [r for r in roots if r.exists()]
    if not roots:
        log.error("No valid roots.")
        return

    client = OllamaClient(args.ollama_host)

    # Memory-friendly defaults (esp. for gemma3:4b)
    ollama_opts = {
        "num_ctx": int(args.ctx),
        "num_predict": int(args.predict),
        "temperature": float(args.temperature),
        # You can also add:
        # "repeat_penalty": 1.1,
        # "top_p": 0.9,
    }

    # Auto-throttle state
    throttle_state = {
        "consec_fail": 0,
        "sleep_min": max(0.0, float(args.sleep_min)),
        "sleep_max": max(float(args.sleep_min), float(args.sleep_max)),
        "auto_throttle": bool(args.auto_throttle),
        "force_sleep": bool(args.force_sleep),
    }

    pairs = _iter_unique_stems(roots)
    if not pairs:
        log.info("No transcripts found under: %s", ", ".join(str(x) for x in roots))
        return

    # Sequential deterministic loop
    total = 0
    ok = 0
    for stem, dp in pairs:
        if args.limit and total >= args.limit:
            break
        tr = _choose_best_transcript(dp, stem)
        if not tr:
            continue
        total += 1
        if args.dry_run:
            # still pace gently when dry-running
            _apply_sleep(throttle_state, success=True, last_duration=0.05)
            continue
        res = summarize_one(
            transcript=tr,
            client=client,
            model=args.model,
            overwrite=args.overwrite,
            options=ollama_opts,
            prompt_chars=int(args.prompt_chars),
            throttle_state=throttle_state,
        )
        if res:
            ok += 1

    log.info("Completed. items=%d ok=%d errors=%d", total, ok, total - ok)

if __name__ == "__main__":
    main()
