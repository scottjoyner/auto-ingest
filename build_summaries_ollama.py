#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSON summaries (sidecars) for existing audio transcription files using Ollama (llama3:latest by default).

- Scans one or more roots for transcripts (prefers *_medium_transcription.txt then *_transcription.txt then *_transcription.csv)
- For each found transcript, asks the LLM for a STRICT-JSON summary payload and writes <stem>_summary.json alongside the transcript
- Never touches Neo4j; this is offline prep for later ingestion
- Safe to re-run: skips items that already have a valid _summary.json unless --overwrite

Tested on macOS (Apple Silicon) with Ollama running locally.

Usage examples:
  python build_summaries_ollama.py --roots /Volumes/Untitled/audio --model llama3:latest --max-concurrent 2
  OLLAMA_HOST=http://127.0.0.1:11434 python build_summaries_ollama.py --roots /mnt/8TB_2025/fileserver/audio --dry-run

Environment:
  OLLAMA_HOST   Default http://127.0.0.1:11434
  OLLAMA_MODEL  Default "llama3:latest" (can be overridden by --model)
"""

import os, re, sys, csv, json, time, math, logging, argparse, pathlib, concurrent.futures
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import http.client
from urllib.parse import urlparse

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summaries")

# ----------------------------------
# CLI
# ----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build JSON summaries from transcripts using Ollama.")
    p.add_argument("--roots", nargs="+", required=True, help="Root directory(ies) to scan (recursive)")
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3:latest"), help="Ollama model name")
    p.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"), help="Ollama base URL")
    p.add_argument("--max-concurrent", type=int, default=max(1, os.cpu_count() or 1)//2, help="Parallel workers")
    p.add_argument("--overwrite", action="store_true", help="Recreate existing _summary.json files")
    p.add_argument("--dry-run", action="store_true", help="Print what would happen, do not call the model or write files")
    p.add_argument("--limit", type=int, default=0, help="Process at most N items (0 = no limit)")
    return p.parse_args()

# ----------------------------------
# File helpers
# ----------------------------------

TRANSCRIPT_TXT_PATTERNS = [
    "{stem}_medium_transcription.txt",
    "{stem}_transcription.txt",
]
TRANSCRIPT_CSV_PATTERNS = [
    "{stem}_transcription.csv",
]

STEM_RE = re.compile(r"^(\d{4}_\d{4}_\d{6}|\d{12,})(?:_[0-9]{6})?", re.ASCII)

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
            # Heuristics: prefer 'text' column; else join all stringy columns
            if rdr.fieldnames is None:
                return None
            prefer = None
            for name in rdr.fieldnames:
                if name.lower() in ("text", "utterance", "transcript", "content"):
                    prefer = name
                    break
            for row in rdr:
                if prefer and row.get(prefer):
                    rows.append(str(row[prefer]))
                else:
                    # Fallback: join non-empty values
                    vals = [str(v) for k,v in row.items() if isinstance(v, str) and v.strip()]
                    if vals:
                        rows.append(" ".join(vals))
        text = "\n".join(rows).strip()
        return text or None
    except Exception as e:
        log.warning("Failed reading CSV %s: %s", p, e)
        return None


def find_best_transcript_for_stem(dirpath: Path, stem: str) -> Optional[Transcript]:
    # Try text patterns first
    for pat in TRANSCRIPT_TXT_PATTERNS:
        cand = dirpath / pat.format(stem=stem)
        if cand.exists():
            txt = _read_text_file(cand)
            if txt:
                return Transcript(stem=stem, text=txt, source_path=cand)
    # Then CSV patterns
    for pat in TRANSCRIPT_CSV_PATTERNS:
        cand = dirpath / pat.format(stem=stem)
        if cand.exists():
            txt = _read_csv_concat_text(cand)
            if txt:
                return Transcript(stem=stem, text=txt, source_path=cand)
    return None


def iter_candidate_stems(root: Path) -> List[Tuple[str, Path]]:
    """Return list of (stem, dirpath) discovered under root.
    Strategy: look for known transcript filename patterns and extract stems.
    """
    out: List[Tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        for name in filenames:
            if name.endswith("_transcription.txt") or name.endswith("_transcription.csv"):
                stem = _detect_stem(Path(name))
                if stem:
                    out.append((stem, dp))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for stem, dp in out:
        key = (stem, str(dp))
        if key not in seen:
            seen.add(key)
            uniq.append((stem, dp))
    return uniq

# ----------------------------------
# Ollama client (no third-party deps)
# ----------------------------------

class OllamaClient:
    def __init__(self, base_url: str):
        u = urlparse(base_url)
        if not u.scheme.startswith("http"):
            raise ValueError("OLLAMA_HOST must be http(s) URL")
        self._is_https = (u.scheme == "https")
        self._host = u.hostname or "127.0.0.1"
        self._port = u.port or (443 if self._is_https else 11434)
        self._base_path = u.path.rstrip("/")

    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None, retries: int = 3, timeout: float = 120.0) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
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
                log.warning("Ollama call failed (attempt %d/%d): %s", attempt, retries, e)
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Ollama call failed after retries")

# ----------------------------------
# Prompt
# ----------------------------------

SUMMARY_SCHEMA_HINT = {
    "version": "1.0",
    "language": "<ISO 639-1 two-letter, e.g., en>",
    "summary": "<5-9 sentence natural-language summary of the whole transcript>",
    "key_points": ["<bullet 1>", "<bullet 2>", "..."] ,
    "topics": ["<topic-1>", "<topic-2>", "..."],
    "people": ["<full name if present>", "..."],
    "organizations": ["<org>", "..."],
    "places": ["<place>", "..."],
    "quality_notes": "<any issues noticed in transcript: mishears, noise, music, language mix, etc.>",
}

PROMPT_TEMPLATE = (
    "You are a precise summarization model. Given a transcript, output STRICT JSON only, no prose, matching this schema: "
    + json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False)
    + "\nRules:\n"
      "- Output ONLY JSON, no markdown, no preface.\n"
      "- Keep arrays deduplicated.\n"
      "- If a field is unknown, use an empty array or empty string accordingly.\n"
      "- Language should reflect the transcript language.\n"
      "\nTranscript begins:\n\n{transcript}\n\nEND."
)

# ----------------------------------
# Main worker
# ----------------------------------

@dataclass
class Task:
    stem: str
    dirpath: Path
    transcript: Transcript
    out_path: Path


def make_task(dirpath: Path, stem: str, overwrite: bool) -> Optional[Task]:
    tr = find_best_transcript_for_stem(dirpath, stem)
    if not tr:
        return None
    out_path = dirpath / f"{stem}_summary.json"
    if out_path.exists() and not overwrite:
        # quick validity check (is it JSON?)
        try:
            json.loads(out_path.read_text(encoding="utf-8"))
            return None  # skip
        except Exception:
            pass  # will rewrite
    return Task(stem=stem, dirpath=dirpath, transcript=tr, out_path=out_path)


def process_task(task: Task, client: OllamaClient, model: str, dry_run: bool) -> Tuple[str, str]:
    stem = task.stem
    src = task.transcript.source_path
    dst = task.out_path
    # Trim very long transcripts to keep prompt under control, but still useful
    transcript_text = task.transcript.text.strip()
    if len(transcript_text) > 120_000:
        log.debug("%s transcript too long (%d chars), truncating head+tail", stem, len(transcript_text))
        head = transcript_text[:80_000]
        tail = transcript_text[-20_000:]
        transcript_text = head + "\n...\n" + tail

    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)

    if dry_run:
        log.info("DRY-RUN would summarize: %s -> %s", src, dst)
        return (stem, "DRY-RUN")

    raw = client.generate(model=model, prompt=prompt, options={"temperature": 0.2})

    # Some models occasionally wrap JSON in code fences; try to strip safely
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n ")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip("\n")

    try:
        payload = json.loads(cleaned)
    except Exception:
        # As a fallback, try to extract the first {...} block
        m = re.search(r"\{.*\}\s*$", cleaned, re.DOTALL)
        if not m:
            raise
        payload = json.loads(m.group(0))

    # Add provenance
    payload["_meta"] = {
        "source_transcript": str(src),
        "generated_by_model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "summary.sidecar.v1",
    }

    tmp = Path(str(dst) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dst)
    return (stem, str(dst))

# ----------------------------------
# Entrypoint
# ----------------------------------

def main() -> None:
    args = parse_args()
    client = OllamaClient(args.ollama_host)
    model = args.model

    # Discover work
    pairs: List[Tuple[str, Path]] = []
    for r in args.roots:
        root = Path(r).expanduser().resolve()
        if not root.exists():
            log.warning("Root does not exist: %s", root)
            continue
        pairs.extend(iter_candidate_stems(root))

    if not pairs:
        log.info("No transcripts found under roots: %s", ", ".join(args.roots))
        return

    # Build tasks
    tasks: List[Task] = []
    for stem, dp in pairs:
        t = make_task(dp, stem, args.overwrite)
        if t:
            tasks.append(t)

    if args.limit and len(tasks) > args.limit:
        tasks = tasks[:args.limit]

    if not tasks:
        log.info("Nothing to do (all summaries exist or no transcripts found).")
        return

    log.info("Work items: %d (model=%s, workers=%d, overwrite=%s)", len(tasks), model, args.max_concurrent, args.overwrite)

    # Parallel execution
    ok = 0
    errs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_concurrent)) as ex:
        futs = [ex.submit(process_task, t, client, model, args.dry_run) for t in tasks]
        for f in concurrent.futures.as_completed(futs):
            try:
                stem, where = f.result()
                ok += 1
                log.info("Done %-20s -> %s", stem, where)
            except Exception as e:
                errs += 1
                log.error("Failed on a task: %s", e)

    log.info("Completed. ok=%d, errors=%d", ok, errs)


if __name__ == "__main__":
    main()
