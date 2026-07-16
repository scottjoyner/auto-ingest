#!/usr/bin/env python3
"""kg_nl_query.py — natural-language questions against the Scott knowledge graph.

Takes a question, asks LM Studio (whichever chat model is loaded) to produce a
read-only Cypher query against the graph, runs it, and returns a plain-language
answer. This is the conversational upgrade to the idle query system: instead of
the fixed curious_agent digest, you can ask "what papers did I collect about
speaker diarization this month?" and get a real answer.

Safety: the generated Cypher is forced read-only (no WRITE/MERGE/DELETE/CREATE
keywords allowed). If the model produces a write, it is rejected.

Model selection: tries DEFAULT_MODELS in order, using whichever is currently
loaded in LM Studio (handles the fact that only some models are in VRAM at once).

Usage:
    .venv/bin/python scripts/kg_nl_query.py "what are my top research themes at home?"
    echo "papers about diffusion models" | .venv/bin/python scripts/kg_nl_query.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LM_STUDIO_URL = "http://100.64.43.123:1234/v1/chat/completions"
# models that can generate Cypher, in preference order. Only ONE needs to be loaded.
# Override the whole list with KG_NL_MODELS="m1,m2" env.
DEFAULT_MODELS = os.environ.get(
    "KG_NL_MODELS",
    "north-mini-code-1.0,bonsai-27b,refinedtoolcallv5-3b,ornith-1.0-35b,"
    "qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored",
).split(",")

SCHEMA_HINT = """Graph: Neo4j, db=neo4j. Key labels & relationships (USE EXACT PATTERNS):
- Paper {arxiv_id,title,abstract,embedding_768}  -[:DISCUSSES {method,score}]-> Concept {name,embedding_768}
- Paper -[:SIMILAR {method:'vector',score}]-> Paper   (paper-to-paper similarity, 768 space)
- Concept {name,embedding_768}
- Summary {text} -[:AT_PLACE]-> SummaryPlace {place_role:'home'|'work'}
- Summary -[:RELATED_CONCEPT {method:'semantic'}]-> Concept   (NOTE: RELATED_CONCEPT is on Summary, not SummaryPlace)
- Scott:Person -[:SENT]-> SignalMessage {body,embedding_768} -[:ABOUT {method,score}]-> Concept
- SummaryLocationCluster {place_name,summary_count}
All timestamps: ingested_at (datetime). Return ONLY a single Cypher query, no prose, no markdown."""


def log(*a):
    print("[kg_nl_query]", *a, file=sys.stderr, flush=True)


def chat(prompt: str, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SCHEMA_HINT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(
        LM_STUDIO_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def chat_with_fallback(prompt: str):
    """Try candidate models in order; return (text, model_used). Raises on total failure."""
    errs = []
    for model in DEFAULT_MODELS:
        try:
            return chat(prompt, model), model
        except urllib.error.HTTPError as e:
            errs.append(f"{model}: {e.code}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{model}: {type(e).__name__}")
    raise RuntimeError("all models failed: " + " | ".join(errs))


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def is_readonly(cypher: str) -> bool:
    up = cypher.upper()
    for banned in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP", "WITH WRITE"]:
        if banned in up:
            return False
    return True


def extract_cypher(text: str) -> str:
    t = text.strip()
    if "```" in t:
        import re
        m = re.search(r"```(?:cypher)?\s*(.*?)```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    return t.split(";")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="natural-language question")
    args = ap.parse_args()
    q = args.question or sys.stdin.read().strip()
    if not q:
        log("no question provided")
        return

    drv, db = get_neo4j()
    try:
        log(f"asking LM Studio: {q[:60]!r}")
        try:
            raw, model = chat_with_fallback(q)
        except RuntimeError as e:
            print(f"⚠️ LM Studio unavailable: {e}")
            return
        log(f"model={model}")
        cypher = extract_cypher(raw)
        log(f"generated: {cypher[:120]}")
        if not cypher:
            # small models sometimes emit a thinking block with no query; one retry
            log("empty query, retrying once")
            try:
                raw, _ = chat_with_fallback(q)
                cypher = extract_cypher(raw)
            except RuntimeError:
                pass
        if not cypher:
            print("⚠️ No Cypher generated (model returned empty after retry). Raw response:")
            print(raw)
            return
        if not is_readonly(cypher):
            print("⚠️ Rejected: generated Cypher was not read-only.")
            print("RAW:", raw)
            return
        try:
            with drv.session(database=db) as s:
                rows = s.run(cypher).data()
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Cypher execution failed: {type(e).__name__}")
            print("GENERATED CYPHER:", cypher)
            return
        if not rows:
            print("No results.")
        elif len(rows) == 1 and len(rows[0]) == 1:
            print(list(rows[0].values())[0])
        else:
            for r in rows[:20]:
                print(r)
    finally:
        drv.close()


if __name__ == "__main__":
    main()
