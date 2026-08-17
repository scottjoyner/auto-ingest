#!/usr/bin/env python3
"""
rag_ask.py — retrieve relevant passages across one or both KGs, then answer via LLM.

The natural capstone over vector_search + serve_search: semantic retrieval feeds an
LLM (LM Studio OpenAI-compatible API) that answers the question and cites passage ids.

Usage:
  rag_ask.py --query "what model achieved the best loss?" --target chunk --topk 8
  rag_ask.py --query "..." --target chunk --kg2-uri bolt://127.0.0.1:17687   # both KGs
"""
from __future__ import annotations
import os
import sys
import re
import json
import argparse
import logging
import urllib.request
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vector_search as vs
from vector_search import search_text

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_llm(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def llm_answer(query: str, passages: list, lm_url: str, model: str, timeout: int = 180) -> str:
    ctx = "\n\n".join(
        f"[{i + 1}] (id={p.get('nid')}) {(p.get('text_snip') or p.get('text') or '')}"
        for i, p in enumerate(passages)
    )
    prompt = (
        "Answer the question using ONLY the passages below. Cite passage numbers "
        "like [1], [2] when you use them. If the answer is not in the passages, "
        "say you don't know.\n\n"
        f"Question: {query}\n\nPassages:\n{ctx}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        lm_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return clean_llm(data["choices"][0]["message"]["content"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--target", default="chunk")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--kg1-uri", default=None)
    ap.add_argument("--kg2-uri", default=None)
    ap.add_argument("--lm-url", default=os.getenv("LM_STUDIO_URL", "http://192.168.1.128:1234/v1/chat/completions"))
    ap.add_argument("--model", default=os.getenv("LM_MODEL", "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled"))
    args = ap.parse_args()

    vs.NEO4J_URI = args.kg1_uri or vs.NEO4J_URI
    d1 = GraphDatabase.driver(vs.NEO4J_URI, auth=(vs.NEO4J_USER, vs.NEO4J_PASSWORD))
    rows = search_text(d1, args.query, args.target, args.topk, False, 400)
    d1.close()
    if args.kg2_uri:
        d2 = GraphDatabase.driver(args.kg2_uri, auth=(vs.NEO4J_USER, vs.NEO4J_PASSWORD))
        r2 = search_text(d2, args.query, args.target, args.topk, False, 400)
        # merge by score, keep topk, tag source
        for r in rows:
            r["kg"] = "kg1"
        for r in r2:
            r["kg"] = "kg2"
        rows = sorted(rows + r2, key=lambda x: x.get("score", 0.0), reverse=True)[:args.topk]
        d2.close()

    answer = llm_answer(args.query, rows, args.lm_url, args.model)
    citations = [r.get("nid") for r in rows]
    print("ANSWER:", answer)
    print("CITATIONS:", citations)
    logging.info("rag_ask done; %d passages", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
