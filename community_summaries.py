#!/usr/bin/env python3
"""
community_summaries.py — turn embedding clusters into human-readable summaries.

Reads :Community nodes produced by graph_communities.py, samples member text for
the largest clusters, asks an LLM (LM Studio OpenAI-compatible API) to summarize
each, and writes the result back to Community.summary. Turns raw vector clusters
into navigable topics.

Usage:
  community_summaries.py --uri bolt://100.64.43.123:7687 --label Chunk \
      --lm-url http://192.168.1.128:1234/v1/chat/completions --model llama-3.1 ...
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

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_llm(text: str) -> str:
    """Reasoning models wrap output in <think>...</think>; keep the final answer."""
    if not text:
        return ""
    return THINK_RE.sub("", text).strip()

DEFAULT_TEXT_PROP = {
    "Chunk": "text", "Summary": "text", "Segment": "text", "Utterance": "text",
    "Transcription": "text", "Entity": "text", "Concept": "name", "Topic": "name",
    "Keyword": "name", "KgNode": "title", "Note": "text",
    "Speaker": "label", "GlobalSpeaker": "display_label",
}


def llm_summary(texts, lm_url: str, model: str, timeout: int = 180) -> str:
    prompt = (
        "Summarize the following related passages as a single concise topic "
        "description (2-3 sentences). Capture the shared subject, not a list:\n\n"
        + "\n---\n".join(t for t in texts if t)
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        lm_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--label", default="Chunk")
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--topk", type=int, default=20, help="number of largest communities to summarize")
    ap.add_argument("--samples", type=int, default=8, help="member texts sampled per community")
    ap.add_argument("--lm-url", default=os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions"))
    ap.add_argument("--model", default=os.getenv("LM_MODEL", "local-model"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text_prop = DEFAULT_TEXT_PROP.get(args.label, "text")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    done = 0
    with driver.session() as s:
        comms = s.run(
            "MATCH (c:Community {label:$label, prop:$prop}) "
            "RETURN c.id AS cid, c.size AS size ORDER BY c.size DESC LIMIT $k",
            label=args.label, prop=args.prop, k=args.topk,
        ).data()
        logging.info("summarizing %d communities (label=%s)", len(comms), args.label)
        for c in comms:
            cid = c["cid"]
            rows = s.run(
                f"MATCH (n:{args.label}) WHERE n.community_id = $cid "
                f"RETURN coalesce(n.{text_prop}, '') AS t LIMIT $lim",
                cid=cid, lim=args.samples,
            ).data()
            texts = [r["t"] for r in rows if r["t"]]
            if not texts:
                continue
            if args.dry_run:
                logging.info("[dry] community %s size=%s (%d samples)", cid, c["size"], len(texts))
                continue
            try:
                summary = clean_llm(llm_summary(texts, args.lm_url, args.model))
            except Exception as e:
                logging.warning("community %s LLM failed: %s", cid, e)
                continue
            s.run(
                "MATCH (c:Community {id:$cid, label:$label, prop:$prop}) "
                "SET c.summary = $summary",
                cid=cid, label=args.label, prop=args.prop, summary=summary,
            )
            done += 1
            logging.info("community %s -> %s", cid, summary[:80])
    driver.close()
    logging.info("DONE summarized %d communities", done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
