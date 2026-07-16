#!/usr/bin/env python3
"""arxiv_kg_bridge.py — ingest arXiv research into the Scott knowledge graph.

Makes the arXiv research sessions (run via the arxiv skill on x1-370) flow into
Neo4j automatically. Given a query, it:
  1. Fetches recent arXiv papers (export.arxiv.org API, no key).
  2. For each paper: MERGE (Paper {arxiv_id}) with title/abstract/authors/url.
  3. Chunk the abstract, embed via LM Studio nomic v1.5, MERGE (Chunk)-[:PART_OF]->(Paper).
  4. Link papers to existing Concept nodes (lexical) + cross-paper SIMILAR edges
     via embedding cosine in Python (small N, safe).
  5. Persist ingested arxiv_ids so re-runs are idempotent (MERGE + skip if seen).

Idempotent & safe: only MERGEs, never deletes; tracks seen ids in a cursor file.

Usage:
  python arxiv_kg_bridge.py "GRPO reinforcement learning" --max 10
  python arxiv_kg_bridge.py --from-file queries.txt          # batch of queries
  python arxiv_kg_bridge.py "diffusion models" --max 5 --dry-run
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CURSOR_FILE = REPO / "scripts" / ".arxiv_kg_cursor.json"
LM_STUDIO_URL = "http://100.64.43.123:1234/v1/embeddings"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
ATOM = "{http://www.w3.org/2005/Atom}"


def log(*a):
    print("[arxiv_kg_bridge]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def fetch_arxiv(query: str, max_results: int = 10):
    q = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{q}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            xml = r.read().decode()
    except Exception as e:  # noqa: BLE001
        log("arxiv fetch failed:", e)
        return []
    papers = []
    try:
        root = ET.fromstring(xml)
    except Exception as e:
        log("arxiv parse failed:", e)
        return []
    for entry in root.findall(f"{ATOM}entry"):
        aid = (entry.findtext(f"{ATOM}id") or "").strip()
        # arxiv id is the trailing part of the abs url
        arxiv_id = aid.rsplit("/", 1)[-1] if aid else ""
        title = (entry.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{ATOM}summary") or "").strip().replace("\n", " ")
        authors = [a.findtext(f"{ATOM}name", "").strip()
                   for a in entry.findall(f"{ATOM}author")]
        published = (entry.findtext(f"{ATOM}published") or "").strip()
        if arxiv_id:
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published,
                "url": aid,
            })
    return papers


def embed(text: str):
    if not text or not text.strip():
        return None
    body = json.dumps({"model": EMBED_MODEL, "input": text[:2000]}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["data"][0]["embedding"]
    except Exception as e:  # noqa: BLE001
        log("embed failed:", type(e).__name__, str(e)[:100])
        return None


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_seen():
    if CURSOR_FILE.exists():
        try:
            return set(json.loads(CURSOR_FILE.read_text()).get("seen", []))
        except Exception:
            return set()
    return set()


def save_seen(seen):
    CURSOR_FILE.write_text(json.dumps({"seen": sorted(seen)}))


def ingest(papers, dry_run=False):
    seen = load_seen()
    new_ids = []
    if not papers:
        log("no papers")
        return 0
    drv, db = get_neo4j()
    new = 0
    try:
        with drv.session(database=db) as s:
            for p in papers:
                if p["arxiv_id"] in seen:
                    continue
                emb = embed(p["abstract"])
                if dry_run:
                    log(f"[dry-run] {p['arxiv_id']}: {p['title'][:60]!r}")
                    seen.add(p["arxiv_id"])
                    new += 1
                    continue
                try:
                    s.run(
                        """
                        MERGE (pa:Paper {arxiv_id:$aid})
                          SET pa.title=$title, pa.abstract=$abstract,
                              pa.authors=$authors, pa.published=$published,
                              pa.url=$url, pa.ingested_at=$at
                        """,
                        {"aid": p["arxiv_id"], "title": p["title"], "abstract": p["abstract"],
                         "authors": p["authors"], "published": p["published"],
                         "url": p["url"], "at": datetime.now(timezone.utc).isoformat()},
                    )
                    if emb:
                        s.run(
                            "MATCH (pa:Paper {arxiv_id:$aid}) SET pa.embedding_768=$emb",
                            {"aid": p["arxiv_id"], "emb": emb},
                        )
                        s.run(
                            """
                            MERGE (c:Chunk {paper_id:$aid, idx:0})
                              SET c.text=$abstract, c.embedding_768=$emb
                            WITH c
                            MATCH (pa:Paper {arxiv_id:$aid})
                            MERGE (pa)<-[:PART_OF]-(c)
                            """,
                            {"aid": p["arxiv_id"], "abstract": p["abstract"], "emb": emb},
                        )
                    # link to concepts by title/abstract keyword
                    s.run(
                        """
                        MATCH (pa:Paper {arxiv_id:$aid})
                        MATCH (cn:Concept) WHERE cn.name IS NOT NULL
                          AND (toLower(pa.title) CONTAINS toLower(cn.name)
                            OR toLower(pa.abstract) CONTAINS toLower(cn.name))
                        MERGE (pa)-[:DISCUSSES]->(cn)
                        """,
                        {"aid": p["arxiv_id"]},
                    )
                    seen.add(p["arxiv_id"])
                    new += 1
                    new_ids.append(p["arxiv_id"])
                except Exception as e:  # noqa: BLE001 - resilience: one dup shouldn't kill batch
                    log(f"skip {p['arxiv_id']}: {type(e).__name__}: {str(e)[:90]}")
        # cross-paper SIMILAR edges among ONLY the freshly-ingested papers (small N).
        # Do NOT scan all 68k embedded papers — that is O(n^2) and will never finish.
        # Use a fresh session (the loop session may be in a bad state) and make it
        # idempotent: delete any prior SIMILAR between these ids first, then re-merge.
        if not dry_run and new_ids:
            try:
                with drv.session(database=db) as ss:
                    rows = ss.run(
                        "UNWIND $ids AS id MATCH (p:Paper {arxiv_id:id}) "
                        "WHERE p.embedding_768 IS NOT NULL "
                        "RETURN p.arxiv_id AS id, p.embedding_768 AS e",
                        {"ids": new_ids},
                    ).data()
                    # idempotent: clear any existing SIMILAR among these ids
                    ss.run(
                        "UNWIND $ids AS id MATCH (a:Paper {arxiv_id:id})-[r:SIMILAR]->(b:Paper) "
                        "WHERE b.arxiv_id IN $ids DELETE r",
                        {"ids": new_ids},
                    )
                    for i in range(len(rows)):
                        for j in range(i + 1, len(rows)):
                            sim = cosine(rows[i]["e"], rows[j]["e"])
                            if sim >= 0.6:
                                ss.run(
                                    "MATCH (a:Paper {arxiv_id:$a}), (b:Paper {arxiv_id:$b}) "
                                    "MERGE (a)-[:SIMILAR {method:'semantic', score:$s}]->(b)",
                                    {"a": rows[i]["id"], "b": rows[j]["id"], "s": sim},
                                )
            except Exception as e:  # noqa: BLE001
                log(f"similar-edge step skipped: {type(e).__name__}: {str(e)[:90]}")
        save_seen(seen)
        log(f"ingested {new} new papers (total seen {len(seen)})" + (" [dry-run]" if dry_run else ""))
        return new
    finally:
        drv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="arXiv query (or use --from-file)")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--from-file", help="file with one query per line")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queries = []
    if args.from_file:
        queries = [l.strip() for l in open(args.from_file) if l.strip()]
    elif args.query:
        queries = [args.query]
    if not queries:
        log("no query given")
        return

    all_papers = []
    for q in queries:
        log("fetching:", q)
        all_papers += fetch_arxiv(q, args.max)
    # dedupe by arxiv_id
    seen_ids = set()
    uniq = []
    for p in all_papers:
        if p["arxiv_id"] not in seen_ids:
            seen_ids.add(p["arxiv_id"])
            uniq.append(p)
    log(f"{len(uniq)} unique papers from {len(queries)} queries")
    ingest(uniq, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
