#!/usr/bin/env python3
"""
Step 5b of neo4j-summary-geo-clustering: SEMANTIC bridge from Summaries to the research
Concept graph, using vector similarity.

WHY A SEPARATE EMBEDDING SPACE:
- Existing `Summary.embedding` is 384-dim (made by an older nomic v1 model, no longer loaded).
- The only embed model currently in LM Studio is `text-embedding-nomic-embed-text-v1.5`
  which outputs 768-dim. The two spaces are NOT comparable (different dims + different model).
- Solution: re-embed BOTH summaries and concepts into a NEW `embedding_768` property (the
  existing 384-dim `embedding` is left untouched — other systems may depend on it). Then
  cosine-match in the unified 768-dim space.

RESULT: `(s)-[:RELATED_CONCEPT {method:'semantic', similarity}]->(c)` for the top-K concepts
per summary above a similarity threshold. Lexical DISCUSSES edges (Step 5) are NOT duplicated
here — semantic edges are kept distinct so you can tell how a link was derived.

Idempotent / resumable: skips summaries/concepts that already have `embedding_768`.
"""
import neo4j
import requests
import numpy as np

# Use the repo's standard Neo4j config resolver (env -> config -> baked-in).
import sys as _sys
from pathlib import Path as _P
_SYS_REPO = str(_P(__file__).resolve().parent.parent)
if _SYS_REPO not in _sys.path:
    _sys.path.insert(0, _SYS_REPO)
from auto_ingest_config import get_neo4j_config as _get_cfg
EMBED_URL = "http://localhost:1234/v1/embeddings"
MODEL = "text-embedding-nomic-embed-text-v1.5"

SIM_THRESHOLD = 0.25   # minimum cosine to create a semantic edge
TOP_K = 5              # max concepts per summary

def embed(texts):
    """Batch embed via LM Studio OpenAI-compatible endpoint. Returns list[np.array]."""
    r = requests.post(EMBED_URL, json={"model": MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return [np.array(d["embedding"], dtype=np.float32) for d in r.json()["data"]]

def norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v

def main():
    _cfg = _get_cfg()
    uri, user, pw = _cfg["uri"], _cfg["user"], _cfg["password"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    print("connected", flush=True)

    # 1. Concepts -> embedding_768
    with driver.session() as s:
        concepts = [(r["name"],) for r in s.run("MATCH (c:Concept) WHERE c.embedding_768 IS NULL RETURN c.name AS name")]
        names = [c[0] for c in concepts]
    if names:
        print(f"embedding {len(names)} concepts...", flush=True)
        B = 100
        with driver.session() as ws:
            for i in range(0, len(names), B):
                chunk = names[i:i+B]
                vecs = embed(chunk)
                with ws.begin_transaction() as tx:
                    for nm, v in zip(chunk, vecs):
                        tx.run("MATCH (c:Concept {name: $n}) SET c.embedding_768 = $v",
                               n=nm, v=v.tolist())
                    tx.commit()
        print(f"  embedded {len(names)} concepts", flush=True)
    else:
        print("concepts already embedded", flush=True)

    # 2. Load all concept vectors (now complete)
    C = {}
    with driver.session() as s:
        for r in s.run("MATCH (c:Concept) WHERE c.embedding_768 IS NOT NULL RETURN c.name AS name, c.embedding_768 AS v"):
            C[r["name"]] = norm(np.array(r["v"], dtype=np.float32))
    cnames = list(C.keys())
    cmat = np.stack([C[n] for n in cnames])  # (n_concepts, 768)
    print(f"loaded {len(cnames)} concept vectors", flush=True)

    # 3. Summaries -> embedding_768 (resumable), then match
    q = """
    MATCH (s:Summary) WHERE s.lat IS NOT NULL AND size(s.text) > 0 AND s.embedding_768 IS NULL
    RETURN s.id AS id, s.text AS text, s.bullets AS bullets
    """
    total_edges = 0
    batch = []  # (sid, vec)

    def flush_summary_embed(batch):
        if not batch: return
        with driver.session() as ws:
            with ws.begin_transaction() as tx:
                for (sid, v) in batch:
                    tx.run("MATCH (s:Summary {id: $id}) SET s.embedding_768 = $v",
                           id=sid, v=v.tolist())
                tx.commit()

    def match_and_link(ids, vecs):
        nonlocal total_edges
        # ids/vecs: this batch of summaries (already embedded). Match each to concepts.
        smat = np.stack([norm(v) for v in vecs])  # (b, 768)
        sims = smat @ cmat.T  # (b, n_concepts)
        edges = []
        for i, sid in enumerate(ids):
            row = sims[i]
            top_idx = np.argsort(row)[::-1][:TOP_K]
            for j in top_idx:
                if row[j] >= SIM_THRESHOLD:
                    edges.append((sid, cnames[j], float(row[j])))
        # write edges (distinct from lexical DISCUSSES)
        with driver.session() as ws:
            with ws.begin_transaction() as tx:
                for (sid, cname, sim) in edges:
                    tx.run(
                        "MATCH (s:Summary {id: $sid}) MATCH (c:Concept {name: $cname}) "
                        "MERGE (s)-[r:RELATED_CONCEPT {method:'semantic'}]->(c) "
                        "SET r.similarity = $sim",
                        sid=sid, cname=cname, sim=sim)
                tx.commit()
        total_edges += len(edges)

    B = 64
    with driver.session() as s:
        rows = list(s.run(q))
    print(f"summaries to embed: {len(rows)}", flush=True)
    ids, vecs, texts = [], [], []
    for k, rec in enumerate(rows):
        # build text
        t = rec["text"]; b = rec["bullets"]
        parts = []
        if isinstance(t, str): parts.append(t)
        elif isinstance(t, list): parts += [x for x in t if isinstance(x, str)]
        if isinstance(b, str): parts.append(b)
        elif isinstance(b, list): parts += [x for x in b if isinstance(x, str)]
        txt = " ".join(parts)[:2000]
        if not txt.strip():
            continue
        texts.append(txt); ids.append(rec["id"])
        if len(texts) >= B:
            vecs = embed(texts)
            flush_summary_embed(list(zip(ids, vecs)))
            match_and_link(ids, vecs)
            ids, vecs, texts = [], [], []
            if (k // B) % 50 == 0:
                print(f"  processed {k}/{len(rows)} summaries, {total_edges} semantic edges so far", flush=True)
    if texts:
        vecs = embed(texts)
        flush_summary_embed(list(zip(ids, vecs)))
        match_and_link(ids, vecs)

    print(f"DONE. semantic RELATED_CONCEPT edges created/merged: {total_edges}", flush=True)
    driver.close()

if __name__ == "__main__":
    main()
