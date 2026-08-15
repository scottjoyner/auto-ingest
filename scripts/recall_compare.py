#!/usr/bin/env python3
"""
recall_compare.py — A/B recall comparison for embedding models on a sample.

Re-embeds N utterances with each candidate model (same pooling via
auto_ingest.embed, so the comparison isolates model quality, not pooling),
writes each to a throwaway vector property + a tuned-HNSW index, then scores
the models against a fixed query set with overlap@k, mean cosine, and
rank-normalized nDCG (using model[0]'s ranking as the relevance reference).

Designed to decide whether a re-take (e.g. MiniLM-L6 -> gte-small) is worth a
full-graph rebuild on either deathstar CPU or x1-370 ROCm.

Usage:
  ./.venv/bin/python3 scripts/recall_compare.py --models all-MiniLM-L6,gte-small --n 2000 --k 10
  ./.venv/bin/python3 scripts/recay_compare.py --models all-MiniLM-L12,gte-small,e5-large --n 3000 --keep

Models are resolved by a short key against HF hub names:
  mini6 -> sentence-transformers/all-MiniLM-L6-v2
  mini12-> sentence-transformers/all-MiniLM-L12-v2
  gte-small -> thenlper/gte-small
  e5-large -> intfloat/multilingual-e5-large
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import torch

torch.set_num_threads(max(2, (os.cpu_count() or 2) - 2))

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(sys_path))

from auto_ingest_config import get_neo4j_config
from auto_ingest.embed import EmbedModel, HNSW_M, HNSW_EF, HNSW_QUANT
from neo4j import GraphDatabase

log = logging.getLogger("recall_compare")
MODELS = {
    "mini6": "sentence-transformers/all-MiniLM-L6-v2",
    "mini12": "sentence-transformers/all-MiniLM-L12-v2",
    "gte-small": "thenlper/gte-small",
    "e5-large": "intfloat/multilingual-e5-large",
}
DEFAULT_QUERIES = [
    "quarterly fiscal year report numbers revenue",
    "router firmware upgrade wlan config",
    "beach house weekend family photos",
    "docker build container kubernetes deploy",
    "medical emergency call notes transcription",
    "stock market portfolio allocation",
    "song lyrics guitar chords summer",
    "project timeline sprint retrospective",
]
PROP_PREFIX = "cmp"  # test props: cmp_<key>


def parse_models(s: str):
    out = []
    for key in s.split(","):
        key = key.strip()
        name = MODELS.get(key.lower(), key)
        out.append((key.strip().replace("/", "_").replace("-", "_"), name))
    return out


def create_index(sess, prop: str, dim: int, label: str = "Utterance"):
    name = f"{prop}_index"
    sess.run(f"DROP INDEX {name} IF EXISTS")
    sess.run(
        f"CREATE VECTOR INDEX {name} FOR (n:{label}) ON (n.{prop}) "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine', "
        f"`vector.hnsw.m`: {HNSW_M}, `vector.hnsw.ef_construction`: {HNSW_EF}"
        + (", `vector.quantization.enabled`: true" if HNSW_QUANT else "") +
        f" }} }}"
    )
    for _ in range(60):
        rows = sess.run("SHOW VECTOR INDEXES YIELD name, state WHERE name = $n RETURN state",
                        n=name).values()
        if rows and rows[0][0] and str(rows[0][0]) == "ONLINE":
            return
        time.sleep(0.5)


def sample_utterances(sess, n: int):
    # Deterministic-ish sample: high-embedding-weight utterances first (the
    # nodes that matter for retrieval). ORDER BY id(n) LIMIT keeps it stable.
    rows = sess.run(
        "MATCH (n:Utterance) WHERE n.text IS NOT NULL AND n.embedding IS NOT NULL "
        "RETURN id(n) AS nid, n.text AS text LIMIT $n", n=n).data()
    return [(r["nid"], r["text"] or "") for r in rows]


def write_vectors(sess, prop: str, rows):
    upd = [{"nid": nid, "vec": vec} for (nid, _), vec in rows]
    sess.run(f"UNWIND $u AS x MATCH (n:Utterance) WHERE id(n)=x.nid SET n.{prop}=x.vec", u=upd)


def query_index(sess, model: EmbedModel, prop: str, q: str, k: int):
    qv = model.embed([q])[0]
    idx = f"{prop}_index"
    rows = sess.run(
        f"CALL db.index.vector.queryNodes('{idx}', $k, $qvec) YIELD node, score "
        f"RETURN id(node) AS nid, score", k=k, qvec=list(qv)).data()
    return [(r["nid"], float(r["score"])) for r in rows]


def ndcg_at_k(ranked_ids, ref_ids, k):
    """Standard nDCG@k vs a reference ranking. Relevance = 1 if id is in the
    reference's top-k, else 0. Discount uses log2(rank+2) per rank i (0-based)."""
    import math
    ref_set = set(ref_ids[:k])
    dcg = 0.0
    for i, nid in enumerate(ranked_ids[:k]):
        if nid in ref_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ref_ids))))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser(description="A/B recall compare for embedding models")
    ap.add_argument("--models", default="mini6,gte-small", help="comma list of model keys")
    ap.add_argument("--n", type=int, default=2000, help="sample size (utterances)")
    ap.add_argument("--k", type=int, default=10, help="retrieve top-k per query")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--queries", default="", help="comma-sep queries; defaults to built-in set")
    ap.add_argument("--keep", action="store_true", help="keep test props/indexes (else drop)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = parse_models(args.models)
    cfg = get_neo4j_config()
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database"))
    queries = [q.strip() for q in args.queries.split(",")] if args.queries else DEFAULT_QUERIES

    log.info("sample %d utterances", args.n)
    with driver.session() as s:
        rows = sample_utterances(s, args.n)
    log.info("got %d utterances", len(rows))

    # Load each model, embed the sample, write to cmp_<key>, build index.
    per_model: dict = {}  # key -> (model, prop, qvecs dict)
    with driver.session() as s:
        for key, name in model_keys:
            model = EmbedModel(name, device="cpu")
            prop = f"{PROP_PREFIX}_{key}"
            texts = [t for _, t in rows]
            t0 = time.perf_counter()
            vecs = model.embed(texts, batch_size=min(args.batch_size, len(texts)))
            log.info("embedded %d with %s (dim=%d, %.1f/s)", len(vecs), name, model.dim,
                     len(vecs) / (time.perf_counter() - t0))
            write_vectors(s, prop, zip(rows, vecs))
            create_index(s, prop, model.dim)
            per_model[key] = (model, prop, dict(zip([nid for nid, _ in rows], vecs)))
            log.info("wrote %s.%s + index", "Utterance", prop)

    # key -> property used on this run (built once, reused for query + cleanup).
    key2prop = {key: f"{PROP_PREFIX}_{key}" for key, _ in model_keys}

    # Query each model for the query set.
    results: dict = {}
    with driver.session() as s:
        for key, (model, prop, _) in per_model.items():
            results[key] = {q: query_index(s, model, prop, q, args.k) for q in queries}

    print("\n===== A/B RECALL (k=%d, n=%d) =====" % (args.k, args.n))
    print("models: " + ", ".join(f"{key}={name} (dim={per_model[key][0].dim})" for key, name in model_keys))
    print("%-40s %9s %9s  overlap@5_vs_%s  ndcg5_vs_%s" % ("query", model_keys[0][0], model_keys[1][0] if len(model_keys) > 1 else "-", model_keys[0][0], model_keys[0][0]))
    print("-" * 100)
    ref_key = model_keys[0][0]
    for q in queries:
        ref = results[ref_key][q]
        ref_ids = [nid for nid, _ in ref]
        row = "%-40.40s" % q
        for key, _ in model_keys:
            top1 = round(results[key][q][0][1], 4) if results[key][q] else 0.0
            row += " %9.4f" % top1
        # overlap@5 + nDCG@5 of each non-ref model vs the reference model[0]
        for key, _ in model_keys[1:]:
            cur_ids = [nid for nid, _ in results[key][q][:5]]
            ov = len(set(cur_ids) & set(ref_ids[:5])) / 5.0
            ndcg = ndcg_at_k(cur_ids, ref_ids, 5)
            row += "  o5=%.2f ndcg5=%.3f" % (ov, ndcg)
        print(row)

    # cleanup
    if not args.keep:
        with driver.session() as s:
            for key, _ in model_keys:
                prop = key2prop[key]
                s.run(f"DROP INDEX {prop}_index IF EXISTS")
                s.run(f"MATCH (n:Utterance) WHERE n.{prop} IS NOT NULL REMOVE n.{prop}")
        log.info("cleaned up test props/indexes")
    driver.close()


if __name__ == "__main__":
    main()
