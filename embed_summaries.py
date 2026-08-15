#!/usr/bin/env python3
"""Embed Summary nodes missing a vector, using the canonical embedding path.

Delegates to auto_ingest.embed (single source of pooling truth) so Summary
vectors are byte-comparable with Segment/Transcription/Utterance vectors from
the ingest path and reembed.py. Writes are batched with UNWIND instead of one
Cypher round-trip per summary.

Usage:
  embed_summaries.py [--prop emb_gte_small] [--model thenlper/gte-small]
                     [--batch 256] [--dry-run]
"""
import os, sys, time

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_ingest_config import get_neo4j_config
from auto_ingest.embed import load_embed_model, DEFAULT_MODEL, HNSW_M, HNSW_EF, HNSW_QUANT

PROP = os.getenv("SUMMARY_EMBED_PROP", "embedding")


def ensure_index(sess, prop: str, dim: int):
    name = f"summary_{prop}_index"
    # Index hygiene: keep the index when it already matches (label/prop/dim).
    existing = sess.run(
        "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, options "
        "WHERE name = $n RETURN labelsOrTypes, properties, options",
        n=name,
    ).single()
    if existing is not None:
        labels = list(existing[0]) if existing[0] else []
        props = list(existing[1]) if existing[1] else []
        dim_existing = int(existing[2].get("indexConfig", {}).get("vector.dimensions", -1))
        if props == [prop] and labels == ["Summary"] and dim_existing == dim:
            print(f"  index {name} already exists (dim={dim}); keeping it")
            return
        sess.run(f"DROP INDEX {name} IF EXISTS")
    sess.run(
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:Summary) ON (n.{prop}) "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine', "
        f"`vector.hnsw.m`: {HNSW_M}, `vector.hnsw.ef_construction`: {HNSW_EF}"
        + (", `vector.quantization.enabled`: true" if HNSW_QUANT else "") +
        f" }} }}"
    )
    while True:
        rows = sess.run(
            "SHOW VECTOR INDEXES YIELD name, state WHERE name = $n RETURN state", n=name
        ).values()
        if rows and str(rows[0][0]) == "ONLINE":
            return
        time.sleep(1.0)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prop", default=PROP)
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "256")))
    ap.add_argument("--engine", choices=["torch", "onnx"], default=os.getenv("EMBED_ENGINE", "torch"),
                    help="Inference engine. onnx runs the same weights via onnxruntime CPU "
                         "(fast; ONNX_QUANTIZE=1 for ~3-5x). Keep ONE engine per prop.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = get_neo4j_config()
    driver = GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database")
    )

    with driver.session() as s:
        total = s.run(
            f"MATCH (s:Summary) WHERE size(s.text)>10 RETURN count(s)"
        ).single().values()[0]
        done = s.run(
            f"MATCH (s:Summary) WHERE size(s.text)>10 AND s.{args.prop} IS NOT NULL "
            f"RETURN count(s)"
        ).single().values()[0]
        need = total - done
        print(f"{need}/{total} Summary nodes missing {args.prop} vector")
        if args.dry_run or need == 0:
            return

        ensure_index(s, args.prop, load_embed_model(args.model, engine=args.engine).dim)
        model = load_embed_model(args.model, engine=args.engine)

        rows = s.run(
            f"MATCH (s:Summary) WHERE size(s.text)>10 AND s.{args.prop} IS NULL "
            f"RETURN s.id AS sid, s.text AS text"
        )
        pending = [(r["sid"], str(r["text"] or "").strip()) for r in rows]

    done_n = 0
    t0 = time.perf_counter()
    for i in range(0, len(pending), args.batch):
        chunk = pending[i:i + args.batch]
        texts = [t for _, t in chunk]
        vecs = model.embed(texts, batch_size=min(args.batch, len(texts)))
        updates = [{"sid": sid, "vec": vec} for (sid, _), vec in zip(chunk, vecs)]
        with driver.session() as s:
            s.run(
                f"UNWIND $u AS x MATCH (s:Summary {{id: x.sid}}) "
                f"SET s.{args.prop} = x.vec",
                u=updates,
            )
        done_n += len(updates)
        if done_n % 500 < args.batch or done_n >= len(pending):
            rate = done_n / (time.perf_counter() - t0)
            print(f"  {done_n}/{len(pending)} (rate={rate:.0f}/s)")
    driver.close()
    print("Done!")


if __name__ == "__main__":
    main()