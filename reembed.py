#!/usr/bin/env python3
"""
reembed.py — re-embed selected node types with an arbitrary model+property.

Designed for the embeddings "re-take" / A:B testing phase: keep several
embedding models side by side on LIVE data and compare recall, rather than
blowing away the canonical `embedding` column.

Usage
-----
  reembed.py Segment      --model sentence-transformers/all-MiniLM-L12-v2 --prop emb_mini12
  reembed.py Transcription --model intfloat/multilingual-e5-large --prop emb_e5_large
  reembed.py Utterance Segment --model thenlper/gte-small --prop emb_gte_small --batch-size 64

Each run:
  * loads the model once (via auto_ingest.embed.EmbedModel, shared pooling math),
  * pages nodes in PK order that have a text field but are missing the target
    property (so it is resumable via --resume),
  * writes vectors and creates/ensures a vector index named
    `<prop>_index` on `<Label>.<prop>` with the model's hidden_size dimension,
  * commits every batch so a Ctrl-C leaves the graph consistent.

Deathstar runs this on CPU (no ROCm on the RX 480). On x1-370 (ROCm, once back)
the same CLI runs GPU-accelerated unchanged.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

from auto_ingest_config import get_neo4j_config
from auto_ingest.embed import EmbedModel, DEFAULT_MODEL, HNSW_M, HNSW_EF, HNSW_QUANT
from auto_ingest.backend import gpu_target_machine, has_rocm

log = logging.getLogger("reembed")

# Node type -> text property we read from.
# Paging uses the internal id(n) (stable integer within a run, monotonic under
# no-deletes) as the cursor; a sidecar .cursor file records the last id so the
# run is resumable. This keeps a single property name out of schema specifics
# (Segment/Utterance use `idx`, Transcription uses `key`) and avoids OFFSET
# which gets O(n) over millions of nodes.
SCHEMA: Dict[str, str] = {
    "Segment": "text",
    "Transcription": "text",
    "Utterance": "text",
}


def ensure_vector_index(sess, label: str, prop: str, dim: int):
    idx_name = f"{prop}_index"
    sess.run(f"DROP INDEX {idx_name} IF EXISTS")
    # Neo4j 5.x vector syntax with tuned HNSW params (recall > defaults).
    sess.run(
        f"CREATE VECTOR INDEX {idx_name} FOR (n:{label}) ON (n.{prop}) "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine', "
        f"`vector.hnsw.m`: {HNSW_M}, `vector.hnsw.ef_construction`: {HNSW_EF}"
        + (", `vector.quantization.enabled`: true" if HNSW_QUANT else "") +
        f" }} }}"
    )
    log.info("  index %s ON %s.%s dim=%d hnsw(m=%d,ef=%d,q=%s)",
             idx_name, label, prop, dim, HNSW_M, HNSW_EF, HNSW_QUANT)
    log.info("waiting for %s to become ONLINE ...", idx_name)
    while True:
        rows = sess.run(
            "SHOW VECTOR INDEXES YIELD name, state WHERE name = $n RETURN state",
            n=idx_name,
        ).values()
        # .values() yields rows as tuples; state is the first column.
        state = rows[0][0] if rows else None
        if str(state) == "ONLINE":
            break
        time.sleep(1.0)


def count_needing(sess, label: str, text_prop: str, prop: str) -> Tuple[int, int]:
    total = sess.run(
        f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL RETURN count(n)"
    ).single().values()[0]
    done = sess.run(
        f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL AND n.{prop} IS NOT NULL RETURN count(n)"
    ).single().values()[0]
    return total - done, total


def _cursor_path(label: str, prop: str) -> str:
    return f".reembed_{label}_{prop}.cursor"


def page_rows(sess, label: str, prop: str, start_id: int, limit: int):
    """Resumable page: nodes still missing `prop`, ordered by internal id(n)."""
    rows = sess.run(
        f"MATCH (n:{label}) "
        f"WHERE n.text IS NOT NULL AND n.{prop} IS NULL AND id(n) > $start "
        f"RETURN id(n) AS nid, n.text AS text "
        f"ORDER BY n.id(n) ASC, id(n) LIMIT $lim",
        start=start_id, lim=limit,
    )
    return [(r["nid"], r["text"]) for r in rows]


def run(label: str, model_name: str, prop: str, batch_size: int, resume: bool):
    cfg = get_neo4j_config()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database")
    )
    text_prop = SCHEMA[label]
    model = EmbedModel(model_name)

    log.info(
        "reembed %s  model=%s  prop=%s  dim=%d  device=%s  rocm=%s  gpu_target=%s",
        label, model_name, prop, model.dim, model.device, has_rocm(),
        (gpu_target_machine() or {}).get("name"),
    )

    cursor_file = _cursor_path(label, prop)
    start_id = 0
    if resume and os.path.exists(cursor_file):
        start_id = int(open(cursor_file).read().strip())
        log.info("resuming %s from id > %d (cursor file)", label, start_id)

    with driver.session() as sess:
        ensure_vector_index(sess, label, prop, model.dim)
        need, total = count_needing(sess, label, text_prop, prop)
        log.info("%s: %d/%d nodes missing %s vector", label, need, total, prop)

        done = 0
        t0 = time.perf_counter()
        while True:
            rows = page_rows(sess, label, prop, start_id, batch_size)
            if not rows:
                break
            texts = [t or "" for _, t in rows]
            vecs = model.embed(texts, batch_size=min(batch_size, len(texts)))
            updates = []
            for (nid, _text), vec in zip(rows, vecs):
                updates.append({"nid": nid, "vec": vec})
            if updates:
                sess.run(
                    f"UNWIND $u AS x "
                    f"MATCH (n:{label}) WHERE id(n) = x.nid "
                    f"SET n.{prop} = x.vec",
                    u=updates,
                )
                start_id = rows[-1][0]
                with open(cursor_file, "w") as f:
                    f.write(str(start_id))
            done += len(updates)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0.0
            log.info(
                "  %s: %d/%d done (%.0f/s, ~%.0fs left)",
                label, done, need, rate, (need - done) / rate if rate else 0.0,
            )
        log.info("%s complete: %d vectors written to .%s", label, done, prop)
    driver.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-embed nodes with a chosen model+property")
    ap.add_argument("labels", nargs="+", choices=list(SCHEMA.keys()), help="Node labels to re-embed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model name (env EMBED_MODEL_NAME default)")
    ap.add_argument("--prop", required=True, help="Destination vector property (e.g. emb_gte_small)")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_BATCH", "32")))
    ap.add_argument("--resume", action="store_true", help="Only populate missing vectors (default behavior)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for label in args.labels:
        run(label, args.model, args.prop, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
