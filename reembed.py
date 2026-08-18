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
import hashlib

# Property that stores a sha1 of the source text, so --stale can detect nodes
# whose text changed after they were embedded (re-embed only those).
HASH_PROP = "embed_hash"


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


# Cap CPU threads so the re-embed doesn't starve the still-running speakers.py
# sweep on deathstar. Override with TORCH_THREADS / OMP_NUM_THREADS.
import torch as _torch
_tn = os.getenv("TORCH_THREADS")
if _tn:
    _torch.set_num_threads(int(_tn))

from auto_ingest_config import get_neo4j_config
from auto_ingest.embed import load_embed_model, DEFAULT_MODEL, HNSW_M, HNSW_EF, HNSW_QUANT
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
    "Chunk": "text",
    "Summary": "text",
    # Entity / taxonomy nodes store text in non-`text` properties
    "Entity": "text",
    "Concept": "name",
    "Topic": "name",
    "Keyword": "name",
    "KgNode": "title",
    "Note": "text",
    "Speaker": "label",
    "GlobalSpeaker": "display_label",
}


def ensure_vector_index(sess, label: str, prop: str, dim: int, drop_first: bool = True):
    # Per-label index name (Neo4j vector indexes target exactly ONE label, so a
    # shared name like `emb_gte_small_index` would silently drop the previous
    # label's index as each label is processed). Matches the ingest convention
    # (`segment_embedding_index`, `utterance_embedding_index`, ...).
    idx_name = f"{label}_{prop}_index"
    # Index hygiene: if the index already exists with the right dimensions, leave it
    # in place — rebuilding from scratch on 400k+ nodes wastes minutes each run.
    existing = sess.run(
        "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, options "
        "WHERE name = $n RETURN labelsOrTypes, properties, options",
        n=idx_name,
    ).single()
    if existing is not None:
        labels = list(existing[0]) if existing[0] else []
        props = list(existing[1]) if existing[1] else []
        dim_existing = int(existing[2].get("indexConfig", {}).get("vector.dimensions", -1))
        if props == [prop] and labels == [label] and dim_existing == dim:
            log.info("index %s already exists (label=%s prop=%s dim=%d); keeping it",
                     idx_name, label, prop, dim)
            return
        if drop_first:
            sess.run(f"DROP INDEX {idx_name} IF EXISTS")
    # Neo4j 5.x vector syntax with tuned HNSW params (recall > defaults).
    sess.run(
        f"CREATE VECTOR INDEX {idx_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{prop}) "
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


def count_needing(sess, label: str, text_prop: str, prop: str, stale: bool = False) -> Tuple[int, int]:
    total = sess.run(
        f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL RETURN count(n)"
    ).single().values()[0]
    if not stale:
        done = sess.run(
            f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL AND n.{prop} IS NOT NULL RETURN count(n)"
        ).single().values()[0]
        return total - done, total
    # --stale: scan and compare stored hashes to current text (full scan, read-only).
    need = 0
    start = -1
    while True:
        rows = sess.run(
            f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL AND id(n) > $start "
            f"RETURN id(n) AS nid, n.{text_prop} AS text, n.{prop} AS vec, n.{HASH_PROP} AS h "
            f"ORDER BY id(n) ASC LIMIT $lim",
            start=start, lim=2000,
        ).data()
        if not rows:
            break
        start = rows[-1]["nid"]
        for r in rows:
            if r["vec"] is None or r["h"] != _text_hash(r["text"]):
                need += 1
    return need, total


def _cursor_path(label: str, prop: str, uri: str = "") -> str:
    # Namespace by URI so the same (label,prop) on DIFFERENT KGs (e.g. bodycam
    # vs research) don't share a resume cursor — bodycam node ids are huge and
    # would otherwise poison a smaller KG's start-id and embed nothing.
    import hashlib
    suf = "_" + hashlib.sha1(uri.encode()).hexdigest()[:10] if uri else ""
    return f".reembed_{label}_{prop}{suf}.cursor"


def page_rows(sess, label: str, prop: str, text_prop: str, start_id: int, limit: int, stale: bool = False):
    """Resumable page of nodes needing `prop` (missing, or --stale: text changed)."""
    if not stale:
        rows = sess.run(
            f"MATCH (n:{label}) "
            f"WHERE n.{text_prop} IS NOT NULL AND n.{prop} IS NULL AND id(n) > $start "
            f"RETURN id(n) AS nid, n.{text_prop} AS text "
            f"ORDER BY id(n) ASC LIMIT $lim",
            start=start_id, lim=limit,
        ).data()
        return [(r["nid"], r["text"]) for r in rows]
    # --stale: fetch nodes with text, filter out ones whose hash still matches.
    rows = sess.run(
        f"MATCH (n:{label}) "
        f"WHERE n.{text_prop} IS NOT NULL AND id(n) > $start "
        f"RETURN id(n) AS nid, n.{text_prop} AS text, n.{prop} AS vec, n.{HASH_PROP} AS h "
        f"ORDER BY id(n) ASC LIMIT $lim",
        start=start_id, lim=limit,
    ).data()
    out = []
    for r in rows:
        if r["vec"] is None or r["h"] != _text_hash(r["text"]):
            out.append((r["nid"], r["text"]))
    return out


def run(label: str, model_name: str, prop: str, batch_size: int, resume: bool,
        verify_only: bool = False, catchup: bool = False, engine: str = "torch",
        stale: bool = False, backfill_hash: bool = False):
    cfg = get_neo4j_config()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database"),
        # Cursor paging depends on id(n) being a monotonic integer (elementId is
        # NOT sortable by creation order — its lexicographic order has no
        # relation to insert sequence). Keep id(n) and silence the deprecation
        # notification instead of switching to a non-monotonic cursor.
        notifications_disabled_classifications=["DEPRECATION"],
    )
    text_prop = SCHEMA[label]
    with driver.session() as sess:
        if verify_only:
            need, total = count_needing(sess, label, text_prop, prop, stale=stale)
            log.info("%s: %d/%d nodes still missing %s vector%s",
                     label, need, total, prop,
                     "  [OK]" if need == 0 else "  [MISSING]")
            driver.close()
            return need

        if backfill_hash:
            n = _backfill_hash(sess, label, text_prop, prop, batch_size)
            log.info("%s: backfilled %s on %d nodes (no re-embed)", label, HASH_PROP, n)
            driver.close()
            return 0

        # Short-circuit: only load the (large) model if something actually needs
        # embedding. Saves a full model load per label on every watcher sweep.
        need, total = count_needing(sess, label, text_prop, prop, stale=stale)
        if need == 0:
            log.info("%s: %d/%d missing %s — nothing to do, skipping model load",
                     label, need, total, prop)
            driver.close()
            return 0

        model = load_embed_model(model_name, engine=engine)
        log.info(
            "reembed %s  model=%s  prop=%s  dim=%d  device=%s  rocm=%s  gpu_target=%s  engine=%s",
            label, model_name, prop, model.dim, model.device, has_rocm(),
            (gpu_target_machine() or {}).get("name"), engine,
        )

        ensure_vector_index(sess, label, prop, model.dim, drop_first=not catchup)
        log.info("%s: %d/%d nodes missing %s vector", label, need, total, prop)

        if catchup:
            # Id-independent sweep: re-scan from 0 repeatedly so nodes created
            # after the main cursor passed their id(n) still get covered. The
            # WHERE ... IS NULL filter makes each pass cheap; we stop when a
            # full pass embeds nothing new (new arrivals were also covered).
            log.info("%s: catch-up mode — sweeping id space for stragglers", label)
            passes = 0
            while True:
                passes += 1
                embedded = _embed_pass(sess, label, model, prop, text_prop, batch_size,
                                        start_id=0, log_every=0, stale=stale)
                log.info("%s: catch-up pass %d embedded %d stragglers",
                         label, passes, embedded)
                if embedded == 0:
                    break
                if passes > 50:
                    log.warning("%s: catch-up not converging after %d passes "
                                "(data arriving faster than we sweep)", label, passes)
                    break
            need_after, _ = count_needing(sess, label, text_prop, prop)
            log.info("%s: catch-up complete — %d/%d still missing%s",
                     label, need_after, total, "" if need_after == 0 else " (non-text nodes excluded)")
            driver.close()
            return need_after

        cursor_file = _cursor_path(label, prop, cfg["uri"])
        start_id = 0
        if resume and os.path.exists(cursor_file):
            start_id = int(open(cursor_file).read().strip())
            log.info("resuming %s from id > %d (cursor file)", label, start_id)

        done = _embed_pass(sess, label, model, prop, text_prop, batch_size, start_id,
                            cursor_file=cursor_file if resume else None,
                            target=need, stale=stale)
        remaining = max(need - done, 0)
        log.info("%s complete: %d vectors written to .%s (%d still missing)",
                 label, done, prop, remaining)
    driver.close()
    return remaining


def _embed_pass(sess, label: str, model, prop: str, text_prop: str, batch_size: int,
                start_id: int, cursor_file: str = None, target: int = 0,
                stale: bool = False, log_every: int = 1) -> int:
    """Embed every node still missing `prop` above `start_id`. Returns count.
    Also stamps embed_hash so a later --stale run can detect text edits."""
    done = 0
    t0 = time.perf_counter()
    while True:
        rows = page_rows(sess, label, prop, text_prop, start_id, batch_size, stale=stale)
        if not rows:
            break
        texts = [t or "" for _, t in rows]
        vecs = model.embed(texts, batch_size=min(batch_size, len(texts)))
        updates = []
        for (nid, text), vec in zip(rows, vecs):
            updates.append({"nid": nid, "vec": vec, "h": _text_hash(text)})
        if updates:
            sess.run(
                f"UNWIND $u AS x "
                f"MATCH (n:{label}) WHERE id(n) = x.nid "
                f"SET n.{prop} = x.vec, n.{HASH_PROP} = x.h",
                u=updates,
            )
            start_id = rows[-1][0]
            if cursor_file:
                with open(cursor_file, "w") as f:
                    f.write(str(start_id))
        done += len(updates)
        if log_every and (done % (batch_size * log_every) < batch_size):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0.0
            log.info(
                "  %s: %d/%d done (%.0f/s, ~%.0fs left)",
                label, done, target, rate, (target - done) / rate if rate else 0.0,
            )
    return done


def _backfill_hash(sess, label: str, text_prop: str, prop: str, batch_size: int) -> int:
    """Write embed_hash on already-embedded nodes without re-embedding.
    Idempotent; prep so a subsequent --stale run won't re-embed unchanged nodes."""
    done = 0
    start = -1
    while True:
        rows = sess.run(
            f"MATCH (n:{label}) WHERE n.{text_prop} IS NOT NULL AND n.{prop} IS NOT NULL "
            f"AND id(n) > $start RETURN id(n) AS nid, n.{text_prop} AS text, n.{HASH_PROP} AS h "
            f"ORDER BY id(n) ASC LIMIT $lim",
            start=start, lim=batch_size,
        ).data()
        if not rows:
            break
        start = rows[-1]["nid"]
        updates = [{"nid": r["nid"], "h": _text_hash(r["text"])}
                   for r in rows if r["h"] != _text_hash(r["text"])]
        if updates:
            sess.run(
                f"UNWIND $u AS x MATCH (n:{label}) WHERE id(n) = x.nid SET n.{HASH_PROP} = x.h",
                u=updates,
            )
            done += len(updates)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-embed nodes with a chosen model+property")
    ap.add_argument("labels", nargs="+", choices=list(SCHEMA.keys()), help="Node labels to re-embed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model name (env EMBED_MODEL_NAME default)")
    ap.add_argument("--prop", required=True, help="Destination vector property (e.g. emb_gte_small)")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_BATCH", "32")))
    ap.add_argument("--resume", action="store_true", help="Only populate missing vectors (default behavior)")
    ap.add_argument("--torch-threads", type=int, default=int(os.getenv("TORCH_THREADS", "6")),
                    help="torch intra-op threads (default 6; leaves headroom on deathstar for the sweep)")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only report per-label counts of nodes still missing the vector; no writes")
    ap.add_argument("--catchup", action="store_true",
                    help="Id-independent sweep: re-scan from 0 until no stragglers remain, "
                         "covering nodes created after the main cursor passed them")
    ap.add_argument("--engine", choices=["torch", "onnx"], default=os.getenv("EMBED_ENGINE", "torch"),
                    help="Inference engine. onnx runs the same weights via onnxruntime CPU "
                         "(fast; ONNX_QUANTIZE=1 for ~3-5x). Keep ONE engine per prop.")
    ap.add_argument("--stale", action="store_true",
                    help="Re-embed nodes whose source text changed (needs embed_hash; run --backfill-hash first)")
    ap.add_argument("--backfill-hash", action="store_true",
                    help="Only write embed_hash on already-embedded nodes (no re-embed); prep for --stale")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _torch.set_num_threads(args.torch_threads)
    log.info("torch threads=%d  model=%s  prop=%s  batch=%d  engine=%s  stale=%s",
             args.torch_threads, args.model, args.prop, args.batch_size, args.engine, args.stale)
    total_missing = 0
    for label in args.labels:
        missing = run(label, args.model, args.prop, args.batch_size, args.resume,
                      verify_only=args.verify_only, catchup=args.catchup, engine=args.engine,
                      stale=args.stale, backfill_hash=args.backfill_hash)
        total_missing += missing
    if args.verify_only:
        raise SystemExit(1 if total_missing else 0)


if __name__ == "__main__":
    main()
