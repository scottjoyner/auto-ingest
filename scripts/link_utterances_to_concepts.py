#!/usr/bin/env python3
"""Link spoken Utterances to research Concepts (W-56).

The knowledge graph separates two embedding spaces:
  * speech/Chunk text  -> 384-dim  all-MiniLM-L6-v2  (Utterance.embedding, Chunk.embedding)
  * research Concept    -> 768-dim  (Concept.embedding_768)

So spoken content and research concepts were never linked (the only MENTIONS
edges today are Transcription->Entity NER). This script bridges them:

  1. Re-embed each Concept into the 384-dim space (same model as Utterance) and
     store it on ``Concept.embedding``. Idempotent: skips Concepts that already
     have a 384-dim embedding.
  2. Create a vector index on ``Concept(embedding)`` (384).
  3. For every Utterance, find the top-K nearest Concepts by cosine and write
     ``Utterance-[:MENTIONS]->Concept`` above a similarity threshold. Resume-safe
     via a state file; a --limit / --sample make it dev-runnable.

Once linked, research-scripted shorts (auto_ingest.shorts) can pull spoken
discussions of a Topic, not just the papers.

Requires the ML image (torch/transformers). Uses get_neo4j_password() (W-53).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure the repo root is importable so `auto_ingest_config` / `vector_search`
# resolve whether run directly or via the auto-ingest CLI.
_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("link_utterances_to_concepts")

EMBED_DIM = 384
CONCEPT_INDEX = "concept_embedding_index"
CONCEPT_BATCH = int(os.getenv("CONCEPT_BATCH", "256"))


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_password
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = get_neo4j_password()
    db = os.getenv("NEO4J_DB", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pw)), db


# --------------------------------------------------------------------------- #
# 1. Re-embed Concepts into 384-dim
# --------------------------------------------------------------------------- #
def embed_concepts(driver, db, *, batch: int = CONCEPT_BATCH, dry_run: bool = False) -> int:
    """Compute Concept.embedding (384-dim) for Concepts missing it. Returns #embedded."""
    from vector_search import embed_texts

    with driver.session(database=db) as sess:
        rows = sess.run(
            """
            MATCH (c:Concept)
            WHERE c.embedding IS NULL OR size(c.embedding) <> $dim
            RETURN c.name AS name, c.definition AS definition
            """,
            dim=EMBED_DIM,
        ).data()
    if not rows:
        log.info("All Concepts already have a %d-dim embedding.", EMBED_DIM)
        return 0

    log.info("Re-embedding %d Concepts into %d-dim space...", len(rows), EMBED_DIM)
    texts = [(r["definition"] if r.get("definition") else r["name"]) or r["name"] for r in rows]
    vecs = embed_texts(texts, batch_size=batch)
    if dry_run:
        log.info("[dry-run] would write %d concept embeddings", len(vecs))
        return len(vecs)

    with driver.session(database=db) as sess:
        for r, vec in zip(rows, vecs):
            sess.run(
                "MATCH (c:Concept {name:$name}) SET c.embedding=$vec, c.embed_model=$m",
                name=r["name"], vec=vec, m="sentence-transformers/all-MiniLM-L6-v2",
            )
    log.info("Wrote %d Concept embeddings.", len(vecs))
    return len(vecs)


# --------------------------------------------------------------------------- #
# 2. Concept vector index
# --------------------------------------------------------------------------- #
def ensure_concept_index(driver, db) -> None:
    with driver.session(database=db) as sess:
        # A vector index requires explicit dimensions + similarity function;
        # a plain CREATE INDEX would create a RANGE index and break queryNodes.
        sess.run(
            f"CREATE VECTOR INDEX {CONCEPT_INDEX} IF NOT EXISTS "
            "FOR (c:Concept) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=EMBED_DIM,
        )
        # wait until the index is online (best-effort)
        for _ in range(30):
            rec = sess.run(
                "SHOW INDEXES YIELD name, state WHERE name=$n RETURN state", n=CONCEPT_INDEX
            ).single()
            if rec and rec["state"] == "ONLINE":
                log.info("Index %s ONLINE.", CONCEPT_INDEX)
                return
            time.sleep(2)
        log.warning("Index %s not ONLINE after wait; continuing.", CONCEPT_INDEX)


# --------------------------------------------------------------------------- #
# 3. Link Utterances -> Concepts
# --------------------------------------------------------------------------- #
def _state_path(path: Optional[str]) -> Path:
    return Path(path or "./utterance_concept_state.json")


def link_utterances(driver, db, *, top_k: int = 3, threshold: float = 0.65,
                    limit: int = 0, sample: int = 0, dry_run: bool = False,
                    state_file: Optional[str] = None) -> int:
    """Write Utterance-[:MENTIONS]->Concept above threshold. Returns #edges written."""
    st_path = _state_path(state_file)
    done: set = set()
    if st_path.exists():
        try:
            done = set(json.loads(st_path.read_text()).get("done", []))
        except Exception:
            done = set()
    log.info("Resuming with %d utterances already linked.", len(done))

    where = "WHERE u.embedding IS NOT NULL"
    if limit:
        where += f" AND id(u) < {limit}"
    elif sample:
        where += f" WITH u, rand() AS r WHERE u.embedding IS NOT NULL AND r < {sample / 709140.0:.8f}"

    with driver.session(database=db) as sess:
        total = sess.run(
            f"MATCH (u:Utterance) {where} RETURN count(u) AS n"
        ).single()["n"]
        log.info("Linking %d Utterances (top_k=%d, thr=%.2f)...", total, top_k, threshold)

        q = (
            f"MATCH (u:Utterance) {where} "
            "CALL db.index.vector.queryNodes($index, $k, u.embedding) "
            "YIELD node AS c, score "
            "WHERE score >= $thr AND c:Concept "
            "RETURN u.id AS uid, collect({name:c.name, score:score}) AS hits"
        )
        written = 0
        batch_state = list(done)
        for i, rec in enumerate(sess.run(q, index=CONCEPT_INDEX, k=top_k, thr=threshold)):
            uid = rec["uid"]
            if uid in done:
                continue
            hits = rec["hits"]
            if not hits:
                done.add(uid)
                continue
            if dry_run:
                written += len(hits)
                done.add(uid)
                continue
            for h in hits:
                sess.run(
                    "MATCH (u:Utterance {id:$uid}) MATCH (c:Concept {name:$name}) "
                    "MERGE (u)-[:MENTIONS {score:$score, model:'all-MiniLM-L6-v2'}]->(c)",
                    uid=uid, name=h["name"], score=float(h["score"]),
                )
                written += 1
            done.add(uid)
            batch_state.append(uid)
            if len(batch_state) >= 2000:
                st_path.write_text(json.dumps({"done": batch_state, "updated_at": time.time()}))
                batch_state = []
            if (i + 1) % 10000 == 0:
                log.info("  progress %d/%d (edges=%d)", i + 1, total, written)
        if batch_state:
            st_path.write_text(json.dumps({"done": batch_state, "updated_at": time.time()}))
    log.info("Wrote %d MENTIONS edges (dry_run=%s).", written, dry_run)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Link spoken Utterances to research Concepts.")
    ap.add_argument("--skip-embed", action="store_true",
                    help="Assume Concept.embedding already exists (skip re-embed step).")
    ap.add_argument("--skip-index", action="store_true",
                    help="Do not (re)create the Concept vector index.")
    ap.add_argument("--top-k", type=int, default=3, help="Max Concepts per Utterance.")
    ap.add_argument("--threshold", type=float, default=0.65, help="Min cosine similarity (0.65 keeps high-confidence links).")
    ap.add_argument("--limit", type=int, default=0, help="Only Utterances with id(u) < N (debug).")
    ap.add_argument("--sample", type=int, default=0, help="Randomly sample N Utterances (debug).")
    ap.add_argument("--no-link", action="store_true",
                    help="Only re-embed + index; do not write MENTIONS edges yet.")
    ap.add_argument("--state-file", type=str, default="./utterance_concept_state.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=CONCEPT_BATCH)
    args = ap.parse_args()

    driver, db = _driver()
    try:
        if not args.skip_embed:
            embed_concepts(driver, db, batch=args.batch, dry_run=args.dry_run)
        if not args.skip_index:
            if not args.dry_run:
                ensure_concept_index(driver, db)
            else:
                log.info("[dry-run] would ensure index %s", CONCEPT_INDEX)
        if not args.no_link:
            link_utterances(
                driver, db, top_k=args.top_k, threshold=args.threshold,
                limit=args.limit, sample=args.sample, dry_run=args.dry_run,
                state_file=args.state_file,
            )
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
