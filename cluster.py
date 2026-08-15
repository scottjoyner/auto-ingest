#!/usr/bin/env python3
"""Cluster Summary nodes by embedding vector -> Cluster nodes in Neo4j.

Replaces the dead cluster.py / cluster_final.py pair (which shelled out to
`docker exec ... cypher-shell` once per embedding, paged with SKIP/LIMIT in
O(n^2), and never wrote results back). This version:

  * reads embeddings through the neo4j driver (one streaming query, no
    per-node subprocess round-trips),
  * picks k with a silhouette sweep on a sample, then fits MiniBatchKMeans,
  * writes all Cluster nodes + CLUSTERED_IN relationships back in batched
    UNWIND transactions (one commit per batch of clusters),
  * labels clusters with keyword topics (TF-IDF) by default, or via LM Studio
    when --label-llm is given and the endpoint is reachable.

Requires Summary nodes to already carry the embedding property — run
`embed_summaries.py` first if they do not.

Usage:
  cluster.py [--prop embedding] [--k 60] [--min-size 3]
             [--batch 2000] [--label-llm] [--drop-existing] [--dry-run]
"""
import os, sys, time, uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_ingest_config import get_neo4j_config

PROP = os.getenv("CLUSTER_PROP", "embedding")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex")
LLM_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "900"))
LABEL_MIN_MEMBERS = 3


def _driver(cfg):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database")
    )


def fetch_vectors(sess, prop: str, min_text: int = 10):
    """Stream (sid, text, vector) for summaries that have an embedding."""
    rows = sess.run(
        f"MATCH (s:Summary) WHERE size(s.text)>$min AND s.{prop} IS NOT NULL "
        f"RETURN s.id AS sid, s.text AS text, s.{prop} AS vec",
        min=min_text,
    )
    out = []
    for r in rows:
        v = r["vec"]
        if v:
            out.append((r["sid"], str(r["text"] or ""), np.asarray(v, dtype=np.float32)))
    return out


def pick_k(X: np.ndarray, k_range):
    """Pick k via silhouette on a stratified sample (sklearn, CPU-friendly).

    The sweep fits MiniBatchKMeans on a fixed random sample of <=4000 rows; the
    sample is drawn once and reused across k so comparisons are apples-to-apples
    and the expensive tokenization/IO stays out of the loop. Returns
    (best_k, best_sil, best_km): the winner's fitted model is reused to
    warm-start the full-data fit, skipping a fresh k-means++ init.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(42)
    n = min(len(X), 4000)
    sample = X[rng.choice(len(X), n, replace=False)] if len(X) > n else X
    best_k, best_sil, best_km = k_range[0], -1.0, None
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024).fit(sample)
        sil = silhouette_score(sample, km.labels_, sample_size=min(2000, n))
        print(f"  k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_k, best_sil, best_km = k, sil, km
    return best_k, best_sil, best_km


_STOP = set("""a an and are as at be but by for from has have he her his i if in is it its
                of on or our that the their they this to was we were will with you your""".split())


def keyword_labels(rows_by_cluster, top_n: int = 3) -> list:
    """Per-cluster top token frequencies as a cheap, deterministic label."""
    from collections import Counter

    labels = []
    for idx in sorted(rows_by_cluster):
        freq = Counter()
        for _, text, _ in rows_by_cluster[idx]:
            for tok in text.lower().split():
                tok = tok.strip(".,;:!?()[]{}'\"-")
                if len(tok) >= 3 and tok not in _STOP and tok.isalpha():
                    freq[tok] += 1
        terms = [t for t, _ in freq.most_common(top_n)]
        labels.append(", ".join(terms) if terms else f"cluster_{idx}")
    return labels


def llm_labels(rows_by_cluster, batch_size: int = 8) -> list:
    import requests, json

    labels = []
    for start in range(0, len(rows_by_cluster), batch_size):
        chunk = [(idx, rows_by_cluster[idx]) for idx in sorted(rows_by_cluster)[start:start + batch_size]]
        texts = []
        for idx, items in chunk:
            merged = " | ".join(t for _, t, _ in items[:8])[:1600]
            texts.append(f"Cluster {idx}: {merged}")
        system = ("You assign short topic labels to conversation clusters. "
                  "Return ONLY JSON: {\"labels\": [\"<label>\", ...]} with one label per cluster.")
        user = "\n\n".join(texts)
        try:
            r = requests.post(f"{LLM_ENDPOINT}/chat/completions", json={
                "model": LLM_MODEL, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ], "temperature": 0.2, "stream": False,
            }, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()["choices"][0]["message"]["content"]
            arr = json.loads(data).get("labels", [])
        except Exception as e:
            print(f"  LLM labeling failed ({e}); falling back to keywords.")
            return keyword_labels(rows_by_cluster)
        for (idx, _), lab in zip(chunk, arr):
            labels.append((idx, str(lab).strip()))
    return [lab for _, lab in sorted(labels)]


def write_clusters(driver, assignments, labels, algorithm: str, batch: int, prop: str):
    """Persist Cluster nodes + CLUSTERED_IN rels in batched transactions."""
    clusters = {}
    for sid, cid in assignments.items():
        clusters.setdefault(cid, []).append(sid)
    total = 0
    keys = sorted(clusters)
    t0 = time.perf_counter()
    with driver.session() as s:
        for start in range(0, len(keys), batch):
            chunk = keys[start:start + batch]
            cluster_rows = [{
                "cid": f"c-{i}",
                "label": labels[i] if i < len(labels) else f"cluster_{i}",
                "size": len(clusters[i]),
                "algorithm": algorithm,
            } for i in chunk]
            s.execute_write(lambda tx: tx.run(
                """UNWIND $rows AS r
                CREATE (c:Cluster {id: r.cid, label: r.label, size: r.size,
                                   algorithm: r.algorithm, created_at: datetime()})""",
                rows=cluster_rows,
            ))
            rel_rows = [{"sid": sid, "cid": f"c-{i}"} for i in chunk for sid in clusters[i]]
            for j in range(0, len(rel_rows), batch):
                sub = rel_rows[j:j + batch]
                s.execute_write(lambda tx, rows=sub: tx.run(
                    """UNWIND $rows AS r
                    MATCH (s:Summary {id:r.sid}) MATCH (c:Cluster {id:r.cid})
                    MERGE (s)-[:CLUSTERED_IN]->(c)""",
                    rows=rows,
                ))
            total += len(chunk)
            rate = total / (time.perf_counter() - t0)
            print(f"  {total}/{len(keys)} clusters ({rate:.0f}/s)")
    return total


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prop", default=PROP)
    ap.add_argument("--k", type=int, default=0, help="0 = silhouette sweep")
    ap.add_argument("--k-min", type=int, default=40)
    ap.add_argument("--k-max", type=int, default=120)
    ap.add_argument("--k-step", type=int, default=20)
    ap.add_argument("--min-size", type=int, default=LABEL_MIN_MEMBERS,
                    help="Drop clusters smaller than this after fitting")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--label-llm", action="store_true")
    ap.add_argument("--drop-existing", action="store_true",
                    help="Delete existing Cluster nodes + CLUSTERED_IN rels first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = get_neo4j_config()
    driver = _driver(cfg)

    with driver.session() as s:
        total = s.run(f"MATCH (s:Summary) WHERE size(s.text)>10 RETURN count(s)").single().values()[0]
        embedded = s.run(
            f"MATCH (s:Summary) WHERE size(s.text)>10 AND s.{args.prop} IS NOT NULL "
            f"RETURN count(s)"
        ).single().values()[0]
        print(f"{embedded}/{total} summaries have {args.prop} vectors")
        if embedded == 0:
            print("No summaries embedded yet — run `embed_summaries.py` first.")
            driver.close()
            return 1
        if args.drop_existing:
            s.run("MATCH (c:Cluster) DETACH DELETE c")
            print("Dropped existing Cluster nodes.")

        rows = fetch_vectors(s, args.prop)
    print(f"Loaded {len(rows)} vectors.")

    X = np.vstack([v for _, _, v in rows])
    dim = X.shape[1]
    print(f"Matrix: {X.shape} (dim={dim})")

    if args.k:
        k = args.k
        init_centers = None
    else:
        k_range = range(args.k_min, args.k_max + 1, args.k_step)
        k, _sil, sweep_km = pick_k(X, k_range)
        # Warm-start the full-data fit with the winning sweep centers instead of
        # re-running k-means++ from scratch on the whole matrix.
        init_centers = sweep_km.cluster_centers_ if sweep_km is not None else None
    print(f"Using k={k}")

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=1024,
        init="k-means++" if init_centers is None else init_centers,
        n_init=1,
    ).fit(X)
    assignments = {sid: int(c) for (sid, _, _), c in zip(rows, km.labels_)}

    rows_by_cluster = {}
    for (sid, txt, _), c in zip(rows, km.labels_):
        rows_by_cluster.setdefault(int(c), []).append((sid, txt, None))

    sizes = {c: len(items) for c, items in rows_by_cluster.items()}
    print(f"Cluster sizes: min={min(sizes.values())}, max={max(sizes.values())}, "
          f"mean={np.mean(list(sizes.values())):.0f}")

    drop = {c for c, n in sizes.items() if n < args.min_size}
    if drop:
        print(f"Dropping {len(drop)} clusters smaller than {args.min_size}: "
              f"({sorted(drop)[:10]}{'...' if len(drop) > 10 else ''})")
        for c in drop:
            rows_by_cluster.pop(c, None)
        for sid, c in list(assignments.items()):
            if c in drop:
                assignments.pop(sid, None)

    print(f"Labeling {len(rows_by_cluster)} clusters...")
    if args.label_llm:
        labels = llm_labels(rows_by_cluster)
    else:
        labels = keyword_labels(rows_by_cluster)
    for idx, lab in zip(sorted(rows_by_cluster), labels):
        print(f"  cluster {idx}: {lab} ({len(rows_by_cluster[idx])} summaries)")

    if args.dry_run:
        print(f"[dry-run] would write {len(rows_by_cluster)} clusters, "
              f"{len(assignments)} memberships")
        driver.close()
        return 0

    n = write_clusters(driver, assignments, labels, "minibatch_kmeans", args.batch, args.prop)
    print(f"Done: wrote {n} Cluster nodes.")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())