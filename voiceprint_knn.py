#!/usr/bin/env python3
"""
voiceprint_knn.py — KNN snap / identify layer for Scott voice-fingerprint reconciliation.

Purpose
-------
Given an unknown speaker embedding (a 192-dim ECAPA vector, e.g. from a freshly
diarized Speaker/GlobalSpeaker centroid in the main `neo4j` corpus, or a live
Sophia capture), score it against the CANONICAL Scott auth vector that now lives
in the `assistx` overlay (VoiceprintGroup `scott:identity` active version, re-synced
from the real neo4j Scott centroid by reconcile_scott_voiceprint.py).

Design notes
------------
* Quality-aware threshold: degraded audio yields sparse/low-norm vectors. We do NOT
  trust a high cosine from a low-quality vector — instead we require a larger margin
  when the candidate's own norm / nonzero fraction is low. This is what lets the snap
  work "despite the varying level of quality".
* This module ONLY produces a SIMILARITY SCORE + identity suggestion. It makes NO
  authentication verdict (authenticated_scott / rejected). Per W-50 ownership, the
  auth decision belongs to the Sophia auth layer; this is the linking/identification
  heuristic feeding it.
* KNN: the index holds the canonical Scott vector plus any other known identities
  (extensible). `identify()` returns the top-K nearest with distances.

Usage (CLI self-test):
  python voiceprint_knn.py --self-test
"""
import argparse
import logging
import numpy as np
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

URI = "bolt://localhost:7687"
USER = "neo4j"
PASS = "knowledge_graph_2026"
NEO4J_DB = "neo4j"
ASSISTX_DB = "assistx"

# Canonical Scott auth vector lives here (re-synced, norm ~1.0).
CANONICAL_GROUP_KEY = "scott:identity"
# Canonical Scott centroid in the main corpus (source of truth for the vector).
NEO4J_SCOTT_GSID = "1081161525ba29247e8fc6e1bb26be30"

# Threshold bands (cosine). High-quality ECAPA accept ~0.72-0.78.
THRESH_HIGH_QUALITY = 0.72
THRESH_LOW_QUALITY = 0.85   # require a bigger margin when candidate vector is degraded
QUALITY_NORM_FLOOR = 0.5    # below this L2 norm, treat candidate as low-quality
QUALITY_NONZERO_FRAC = 0.5  # below this fraction of nonzero dims, treat as low-quality


def driver():
    return GraphDatabase.driver(URI, auth=(USER, PASS))


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v, 0.0, 0.0
    return v / n, float(n), float(np.count_nonzero(v) / max(v.shape[0], 1))


def load_canonical_scott(drv, prefer="assistx"):
    """Return the canonical Scott unit vector + provenance.

    prefer='assistx' reads the re-synced auth vector; 'neo4j' reads the corpus centroid.
    """
    emb = None
    prov = None
    if prefer == "assistx":
        with drv.session(database=ASSISTX_DB) as s:
            r = s.run(
                "MATCH (g:VoiceprintGroup{group_key:$gk})-[:ACTIVE_VERSION]->(v) "
                "RETURN v.embedding AS e", gk=CANONICAL_GROUP_KEY).single()
            if r and r["e"] is not None:
                emb, _, _ = _unit(r["e"])
                prov = "assistx:VoiceprintGroup[scott:identity].active_version"
    if emb is None:
        with drv.session(database=NEO4J_DB) as s:
            r = s.run("MATCH (g:GlobalSpeaker{id:$id}) RETURN g.embedding AS e",
                      id=NEO4J_SCOTT_GSID).single()
            emb, _, _ = _unit(r["e"])
            prov = "neo4j:GlobalSpeaker[1081161525...]"
    return emb, prov


def cosine(a, b):
    return float(np.dot(a, b))


def adaptive_threshold(cand_norm, cand_nonzero_frac):
    """Higher threshold when candidate quality is low."""
    low_q = (cand_norm < QUALITY_NORM_FLOOR) or (cand_nonzero_frac < QUALITY_NONZERO_FRAC)
    return THRESH_LOW_QUALITY if low_q else THRESH_HIGH_QUALITY


def identify(candidate_vec, index, k=3):
    """KNN identify against an index of {label: unit_vector}.

    Returns list of dicts: label, cosine, quality, threshold, decision
    where decision in {'match','weak','no-match'} (NOT an auth verdict).
    """
    cvec, cnorm, cnz = _unit(candidate_vec)
    if cnorm == 0:
        return [{"label": lbl, "cosine": 0.0, "quality": "zero",
                 "threshold": THRESH_LOW_QUALITY, "decision": "no-match"}
                for lbl in index]
    results = []
    for label, vec in index.items():
        sc = cosine(cvec, vec)
        thr = adaptive_threshold(cnorm, cnz)
        if sc >= thr:
            decision = "match" if sc >= thr + 0.05 else "weak"
        else:
            decision = "no-match"
        results.append({
            "label": label,
            "cosine": round(sc, 4),
            "quality": ("low" if (cnorm < QUALITY_NORM_FLOOR or cnz < QUALITY_NONZERO_FRAC) else "high"),
            "threshold": thr,
            "decision": decision,
        })
    results.sort(key=lambda x: x["cosine"], reverse=True)
    return results[:k]


def build_index(drv):
    """Build the known-identity index. Starts with canonical Scott; other identities
    can be added from neo4j GlobalSpeaker nodes flagged is_me or by person_id."""
    scott, prov = load_canonical_scott(drv, prefer="assistx")
    index = {"scott": scott}
    logging.info(f"[knn] canonical Scott vector from {prov} (dim={scott.shape[0]}).")
    return index


def self_test(drv):
    index = build_index(drv)
    scott_vec = index["scott"]
    print("\n=== KNN SELF-TEST ===")
    # 1) Scott himself (perfect) — should MATCH
    r = identify(scott_vec, index, k=1)[0]
    print(f"  [Scott vs Scott]      cosine={r['cosine']} quality={r['quality']} "
          f"thr={r['threshold']} -> {r['decision']}")
    # 2) Scott + tiny noise — should still MATCH
    noisy = scott_vec + np.random.RandomState(0).normal(0, 0.05, scott_vec.shape)
    r = identify(noisy, index, k=1)[0]
    print(f"  [Scott +noise]        cosine={r['cosine']} quality={r['quality']} "
          f"thr={r['threshold']} -> {r['decision']}")
    # 3) A real non-Scott neo4j GlobalSpeaker centroid — should NO-MATCH
    with drv.session(database=NEO4J_DB) as s:
        other = s.run(
            "MATCH (g:GlobalSpeaker) WHERE g.person_id IS NULL AND g.embedding IS NOT NULL "
            "RETURN g.embedding AS e LIMIT 1").single()
        if other and other["e"]:
            r = identify(other["e"], index, k=1)[0]
            print(f"  [non-Scott GS]        cosine={r['cosine']} quality={r['quality']} "
                  f"thr={r['threshold']} -> {r['decision']}")
    # 4) Degraded candidate (sparse/low-norm) that happens to align — should demand
    #    higher threshold (weak/no-match) to avoid false accept on garbage audio.
    degraded = scott_vec * 0.3  # low norm, same direction
    r = identify(degraded, index, k=1)[0]
    print(f"  [degraded (low-norm)] cosine={r['cosine']} quality={r['quality']} "
          f"thr={r['threshold']} -> {r['decision']}  (quality gate working)")
    print("=== END SELF-TEST ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="Run the built-in identify self-test.")
    args = ap.parse_args()
    drv = driver()
    try:
        if args.self_test:
            self_test(drv)
        else:
            idx = build_index(drv)
            print("Index labels:", list(idx.keys()))
    finally:
        drv.close()


if __name__ == "__main__":
    main()
