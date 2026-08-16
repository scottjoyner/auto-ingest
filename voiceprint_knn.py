#!/usr/bin/env python3
"""KNN snap/identify layer for Scott voice-fingerprint reconciliation.

This module produces similarity scores and identity suggestions only; it does
not make the final authentication decision.
"""
import argparse
import logging

import numpy as np
from neo4j import GraphDatabase

from auto_ingest_config import get_neo4j_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

URI, USER, PASS, NEO4J_DB = get_neo4j_env()
ASSISTX_DB = "assistx"
CANONICAL_GROUP_KEY = "scott:identity"
NEO4J_SCOTT_GSID = "1081161525ba29247e8fc6e1bb26be30"

THRESH_HIGH_QUALITY = 0.72
THRESH_LOW_QUALITY = 0.85
QUALITY_NORM_FLOOR = 0.5
QUALITY_NONZERO_FRAC = 0.5


def driver():
    return GraphDatabase.driver(URI, auth=(USER, PASS))


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v, 0.0, 0.0
    return v / n, float(n), float(np.count_nonzero(v) / max(v.shape[0], 1))


def load_canonical_scott(drv, prefer="assistx"):
    emb = None
    prov = None
    if prefer == "assistx":
        with drv.session(database=ASSISTX_DB) as s:
            r = s.run(
                "MATCH (g:VoiceprintGroup{group_key:$gk})-[:ACTIVE_VERSION]->(v) "
                "RETURN v.embedding AS e",
                gk=CANONICAL_GROUP_KEY,
            ).single()
            if r and r["e"] is not None:
                emb, _, _ = _unit(r["e"])
                prov = "assistx:VoiceprintGroup[scott:identity].active_version"
    if emb is None:
        with drv.session(database=NEO4J_DB) as s:
            r = s.run(
                "MATCH (g:GlobalSpeaker{id:$id}) RETURN g.embedding AS e",
                id=NEO4J_SCOTT_GSID,
            ).single()
            if not r or r["e"] is None:
                raise RuntimeError("canonical Scott embedding not found")
            emb, _, _ = _unit(r["e"])
            prov = "neo4j:GlobalSpeaker[1081161525...]"
    return emb, prov


def cosine(a, b):
    return float(np.dot(a, b))


def adaptive_threshold(cand_norm, cand_nonzero_frac):
    low_q = (cand_norm < QUALITY_NORM_FLOOR) or (
        cand_nonzero_frac < QUALITY_NONZERO_FRAC
    )
    return THRESH_LOW_QUALITY if low_q else THRESH_HIGH_QUALITY


def identify(candidate_vec, index, k=3):
    cvec, cnorm, cnz = _unit(candidate_vec)
    if cnorm == 0:
        return [
            {
                "label": lbl,
                "cosine": 0.0,
                "quality": "zero",
                "threshold": THRESH_LOW_QUALITY,
                "decision": "no-match",
            }
            for lbl in index
        ]

    results = []
    for label, vec in index.items():
        sc = cosine(cvec, vec)
        thr = adaptive_threshold(cnorm, cnz)
        if sc >= thr:
            decision = "match" if sc >= thr + 0.05 else "weak"
        else:
            decision = "no-match"
        results.append(
            {
                "label": label,
                "cosine": round(sc, 4),
                "quality": (
                    "low"
                    if (cnorm < QUALITY_NORM_FLOOR or cnz < QUALITY_NONZERO_FRAC)
                    else "high"
                ),
                "threshold": thr,
                "decision": decision,
            }
        )
    results.sort(key=lambda x: x["cosine"], reverse=True)
    return results[:k]


def build_index(drv):
    scott, prov = load_canonical_scott(drv, prefer="assistx")
    index = {"scott": scott}
    logging.info("[knn] canonical Scott vector from %s (dim=%d)", prov, scott.shape[0])
    return index


def self_test(drv):
    index = build_index(drv)
    scott_vec = index["scott"]
    print("\n=== KNN SELF-TEST ===")

    r = identify(scott_vec, index, k=1)[0]
    print(
        f"  [Scott vs Scott] cosine={r['cosine']} quality={r['quality']} "
        f"thr={r['threshold']} -> {r['decision']}"
    )

    noisy = scott_vec + np.random.RandomState(0).normal(0, 0.05, scott_vec.shape)
    r = identify(noisy, index, k=1)[0]
    print(
        f"  [Scott +noise] cosine={r['cosine']} quality={r['quality']} "
        f"thr={r['threshold']} -> {r['decision']}"
    )

    with drv.session(database=NEO4J_DB) as s:
        other = s.run(
            "MATCH (g:GlobalSpeaker) WHERE g.person_id IS NULL AND g.embedding IS NOT NULL "
            "RETURN g.embedding AS e LIMIT 1"
        ).single()
        if other and other["e"]:
            r = identify(other["e"], index, k=1)[0]
            print(
                f"  [non-Scott GS] cosine={r['cosine']} quality={r['quality']} "
                f"thr={r['threshold']} -> {r['decision']}"
            )

    degraded = scott_vec * 0.3
    r = identify(degraded, index, k=1)[0]
    print(
        f"  [degraded] cosine={r['cosine']} quality={r['quality']} "
        f"thr={r['threshold']} -> {r['decision']}"
    )
    print("=== END SELF-TEST ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
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
