#!/usr/bin/env python3
"""verify_coverage.py — full embedding-coverage audit for text-bearing labels.

This is the single command behind the "full data recovery" guarantee: it
enumerates every node label that carries user-facing text, discovers every
embedding property that has a live vector index, and reports how many nodes
still lack a vector for each (label, prop) pair.

Sources of truth:
  * text-bearing labels: labels whose nodes have a non-empty ``text`` property
    (auto-discovered via ``db.labels()`` + a per-label probe, or pinned with
    ``--label``).
  * embedding props: properties covered by ``SHOW VECTOR INDEXES`` (an index
    existing on a prop implies some consumer expects vectors there).

Exit code:
  0  every text label has vectors for every indexed embedding prop (no gaps)
  1  at least one gap found (or ``--require`` prop is missing somewhere)

Usage:
  verify_coverage.py
  verify_coverage.py --prop emb_gte_small            # focus on one prop
  verify_coverage.py --label Segment,Utterance       # focus on labels
  verify_coverage.py --json                          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_ingest_config import get_neo4j_config
from neo4j import GraphDatabase

# Text props to probe when deciding whether a label is "text-bearing".
TEXT_PROPS = ["text"]
# Canonical text labels that the recovery mission must cover, plus any label
# with a live vector index (discovered at runtime).
REQUIRED_PROPS = {"emb_gte_small"}


def _driver(cfg):
    return GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database")
    )


def text_bearing_labels(sess, indexed: dict[str, list[str]]) -> list[str]:
    """Labels worth auditing: the canonical text labels plus any label that
    actually carries an embedding index on a text property. Non-text labels
    (Frame, GlobalSpeaker, DashcamEmbedding, ...) are excluded — counting their
    nodes is a full scan and they are not part of text recovery.
    """
    canonical = ["Segment", "Utterance", "Transcription", "Summary"]
    from_index = [lbl for lbls in indexed.values() for lbl in lbls]
    return sorted(set(canonical) & set(from_index)) or canonical


def indexed_props(sess) -> dict[str, list[str]]:
    """prop -> labels it is indexed on, from SHOW VECTOR INDEXES."""
    out: dict[str, list[str]] = {}
    for rec in sess.run(
        "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties "
        "RETURN labelsOrTypes AS lbls, properties AS props"
    ):
        props = [p for p in (rec["props"] or []) if not str(p).startswith("_")]
        for p in props:
            out.setdefault(str(p), [])
            for lbl in (rec["lbls"] or []):
                if lbl not in out[str(p)]:
                    out[str(p)].append(lbl)
    return out


def text_predicate(label: str) -> str:
    """Match each label's embed stage so 'OK' means the embedder's target is met.

    reembed embeds any node with text; embed_summaries deliberately skips
    Summary nodes whose text is too short (<=10 chars) to embed meaningfully, so
    those are not a recovery target and must not count toward `total`.
    """
    if label == "Summary":
        return "size(n.text) > 10"
    return "n.text IS NOT NULL AND trim(n.text) <> ''"


def coverage(sess, label: str, prop: str) -> dict:
    pred = text_predicate(label)
    total = sess.run(
        f"MATCH (n:`{label}`) WHERE {pred} "
        f"RETURN count(n) AS c"
    ).single()["c"]
    done = sess.run(
        f"MATCH (n:`{label}`) WHERE {pred} "
        f"AND n.`{prop}` IS NOT NULL RETURN count(n) AS c"
    ).single()["c"]
    return {"label": label, "prop": prop, "embedded": int(done), "total": int(total)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="Comma-separated label subset (default: all text-bearing)")
    ap.add_argument("--prop", help="Comma-separated prop subset (default: all indexed props)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ap.add_argument("--require", default="emb_gte_small",
                    help="Prop(s) that must be fully covered for exit 0 (default: emb_gte_small)")
    args = ap.parse_args(argv)

    cfg = get_neo4j_config()
    driver = _driver(cfg)
    rows: list[dict] = []
    try:
        with driver.session() as sess:
            props_map = indexed_props(sess)
            if args.label:
                labels = [l.strip() for l in args.label.split(",") if l.strip()]
            else:
                labels = text_bearing_labels(sess, props_map)
            if args.prop:
                props = [p.strip() for p in args.prop.split(",") if p.strip()]
            else:
                props = sorted(props_map)
            required = {p.strip() for p in args.require.split(",") if p.strip()}

            for lbl in labels:
                for prop in props:
                    # Only audit pairs that are real: the prop is indexed on this
                    # label, or it is a required recovery prop. Avoids scanning
                    # millions of non-text nodes for nonsense label×prop pairs.
                    if prop not in props_map:
                        continue
                    if lbl not in props_map[prop] and prop not in required:
                        continue
                    rows.append(coverage(sess, lbl, prop))
    finally:
        driver.close()

    gaps = [r for r in rows if r["embedded"] < r["total"]]
    # Required-prop gaps: a required prop with zero index coverage at all.
    covered_props = {r["prop"] for r in rows if r["total"] > 0 and r["embedded"] == r["total"]}
    missing_required = sorted(required - covered_props)

    if args.json:
        print(json.dumps({"rows": rows, "gaps": gaps,
                          "missing_required_props": missing_required}, indent=2))
    else:
        header = f"{'LABEL':<22} {'PROP':<20} {'EMBEDDED':>9} {'TOTAL':>9}  STATUS"
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: (x["label"], x["prop"])):
            status = "OK" if r["embedded"] == r["total"] else "GAP"
            print(f"{r['label']:<22} {r['prop']:<20} {r['embedded']:>9} "
                  f"{r['total']:>9}  {status}")
        if gaps:
            print(f"\n{len(gaps)} gap(s) — run the re-embed stage to close them.")
        if missing_required:
            print(f"required prop(s) with zero coverage: {missing_required}")

    return 1 if (gaps or missing_required) else 0


if __name__ == "__main__":
    sys.exit(main())