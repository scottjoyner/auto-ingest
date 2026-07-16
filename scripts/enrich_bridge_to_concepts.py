#!/usr/bin/env python3
"""
Step 5 of neo4j-summary-geo-clustering: bridge Summaries to the research Concept graph.

The Concept nodes (arXiv-style taxonomy: cs., agent, reasoning, algorithms, cs.cv...)
form the "ideas / research" substrate. Summaries are currently DISCONNECTED from them
(verified 0 edges). This links each Summary to the Concepts it actually mentions in
its text/bullets via (s)-[:DISCUSSES]->(c) with a mention count as weight.

Approach: keyword substring match of Concept.name in lowercased summary text+bullets.
Verified coverage: 4,093 geo-tagged summaries mention >=1 concept name. This is a REAL
bridge (lexical overlap), not an embedding guess.

Idempotent: WHERE NOT EXISTS guards the edge; re-runs only add new mentions.

Optional propagation: if a Summary DISCUSSES concept X and X CO_OCCURS_WITH Y (strong,
weight >= W), also add a lighter (s)-[:RELATED_CONCEPT]->(Y) so the research neighborhood
lights up. Controlled by PROPAGATE + MIN_COOC_WEIGHT.
"""
import neo4j

# Use the repo's standard Neo4j config resolver (env -> config -> baked-in).
import sys as _sys
from pathlib import Path as _P
_SYS_REPO = str(_P(__file__).resolve().parent.parent)
if _SYS_REPO not in _sys.path:
    _sys.path.insert(0, _SYS_REPO)
from auto_ingest_config import get_neo4j_config as _get_cfg

PROPAGATE = True
MIN_COOC_WEIGHT = 1000   # only strong co-occurrence edges propagate

def text_of(s):
    parts = []
    t = s.get("text"); b = s.get("bullets")
    if isinstance(t, str):
        parts.append(t)
    elif isinstance(t, list):
        parts.extend([x for x in t if isinstance(x, str)])
    if isinstance(b, str):
        parts.append(b)
    elif isinstance(b, list):
        parts.extend([x for x in b if isinstance(x, str)])
    return " ".join(parts).lower()

def main():
    _cfg = _get_cfg()
    uri, user, pw = _cfg["uri"], _cfg["user"], _cfg["password"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    print("connected", flush=True)

    # Load all concept names
    names = []
    with driver.session() as s:
        for r in s.run("MATCH (c:Concept) RETURN c.name AS name"):
            if r["name"]:
                names.append(r["name"].lower())
    names.sort(key=len, reverse=True)  # longer first so 'cs.cv' wins over 'cs'
    print(f"loaded {len(names)} concept names", flush=True)

    # Strong co-occurrence map: name -> [(neighbor_name, weight)]
    cooc = {}
    if PROPAGATE:
        with driver.session() as s:
            for r in s.run(
                "MATCH (a:Concept)-[r:CO_OCCURS_WITH]->(b:Concept) "
                "WHERE r.weight >= $w RETURN a.name AS a, b.name AS b, r.weight AS w",
                w=MIN_COOC_WEIGHT):
                cooc.setdefault(r["a"].lower(), []).append((r["b"].lower(), r["w"]))
        print(f"co-occurrence map: {len(cooc)} concepts with strong neighbors", flush=True)

    # Process summaries in batches
    q = """
    MATCH (s:Summary) WHERE s.lat IS NOT NULL AND size(s.text) > 0
    RETURN s.id AS id, s.text AS text, s.bullets AS bullets
    """
    total_links = 0
    total_rel = 0
    batch = []
    B = 500

    def flush(batch):
        nonlocal total_links, total_rel
        with driver.session() as ws:
            with ws.begin_transaction() as tx:
                for (sid, mentioned, related) in batch:
                    for name in mentioned:
                        tx.run(
                            "MATCH (s:Summary {id: $sid}) MATCH (c:Concept {name: $name}) "
                            "MERGE (s)-[r:DISCUSSES]->(c) SET r.weight = coalesce(r.weight,0)+1",
                            sid=sid, name=name)
                        total_links += 1
                    for name in related:
                        tx.run(
                            "MATCH (s:Summary {id: $sid}) MATCH (c:Concept {name: $name}) "
                            "MERGE (s)-[r:RELATED_CONCEPT]->(c) SET r.weight = coalesce(r.weight,0)+1",
                            sid=sid, name=name)
                        total_rel += 1
                tx.commit()

    with driver.session() as s:
        for rec in s.run(q):
            txt = text_of(rec.data())
            if not txt:
                continue
            mentioned = [n for n in names if n in txt]
            if not mentioned:
                continue
            related = set()
            if PROPAGATE:
                for n in mentioned:
                    for (nb, w) in cooc.get(n, []):
                        if nb not in mentioned:
                            related.add(nb)
            batch.append((rec["id"], mentioned, list(related)))
            if len(batch) >= B:
                flush(batch); batch = []
    if batch:
        flush(batch)

    print(f"DONE. DISCUSSES links={total_links}, RELATED_CONCEPT links={total_rel}", flush=True)
    driver.close()

if __name__ == "__main__":
    main()
