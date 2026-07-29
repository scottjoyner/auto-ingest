#!/usr/bin/env python3
"""
Fix-up pass after reconcile_scott_voiceprint.py --apply:

1) Remove the STALE ACTIVE_VERSION edge to the old (superseded, corrupt) version,
   keeping only the new reconcile_resync active version. The HAS_VERSION lineage
   edge to the old version is preserved (history kept).
2) Set aliases + canonical_uid on the 'Scott' VoiceIdentity so it's clearly an
   alias of canonical 'scott' (the first apply's CASE/SET didn't persist due to
   WITH-scoping of the deleted relationship).

Dry-run by default; --apply to write.
"""
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
URI = "bolt://localhost:7687"; USER = "neo4j"; PASS = "knowledge_graph_2026"
ASSISTX_DB = "assistx"
CANONICAL_GROUP_KEY = "scott:identity"


def driver():
    return GraphDatabase.driver(URI, auth=(USER, PASS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", dest="dry_run", action="store_false")
    args = ap.parse_args()
    drv = driver()
    try:
        with drv.session(database=ASSISTX_DB) as s:
            # 1) identify stale ACTIVE_VERSION edges (any except the reconcile_resync active one)
            stale = s.run("""
                MATCH (g:VoiceprintGroup{group_key:$gk})-[r:ACTIVE_VERSION]->(v:VoiceprintVersion)
                WHERE v.source <> 'reconcile_resync'
                RETURN v.version_id AS vid
            """, gk=CANONICAL_GROUP_KEY).data()
            stale_ids = [r["vid"] for r in stale]
            logging.info(f"[fixup] stale ACTIVE_VERSION edges (will remove): {stale_ids}")
            if not args.dry_run and stale_ids:
                s.run("""
                    MATCH (g:VoiceprintGroup{group_key:$gk})-[r:ACTIVE_VERSION]->(v:VoiceprintVersion)
                    WHERE v.source <> 'reconcile_resync'
                    DELETE r
                """, gk=CANONICAL_GROUP_KEY)
                logging.info(f"[fixup] removed {len(stale_ids)} stale ACTIVE_VERSION edge(s).")

            # 2) set alias metadata on 'Scott' VoiceIdentity
            info = s.run("""
                MATCH (vi:VoiceIdentity{user_id:'Scott'})
                RETURN vi.aliases AS al, vi.canonical_uid AS cu
            """).single()
            al = info["al"] if info else None
            cu = info["cu"] if info else None
            logging.info(f"[fixup] 'Scott' VI before: aliases={al} canonical_uid={cu}")
            if not args.dry_run:
                # pure Cypher: ensure 'Scott' is in the aliases list, no APOC needed
                s.run("""
                    MATCH (vi:VoiceIdentity{user_id:'Scott'})
                    SET vi.aliases = CASE
                        WHEN vi.aliases IS NULL THEN ['Scott']
                        WHEN NOT 'Scott' IN vi.aliases THEN vi.aliases + 'Scott'
                        ELSE vi.aliases END,
                        vi.canonical_uid = 'scott',
                        vi.is_alias = true
                """)
                logging.info("[fixup] set aliases=['Scott'], canonical_uid='scott', is_alias=true on 'Scott' VI.")

        mode = "DRY-RUN" if args.dry_run else "APPLIED"
        logging.info(f"=== DONE [{mode}] ===")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
