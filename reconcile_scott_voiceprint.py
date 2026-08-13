#!/usr/bin/env python3
"""Reconcile Scott's voice fingerprint identity across Neo4j databases.

Two safe, idempotent passes are supported and dry-run is the default:
A) consolidate duplicate identity/group relationships in the assistx overlay;
B) re-sync the canonical 192-dim ECAPA centroid into the auth overlay.
"""
import argparse
import logging

import numpy as np
from neo4j import GraphDatabase

from auto_ingest_config import get_neo4j_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

URI, USER, PASS, NEO4J_DB = get_neo4j_env()
ASSISTX_DB = "assistx"
NEO4J_SCOTT_GSID = "1081161525ba29247e8fc6e1bb26be30"
CANONICAL_UID = "scott"
CANONICAL_GROUP_KEY = "scott:identity"


def driver():
    return GraphDatabase.driver(URI, auth=(USER, PASS))


def fetch_centroid(drv):
    with drv.session(database=NEO4J_DB) as s:
        r = s.run(
            "MATCH (g:GlobalSpeaker{id:$id}) RETURN g.embedding AS e",
            id=NEO4J_SCOTT_GSID,
        ).single()
        if not r or r["e"] is None:
            raise SystemExit(f"neo4j Scott centroid {NEO4J_SCOTT_GSID} missing embedding")
        e = np.asarray(r["e"], dtype=np.float32)
        n = np.linalg.norm(e)
        if n == 0:
            raise SystemExit("neo4j Scott centroid is zero vector")
        unit = (e / n).tolist()
        logging.info(
            "[centroid] dim=%d nonzero=%d raw_norm=%.3f unit_norm=%.4f",
            e.shape[0], int(np.count_nonzero(e)), n, np.linalg.norm(unit),
        )
        return unit


def pass_a_consolidate(drv, dry_run):
    with drv.session(database=ASSISTX_DB) as s:
        vis = s.run("MATCH (v:VoiceIdentity) RETURN v.user_id AS uid").data()
        uids = [r["uid"] for r in vis]
        logging.info("[A] VoiceIdentity nodes: %s", uids)
        if CANONICAL_UID not in uids:
            raise SystemExit(f"Canonical VoiceIdentity '{CANONICAL_UID}' not found: {uids}")

        canon = s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            OPTIONAL MATCH (g)-[:ACTIVE_VERSION]->(av:VoiceprintVersion)
            OPTIONAL MATCH (vi:VoiceIdentity{user_id:$uid})-[:IS_GLOBAL_SPEAKER]->(gs:GlobalSpeaker)
            RETURN av.version_id AS avid, gs.id AS gsid
        """, gk=CANONICAL_GROUP_KEY, uid=CANONICAL_UID).single()
        if not canon:
            raise SystemExit("Canonical group/active-version/GlobalSpeaker not resolvable")
        canon_gsid = canon["gsid"]
        logging.info(
            "[A] canonical group=%s active_version=%s GlobalSpeaker=%s",
            CANONICAL_GROUP_KEY, canon["avid"], canon_gsid,
        )

        dup = s.run("""
            MATCH (vi:VoiceIdentity{user_id:'Scott'})
            OPTIONAL MATCH (vi)-[:HAS_GROUP]->(dg:VoiceprintGroup)
            OPTIONAL MATCH (vi)-[:IS_GLOBAL_SPEAKER]->(dgs:GlobalSpeaker)
            RETURN vi.id AS via, dg.group_key AS dgk, dgs.id AS dgsid
        """).single()
        logging.info("[A] duplicate Scott identity: %s", dup)

        if dry_run:
            logging.info("[A][dry-run] would repoint duplicate identity to canonical group/speaker")
            return

        if dup and dup["via"]:
            s.run("""
                MATCH (vi:VoiceIdentity{user_id:'Scott'})
                OPTIONAL MATCH (vi)-[r:HAS_GROUP]->(:VoiceprintGroup)
                DELETE r
                WITH vi
                MATCH (cg:VoiceprintGroup{group_key:$gk})
                MERGE (vi)-[:HAS_GROUP]->(cg)
                SET vi.aliases = CASE
                        WHEN vi.aliases IS NULL THEN ['Scott']
                        WHEN NOT 'Scott' IN vi.aliases THEN vi.aliases + 'Scott'
                        ELSE vi.aliases END,
                    vi.canonical_uid = $uid
            """, gk=CANONICAL_GROUP_KEY, uid=CANONICAL_UID)
            s.run("""
                MATCH (vi:VoiceIdentity{user_id:'Scott'})
                OPTIONAL MATCH (vi)-[r:IS_GLOBAL_SPEAKER]->(:GlobalSpeaker)
                DELETE r
                WITH vi
                MATCH (cgs:GlobalSpeaker{id:$gsid})
                MERGE (vi)-[:IS_GLOBAL_SPEAKER]->(cgs)
            """, gsid=canon_gsid)

        s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            SET g.is_canonical = true, g.primary = true, g.updated_at = datetime()
        """, gk=CANONICAL_GROUP_KEY)
        s.run("""
            MATCH (g:VoiceprintGroup{group_key:'Scott:identity'})
            SET g.is_canonical = false, g.alias_of = 'scott:identity',
                g.deprecated = true, g.updated_at = datetime()
        """)
        s.run("""
            MATCH (vi:VoiceIdentity)
            WHERE vi.user_id IN ['Scott','scott']
            OPTIONAL MATCH (vi)-[r:IS_GLOBAL_SPEAKER]->(:GlobalSpeaker)
            DELETE r
            WITH vi
            MATCH (cgs:GlobalSpeaker{id:$gsid})
            MERGE (vi)-[:IS_GLOBAL_SPEAKER]->(cgs)
        """, gsid=canon_gsid)


def pass_b_resync(drv, centroid, dry_run):
    with drv.session(database=ASSISTX_DB) as s:
        rec = s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            OPTIONAL MATCH (g)-[:ACTIVE_VERSION]->(av:VoiceprintVersion)
            OPTIONAL MATCH (gs:GlobalSpeaker{is_owner_voiceprint:true})
            RETURN av.version_id AS avid, gs.id AS gsid
            LIMIT 1
        """, gk=CANONICAL_GROUP_KEY).single()
        if not rec:
            raise SystemExit("Canonical voiceprint group is missing")
        avid, gsid = rec["avid"], rec["gsid"]

        if dry_run:
            logging.info(
                "[B][dry-run] would write centroid dim=%d to group/version/speaker",
                len(centroid),
            )
            return

        new_vid = __import__("hashlib").md5(
            ("reconcile_resync|" + CANONICAL_GROUP_KEY).encode()
        ).hexdigest()
        if avid:
            s.run(
                "MATCH (v:VoiceprintVersion{version_id:$avid}) "
                "SET v.active=false, v.superseded_at=datetime()",
                avid=avid,
            )
        s.run("""
            CREATE (nv:VoiceprintVersion {
                version_id: $vid, user_id: $uid, scope: 'identity',
                group_key: $gk, source: 'reconcile_resync', lineage_mode: 'replace',
                active: true, embedding: $emb, threshold: 0.6,
                created_at: datetime(), sample_count: 1, append: false
            })
            WITH nv
            MATCH (g:VoiceprintGroup{group_key:$gk})
            MERGE (g)-[:HAS_VERSION]->(nv)
            MERGE (g)-[:ACTIVE_VERSION]->(nv)
            SET g.current_version_id = $vid, g.active_version_id = $vid,
                g.embedding = $emb, g.updated_at = datetime()
        """, vid=new_vid, uid=CANONICAL_UID, gk=CANONICAL_GROUP_KEY, emb=centroid)
        if gsid:
            s.run(
                "MATCH (g:GlobalSpeaker{id:$gsid}) "
                "SET g.embedding=$emb, g.updated_at=datetime(), "
                "g.display_label='Scott', g.is_owner_voiceprint=true",
                gsid=gsid,
                emb=centroid,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", dest="dry_run", action="store_false")
    args = ap.parse_args()

    drv = driver()
    try:
        centroid = fetch_centroid(drv)
        pass_a_consolidate(drv, args.dry_run)
        pass_b_resync(drv, centroid, args.dry_run)
        logging.info("=== DONE [%s] ===", "DRY-RUN" if args.dry_run else "APPLIED")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
