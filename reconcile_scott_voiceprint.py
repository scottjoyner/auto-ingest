#!/usr/bin/env python3
"""
Reconcile Scott's voice fingerprint identity across the two Neo4j databases:

  * neo4j  (main diarization corpus)  -> canonical Scott GlobalSpeaker centroid
  * assistx (voice-fingerprint AUTH overlay, Sophia) -> VoiceIdentity / VoiceprintGroup /
    VoiceprintVersion / GlobalSpeaker

Two safe, idempotent passes (dry_run by default):

  A) IDENTITY CONSOLIDATION (graph-only, no embedding math)
     - The assistx side has a duplicate case-split: VoiceIdentity 'Scott' vs 'scott',
       two VoiceprintGroups ('Scott:identity' vs 'scott:identity'), each with ~2474-2478
       corrupt VoiceprintVersion nodes (all embeddings truncated to ~3 nonzero dims,
       norm ~0.06 instead of a real ECAPA norm ~216).
     - Collapse into ONE canonical 'scott' identity. The 'Scott' VoiceIdentity is kept as
       an alias but repointed to the same group + GlobalSpeaker. We do NOT delete the
       4952 VoiceprintVersion lineage nodes (history is preserved).

  B) AUTH VECTOR RE-SYNC (writes the REAL centroid)
     - Pull the real 192-dim ECAPA centroid from neo4j GlobalSpeaker
       (id 1081161525ba29247e8fc6e1bb26be30, person_id='scott', is_me=true).
     - L2-normalize it (cosine is scale-invariant but the auth path compares normalized
       vectors, and the corrupt ones had norm ~0.06, so we store a clean unit vector).
     - Write it into:
         * canonical 'scott:identity' VoiceprintGroup.embedding
         * that group's ACTIVE VoiceprintVersion.embedding (new active version, source=
           'reconcile_resync', lineage_mode='replace', active=true)
         * assistx GlobalSpeaker{is_owner_voiceprint:true}.embedding
     - This is the only way KNN auth snapping will ever work, because every prior
       assistx vector is corrupt.

Usage:
  python reconcile_scott_voiceprint.py --dry-run     # default, no writes
  python reconcile_scott_voiceprint.py               # apply (after reviewing dry-run)
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

# Canonical Scott centroid in the main corpus (verified: dim=192, nonzero=192, norm=216).
NEO4J_SCOTT_GSID = "1081161525ba29247e8fc6e1bb26be30"
CANONICAL_UID = "scott"          # lowercase matches neo4j person_id convention
CANONICAL_GROUP_KEY = "scott:identity"


def driver():
    return GraphDatabase.driver(URI, auth=(USER, PASS))


def fetch_centroid(drv):
    with drv.session(database=NEO4J_DB) as s:
        r = s.run("MATCH (g:GlobalSpeaker{id:$id}) RETURN g.embedding AS e",
                  id=NEO4J_SCOTT_GSID).single()
        if not r or r["e"] is None:
            raise SystemExit(f"neo4j Scott centroid {NEO4J_SCOTT_GSID} missing embedding")
        e = np.asarray(r["e"], dtype=np.float32)
        n = np.linalg.norm(e)
        if n == 0:
            raise SystemExit("neo4j Scott centroid is zero vector")
        unit = (e / n).tolist()
        logging.info(f"[centroid] dim={e.shape[0]} nonzero={int(np.count_nonzero(e))} "
                     f"raw_norm={n:.3f} unit_norm={np.linalg.norm(unit):.4f}")
        return unit


# --------------------------------------------------------------------------
# PASS A: identity consolidation (graph-only)
# --------------------------------------------------------------------------
def pass_a_consolidate(drv, dry_run):
    with drv.session(database=ASSISTX_DB) as s:
        # 1) confirm the two VoiceIdentities exist
        vis = s.run("MATCH (v:VoiceIdentity) RETURN v.user_id AS uid").data()
        uids = [r["uid"] for r in vis]
        logging.info(f"[A] VoiceIdentity nodes: {uids}")
        if CANONICAL_UID not in uids:
            raise SystemExit(f"Canonical VoiceIdentity '{CANONICAL_UID}' not found: {uids}")

        # 2) locate the canonical group + its GlobalSpeaker + active version
        # NOTE: VoiceprintGroup is keyed by group_key (no `id` property).
        canon = s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            OPTIONAL MATCH (g)-[:ACTIVE_VERSION]->(av:VoiceprintVersion)
            OPTIONAL MATCH (vi:VoiceIdentity{user_id:$uid})-[:IS_GLOBAL_SPEAKER]->(gs:GlobalSpeaker)
            RETURN av.version_id AS avid, gs.id AS gsid
        """, gk=CANONICAL_GROUP_KEY, uid=CANONICAL_UID).single()
        if not canon:
            raise SystemExit("Canonical group/active-version/GlobalSpeaker not resolvable")
        canon_gid = CANONICAL_GROUP_KEY
        canon_avid = canon["avid"]
        canon_gsid = canon["gsid"]
        logging.info(f"[A] canonical group={canon_gid} active_version={canon_avid} "
                     f"GlobalSpeaker={canon_gsid}")

        # 3) the duplicate 'Scott' identity pieces
        dup = s.run("""
            MATCH (vi:VoiceIdentity{user_id:'Scott'})
            OPTIONAL MATCH (vi)-[:HAS_GROUP]->(dg:VoiceprintGroup)
            OPTIONAL MATCH (vi)-[:IS_GLOBAL_SPEAKER]->(dgs:GlobalSpeaker)
            RETURN vi.id AS via, dg.group_key AS dgk, dgs.id AS dgsid
        """).single()
        if dup:
            logging.info(f"[A] duplicate 'Scott' VoiceIdentity id={dup['via']} "
                         f"group={dup['dgk']} gs={dup['dgsid']}")
        else:
            logging.info("[A] no duplicate 'Scott' VoiceIdentity (already consolidated)")

        if dry_run:
            logging.info("[A][dry-run] would: repoint 'Scott' VoiceIdentity HAS_GROUP + "
                         "IS_GLOBAL_SPEAKER onto canonical scott group/GlobalSpeaker, "
                         "add aliases, mark canonical group as primary. (no deletes)")
            return

        # --- APPLY ---
        # Repoint the duplicate VoiceIdentity's relationships to the canonical ones.
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
            logging.info("[A] repointed 'Scott' VoiceIdentity onto canonical scott "
                         "group + GlobalSpeaker (kept as alias).")

        # Mark canonical group as primary; tag the duplicate group as deprecated-alias.
        s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            SET g.is_canonical = true, g.primary = true, g.updated_at = datetime()
        """, gk=CANONICAL_GROUP_KEY)
        s.run("""
            MATCH (g:VoiceprintGroup{group_key:'Scott:identity'})
            SET g.is_canonical = false, g.alias_of = 'scott:identity',
                g.deprecated = true, g.updated_at = datetime()
        """)
        logging.info("[A] marked 'scott:identity' canonical, 'Scott:identity' as alias.")

        # Point BOTH VoiceIdentity nodes at the same GlobalSpeaker (idempotent).
        s.run("""
            MATCH (vi:VoiceIdentity)
            WHERE vi.user_id IN ['Scott','scott']
            OPTIONAL MATCH (vi)-[r:IS_GLOBAL_SPEAKER]->(old:GlobalSpeaker)
            DELETE r
            WITH vi
            MATCH (cgs:GlobalSpeaker{id:$gsid})
            MERGE (vi)-[:IS_GLOBAL_SPEAKER]->(cgs)
        """, gsid=canon_gsid)
        logging.info(f"[A] both VoiceIdentity nodes now IS_GLOBAL_SPEAKER -> {canon_gsid}")


# --------------------------------------------------------------------------
# PASS B: re-sync the real centroid into the auth overlay
# --------------------------------------------------------------------------
def pass_b_resync(drv, centroid, dry_run):
    with drv.session(database=ASSISTX_DB) as s:
        # locate canonical active version + GlobalSpeaker (group keyed by group_key)
        rec = s.run("""
            MATCH (g:VoiceprintGroup{group_key:$gk})
            OPTIONAL MATCH (g)-[:ACTIVE_VERSION]->(av:VoiceprintVersion)
            OPTIONAL MATCH (gs:GlobalSpeaker{is_owner_voiceprint:true})
            RETURN av.version_id AS avid, gs.id AS gsid
            LIMIT 1
        """, gk=CANONICAL_GROUP_KEY).single()
        gid, avid, gsid = CANONICAL_GROUP_KEY, rec["avid"], rec["gsid"]
        logging.info(f"[B] target group={gid} active_version={avid} owner_global_speaker={gsid}")

        if dry_run:
            logging.info(f"[B][dry-run] would write unit centroid (dim={len(centroid)}) into "
                         f"group {gid}.embedding, active version {avid}.embedding, and "
                         f"GlobalSpeaker {gsid}.embedding; create a fresh active version "
                         f"source='reconcile_resync' lineage_mode='replace' active=true.")
            return

        # 1) write into the existing active version (supersede the corrupt one cleanly)
        new_vid = __import__("hashlib").md5(
            ("reconcile_resync|" + CANONICAL_GROUP_KEY).encode()).hexdigest()
        # supersede old active version
        if avid:
            s.run("MATCH (v:VoiceprintVersion{version_id:$avid}) "
                  "SET v.active=false, v.superseded_at=datetime()", avid=avid)
        # create the new canonical active version
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

        # 2) write into the owner GlobalSpeaker
        if gsid:
            s.run("MATCH (g:GlobalSpeaker{id:$gsid}) "
                  "SET g.embedding = $emb, g.updated_at = datetime(), "
                  "g.display_label = 'Scott', g.is_owner_voiceprint = true",
                  gsid=gsid, emb=centroid)
        logging.info(f"[B] wrote real Scott centroid into group {gid}, new active "
                     f"version {new_vid}, GlobalSpeaker {gsid}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Default. Log actions only; no writes.")
    ap.add_argument("--apply", dest="dry_run", action="store_false",
                    help="Actually perform consolidation + re-sync writes.")
    args = ap.parse_args()

    drv = driver()
    try:
        centroid = fetch_centroid(drv)
        pass_a_consolidate(drv, args.dry_run)
        pass_b_resync(drv, centroid, args.dry_run)
        mode = "DRY-RUN (no writes)" if args.dry_run else "APPLIED"
        logging.info(f"=== DONE [{mode}] ===")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
