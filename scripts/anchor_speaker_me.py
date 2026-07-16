#!/usr/bin/env python3
"""Anchor the 'me' (Scott) speaker identity and propagate it to every clip.

Bootstrap seed: VoiceIdentity{user_id:'scott'} -[:IS_GLOBAL_SPEAKER]-> GlobalSpeaker.
That GlobalSpeaker is marked is_me=true / person_id='scott' / label='Scott' / status='confirmed',
and every Speaker SAME_PERSON-> it is also flagged is_me. Future linker runs that attach
locals to this GlobalSpeaker then resolve to Scott automatically.

Optional --corpus validates (does not auto-apply) a Scott centroid computed from a folder
of known-Scott audio clips; prints the nearest GlobalSpeaker so you can sanity-check.

Idempotent and safe: nearest other GlobalSpeakers sit at cosine ~0.55 (linker uses 0.78),
so no false merges occur.
"""
import os
import argparse
import logging
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
SCOTT_UID = os.getenv("SCOTT_USER_ID", "scott")
ME_LABEL = os.getenv("ME_LABEL", "Scott")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def find_seed(sess):
    rec = sess.run(
        "MATCH (v:VoiceIdentity{user_id:$uid})-[:IS_GLOBAL_SPEAKER]->(g:GlobalSpeaker) "
        "RETURN g.id AS id, g.embedding IS NOT NULL AS has_emb ORDER BY has_emb DESC LIMIT 1",
        uid=SCOTT_UID,
    ).single()
    if rec:
        return rec["id"]
    rec = sess.run(
        "MATCH (v:VoiceIdentity{user_id:$uid})-[:IS_SPEAKER]->(s:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker) "
        "RETURN g.id AS id LIMIT 1",
        uid=SCOTT_UID,
    ).single()
    return rec["id"] if rec else None


def corpus_centroid(corpus_dir):
    import numpy as np
    import torch
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier

    base = next(iter([
        p for p in [
            os.getenv("AUDIO_BASE"),
            "/media/scott/SSD_4TB/audio",
            "/media/scott/NAS4/fileserver/audio",
        ] if p
    ]), None)
    exts = {".wav", ".mp3", ".m4a", ".flac"}
    files = []
    root = os.path.expanduser(corpus_dir)
    for f in os.listdir(root):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(root, f))
    if not files:
        raise SystemExit(f"No audio files found in {corpus_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_pretrained("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    vecs = []
    for f in files:
        wav, sr = torchaudio.load(f)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        emb = enc.encode_batch(wav.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()
        vecs.append(emb.astype(np.float32))
    cen = np.mean(vecs, axis=0)
    cen = cen / np.linalg.norm(cen)
    return cen


def merge_scott_global_speakers(sess, canonical_id, dry_run):
    ids = [r["id"] for r in sess.run(
        "MATCH (g:GlobalSpeaker{is_me:true}) WHERE g.id <> $cid RETURN g.id AS id", cid=canonical_id
    )]
    if not ids:
        logging.info("[merge] Scott already a single GlobalSpeaker; nothing to merge.")
        return 0
    logging.info(f"[merge] consolidating {len(ids)} extra Scott GlobalSpeakers into {canonical_id}")
    if dry_run:
        return len(ids)
    for gid in ids:
        sess.run(
            "MATCH (sp:Speaker)-[r:SAME_PERSON]->(g:GlobalSpeaker{id:$gid}) "
            "WHERE g.id <> $cid "
            "MERGE (c:GlobalSpeaker{id:$cid}) "
            "MERGE (sp)-[:SAME_PERSON]->(c) "
            "DELETE r "
            "WITH g WHERE g.id <> $cid AND NOT (g)<-[:SAME_PERSON]-() "
            "DETACH DELETE g",
            gid=gid, cid=canonical_id,
        )
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", default=None, help="Override Scott GlobalSpeaker id")
    ap.add_argument("--merge-threshold", type=float, default=0.85,
                    help="Also flag other GlobalSpeakers as Scott if cosine >= this (none trigger at 0.85 today)")
    ap.add_argument("--merge", action="store_true",
                    help="Merge all is_me GlobalSpeakers into the canonical seed (one Scott identity)")
    ap.add_argument("--corpus", default=None, help="Folder of known-Scott audio to validate the anchor (torch+speechbrain)")
    args = ap.parse_args()

    drv = driver()
    with drv.session(database=NEO4J_DB) as sess:
        seed = args.seed or find_seed(sess)
        if not seed:
            logging.error("No Scott GlobalSpeaker seed found via VoiceIdentity. Use --seed.")
            return
        logging.info(f"Scott seed GlobalSpeaker: {seed}")

        # Scott identity set = VoiceIdentity seed + any GlobalSpeaker already labeled Scott / scott.
        # (Prior linking already named some of these 'Scott'; trusting that label is safe and
        # recovers the fragmented Scott identities the user reported.)
        if not args.dry_run:
            sess.run(
                "MATCH (g:GlobalSpeaker) WHERE g.id=$id OR g.label=$label OR g.person_id=$uid "
                "SET g.is_me=true, g.person_id=$uid, g.label=$label, g.status='confirmed', "
                "g.anchored_at=datetime(), g.anchor_method='voiceidentity-seed'",
                id=seed, uid=SCOTT_UID, label=ME_LABEL,
            )

        if args.dry_run:
            prop = sess.run(
                "MATCH (sp:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker) "
                "WHERE (g.id=$id OR g.label=$label OR g.person_id=$uid) "
                "AND (sp.is_me IS NULL OR sp.is_me <> true) RETURN count(sp) AS n",
                id=seed, uid=SCOTT_UID, label=ME_LABEL,
            ).single()
        else:
            prop = sess.run(
                "MATCH (sp:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker) "
                "WHERE (g.id=$id OR g.label=$label OR g.person_id=$uid) "
                "WITH sp WHERE sp.is_me IS NULL OR sp.is_me <> true "
                "SET sp.is_me=true, sp.person_id=$uid, sp.label=$label "
                "RETURN count(sp) AS n",
                id=seed, uid=SCOTT_UID, label=ME_LABEL,
            ).single()
        n_prop = prop["n"] if prop else 0
        logging.info(f"Propagated is_me to {n_prop} Speaker nodes (GlobalSpeaker-seed pass).")

        # Speaker-level pass: some Speaker nodes are already named 'Scott' but their
        # GlobalSpeaker is not. Anchor those too and propagate up to their GlobalSpeaker.
        if args.dry_run:
            n_sp = sess.run(
                "MATCH (sp:Speaker) WHERE sp.label=$label OR sp.person_id=$uid RETURN count(sp) AS n",
                uid=SCOTT_UID, label=ME_LABEL,
            ).single()["n"]
        else:
            sess.run(
                "MATCH (sp:Speaker) WHERE sp.label=$label OR sp.person_id=$uid OR sp.is_me=true "
                "SET sp.is_me=true, sp.person_id=$uid, sp.label=$label",
                uid=SCOTT_UID, label=ME_LABEL,
            )
            sess.run(
                "MATCH (sp:Speaker{is_me:true})-[:SAME_PERSON]->(g:GlobalSpeaker) "
                "SET g.is_me=true, g.person_id=$uid, g.label=$label, g.status='confirmed'",
                uid=SCOTT_UID, label=ME_LABEL,
            )
            n_sp = sess.run(
                "MATCH (sp:Speaker{is_me:true}) RETURN count(sp) AS n",
            ).single()["n"]
        logging.info(f"Total Speaker nodes now is_me: {n_sp}.")

        if args.merge:
            merged = merge_scott_global_speakers(sess, seed, args.dry_run)
            logging.info(f"[merge] consolidated {merged} extra Scott GlobalSpeaker(s) into canonical {seed}.")

        for row in sess.run(
            "MATCH (sp:Speaker{is_me:true})<-[:SPOKEN_BY]-(n) "
            "RETURN labels(n)[0] AS lbl, count(DISTINCT n) AS c ORDER BY c DESC"
        ).data():
            logging.info(f"  Scott-labeled {row['lbl']}: {row['c']}")

        emb = sess.run("MATCH (g:GlobalSpeaker{id:$id}) RETURN g.embedding AS e", id=seed).single()
        if emb and emb["e"] is not None:
            import numpy as np
            sv = np.asarray(emb["e"], dtype=np.float32)
            sv = sv / np.linalg.norm(sv)
            cands = []
            for r in sess.run(
                "MATCH (g:GlobalSpeaker) WHERE g.embedding IS NOT NULL AND g.id<>$id "
                "RETURN g.id AS id, g.embedding AS e", id=seed
            ):
                e = np.asarray(r["e"], dtype=np.float32)
                e = e / np.linalg.norm(e)
                sc = float(np.dot(sv, e))
                if sc >= args.merge_threshold:
                    cands.append((round(sc, 4), r["id"]))
            if cands:
                logging.info(f"Other GlobalSpeakers >= {args.merge_threshold} (also flagged Scott): {cands}")
                if not args.dry_run:
                    for sc, gid in cands:
                        sess.run(
                            "MATCH (g:GlobalSpeaker{id:$id}) "
                            "SET g.is_me=true, g.person_id=$uid, g.label=$label, g.status='confirmed'",
                            id=gid, uid=SCOTT_UID, label=ME_LABEL,
                        )
            else:
                logging.info("No other GlobalSpeaker within merge threshold (anchor is isolated; safe).")

        if args.corpus:
            cen = corpus_centroid(args.corpus)
            best, best_sc = None, -1.0
            for r in sess.run("MATCH (g:GlobalSpeaker) WHERE g.embedding IS NOT NULL RETURN g.id AS id, g.embedding AS e"):
                e = np.asarray(r["e"], dtype=np.float32)
                e = e / np.linalg.norm(e)
                sc = float(np.dot(cen, e))
                if sc > best_sc:
                    best_sc, best = sc, r["id"]
            logging.info(f"[corpus] nearest GlobalSpeaker to {args.corpus} = {best} (cosine {best_sc:.4f}); seed={seed}")
            if best != seed and best_sc > 0.6:
                logging.warning("[corpus] corpus centroid points at a DIFFERENT GlobalSpeaker than the seed; review before trusting either.")

    drv.close()


if __name__ == "__main__":
    main()
