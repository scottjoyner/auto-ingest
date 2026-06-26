#!/usr/bin/env python3
"""Speaker voice clustering on SSD_4TB audio using WAVLM embeddings."""

import os
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio.pipelines as ta_bundles
import librosa
import neo4j
import sklearn.cluster
from sklearn.metrics import silhouette_score


AUDIO_ROOT = os.environ.get("AUDIO_ROOT", "/media/scott/SSD_4TB/audio")
NEO_URI    = "bolt://localhost:7687"
NEO_AUTH=("neo4j", "knowledge_graph_2026")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_rttm(path: Path):
    """Parse RTTM into list of (speaker, start_sec, duration_sec)."""
    entries = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 10:
                continue
            speaker = parts[7]
            start   = float(parts[9])
            dur     = float(parts[10])
            entries.append((speaker, start, dur))
    return entries


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

logging.info("Loading WavLM via torchaudio...")
bundle = ta_bundles.WAVLM_BASE_PLUS.get_model()
del ta_bundles                             # free the bundle ref
# no CUDA on this machine — CPU is fine
logging.info("Model loaded.")


def embed_mp3(path: Path):
    """Load an MP3 and produce a (T, 512) embedding array."""
    audio, sr = librosa.load(str(path), sr=16000)
    wav = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        feats = bundle(wav)                   # (batch, T, C)
    return feats.mean(dim=1).flatten()         # (T, 512)


# ---------------------------------------------------------------------------
# Index audio / RTTM pairs
# ---------------------------------------------------------------------------

pairs: list[tuple[Path, Path]] = []  # (mp3_path, rttm_path)

for root, dirs, files in os.walk(AUDIO_ROOT):
    if "chunks" in root.split(os.sep):
        continue
    mp3s = [f for f in files if f.endswith(".mp3")]
    base_names = {Path(f).stem: Path(f) for f in mp3s}
    for name, mp3_file in base_names.items():
        rttm_file = mp3_file.with_name(name + "_speakers.rttm")
        if rttm_file.exists():
            pairs.append((mp3_file, rttm_file))

pairs.sort(key=lambda p: str(p[0]))
logging.info("Indexed %d audio+RTTM pairs.", len(pairs))


# ---------------------------------------------------------------------------
# Neo4j: existing Speaker nodes and Segment-Speaker links
# ---------------------------------------------------------------------------

driver = neo4j.GraphDatabase.driver(NEO_URI, auth=NEO_AUTH)

with driver.session() as session:
    rows = list(session.execute_read(lambda s: list(s.run("""
        MATCH (s:Speaker)-[r:HAS_SPEAKER]->(seg:Segment)
        RETURN id(seg) AS seg_id
    """))))
    existing_segments = {r["seg_id"] for r in rows}
    logging.info("Found %d existing Segment-Speaker links.", len(existing_segments))


# ---------------------------------------------------------------------------
# Process each file
# ---------------------------------------------------------------------------

all_embeddings: list[np.ndarray] = []  # (T, 512) per segment
meta: list[dict] = []                   # {"file", "speaker"}

with driver.session() as session:
    for mp3_path, rttm_path in pairs:
        logging.info("Processing %s …", mp3_path.name)
        segments = parse_rttm(rttm_path)
        if not segments:
            continue
        emb = embed_mp3(mp3_path)  # (T, 512)

        for speaker_name, start_sec, dur in segments:
            seg_id = session.execute_write(
                lambda s, mp=str(mp3_path), sp=speaker_name, st=start_sec, du=dur:
                    s.run("""
                        CREATE (:Segment {id:$mp, start_time:$st, duration:$du})
                    """, mp=mp, st=st, du=du).consume()
            )

            all_embeddings.append(emb.numpy())
            meta.append({"file": str(mp3_path), "speaker": speaker_name})


# ---------------------------------------------------------------------------
# KMeans clustering on pooled embeddings
# ---------------------------------------------------------------------------

X = np.vstack(all_embeddings).astype("float32")  # (N, 512)

logging.info("Clustering %d samples …", X.shape[0])

silhouettes: list[float] = []
for k in range(2, 21):
    km = sklearn.cluster.KMeans(n_clusters=k, random_state=42).fit(X)
    silhouettes.append(silhouette_score(X, km.labels_))

best_k = list(range(2, 21))[int(np.argmax(silhouettes))]
logging.info("Best k=%d (sil=%.3f)", best_k, max(silhouettes))

km_final = sklearn.cluster.KMeans(n_clusters=best_k, random_state=42).fit(X)
for m, lab in zip(meta, km_final.labels_):
    logging.info(
        "  %-50s speaker=%-8s → cluster %d",
        Path(m["file"]).name, m["speaker"], lab,
    )


# ---------------------------------------------------------------------------
# Persist cluster assignments back to Neo4j
# ---------------------------------------------------------------------------

with driver.session() as session:
    for m, lab in zip(meta, km_final.labels_):
        rec = session.execute_write(
            lambda s, mp=str(m["file"]):
                s.run("""
                    MATCH (seg:Segment {id:$mp}) RETURN id(seg) AS seg_id
                """, mp=mp).single()
        )

        if rec is not None and rec["seg_id"] is not None:
            session.execute_write(
                lambda s, sid=rec["seg_id"], cl=lab:
                    s.run("""
                        MERGE (c:VoiceCluster {cluster_id:$cl})
                        MATCH (seg) WHERE id(seg) = $sid
                        CREATE (seg)-[:IN_CLUSTER]->(c)
                    """, sid=sid, cl=cl),
            )

driver.close()
logging.info("Done. Total segments processed: %d", len(meta))
