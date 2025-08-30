#!/usr/bin/env python3
"""
Link local per-transcription Speaker nodes to global identities.

Pipeline:
  1) Query Neo4j for (Speaker)--(Utterance)--(Transcription) with u.start/u.end and t.key
  2) Locate corresponding audio file for each t.key
  3) Clip N short utterance snippets per Speaker (with left/right padding)
  4) Compute speaker embeddings (SpeechBrain ECAPA or pyannote-embedding fallback)
  5) Average to per-Speaker voiceprints
  6) Build similarity graph; cluster with a cosine threshold
  7) Create/merge (gs:GlobalSpeaker) and connect (Speaker)-[:SAME_PERSON {score:.., method:"ecapa"}]->(gs)

Notes:
  - Requires: neo4j, torchaudio, soundfile, numpy, torch, speechbrain OR pyannote.audio
  - Audio discovery is heuristic; adjust AUDIO_BASE/EXTS if needed.
"""

import os, sys, re, math, json, random, logging, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torchaudio
import soundfile as sf
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# -----------------------
# Config
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AUDIO_BASE = Path(os.getenv("AUDIO_BASE", "/mnt/8TB_2025/fileserver/audio"))
ALT_AUDIO_BASES = [
    Path("/mnt/8TB_2025/fileserver/dashcam/audio"),
    Path("/mnt/8TB_2025/fileserver/dashcam"),  # last-resort scan
]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac"}

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB       = os.getenv("NEO4J_DB", "neo4j")

# Embedding model selection
SPK_MODEL = os.getenv("SPK_MODEL", "speechbrain")  # "speechbrain" | "pyannote"
ECAPA_NAME = os.getenv("ECAPA_NAME", "speechbrain/spkrec-ecapa-voxceleb")  # HuggingFace id

# Defaults
DEFAULT_MIN_UTT = 0.7       # seconds, minimum utterance duration to consider
DEFAULT_MAX_SNIPS = 8       # per local Speaker
DEFAULT_PAD = 0.2           # seconds left/right around utterance for the snip
DEFAULT_SR = 16000          # target sample rate for embeddings
DEFAULT_THRESH = 0.72       # cosine similarity threshold for "same speaker"
DEFAULT_LIMIT_SPEAKERS = 0  # 0 = no limit
DEFAULT_DRY_RUN = False
DEFAULT_WRITE_SNIPS = False

# Optional trivial music filter heuristic (text-level): drop snips with almost no alpha chars
TEXT_ALPHA_MIN = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Neo4j helpers
# -----------------------
def neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ensure_schema(driver):
    cy = [
        "CREATE CONSTRAINT global_speaker_id IF NOT EXISTS FOR (g:GlobalSpeaker) REQUIRE g.id IS UNIQUE",
        "CREATE INDEX speaker_label IF NOT EXISTS FOR (s:Speaker) ON (s.label)",
    ]
    with driver.session(database=NEO4J_DB) as sess:
        for q in cy:
            sess.run(q)

def fetch_speakers_and_utts(driver, limit_speakers: int = 0, min_utt_sec: float = DEFAULT_MIN_UTT):
    """
    Return structure:
      speakers: {speaker_id: {'label':..., 'key_set': set([...]), 'utts': [(key,start,end,text), ...]}}
    """
    cypher = f"""
    MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance)-[:SPOKEN_BY]->(s:Speaker)
    WHERE coalesce(u.end,0) - coalesce(u.start,0) >= $min_utt
    RETURN s.id AS speaker_id, s.label AS label, t.key AS file_key,
           u.start AS start, u.end AS end, u.text AS text
    ORDER BY speaker_id, file_key, start
    """
    speakers: Dict[str, Dict] = {}
    with driver.session(database=NEO4J_DB) as sess:
        for rec in sess.run(cypher, min_utt=min_utt_sec):
            sid = rec["speaker_id"]
            sp = speakers.setdefault(sid, {"label": rec["label"], "key_set": set(), "utts": []})
            sp["key_set"].add(rec["file_key"])
            sp["utts"].append((rec["file_key"], float(rec["start"] or 0.0), float(rec["end"] or 0.0), (rec["text"] or "")))
    # optional limit
    if limit_speakers > 0 and len(speakers) > limit_speakers:
        keep = dict(list(speakers.items())[:limit_speakers])
        logging.info(f"Limiting speakers: {len(speakers)} → {len(keep)}")
        speakers = keep
    return speakers

def write_links(driver, groups: List[List[str]], scores: Dict[Tuple[str,str], float], method: str, dry_run: bool):
    """
    groups: list of clusters; each is list of speaker_ids
    scores: pairwise best scores (sidA,sidB) -> cosine
    """
    with driver.session(database=NEO4J_DB) as sess:
        for cluster in groups:
            cluster_sorted = sorted(cluster)
            gid_hash = hashlib.md5("|".join(cluster_sorted).encode("utf-8")).hexdigest()
            if dry_run:
                logging.info(f"DRY-RUN cluster size={len(cluster_sorted)} id={gid_hash} members={cluster_sorted}")
                continue

            # Create/merge the GlobalSpeaker
            sess.run("""
              MERGE (g:GlobalSpeaker {id:$gid})
              ON CREATE SET g.created_at=datetime()
              ON MATCH  SET g.updated_at=datetime()
            """, gid=gid_hash)

            # Link each local Speaker to global node with best-pair score
            for sid in cluster_sorted:
                # choose best score vs any other in cluster (or 1.0 if alone)
                best = 1.0 if len(cluster_sorted) == 1 else max(
                    scores.get((min(sid, x), max(sid, x)), 0.0) for x in cluster_sorted if x != sid
                )
                sess.run("""
                  MATCH (s:Speaker {id:$sid}), (g:GlobalSpeaker {id:$gid})
                  MERGE (s)-[r:SAME_PERSON]->(g)
                  ON CREATE SET r.method=$method, r.score=$score, r.created_at=datetime()
                  ON MATCH  SET r.method=$method, r.score=$score, r.updated_at=datetime()
                """, sid=sid, gid=gid_hash, method=method, score=float(best))

# -----------------------
# Audio helpers
# -----------------------
def _norm_stem(key: str) -> str:
    # unify legacy keys if needed (e.g., trailing "_R")
    return re.sub(r"_R$", "", key)

def find_audio_for_key(key: str) -> Optional[Path]:
    """
    Strategy:
      1) AUDIO_BASE/**/{key}.{ext}
      2) ALT_AUDIO_BASES/**/{key}.{ext}
      3) Fallback: scan for files whose stem startswith key (rare)
    """
    stem = _norm_stem(key)
    # pass 1
    for ext in AUDIO_EXTS:
        cand = list(AUDIO_BASE.rglob(f"**/{stem}{ext}"))
        if cand:
            return cand[0]
    # pass 2
    for base in ALT_AUDIO_BASES:
        for ext in AUDIO_EXTS:
            cand = list(base.rglob(f"**/{stem}{ext}"))
            if cand:
                return cand[0]
    # pass 3 loose
    cand = list(AUDIO_BASE.rglob(f"**/{stem}*"))
    return cand[0] if cand else None

def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr

def clip_wave(wav: torch.Tensor, sr: int, start: float, end: float, pad: float) -> torch.Tensor:
    s = max(0, int(round((start - pad) * sr)))
    e = min(wav.numel(), int(round((end + pad) * sr)))
    if e <= s:
        return torch.zeros(0)
    return wav[s:e].clone()

def write_debug_wav(out_dir: Path, name: str, wav: torch.Tensor, sr: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f"{name}.wav"), wav.cpu().numpy(), sr, subtype="PCM_16")

# -----------------------
# Speaker embedding backends
# -----------------------
class SpkEmbedder:
    def __init__(self, backend: str = SPK_MODEL):
        self.backend = backend
        self._ecapa = None
        self._pyannote = None

    def _ensure_ecapa(self):
        if self._ecapa is None:
            from speechbrain.pretrained import EncoderClassifier
            logging.info("Loading SpeechBrain ECAPA speaker encoder…")
            self._ecapa = EncoderClassifier.from_hparams(
                source=ECAPA_NAME, run_opts={"device": DEVICE}
            )

    def _ensure_pyannote(self):
        if self._pyannote is None:
            from pyannote.audio import Inference
            logging.info("Loading pyannote speaker embedding model…")
            # embeddings/speaker-diarization default; change if you prefer another
            self._pyannote = Inference("pyannote/embedding", device=DEVICE)

    @torch.inference_mode()
    def embed(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        if wav.numel() == 0:
            return np.zeros((192,), dtype=np.float32) if self.backend == "pyannote" else np.zeros((192,), dtype=np.float32)

        if self.backend == "speechbrain":
            self._ensure_ecapa()
            # speechbrain expects [batch, time]
            emb = self._ecapa.encode_batch(wav.unsqueeze(0).to(DEVICE)).squeeze(0).squeeze(0).cpu().numpy()
            return emb.astype(np.float32)
        else:
            self._ensure_pyannote()
            # pyannote Inference accepts numpy; ensure mono float32
            arr = wav.cpu().numpy().astype(np.float32)
            emb = self._pyannote(arr, sample_rate=sr)
            return np.asarray(emb, dtype=np.float32)

# -----------------------
# Clustering
# -----------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cluster_by_threshold(centroids: Dict[str, np.ndarray], thresh: float) -> Tuple[List[List[str]], Dict[Tuple[str,str], float]]:
    """
    Simple union-find clustering: connect if cosine >= thresh, then take components.
    Returns (groups, pair_scores)
    """
    ids = list(centroids.keys())
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    scores: Dict[Tuple[str,str], float] = {}
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            s = cosine(centroids[a], centroids[b])
            scores[(a, b)] = s
            if s >= thresh:
                union(a, b)

    comps: Dict[str, List[str]] = {}
    for x in ids:
        r = find(x)
        comps.setdefault(r, []).append(x)
    groups = list(comps.values())
    groups.sort(key=len, reverse=True)
    return groups, scores

# -----------------------
# Main logic
# -----------------------
@dataclass
class Args:
    min_utt: float
    max_snips: int
    pad: float
    thresh: float
    limit_speakers: int
    backend: str
    write_snips: bool
    dry_run: bool
    out_snips: Optional[Path]

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Link local Speakers to GlobalSpeaker identities via voice embeddings.")
    p.add_argument("--min-utt", type=float, default=DEFAULT_MIN_UTT, help="Min utterance seconds to consider")
    p.add_argument("--max-snips", type=int, default=DEFAULT_MAX_SNIPS, help="Max snippets per local Speaker")
    p.add_argument("--pad", type=float, default=DEFAULT_PAD, help="Padding seconds around utterance")
    p.add_argument("--thresh", type=float, default=DEFAULT_THRESH, help="Cosine threshold for same-speaker")
    p.add_argument("--limit-speakers", type=int, default=DEFAULT_LIMIT_SPEAKERS, help="Process at most N speakers (debug)")
    p.add_argument("--backend", type=str, choices=["speechbrain","pyannote"], default=SPK_MODEL, help="Embedding backend")
    p.add_argument("--write-snips", action="store_true", help="Write clipped WAV snips (debug)")
    p.add_argument("--dry-run", action="store_true", help="Don’t write to Neo4j; print plan only")
    p.add_argument("--snips-dir", type=str, default=None, help="Directory to dump snips if --write-snips")
    return Args(**vars(p.parse_args()))

def main():
    args = parse_args()
    driver = neo4j_driver()
    ensure_schema(driver)

    logging.info("Querying Neo4j for speakers & utterances…")
    speakers = fetch_speakers_and_utts(driver, limit_speakers=args.limit_speakers, min_utt_sec=args.min_utt)
    logging.info(f"Found {len(speakers)} local Speaker nodes with qualifying utterances.")

    # Preload audio per key
    cache_audio: Dict[str, Tuple[torch.Tensor, int, Path]] = {}
    def get_audio_for_key(key: str) -> Optional[Tuple[torch.Tensor, int, Path]]:
        if key in cache_audio:
            return cache_audio[key]
        p = find_audio_for_key(key)
        if not p or not p.exists():
            logging.warning(f"Audio missing for key={key}")
            cache_audio[key] = None
            return None
        wav, sr = load_audio(p, DEFAULT_SR)
        cache_audio[key] = (wav, sr, p)
        return cache_audio[key]

    # Build snippets & embeddings per local speaker
    embedder = SpkEmbedder(args.backend)
    rng = random.Random(1337)

    centroids: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    if args.write_snips:
        out_dir = Path(args.snips_dir or "./speaker_snips")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    logging.info("Clipping and embedding per-speaker snippets…")
    for sid, meta in speakers.items():
        utts = meta["utts"]  # list of (key, start, end, text)
        # light text/music heuristic: require a couple of alpha chars to avoid pure noise/music
        filt = [u for u in utts if sum(c.isalpha() for c in (u[3] or "")) >= TEXT_ALPHA_MIN]
        if not filt:
            filt = utts
        if not filt:
            logging.warning(f"No usable utterances for speaker {sid}")
            continue

        # prefer longer chunks; sample up to max_snips
        filt.sort(key=lambda x: (x[2] - x[1]), reverse=True)
        chosen = filt[:args.max_snips] if len(filt) > args.max_snips else filt
        rng.shuffle(chosen)  # avoid systematic bias across files

        embs = []
        for (key, s, e, text) in chosen:
            pack = get_audio_for_key(key)
            if pack is None:
                continue
            wav, sr, wavpath = pack
            clip = clip_wave(wav, sr, s, e, args.pad)
            if clip.numel() == 0:
                continue
            if args.write_snips and out_dir is not None:
                name = f"{sid}_{_norm_stem(key)}_{s:.2f}-{e:.2f}"
                write_debug_wav(out_dir, name, clip, sr)
            emb = embedder.embed(clip, sr)
            embs.append(emb)

        if not embs:
            logging.warning(f"No embeddings generated for {sid}")
            continue

        arr = np.stack(embs, axis=0).astype(np.float32)
        cen = arr.mean(axis=0)
        n = np.linalg.norm(cen)
        if n > 0:
            cen = cen / n
        centroids[sid] = cen
        counts[sid] = arr.shape[0]

    logging.info(f"Built centroids for {len(centroids)} speakers (snips per speaker median={np.median(list(counts.values())) if counts else 0}).")

    if not centroids:
        logging.warning("Nothing to cluster; exiting.")
        return

    logging.info(f"Clustering with cosine threshold {args.thresh:.3f} …")
    groups, pair_scores = cluster_by_threshold(centroids, args.thresh)
    for i, g in enumerate(groups, 1):
        logging.info(f"Cluster {i}: size={len(g)} -> {g[:10]}{' …' if len(g)>10 else ''}")

    logging.info("Writing links to Neo4j…" if not args.dry_run else "DRY-RUN; not writing to Neo4j.")
    write_links(driver, groups, pair_scores, method=("ecapa" if args.backend=="speechbrain" else "pyannote"), dry_run=args.dry_run)

    logging.info("Done.")

if __name__ == "__main__":
    main()
