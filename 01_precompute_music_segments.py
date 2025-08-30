#!/usr/bin/env python3
import os, json, sys, argparse, re, time, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from inaSpeechSegmenter import Segmenter
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------- Config ----------------
AUDIO_ROOTS = [
    "/mnt/8TB_2025/fileserver/audio",
    "/mnt/8TB_2025/fileserver/dashcam/audio",
    "/mnt/8TB_2025/fileserver/bodycam/audio",
]  # add/remove as needed

SIDEcar_SUFFIX = ".music.json"          # where we store music segments per audio
ALLOW_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4"}

# Neo4j (optional)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASS") or "livelongandprosper"

# ----------------------------------------

@dataclass
class AudioRef:
    key: str
    path: str

def stem(p: str) -> str:
    b = os.path.basename(p)
    s, _ = os.path.splitext(b)
    return s

def build_fs_index(roots: List[str]) -> Dict[str, str]:
    idx = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for root, _, files in os.walk(r):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOW_EXT:
                    s = os.path.splitext(f)[0]
                    # If duplicate stems exist, last write wins (we also try exact key match first later)
                    idx[s] = os.path.join(root, f)
    return idx

def resolve_audio_path(audio_key: str, fs_index: Dict[str, str]) -> Optional[str]:
    if not audio_key:
        return None
    # 1) exact stem match
    if audio_key in fs_index:
        return fs_index[audio_key]
    # 2) relaxed: find a stem that contains the key (can be expensive if many; keep first best)
    for s, p in fs_index.items():
        if audio_key in s:
            return p
    return None

def segment_music(seg: Segmenter, path: str) -> List[Tuple[float, float]]:
    """
    Returns list of (start_sec, end_sec) music intervals.
    inaSpeech labels: speech, music, noEnergy, noise, male, female, jingle, ...
    We treat 'music' and 'jingle' as music.
    """
    res = seg(path)
    music = []
    for label, start, end in res:
        if label in ("music", "jingle"):
            music.append((float(start), float(end)))
    return music

def write_sidecar(path: str, music: List[Tuple[float,float]]) -> str:
    sc = path + SIDEcar_SUFFIX
    with open(sc, "w") as f:
        json.dump({"music": music}, f)
    return sc

def push_to_neo4j(audiokey: str, path: str, music: List[Tuple[float,float]]):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as sess:
        sess.execute_write(_write_segments, audiokey, path, music)
    driver.close()

def _write_segments(tx, audiokey: str, path: str, music: List[Tuple[float,float]]):
    # AudioFile + segments; wipe old segments for idempotency
    q1 = """
    MERGE (af:AudioFile {key:$key})
      ON CREATE SET af.path = $path
      ON MATCH  SET af.path = coalesce(af.path, $path)
    WITH af
    OPTIONAL MATCH (af)-[r:HAS_SEGMENT]->(s:AudioSegment)
    DELETE r, s
    """
    tx.run(q1, key=audiokey, path=path)

    if music:
        q2 = """
        MERGE (af:AudioFile {key:$key})
        WITH af
        UNWIND $segs AS seg
        CREATE (s:AudioSegment {kind:"music", start: seg.start, end: seg.end})
        MERGE (af)-[:HAS_SEGMENT]->(s)
        """
        tx.run(q2, key=audiokey, segs=[{"start": s, "end": e} for s, e in music])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-csv", help="Optional CSV with one audio_key per line to limit processing.")
    ap.add_argument("--push-neo4j", action="store_true", help="Also write segments into Neo4j.")
    ap.add_argument("--force", action="store_true", help="Recompute even if sidecar exists.")
    args = ap.parse_args()

    fs_index = build_fs_index(AUDIO_ROOTS)
    if not fs_index:
        print("No audio files found under AUDIO_ROOTS.", file=sys.stderr)

    keys = None
    if args.keys_csv and os.path.exists(args.keys_csv):
        with open(args.keys_csv) as f:
            keys = [ln.strip() for ln in f if ln.strip()]

    seg = Segmenter()

    todo: List[AudioRef] = []
    if keys:
        for k in keys:
            p = resolve_audio_path(k, fs_index)
            if p:
                todo.append(AudioRef(k, p))
            else:
                print(f"[WARN] Could not resolve audio_key={k} in filesystem index.")
    else:
        # Process everything we can see; key=stem
        for s, p in fs_index.items():
            todo.append(AudioRef(s, p))

    for ar in tqdm(todo, desc="music-seg"):
        sc = ar.path + SIDEcar_SUFFIX
        if (not args.force) and os.path.exists(sc):
            # already computed
            if args.push_neo4j:
                with open(sc) as f:
                    music = json.load(f).get("music", [])
                push_to_neo4j(ar.key, ar.path, music)
            continue

        try:
            music = segment_music(seg, ar.path)
            write_sidecar(ar.path, music)
            if args.push_neo4j:
                push_to_neo4j(ar.key, ar.path, music)
        except Exception as e:
            print(f"[ERROR] {ar.path}: {e}")

if __name__ == "__main__":
    main()
