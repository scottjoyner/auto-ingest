"""CLIP embedding + vector-index helpers for personal media recall.

Mirrors the CLIP embedder in ``ingest_media.py`` WITHOUT importing that module
(avoids pulling in its heavy pipeline deps). Both image and text embeddings land
in the SAME 512-dim CLIP space (open_clip ViT-B/32, laion2b_s34b_b79k), so a
text query embedding can be compared against stored image embeddings.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from auto_ingest.shorts.db_retry import with_driver

log = logging.getLogger("personal.embed")

CLIP_DIM = 512

# Module-global cache: None = not loaded, False = unavailable, else bundle.
_CLIP = None


def _register_heif() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        pass


def _clip_model():
    """Lazy-load open_clip ViT-B/32; cache in module global. Mirror ingest_media."""
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    try:
        import open_clip
        import torch  # noqa: F401 - imported to fail fast if unavailable
    except Exception as e:
        log.warning("open_clip unavailable, skipping embedding: %s", e)
        _CLIP = False
        return None
    log.info("loading CLIP ViT-B/32 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B/32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B/32")
    model.eval()
    _CLIP = (model, preprocess, tok)
    return _CLIP


def embed_image(path) -> Optional[List[float]]:
    """Return a 512-dim CLIP image embedding (raw, unnormalized) or None."""
    bundle = _clip_model()
    if not bundle:
        return None
    import torch
    from PIL import Image
    _register_heif()
    try:
        model, preprocess, _ = bundle
        with Image.open(Path(path)) as im:
            img = preprocess(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            vec = model.encode_image(img)
        return vec.cpu().numpy().flatten().tolist()
    except Exception as e:
        log.warning("clip image embed failed: %s", e)
        return None


def embed_text(text: str) -> Optional[List[float]]:
    """Return a 512-dim CLIP text embedding (raw, same space as images) or None."""
    bundle = _clip_model()
    if not bundle:
        return None
    import torch
    try:
        model, _, tok = bundle
        tokens = tok(text, context_length=77)
        with torch.no_grad():
            vec = model.encode_text(tokens)
        return vec.cpu().numpy().flatten().tolist()
    except Exception as e:
        log.warning("clip text embed failed: %s", e)
        return None


def ensure_media_indexes(driver) -> None:
    """Idempotently create vector + GPS indexes on :MediaFile. Non-fatal errors ignored."""
    db = os.getenv("NEO4J_DB", "neo4j")
    stmts = [
        (
            "CREATE VECTOR INDEX media_embedding_index IF NOT EXISTS "
            "FOR (n:MediaFile) ON (n.embedding) "
            "OPTIONS { indexConfig: { `vector.dimensions`: 512, "
            "`vector.similarity_function`: 'cosine' } }"
        ),
        (
            "CREATE INDEX media_file_gps IF NOT EXISTS "
            "FOR (m:MediaFile) ON (m.gps_lat, m.gps_lon)"
        ),
    ]
    with driver.session(database=db) as sess:
        for stmt in stmts:
            try:
                sess.run(stmt).consume()
            except Exception as e:
                log.info("ensure_media_indexes non-fatal: %s", e)


def ensure_media_indexes_with_retry() -> None:
    """Run :func:`ensure_media_indexes` via the resilient ``with_driver`` wrapper."""
    with_driver(ensure_media_indexes)
