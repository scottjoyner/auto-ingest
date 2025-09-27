"""Embedding utilities shared by the quality tooling."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import torch
from transformers import AutoTokenizer, AutoModel

from .config import settings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding.model_name)
    model = AutoModel.from_pretrained(settings.embedding.model_name).to(DEVICE).eval()
    return tokenizer, model


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Return sentence embeddings for *texts* as python lists."""
    texts = [t for t in texts if t]
    if not texts:
        return []

    tokenizer, model = _load_model()
    vectors: List[List[float]] = []

    with torch.no_grad():
        for i in range(0, len(texts), settings.embedding.batch_size):
            batch = texts[i : i + settings.embedding.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.embedding.max_length,
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = _mean_pooling(output.last_hidden_state, encoded["attention_mask"])
            pooled = _normalize(pooled)
            vectors.extend(pooled.cpu().numpy().tolist())

    return vectors

