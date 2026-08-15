"""
auto_ingest.embed — single source of truth for dense text embeddings.

Centralizes the model load + *pooling math* used by the ingest path, the
re-embed CLI (`reembed.py`), and the real-time search server. Every consumer
must use the SAME pooling/normalization so vectors written during re-embed
remain directly comparable to those written during normal ingest; otherwise
retrieval recall silently diverges.

Pooling is intentionally identical to auto_ingest.ingest.transcripts
(mean-pool over the last hidden state, then L2-normalize).

Backend
-------
Model is env-driven (`EMBED_MODEL_NAME`, default the canonical MiniLM-L6)
and the torch device resolves via auto_ingest.backend.torch_device(), so the
same code path runs CPU on deathstar and ROCm (exposing the CUDA API) on x1-370
once it is back online.

Model lifecycle
---------------
- `embed_texts(texts, batch_size, max_length)` — fire-and-forget; loads the
  default model lazily and caches it process-wide.
- `EmbedModel(name)` — explicit model handle for A/B testing several models in
  one process (each keeps its own tokenizer/weights).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from .backend import torch_device

# Canonical default — mirrors the ingest path default in auto_ingest_config.
DEFAULT_MODEL = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Cache the default model so the ingest path and ad-hoc callers share it.
_DEFAULT: Optional["EmbedModel"] = None


def _resolve_device() -> str:
    dev = torch_device()
    return dev if dev in ("cuda", "mps", "cpu") else "cpu"


# --- HNSW index tuning (env-configurable) ---
# Neo4j 5.x vector indexes are HNSW. The original indexes were built with the
# provider defaults (m=16, ef_construction=100), which trade recall for speed.
# Raising these improves ANN precision at query time at the cost of build time
# and memory — worth it for the retrieval "re-take". Exposed as env vars so
# reembed.py and the ingest path build indexes identically.
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF = int(os.getenv("HNSW_EF", "200"))
HNSW_QUANT = os.getenv("HNSW_QUANT", "true") == "true"


class EmbedModel:
    """Wrapper around a transformers AutoModel with the shared pooling."""

    def __init__(self, name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.name = name
        self.device = device or _resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        # hidden_size is the ground-truth embedding dimension for the model;
        # prefer it over any hardcoded table so new models "just work".
        cfg = AutoConfig.from_pretrained(name)
        self.dim = int(getattr(cfg, "hidden_size", 0) or 384)

    def embed(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[List[float]]:
        """Mean-pool + L2-normalize. Returns dim-`self.dim` vectors, same math as ingest."""
        if not texts:
            return []
        vectors: List[List[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().numpy().tolist())
        return vectors


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over tokens — byte-for-byte identical to transcripts.mean_pooling."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def get_default_model() -> EmbedModel:
    """Singleton default EmbedModel (env-driven name + engine)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_embed_model(DEFAULT_MODEL)
    return _DEFAULT


def load_embed_model(name: str, engine: str = "torch") -> EmbedModel:
    """Build an EmbedModel for `name`, dispatching on engine.

    - "torch" (default): transformers AutoModel on torch_device().
    - "onnx": same weights exported once to ONNX and run via onnxruntime CPU.
      Optional dynamic int8 weight quantization (ONNX_QUANTIZE=1) for ~3-5x CPU
      throughput at a small recall cost.

    NOTE: engine is a *data-affecting* choice. Keep one engine per vector
    property (e.g. do not mix torch emb_gte_small and onnx emb_gte_small) or
    retrieval recall silently diverges.
    """
    if engine == "onnx":
        return OrtEmbedModel(name)
    return EmbedModel(name)


def embed_texts(texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[List[float]]:
    """Embed texts with the default model; lazy singleton load."""
    return get_default_model().embed(texts, batch_size=batch_size, max_length=max_length)


# ---------------------------------------------------------------------------
# ONNX engine
# ---------------------------------------------------------------------------
# Model weights are exported to ONNX once (cached under ONNX_CACHE_DIR) and run
# with onnxruntime's CPUExecutionProvider. Pooling math is identical to the torch
# path (_mean_pool + L2 normalize), so vectors are directly comparable to torch
# embeddings from the same weights (cosine > 0.999 in practice; int8 quantization
# trades a little recall for several-x CPU throughput).
ONNX_CACHE_DIR = Path(os.getenv("ONNX_CACHE_DIR", "~/.cache/auto_ingest_onnx")).expanduser()
ONNX_QUANTIZE = os.getenv("ONNX_QUANTIZE", "0") == "1"
ONNX_OPSET = int(os.getenv("ONNX_OPSET", "14"))


class OrtEmbedModel(EmbedModel):
    """EmbedModel backed by an onnxruntime CPU session (same weights + pooling)."""

    def __init__(self, name: str = DEFAULT_MODEL):
        import onnxruntime as ort

        self.name = name
        self.device = "cpu"
        cfg = AutoConfig.from_pretrained(name)
        self.dim = int(getattr(cfg, "hidden_size", 0) or 384)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model_dir = ONNX_CACHE_DIR / name.replace("/", "--")
        model_dir.mkdir(parents=True, exist_ok=True)
        base = model_dir / "model.onnx"
        quant = model_dir / "model_quant.onnx" if ONNX_QUANTIZE else None
        target = quant if (quant and quant.exists()) else base
        if not target.exists():
            self._export_onnx(base, quant)
            target = quant if (quant and quant.exists()) else base
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0")) or 0
        self.sess = ort.InferenceSession(
            str(target), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def _export_onnx(self, base: Path, quant: Optional[Path]):
        """Export the HF model to ONNX with dynamic batch/sequence axes."""
        import onnxruntime as ort
        import inspect

        model = AutoModel.from_pretrained(self.name).eval()
        dummy = self.tokenizer(["warmup sentence"], return_tensors="pt")
        # Order positional args by the model's forward signature, NOT the tokenizer
        # dict order (BERT forward is (input_ids, attention_mask, token_type_ids),
        # tokenizer returns (input_ids, token_type_ids, attention_mask)).
        fwd_params = list(inspect.signature(model.forward).parameters)
        input_names = [p for p in fwd_params if p in dummy]
        args = tuple(dummy[p] for p in input_names)
        dynamic = {n: {0: "batch", 1: "seq"} for n in input_names}
        dynamic["last_hidden_state"] = {0: "batch", 1: "seq"}
        with torch.no_grad():
            torch.onnx.export(
                model,
                args,
                str(base),
                input_names=input_names,
                output_names=["last_hidden_state"],
                dynamic_axes=dynamic,
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                # Torch 2.6+ defaults to the dynamo-based exporter, which can emit
                # a wrong graph for some HF models; the legacy exporter is reliable.
                dynamo=False,
            )
        if quant is not None:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(str(base), str(quant), weight_type=QuantType.QInt8)

    def embed(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="np"
            )
            feeds = {k: v for k, v in enc.items()}
            last_hidden = self.sess.run(["last_hidden_state"], feeds)[0]  # (B, T, H)
            mask = enc["attention_mask"][:, :, None].astype(np.float32)
            pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            vectors.extend(pooled.tolist())
        return vectors
