"""
auto_ingest.backend — runtime compute-backend detection (single source of truth).

Backend contract
================
This module probes the host once (lazily, on first use) and exposes the best
available ML compute backend without any machine-specific hardcoding.

Public API
----------
- ``detect_backend() -> str``        One of "cuda", "rocm", "mlx", "onnx".
- ``torch_device() -> str``         Maps the backend to a torch device string
                                    ("cuda", "cuda", "mps", "cpu"). Use this
                                    wherever a torch model is placed with
                                    ``model.to(...)`` / ``tensor.to(...)``.
- ``recommended_torch_device() -> str`` Same concrete device string as
                                    ``torch_device()`` but named for call
                                    sites that ask "which device should I use"
                                    (e.g. the speaker linker). Always one of
                                    "cuda"/"mps"/"cpu", never None.
- ``backend_info() -> dict``        Structured profile: backend, torch_device,
                                    and availability flags for each backend plus
                                    the GPU name (when known).
- ``prefer_onnx() -> bool``         True when the chosen backend is "onnx"
                                    (CPU) — i.e. ONNX Runtime / CPU providers
                                    should be preferred for whisper/pyannote
                                    style workloads that support an explicit
                                    provider selection.

Resolution order
----------------
1. Apple Silicon (Darwin + arm) with MPS available  -> "mlx"
   (MLX is Apple's framework; torch's MPS backend is the path we actually use,
   so we probe for the ``mlx`` package but label the backend "mlx" either way.)
2. NVIDIA CUDA (nvidia-smi present AND torch.cuda)  -> "cuda"
3. AMD ROCm   (rocminfo present AND torch.cuda, which ROCm exposes) -> "rocm"
4. Otherwise                                               -> "onnx" (CPU fallback)

Every heavy import (torch, mlx, and the CLI probes for nvidia-smi/rocminfo)
is guarded so this module imports cleanly even when the ML stack is absent
(CI/tests). Detection runs once and is cached at module level.
"""
from __future__ import annotations

import platform
import shutil
from typing import Dict

# ---------------------------------------------------------------------------
# Cached detection result
# ---------------------------------------------------------------------------
_BACKEND: str | None = None
_FAISS_BACKEND: str | None = None
_FAISS_AVAILABLE: bool | None = None
_FAISS_GPU: bool | None = None


def detect_backend() -> str:
    """Return the best available compute backend: "cuda" | "rocm" | "mlx" | "onnx"."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    system = platform.system()
    processor = platform.processor()

    # Import torch defensively; absence short-circuits us to the CPU/onnx path.
    torch = None
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    # 1) Apple Silicon (MLX / MPS)
    if system == "Darwin" and processor == "arm":
        mps_ok = bool(torch is not None and getattr(torch.backends, "mps", None)
                      and torch.backends.mps.is_available())
        if mps_ok:
            try:
                import mlx  # noqa: F401  (MLX lib present? informational only)
            except Exception:
                pass
            _BACKEND = "mlx"
            return _BACKEND

    # 2) NVIDIA CUDA
    nvidia = shutil.which("nvidia-smi")
    if nvidia and torch is not None and torch.cuda.is_available():
        _BACKEND = "cuda"
        return _BACKEND

    # 3) AMD ROCm (exposes the CUDA API via torch.cuda)
    rocm = shutil.which("rocminfo")
    if rocm and torch is not None and torch.cuda.is_available():
        _BACKEND = "rocm"
        return _BACKEND

    # 4) Fallback: ONNX / CPU
    _BACKEND = "onnx"
    return _BACKEND


def faiss_backend() -> str:
    """Return "gpu" if faiss is importable and reports >=1 GPU, else "cpu"."""
    global _FAISS_BACKEND, _FAISS_AVAILABLE, _FAISS_GPU
    if _FAISS_BACKEND is not None:
        return _FAISS_BACKEND

    faiss = None
    try:
        faiss = __import__("faiss")
        _FAISS_AVAILABLE = True
    except Exception:
        faiss = None
        _FAISS_AVAILABLE = False
        _FAISS_GPU = False
        _FAISS_BACKEND = "cpu"
        return _FAISS_BACKEND

    # faiss is present; probe for GPU support.
    gpu = False
    try:
        gpu = bool(faiss.get_num_gpus() > 0)  # type: ignore
    except Exception:
        gpu = False
    _FAISS_GPU = gpu
    _FAISS_BACKEND = "gpu" if gpu else "cpu"
    return _FAISS_BACKEND


def torch_device() -> str:
    """Map the detected backend to a torch device string."""
    backend = detect_backend()
    return {
        "cuda": "cuda",
        "rocm": "cuda",   # ROCm is driven through torch's CUDA API
        "mlx": "mps",     # Apple Silicon -> Metal Performance Shaders
        "onnx": "cpu",
    }.get(backend, "cpu")


def recommended_torch_device() -> str:
    """Return the torch device to place models on: 'cuda' | 'mps' | 'cpu'.

    Thin, descriptive alias over ``torch_device()`` for call sites (e.g. the
    speaker linker) that want an explicit "what device should I use" question.
    Always resolves to a concrete, valid torch device string with CPU as the
    safe fallback when no accelerator is present.
    """
    dev = torch_device()
    return dev if dev in ("cuda", "mps", "cpu") else "cpu"


def backend_info() -> Dict:
    """Structured profile of the detected backend and its capabilities."""
    backend = detect_backend()
    device = torch_device()

    torch = None
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    rocm_available = bool(
        shutil.which("rocminfo") and cuda_available
    )
    mps_available = bool(
        torch is not None and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    mlx_available = (backend == "mlx")
    onnx_available = (backend == "onnx")

    # faiss_backend() populates the cache on first call; read it back here.
    faiss_available = bool(_FAISS_AVAILABLE)
    faiss_gpu = bool(_FAISS_GPU)

    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)  # type: ignore
        except Exception:
            gpu_name = None

    return {
        "backend": backend,
        "torch_device": device,
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "mps_available": mps_available,
        "mlx_available": mlx_available,
        "onnx_available": onnx_available,
        "faiss_available": faiss_available,
        "faiss_gpu": faiss_gpu,
        "gpu_name": gpu_name,
    }


def prefer_onnx(*, force: bool = False) -> bool:
    """
    True when the CPU/ONNX fallback backend is in use.

    Use this to decide whether whisper/pyannote style workloads should select
    an explicit ONNX Runtime CPU provider. ``force`` lets a caller override to
    always prefer ONNX providers (e.g. for reproducible CI runs).
    """
    if force:
        return True
    return detect_backend() == "onnx"
