"""Hermetic regression tests for backend_info()'s faiss capability flags.

``backend_info()`` is documented as a "structured profile ... availability
flags for each backend".  Its faiss flags must describe the host, not whatever
happens to be in the module-level probe caches at call time.  On a fresh import
those caches are ``None``, and ``bool(None)`` silently reports "no faiss / no
faiss GPU" even when a GPU-capable faiss is installed -- until some unrelated
caller happens to invoke ``faiss_backend()`` first.

No real faiss, torch, GPU or filesystem state is used: a tiny in-memory
stand-in is injected into ``sys.modules`` and the caches are reset to their
fresh-import values.
"""
from __future__ import annotations

import importlib
import sys

import pytest

import auto_ingest.backend as backend


class _FakeFaiss:
    """Minimal stand-in for the only faiss call backend.py makes."""

    def __init__(self, n_gpus: int) -> None:
        self._n_gpus = n_gpus

    def get_num_gpus(self) -> int:
        return self._n_gpus


@pytest.fixture
def fresh_backend(monkeypatch):
    """A backend module in its pristine, never-probed state (faiss has 2 GPUs)."""
    fake = _FakeFaiss(2)
    monkeypatch.setitem(sys.modules, "faiss", fake)
    backend._BACKEND = None
    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    yield backend
    backend._BACKEND = None
    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    importlib.reload(backend)


def test_fixture_is_truly_fresh(fresh_backend):
    """Guard the premise: no faiss probe has run before backend_info()."""
    assert fresh_backend._FAISS_BACKEND is None
    assert fresh_backend._FAISS_AVAILABLE is None
    assert fresh_backend._FAISS_GPU is None


def test_backend_info_reports_faiss_gpu_on_fresh_module(fresh_backend):
    """First call to backend_info() must not report a phantom CPU-only host."""
    info = fresh_backend.backend_info()
    assert info["faiss_available"] is True
    assert info["faiss_gpu"] is True


def test_backend_info_populates_faiss_cache_on_fresh_module(fresh_backend):
    """Reading the profile must perform the probe, not read uninitialised state."""
    fresh_backend.backend_info()
    assert fresh_backend._FAISS_BACKEND == "gpu"


def test_backend_info_reports_faiss_cpu_when_no_gpu(monkeypatch):
    """A present-but-CPU-only faiss must be reported as available, non-GPU."""
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss(0))
    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    try:
        info = backend.backend_info()
        assert info["faiss_available"] is True
        assert info["faiss_gpu"] is False
    finally:
        backend._FAISS_BACKEND = None
        backend._FAISS_AVAILABLE = None
        backend._FAISS_GPU = None
        importlib.reload(backend)


def test_backend_info_reports_faiss_missing(monkeypatch):
    """Preserve the existing fail-safe behaviour when faiss cannot be imported."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def no_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss missing")
        return real_import(name, *args, **kwargs)

    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    with monkeypatch.context() as m:
        m.setattr("builtins.__import__", no_faiss)
        info = backend.backend_info()
    assert info["faiss_available"] is False
    assert info["faiss_gpu"] is False
    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    importlib.reload(backend)
