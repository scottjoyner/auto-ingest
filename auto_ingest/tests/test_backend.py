"""
Tests for auto_ingest.backend.

Covers detect_backend / torch_device / backend_info / faiss_backend /
prefer_onnx across all four simulated backends (cuda, rocm, mlx, onnx),
caching stability, and the import-guard fallback path.
"""
from __future__ import annotations

import importlib
import platform
import shutil
from unittest import mock

import pytest

import auto_ingest.backend as backend


def _reload_clear():
    """Clear caches and reload the module fresh."""
    backend._BACKEND = None
    backend._FAISS_BACKEND = None
    backend._FAISS_AVAILABLE = None
    backend._FAISS_GPU = None
    importlib.reload(backend)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    # Always restore a clean state.
    _reload_clear()


# ---------------------------------------------------------------------------
# Real host (CPU / onnx on this machine)
# ---------------------------------------------------------------------------
def test_real_host_detect():
    _reload_clear()
    b = backend.detect_backend()
    assert b in {"cuda", "rocm", "mlx", "onnx"}


def test_real_host_recommended_torch_device():
    _reload_clear()
    dev = backend.recommended_torch_device()
    # Must be a concrete, valid torch device string — never None/empty.
    assert dev in {"cuda", "mps", "cpu"}


def test_real_host_device_mapping():
    _reload_clear()
    dev = backend.torch_device()
    backend_name = backend.detect_backend()
    expected = {
        "cuda": "cuda",
        "rocm": "cuda",
        "mlx": "mps",
        "onnx": "cpu",
    }[backend_name]
    assert dev == expected


def test_real_host_backend_info_keys():
    _reload_clear()
    info = backend.backend_info()
    for key in (
        "backend", "torch_device", "cuda_available", "rocm_available",
        "mps_available", "mlx_available", "onnx_available",
        "faiss_available", "faiss_gpu", "gpu_name",
    ):
        assert key in info
    assert info["backend"] == backend.detect_backend()
    assert info["torch_device"] == backend.torch_device()


def test_real_host_faiss_backend():
    _reload_clear()
    fb = backend.faiss_backend()
    assert fb in {"cpu", "gpu"}


def test_prefer_onnx_real():
    _reload_clear()
    # force=True always returns True
    assert backend.prefer_onnx(force=True) is True
    # non-force reflects detection
    assert backend.prefer_onnx() == (backend.detect_backend() == "onnx")


# ---------------------------------------------------------------------------
# Simulated: CUDA
# ---------------------------------------------------------------------------
def test_simulated_cuda():
    _reload_clear()
    with mock.patch.object(shutil, "which", return_value="/usr/bin/nvidia-smi"), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_name", return_value="Fake GPU"):
        importlib.reload(backend)
        assert backend.detect_backend() == "cuda"
        assert backend.torch_device() == "cuda"
        assert backend.recommended_torch_device() == "cuda"
        info = backend.backend_info()
        assert info["backend"] == "cuda"
        assert info["cuda_available"] is True
        assert info["gpu_name"] == "Fake GPU"
        assert backend.prefer_onnx() is False


# ---------------------------------------------------------------------------
# Simulated: ROCm
# ---------------------------------------------------------------------------
def test_simulated_rocm():
    _reload_clear()
    # nvidia-smi absent, rocminfo present, torch.cuda available (ROCm).
    def which_side(cmd):
        return "/usr/bin/rocminfo" if cmd == "rocminfo" else None

    with mock.patch.object(shutil, "which", side_effect=which_side), \
         mock.patch("torch.cuda.is_available", return_value=True):
        importlib.reload(backend)
        assert backend.detect_backend() == "rocm"
        assert backend.torch_device() == "cuda"
        assert backend.recommended_torch_device() == "cuda"
        info = backend.backend_info()
        assert info["backend"] == "rocm"
        assert info["rocm_available"] is True
        assert backend.prefer_onnx() is False


# ---------------------------------------------------------------------------
# Simulated: MLX (Apple Silicon)
# ---------------------------------------------------------------------------
def test_simulated_mlx():
    _reload_clear()
    with mock.patch.object(platform, "system", return_value="Darwin"), \
         mock.patch.object(platform, "processor", return_value="arm"), \
         mock.patch("torch.backends.mps.is_available", return_value=True), \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch.dict("sys.modules", {"mlx": object()}):
        importlib.reload(backend)
        assert backend.detect_backend() == "mlx"
        assert backend.torch_device() == "mps"
        assert backend.recommended_torch_device() == "mps"
        info = backend.backend_info()
        assert info["backend"] == "mlx"
        assert info["mlx_available"] is True
        assert info["mps_available"] is True
        assert backend.prefer_onnx() is False


# ---------------------------------------------------------------------------
# Simulated: ONNX (no ML stack)
# ---------------------------------------------------------------------------
def test_simulated_onnx():
    _reload_clear()
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def block_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(shutil, "which", return_value=None), \
         mock.patch.dict("sys.modules", {"torch": None}), \
         mock.patch("builtins.__import__", side_effect=block_torch):
        importlib.reload(backend)
        assert backend.detect_backend() == "onnx"
        assert backend.torch_device() == "cpu"
        assert backend.recommended_torch_device() == "cpu"
        info = backend.backend_info()
        assert info["backend"] == "onnx"
        assert info["onnx_available"] is True
        assert info["cuda_available"] is False
        assert info["mlx_available"] is False
        assert backend.prefer_onnx() is True


# ---------------------------------------------------------------------------
# Caching stability
# ---------------------------------------------------------------------------
def test_caching_stable():
    _reload_clear()
    with mock.patch.object(shutil, "which", return_value="/usr/bin/nvidia-smi"), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_name", return_value="Fake GPU"):
        importlib.reload(backend)
        first = backend.detect_backend()
        # Force torch.cuda.is_available to now return False; cached value wins.
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert backend.detect_backend() == first
            assert backend.backend_info()["backend"] == first
            assert backend.torch_device() == backend.torch_device()


# ---------------------------------------------------------------------------
# Import-guard path: missing optional lib (faiss) should not crash
# ---------------------------------------------------------------------------
def test_faiss_import_guard():
    _reload_clear()
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def guarded(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss missing")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=guarded):
        importlib.reload(backend)
        # faiss_backend must not raise; returns "cpu" and marks unavailable.
        assert backend.faiss_backend() == "cpu"
        info = backend.backend_info()
        assert info["faiss_available"] is False
        assert info["faiss_gpu"] is False
