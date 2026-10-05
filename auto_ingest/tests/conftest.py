"""Test-environment shims for `auto_ingest/tests/`.

This directory is run by CI (`python -m pytest auto_ingest/tests/ -q`), which
deliberately installs no ML stack — the comment in `.github/workflows/ci.yml`
says "lightweight test deps (no ML stack)" so the suite runs without
torch/transformers/ultralytics. That is the right call for cost, but it means
these tests must not *require* the stack either.

Four tests in `test_backend.py` were failing here for exactly that reason, and
they were invisible because CI only ran `tests/`. They use
`mock.patch("torch.cuda.is_available")`, and `mock.patch` resolves its target by
importing `torch` — which raises `ModuleNotFoundError` when the stack is absent.
The tests were not wrong; the environment was.

The fix mirrors the convention `test_ml_pure.py` already established with
`_mock_import_module`: a `MagicMock` in `sys.modules` covers every attribute
access. A stub is installed ONLY when torch is genuinely missing, so a developer
or CI job that does have the real stack still exercises the real thing.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

#: Dotted paths that must resolve to the SAME objects the stubbed root exposes.
#: `mock.patch("torch.cuda.get_device_name")` resolves the dotted path by
#: importing "torch" and walking attributes; `auto_ingest.backend` reads
#: `torch.cuda.<attr>` the same way. If sys.modules["torch.cuda"] were a
#: different mock from `sys.modules["torch"].cuda`, the test would patch one
#: object and the code would read the other - and the test would fail with a
#: MagicMock leaking into an assertion. That is exactly what happened first.
_TORCH_STUBS = (
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "torch.nn",
    "torch.nn.functional",
    "torch.fx",
)

#: Child attribute on the stub root -> dotted path that must be the same object.
_TORCH_CHILDREN = {
    "cuda": "torch.cuda",
    "backends": "torch.backends",
    "nn": "torch.nn",
    "fx": "torch.fx",
}


def _torch_importable() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(autouse=True)
def stub_torch_for_patch_targets():
    """Let `mock.patch("torch.…")` resolve when torch is not installed.

    Scoped to this directory and opt-out-able: a test that needs real torch can
    request it, and any test that wants the un-stubbed import failure can mark
    itself. Restored afterwards so a stub never leaks between test modules.
    """
    if _torch_importable():
        yield  # real stack present: nothing to do, and nothing is masked
        return

    saved = {name: sys.modules.get(name) for name in _TORCH_STUBS}

    nodes = {name: mock.MagicMock(name=name) for name in _TORCH_STUBS}
    root = nodes["torch"]
    # One tree: `torch.cuda` must be the same object whether reached by
    # attribute access or found in sys.modules.
    for attr, dotted in _TORCH_CHILDREN.items():
        setattr(root, attr, nodes[dotted])
    setattr(nodes["torch.backends"], "mps", nodes["torch.backends.mps"])
    setattr(nodes["torch.nn"], "functional", nodes["torch.nn.functional"])

    for name, node in nodes.items():
        sys.modules[name] = node
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def real_torch_required():
    """Skip rather than silently pass when a test needs the actual ML stack."""
    if not _torch_importable():
        pytest.skip("requires a real torch install")
