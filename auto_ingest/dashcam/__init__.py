"""Dashcam subpackage: YOLO vehicle detection + embedding ingestion.

Imports are lazy, and that is a correctness fix rather than a nicety. `yolo_embeddings` pulls moviepy, cv2, ultralytics and sentence_transformers at
module scope, so importing it here made the whole subpackage unimportable on any
machine without a GPU image - including `media_pairing`, which is pure filesystem
logic and is the part the detector's own tests need. The two halves of this subpackage
do not share a dependency, and the package boundary was erasing that.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .yolo_embeddings import main as run_yolo_embeddings

__all__ = ["run_yolo_embeddings", "yolo_embeddings"]


def __getattr__(name: str) -> Any:
    if name in ("run_yolo_embeddings", "yolo_embeddings"):
        # importlib, not `from . import x`. The latter consults __getattr__ for the
        # name before the submodule is bound, so a handler that resolves it by
        # importing recurses until the stack gives out. import_module binds the
        # submodule on the package, so this runs once.
        module = importlib.import_module(f"{__name__}.yolo_embeddings")
        return module if name == "yolo_embeddings" else getattr(module, "main")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
