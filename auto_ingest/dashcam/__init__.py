"""Dashcam subpackage: YOLO vehicle detection + embedding ingestion.

Imports are lazy, and that is a correctness fix rather than a nicety.
``yolo_embeddings`` pulls moviepy, cv2, ultralytics and sentence_transformers at
module scope, so importing it here made the entire subpackage unimportable on any
machine without a GPU image - including ``media_pairing``, which is pure
filesystem logic and is the part the detector's own tests, and
``auto_ingest.custody.staging``, need. The two halves of this subpackage do not
share a dependency, and the package boundary was erasing that.

The names this module exported are unchanged; they resolve on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .yolo_embeddings import main as run_yolo_embeddings

__all__ = ["run_yolo_embeddings", "yolo_embeddings"]


def __getattr__(name: str) -> Any:
    if name in ("run_yolo_embeddings", "yolo_embeddings"):
        from . import yolo_embeddings as _mod

        if name == "yolo_embeddings":
            # Resolves to the submodule, which is what the eager import used to
            # bind here once the submodule had been imported.
            return _mod
        return _mod.main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))