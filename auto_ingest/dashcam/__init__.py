"""Dashcam subpackage: YOLO vehicle detection + embedding ingestion."""

from __future__ import annotations

from .yolo_embeddings import main as run_yolo_embeddings

__all__ = ["run_yolo_embeddings", "yolo_embeddings"]
