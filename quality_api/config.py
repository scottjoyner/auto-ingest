"""Configuration helpers for the quality API package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
    database: str = os.getenv("NEO4J_DB", "neo4j")
    transcription_index_v1: str = os.getenv("IDX_TRANS_V1", "transcription_embedding_index")
    transcription_index_v2: str = os.getenv("IDX_TRANS_V2", "transcription_embedding_v2_index")
    segment_index: str = os.getenv("IDX_SEGMENT", "segment_embedding_index")
    utterance_index: str = os.getenv("IDX_UTTER", "utterance_embedding_index")


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = int(os.getenv("EMBED_DIM", "384"))
    batch_size: int = int(os.getenv("EMBED_BATCH", "32"))
    max_length: int = int(os.getenv("EMBED_MAX_LENGTH", "512"))


@dataclass(frozen=True)
class UiSettings:
    default_mode: str = os.getenv("QUALITY_UI_DEFAULT_MODE", "both")
    default_top_k: int = int(os.getenv("QUALITY_UI_DEFAULT_TOP_K", "10"))
    default_pool_size: int = int(os.getenv("QUALITY_UI_DEFAULT_POOL_SIZE", "200"))


@dataclass(frozen=True)
class MediaSettings:
    base_path: str = os.getenv("MEDIA_BASE", "/mnt/8TB_2025/fileserver/audio")


@dataclass(frozen=True)
class QualityValidationSettings:
    max_mph_change: float = float(os.getenv("QUALITY_MAX_MPH_CHANGE", "20"))
    max_lat_change: float = float(os.getenv("QUALITY_MAX_LAT_CHANGE", "1"))
    max_long_change: float = float(os.getenv("QUALITY_MAX_LONG_CHANGE", "1"))
    expected_frame_increment: float = float(os.getenv("QUALITY_EXPECT_FRAME_INCREMENT", "60"))
    enforce_mph: bool = _str_to_bool(os.getenv("QUALITY_ENFORCE_MPH"), False)
    enforce_frame_increment: bool = _str_to_bool(
        os.getenv("QUALITY_ENFORCE_FRAME_INCREMENT"), False
    )


@dataclass(frozen=True)
class Settings:
    debug: bool = _str_to_bool(os.getenv("QUALITY_DEBUG"), False)
    neo4j: Neo4jSettings = Neo4jSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    media: MediaSettings = MediaSettings()
    ui: UiSettings = UiSettings()
    validation: QualityValidationSettings = QualityValidationSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()

