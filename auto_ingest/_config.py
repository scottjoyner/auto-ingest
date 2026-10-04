"""
auto_ingest._config - internal shim exposing the repo-root ``auto_ingest_config``.

The canonical config module ``auto_ingest_config.py`` lives at the repo root and
is imported by the flat scripts (and by ``bin/auto-ingest`` which runs with the
repo root on ``sys.path``). This shim lets the packaged modules import the same
config regardless of cwd by re-exporting the root module when available, or
falling back to a minimal inline implementation so the package stays importable
in tests/CI without the full ML stack.
"""

from __future__ import annotations

try:  # preferred: repo-root module (full implementation)
    from auto_ingest_config import (  # type: ignore
        get_fileserver_path,
        get_fileserver_root,
        get_hot_root,
        get_neo4j_config,
        get_nextcloud_root,
        get_nextcloud_webdav,
        get_shared_path,
        get_shared_root,
        get_storage_layout,
    )
except Exception:  # pragma: no cover - CI / out-of-repo import fallback
    import os

    def get_fileserver_root() -> str:
        root = os.environ.get("FILESERVER_ROOT")
        if not root:
            raise RuntimeError("FILESERVER_ROOT is not set")
        return root

    def get_fileserver_path(*parts: str) -> str:
        import os as _os

        return _os.path.join(get_fileserver_root(), *parts)

    def get_shared_root() -> str:
        root = os.environ.get("NAS_ROOT")
        if not root:
            raise RuntimeError("Shared NAS root is not configured (set NAS_ROOT)")
        return root

    def get_shared_path(*parts: str) -> str:
        import os as _os

        root = _os.path.realpath(get_shared_root())
        candidate = _os.path.realpath(_os.path.join(root, *parts))
        if _os.path.commonpath([root, candidate]) != root:
            raise ValueError(f"shared path escapes NAS root: {parts!r}")
        return candidate

    def get_storage_layout() -> dict:
        return {
            "shared_root": os.environ.get("NAS_ROOT", ""),
            "hot_root": os.environ.get("HOT_STORAGE_ROOT", ""),
            "cold_root": os.environ.get("COLD_STORAGE_ROOT") or os.environ.get("NAS_ROOT", ""),
            "fileserver_root": os.environ.get("FILESERVER_ROOT", ""),
        }

    def get_hot_root() -> str:
        root = get_storage_layout()["hot_root"]
        if not root:
            raise RuntimeError("HOT_STORAGE_ROOT is not set")
        return root

    def get_neo4j_config() -> dict:
        return {
            "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.environ.get("NEO4J_USER", "neo4j"),
            "password": os.environ.get("NEO4J_PASSWORD", ""),
        }

    def get_nextcloud_root() -> str:
        return os.environ.get("NEXTCLOUD_ROOT", "/media/scott/SSD_4TB/nextcloud")

    def get_nextcloud_webdav():
        return (None, None, None)
