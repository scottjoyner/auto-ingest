"""
auto_ingest_config - Machine-agnostic configuration for auto-ingest scripts.

Provides fileserver root path and Neo4j connection settings.
Auto-detects the current machine based on hostname, but allows override via
the FILESERVER_ROOT environment variable.

Usage:
    from auto_ingest_config import get_fileserver_root, get_neo4j_config
    root = get_fileserver_root()
    neo4j_cfg = get_neo4j_config()
"""

import os
import socket
from pathlib import Path

import yaml


def _resolve_env(value, default=None):
    """Resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        return os.environ.get(value[2:-1], default)
    return value if value not in (None, '') else default


def _find_config_path():
    """Find config.yaml relative to this module or in CWD."""
    candidates = [
        Path(__file__).parent / 'config.yaml',
        Path.cwd() / 'config.yaml',
        Path.home() / 'git' / 'auto-ingest' / 'config.yaml',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_config():
    """Load the shared auto-ingest config if available."""
    config_path = _find_config_path()
    if not config_path:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_storage_layout():
    """Return the current hot/cold storage topology.

    The layout is sourced from environment variables first, then config.yaml,
    then sensible machine-local fallbacks.
    """
    cfg = _load_config()
    knowledge_map = cfg.get('knowledge_map', {})
    machine_paths = cfg.get('machine_paths', {})
    hostname = socket.gethostname()

    matched_machine = None
    for vals in machine_paths.values():
        if vals.get('hostname_pattern') and vals['hostname_pattern'] in hostname:
            matched_machine = vals
            break
    if matched_machine is None and machine_paths:
        # Try 'any' first (shared mounts), then fall back to first entry
        if 'any' in machine_paths:
            matched_machine = machine_paths['any']
        else:
            matched_machine = list(machine_paths.values())[0]

    return {
        'hot_root': os.environ.get('HOT_STORAGE_ROOT')
        or (matched_machine or {}).get('hot_ssd_root')
        or knowledge_map.get('hot_layer_path')
        or '/media/scott/SSD_4TB/audio',
        'cold_root': os.environ.get('COLD_STORAGE_ROOT')
        or (matched_machine or {}).get('nas_mirror_root')
        or knowledge_map.get('mirror_vault_path')
        or '/media/scott/SSD_4TB/fileserver',
        'fileserver_root': os.environ.get('FILESERVER_ROOT')
        or (matched_machine or {}).get('fileserver_root')
        or '/media/scott/SSD_4TB/fileserver',
    }


def get_hot_root():
    """Get the SSD hot layer root path."""
    return get_storage_layout()['hot_root']


def get_fileserver_root():
    """Get the fileserver root path for the current machine.

    Priority:
    1. FILESERVER_ROOT environment variable
    2. Auto-detected from config.yaml based on hostname
    3. Fallback to x1-370 default
    """
    return get_storage_layout()['fileserver_root']


def get_neo4j_config():
    """Get Neo4j connection settings for the current machine.

    Priority for each field: environment variable -> config.yaml (machine match,
    then first entry) -> built-in default. Env always wins so a single export
    can retarget every script regardless of the committed config.
    """
    cfg_uri = cfg_user = cfg_pass = None

    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        hostname = socket.gethostname()
        matched = None
        for vals in cfg.get('machine_paths', {}).values():
            if vals.get('hostname_pattern') and vals['hostname_pattern'] in hostname:
                matched = vals
                break
        if matched is None:
            vals_list = list(cfg.get('machine_paths', {}).values())
            matched = vals_list[0] if vals_list else None

        if matched:
            cfg_uri = matched.get('neo4j_uri')
            cfg_user = matched.get('neo4j_user')
            cfg_pass = _resolve_env(
                matched.get('neo4j_password'), os.environ.get('NEO4J_PASSWORD', '')
            )

    return {
        'uri': os.environ.get('NEO4J_URI') or cfg_uri or 'bolt://100.64.43.123:7687',
        'user': os.environ.get('NEO4J_USER') or cfg_user or 'neo4j',
        'password': os.environ.get('NEO4J_PASSWORD') or cfg_pass or '',
    }


def get_neo4j_db():
    """Return the target Neo4j database name (default ``neo4j``)."""
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        db = cfg.get("neo4j_db")
        if db:
            return db
    return os.environ.get("NEO4J_DB", "neo4j")


def get_neo4j_env():
    """Return the full (uri, user, password, db) Neo4j tuple.

    Single source of truth for every ingest script (W-45). Replaces the ~30
    inline ``NEO4J_URI/USER/PASSWORD/DB`` constant blocks that hardcoded the
    ``knowledge_graph_2026`` password fallback. Prefer this over re-declaring
    the four module-level globals.
    """
    cfg = get_neo4j_config()
    return (
        cfg.get("uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
        cfg.get("user", os.environ.get("NEO4J_USER", "neo4j")),
        cfg.get("password", os.environ.get("NEO4J_PASSWORD", "")),
        get_neo4j_db(),
    )


def get_nextcloud_root():
    """Path to the Nextcloud store (true iPhone images/videos/audio).

    The Nextcloud data is on a separate store and is only mounted on some
    machines. Priority: NEXTCLOUD_ROOT env -> config.yaml 'nextcloud_root' ->
    sensible default mount point. Callers must handle the directory being
    absent (graceful skip).
    """
    cfg = _load_config()
    root = (os.environ.get("NEXTCLOUD_ROOT")
           or cfg.get("nextcloud_root")
           or "/media/scott/SSD_4TB/nextcloud")
    return root


def get_nextcloud_webdav():
    """Return Nextcloud WebDAV connection as (url, user, password).

    url/user come from config.yaml 'nextcloud:' (or NEXTCLOUD_URL /
    NEXTCLOUD_USER env); password from NEXTCLOUD_PASS env. Returns
    (None, None, None) if not configured.
    """
    cfg = _load_config()
    nc = cfg.get("nextcloud", {}) if isinstance(cfg, dict) else {}
    url = os.environ.get("NEXTCLOUD_URL") or nc.get("webdav_url")
    user = os.environ.get("NEXTCLOUD_USER") or nc.get("user")
    pass_env = nc.get("pass_env", "NEXTCLOUD_PASS")
    password = os.environ.get("NEXTCLOUD_PASS") or os.environ.get(pass_env)
    if not url or not user:
        return (None, None, None)
    return (url.rstrip("/"), user, password)


def build_artifact_ref(path: str, storage_root: str = "local-ssd",
                       retention_class: str = "keep",
                       tailscale_host_hint: str | None = None) -> dict:
    """Build a UNIFICATION §12.3 artifact provenance reference for a sidecar/artifact.

    Returns keys: storage_root, relative_path, host_path, container_path,
    tailscale_host_hint, sha256 (None; caller fills), retention_class.
    Dependency-light: only uses stdlib + this module's own path resolvers.
    """
    p = os.path.abspath(os.path.expanduser(str(path)))
    rel = p
    container = None
    # Relative to known roots if under them.
    for root in (get_fileserver_root(), get_storage_layout()["hot_root"],
                 get_storage_layout()["cold_root"]):
        rp = os.path.relpath(p, root)
        if not rp.startswith(".."):
            rel = rp
            container = "/nas/" + os.path.relpath(p, "/").lstrip("/")
            break
    return {
        "storage_root": storage_root,
        "relative_path": rel,
        "host_path": p,
        "container_path": container,
        "tailscale_host_hint": tailscale_host_hint,
        "sha256": None,
        "retention_class": retention_class,
    }


def get_fileserver_path(*parts):
    """Get a full fileserver path for the given sub-paths."""
    root = get_fileserver_root()
    return os.path.join(root, *parts)


def get_neo4j_driver(database=None):
    """Return a connected Neo4j driver using get_neo4j_config().

    This is the single import point for leaf scripts that need a live
    connection. Credentials are resolved from config.yaml / env
    (never hardcoded here). If neo4j is not importable, returns None
    so standalone scripts can fall back to their own argparse handling.

    Args:
        database: optional database name override.

    Returns:
        neo4j.Driver instance, or None if the neo4j driver is unavailable.
    """
    try:
        from neo4j import GraphDatabase
    except Exception:
        return None
    cfg = get_neo4j_config()
    uri = cfg.get('uri') or os.environ.get('NEO4J_URI', 'bolt://100.64.43.123:7687')
    user = cfg.get('user') or os.environ.get('NEO4J_USER', 'neo4j')
    password = cfg.get('password') or os.environ.get('NEO4J_PASSWORD', '')
    return GraphDatabase.driver(uri, auth=(user, password))


def neo4j_session(database=None):
    """Return a connected Neo4j session using get_neo4j_config().

    Convenience wrapper around get_neo4j_driver() for scripts that just
    want a session. Returns None if the driver is unavailable.
    """
    drv = get_neo4j_driver(database=database)
    if drv is None:
        return None
    return drv.session(database=database or os.environ.get('NEO4J_DB', 'neo4j'))
