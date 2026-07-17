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

# ---------------------------------------------------------------------------
# Canonical Neo4j password default (single source of truth).
#
# Every script that used to inline the literal ``knowledge_graph_2026`` now
# resolves the password through get_neo4j_password() with this order:
#     NEO4J_PASSWORD  ->  NEO4J_PASSWORD_DEFAULT  ->  _BAKED_IN_NEO4J_PASSWORD
# The baked-in literal lives here (and ONLY here) so the historical default
# keeps working out of the box, while any machine can override it with a single
# `export NEO4J_PASSWORD_DEFAULT=...` (or the stronger per-run NEO4J_PASSWORD).
# ---------------------------------------------------------------------------
_BAKED_IN_NEO4J_PASSWORD = "knowledge_graph_2026"


def get_neo4j_password(config_value: str | None = None) -> str:
    """Resolve the Neo4j password from the standard sources.

    Order of precedence:
      1. NEO4J_PASSWORD           (per-run / per-shell override)
      2. config.yaml value        (passed in as ``config_value``; already
                                    ${ENV}-resolved by the caller)
      3. NEO4J_PASSWORD_DEFAULT   (machine-wide default, e.g. from .env)
      4. baked-in historical default (``knowledge_graph_2026``)
    """
    return (
        os.environ.get("NEO4J_PASSWORD")
        or (config_value or None)
        or os.environ.get("NEO4J_PASSWORD_DEFAULT")
        or _BAKED_IN_NEO4J_PASSWORD
    )


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
        or '/media/scott/NAS5',
        'fileserver_root': os.environ.get('FILESERVER_ROOT')
        or (matched_machine or {}).get('fileserver_root')
        or '/media/scott/SSD_4TB/fileserver',
    }


def _first_existing_path(*candidates):
    """Return the first usable existing path, else the first non-empty candidate.

    Empty unmounted mountpoint directories are ignored so a failed CIFS/NFS mount
    does not win over an older-but-live fallback.
    """
    expanded = [str(Path(p).expanduser()) for p in candidates if p]
    for p in expanded:
        path = Path(p)
        if not path.exists():
            continue
        if path.is_dir() and not os.path.ismount(path):
            try:
                if not any(path.iterdir()):
                    continue
            except OSError:
                continue
        return p
    return expanded[0] if expanded else ''


def _subpath_or_root(root, subdir):
    """Return root if it already ends with subdir, otherwise root/subdir."""
    root_path = Path(str(root))
    if root_path.name == subdir:
        return str(root_path)
    return str(root_path / subdir)


def _require_mounted(path):
    """Fail closed if a configured mount root is missing or only an empty directory."""
    p = Path(str(path))
    if not p.exists():
        raise RuntimeError(f"Required mount path does not exist: {p}")
    if not os.path.ismount(p):
        try:
            is_empty = p.is_dir() and not any(p.iterdir())
        except OSError:
            is_empty = True
        if is_empty:
            raise RuntimeError(f"Required mount path is not mounted: {p}")
    return str(p)


def get_hot_root():
    """Get the SSD hot layer root path."""
    return get_storage_layout()['hot_root']


def get_cold_root():
    """Get the NAS/mirror root path.

    Fail closed: NAS5 is the only accepted NAS mount on deathstar. Do not fall
    back to legacy NAS3 because it may cause new data to be loaded there.
    """
    return _require_mounted(_first_existing_path(get_storage_layout()['cold_root'], '/media/scott/NAS5'))


def get_fileserver_root():
    """Get the fileserver root path for the current machine.

    Priority:
    1. FILESERVER_ROOT environment variable
    2. Auto-detected from config.yaml based on hostname
    3. Fallback to x1-370/default storage layout
    """
    root = get_storage_layout()['fileserver_root']
    return _first_existing_path(root, '/mnt/8TB_2025/fileserver', '/media/scott/SSD_4TB/fileserver')


def get_audio_root():
    """Get the canonical destination/root for organized audio."""
    env = os.environ.get('AUDIO_ROOT')
    layout = get_storage_layout()
    hot_audio = _subpath_or_root(layout.get('hot_root'), 'audio')
    return _first_existing_path(env, hot_audio, '/mnt/8TB_2025/fileserver/audio', '/media/scott/SSD_4TB/audio', '/media/scott/NAS5/audio')


def get_dashcam_root():
    """Get the dashcam processing/cache root.

    On deathstar, process from the local 8TB cache first. NAS5 is cold/durable
    storage and should be populated by a mirror/archive step, not used as the
    first processing path when local cache is available.
    """
    env = os.environ.get('DASHCAM_ROOT')
    if env:
        return env
    fileserver_root = Path(get_fileserver_root())
    local_cache = fileserver_root / 'dashcam'
    if fileserver_root.exists():
        return str(local_cache)
    cold = get_cold_root()
    return _first_existing_path(_subpath_or_root(cold, 'dashcam'), '/media/scott/NAS5/dashcam')


def get_bodycam_root():
    """Get the bodycam processing/cache root.

    On deathstar, process from the local 8TB cache first. NAS5 is cold/durable
    storage and should be populated by a mirror/archive step, not used as the
    first processing path when local cache is available.
    """
    env = os.environ.get('BODYCAM_ROOT')
    if env:
        return env
    fileserver_root = Path(get_fileserver_root())
    local_cache = fileserver_root / 'bodycam'
    if fileserver_root.exists():
        return str(local_cache)
    cold = get_cold_root()
    return _first_existing_path(_subpath_or_root(cold, 'bodycam'), '/media/scott/NAS5/bodycam')


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
            # ${NEO4J_PASSWORD} placeholders resolve to that env var; leave
            # unresolved (None) so get_neo4j_password() can apply the full
            # precedence chain (env -> config -> NEO4J_PASSWORD_DEFAULT -> baked).
            cfg_pass = _resolve_env(matched.get('neo4j_password'), None)

    return {
        'uri': os.environ.get('NEO4J_URI') or cfg_uri or 'bolt://localhost:7687',
        'user': os.environ.get('NEO4J_USER') or cfg_user or 'neo4j',
        'password': get_neo4j_password(cfg_pass),
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
    uri = cfg.get('uri') or os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
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
