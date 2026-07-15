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
import yaml
from pathlib import Path


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
        or knowledge_map.get('hot_layer_path')
        or (matched_machine or {}).get('hot_ssd_root')
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
    """Get Neo4j connection settings for the current machine."""
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        hostname = socket.gethostname()
        for key, vals in cfg.get('machine_paths', {}).items():
            if vals.get('hostname_pattern') and vals['hostname_pattern'] in hostname:
                return {
                    'uri': vals['neo4j_uri'],
                    'user': vals['neo4j_user'],
                    'password': _resolve_env(vals.get('neo4j_password'), os.environ.get('NEO4J_PASSWORD', '')),
                }
        
        vals_list = list(cfg.get('machine_paths', {}).values())
        if vals_list:
            return {
                'uri': vals_list[0]['neo4j_uri'],
                'user': vals_list[0]['neo4j_user'],
                'password': _resolve_env(vals_list[0].get('neo4j_password'), os.environ.get('NEO4J_PASSWORD', '')),
            }
    
    return {
        'uri': os.environ.get('NEO4J_URI', 'bolt://100.64.43.123:7687'),
        'user': os.environ.get('NEO4J_USER', 'neo4j'),
        'password': os.environ.get('NEO4J_PASSWORD', ''),
    }


def get_fileserver_path(*parts):
    """Get a full fileserver path for the given sub-paths."""
    root = get_fileserver_root()
    return os.path.join(root, *parts)
