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
        matched_machine = list(machine_paths.values())[0]

    return {
        'hot_root': os.environ.get('HOT_STORAGE_ROOT')
        or knowledge_map.get('hot_layer_path')
        or (matched_machine or {}).get('hot_ssd_root')
        or '/media/scott/S',
        'cold_root': os.environ.get('COLD_STORAGE_ROOT')
        or knowledge_map.get('mirror_vault_path')
        or (matched_machine or {}).get('nas_mirror_root')
        or '/media/scott/NAS2/fileserver',
        'fileserver_root': os.environ.get('FILESERVER_ROOT')
        or (matched_machine or {}).get('fileserver_root')
        or '/media/scott/NAS2/fileserver',
        'vault_path': os.environ.get('VAULT_PATH')
        or knowledge_map.get('central_vault_path')
        or os.path.expanduser(knowledge_map.get('local_vault_path', '~/knowledge')),
        'mirror_vault_path': knowledge_map.get('mirror_vault_path')
        or '/media/scott/NAS2/fileserver/shared-knowledge',
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
