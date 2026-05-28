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


def get_fileserver_root():
    """Get the fileserver root path for the current machine.
    
    Priority:
    1. FILESERVER_ROOT environment variable
    2. Auto-detected from config.yaml based on hostname
    3. Fallback to x1-370 default
    """
    env = os.environ.get('FILESERVER_ROOT')
    if env:
        return env
    
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        hostname = socket.gethostname()
        for key, vals in cfg.get('machine_paths', {}).items():
            if vals.get('hostname_pattern') and vals['hostname_pattern'] in hostname:
                return vals['fileserver_root']
        
        # Fallback to first entry
        vals_list = list(cfg.get('machine_paths', {}).values())
        if vals_list:
            return vals_list[0]['fileserver_root']
    
    # Ultimate fallback
    return '/media/scott/NAS2/fileserver'


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
                    'password': vals['neo4j_password'],
                }
        
        vals_list = list(cfg.get('machine_paths', {}).values())
        if vals_list:
            return {
                'uri': vals_list[0]['neo4j_uri'],
                'user': vals_list[0]['neo4j_user'],
                'password': vals_list[0]['neo4j_password'],
            }
    
    return {
        'uri': 'bolt://100.64.43.123:7687',
        'user': 'neo4j',
        'password': 'knowledge_graph_2026',
    }


def get_fileserver_path(*parts):
    """Get a full fileserver path for the given sub-paths."""
    root = get_fileserver_root()
    return os.path.join(root, *parts)
