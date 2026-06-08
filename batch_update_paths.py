#!/usr/bin/env python3
"""
Batch update auto-ingest scripts to use config.yaml instead of hardcoded paths.
"""
import os
import re
from pathlib import Path

REPO = Path('/home/scott/git/auto-ingest')

# Define the replacements for Python files
PYTHON_PATH_PATTERNS = [
    r'/media/scott/NAS/fileserver/',
    r'/media/scott/NAS2/fileserver/',
    r'/media/scott/NAS/8TB_2025/fileserver/',
    r'/media/scott/NAS/8TBHDD/fileserver/',
]

# Shell script path replacements
SHELL_PATH_PATTERNS = [
    r'/media/scott/NAS/fileserver/',
    r'/media/scott/NAS2/fileserver/',
    r'/media/scott/NAS/8TB_2025/fileserver/',
    r'/media/scott/NAS/8TBHDD/fileserver/',
]


def update_python_file(filepath):
    """Update a Python file to use config-based paths."""
    with open(filepath) as f:
        content = f.read()
    
    original = content
    
    # Check if file already imports auto_ingest_config
    needs_import = 'auto_ingest_config' not in content
    
    # Replace hardcoded paths with get_fileserver_path() calls
    for pattern in PYTHON_PATH_PATTERNS:
        content = re.sub(pattern, 'get_fileserver_path(', content)
    
    # Fix paths that now have unclosed quotes
    # Pattern: get_fileserver_path("subpath", "subpath") → add closing paren and quote
    content = re.sub(
        r'get_fileserver_path\(([^)]+)\)',
        lambda m: f'get_fileserver_path({m.group(1)})',
        content
    )
    
    # Add import if needed
    if needs_import and content != original:
        # Find the first import block or add at top
        import_pattern = r'(import\s+os\s*\n)'
        match = re.search(import_pattern, content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + 'from auto_ingest_config import get_fileserver_path\n' + content[insert_pos:]
        else:
            content = 'from auto_ingest_config import get_fileserver_path\n' + content
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def update_shell_file(filepath):
    """Update a shell script to use FILESERVER_ROOT variable."""
    with open(filepath) as f:
        content = f.read()
    
    original = content
    
    # Add FILESERVER_ROOT definition at the top if not present
    if 'FILESERVER_ROOT' not in content:
        # Find first line that uses fileserver path
        lines = content.split('\n')
        insert_line = None
        for i, line in enumerate(lines):
            if 'fileserver' in line.lower() and not line.strip().startswith('#'):
                insert_line = i
                break
        
        if insert_line:
            # Add FILESERVER_ROOT detection before the first fileserver reference
            indent = ''
            for ch in lines[insert_line]:
                if ch in (' ', '\t'):
                    indent += ch
                else:
                    break
            
            header = f"""# Auto-detect fileserver root
FILESERVER_ROOT=$(python3 -c "
import socket, yaml
from pathlib import Path
config_path = Path('{config_yaml}')
if config_path.exists():
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    hostname = socket.gethostname()
    for key, vals in cfg.get('machine_paths', {{}}).items():
        if vals.get('hostname_pattern') and vals['hostname_pattern'] in hostname:
            print(vals['fileserver_root'])
            exit()
    print(list(cfg['machine_paths'].values())[0]['fileserver_root'])
")
"""
            lines.insert(insert_line, header)
            content = '\n'.join(lines)
    
    # Replace hardcoded paths with FILESERVER_ROOT
    for pattern in SHELL_PATH_PATTERNS:
        content = re.sub(pattern, 'FILESERVER_ROOT/', content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


if __name__ == '__main__':
    config_yaml = str(REPO / 'config.yaml')
    
    updated_py = 0
    updated_sh = 0
    
    for f in REPO.iterdir():
        if f.suffix == '.py' and f.is_file():
            if update_python_file(f):
                print(f"Updated: {f.name}")
                updated_py += 1
        elif f.suffix == '.sh' and f.is_file():
            if update_shell_file(f):
                print(f"Updated: {f.name}")
                updated_sh += 1
    
    print(f"\nTotal: {updated_py} Python files, {updated_sh} shell scripts updated")
