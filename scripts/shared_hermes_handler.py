#!/usr/bin/env python3
"""
Shared Hermes Data Handler - Slash command interface
Called via /shared-hermes setup, status, migrate
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

SHARED_HERMES = "/media/scott/S/shared-hermes"
HERMES_HOME = Path.home() / ".hermes"

def log(message: str):
    """Log message to stdout"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(cmd: list, cwd: str = None) -> tuple:
    """Run shell command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def setup_symlinks(force: bool = False) -> dict:
    """Create symlinks from ~/.hermes to shared location"""
    log("Setting up shared .hermes symlinks...")
    
    # Check if shared location exists
    if not Path(SHARED_HERMES).exists():
        return {
            "success": False, 
            "error": f"Shared location not found: {SHARED_HERMES}"
        }
    
    # Create ~/.hermes if it doesn't exist
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
    
    symlinks = [
        ("chats", "chats"),
        ("skills", "skills"), 
        ("config", "config")
    ]
    
    results = []
    for target, link_name in symlinks:
        source = Path(SHARED_HERMES) / target
        link_path = HERMES_HOME / link_name
        
        # Remove existing symlink if force or doesn't exist
        if force and link_path.exists():
            log(f"Removing existing {link_name} symlink")
            link_path.unlink()
        
        # Create new symlink
        if not link_path.exists() or (force and link_path.is_symlink()):
            try:
                link_path.symlink_to(source, target_ok=True)
                results.append({"target": target, "success": True})
                log(f"✓ Created symlink: {link_name} -> {source}")
            except FileExistsError:
                results.append({"target": target, "success": False, "error": "File exists"})
                log(f"✗ Failed to create {link_name} symlink")
        else:
            results.append({"target": target, "success": True, "skipped": True})
            log(f"- Skipped {link_name} (already linked)")
    
    return {"success": all(r["success"] for r in results), "results": results}

def show_status(detailed: bool = False) -> dict:
    """Show current symlink status"""
    log("Checking shared .hermes status...")
    
    symlinks = ["chats", "skills", "config"]
    status = []
    
    for link_name in symlinks:
        link_path = HERMES_HOME / link_name
        
        if link_path.is_symlink():
            target = Path(link_path.resolve())
            exists = target.exists()
            
            info = {
                "name": link_name,
                "linked": True,
                "target": str(target),
                "exists": exists
            }
            
            if detailed and exists:
                # Count files in shared location
                try:
                    file_count = len(list(target.iterdir()))
                    info["file_count"] = file_count
                except PermissionError:
                    info["file_count"] = "?"
            
            status.append(info)
        else:
            status.append({
                "name": link_name,
                "linked": False,
                "target": str(Path(SHARED_HERMES) / link_name),
                "exists": Path(SHARED_HERMES).joinpath(link_name).exists()
            })
    
    return {"symlinks": status}

def migrate_data(source: str = None) -> dict:
    """Migrate existing ~/.hermes data to shared location"""
    log("Migrating .hermes data...")
    
    if not source:
        # Migrate all directories
        dirs_to_migrate = ["chats", "skills"]
    else:
        dirs_to_migrate = [source]
    
    results = []
    for dir_name in dirs_to_migrate:
        src_dir = HERMES_HOME / dir_name
        
        if not src_dir.exists():
            log(f"- {dir_name}: No data to migrate")
            continue
        
        dst_dir = Path(SHARED_HERMES) / dir_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files (not directories recursively for now)
        try:
            import shutil
            if src_dir.is_dir():
                for item in src_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst_dir / item.name)
            
            log(f"✓ Migrated {dir_name} to shared location")
            results.append({"name": dir_name, "success": True})
        except Exception as e:
            log(f"✗ Failed to migrate {dir_name}: {e}")
            results.append({"name": dir_name, "success": False, "error": str(e)})
    
    return {"success": all(r["success"] for r in results), "results": results}

def main():
    """Main entry point for slash command handler"""
    if len(sys.argv) < 2:
        print("Usage: shared_hermes_handler.py [command] [args]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "setup":
        force = "--force" in sys.argv
        result = setup_symlinks(force=force)
        print(f"Setup complete: {'success' if result['success'] else 'failed'}")
        
    elif command == "status":
        detailed = "--detailed" in sys.argv
        result = show_status(detailed=detailed)
        print("Shared .hermes status:")
        for link in result["symlinks"]:
            status = "✓ linked" if link["linked"] else "○ not linked"
            exists = "exists" if link.get("exists") else "missing"
            print(f"  {link['name']}: {status} ({exists})")
        
    elif command == "migrate":
        source = sys.argv[2] if len(sys.argv) > 2 else None
        result = migrate_data(source=source)
        print(f"Migration complete: {'success' if result['success'] else 'failed'}")
        
    elif command == "help":
        print("""Shared Hermes Commands:
  setup           - Create symlinks to shared .hermes location
  status          - Show current symlink status
  migrate         - Migrate existing data to shared location
  help            - Show this help message""")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
