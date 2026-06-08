#!/usr/bin/env python3
"""
Knowledge Sync Handler - Slash command interface for multi-machine sync
Called via /knowledge-sync, /knowledge-neo4j, /knowledge-conflicts
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

MASTER_PATH = os.environ.get("KNOWLEDGE_VAULT_PATH", "/home/scott/knowledge")
MIRROR_PATH = os.environ.get("KNOWLEDGE_MIRROR_PATH", "/media/scott/NAS2/fileserver/shared-knowledge")
AUTO_INGEST_PATH = "/home/scott/git/auto-ingest"
LOG_FILE = "/home/scott/logs/knowledge_sync.log"

MACHINES = [
    ("deathstar", "100.78.106.121"),
    ("demo-1", "100.65.68.58"),
    ("destroyer", "100.81.57.77"),
    ("optiplex", "100.69.158.114"),
]

def log(message: str):
    """Log message to file and stdout"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    # Ensure log directory exists
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")
    
    print(log_line)

def run_command(cmd: list, cwd: str = None) -> tuple:
    """Run shell command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def mirror_to_nas() -> dict:
    """Mirror the working vault to the NAS durable layer."""
    target = Path(MIRROR_PATH)
    target.mkdir(parents=True, exist_ok=True)
    log(f"Mirroring {MASTER_PATH} -> {MIRROR_PATH}")
    success, stdout, stderr = run_command([
        "rsync",
        "-a",
        "--delete",
        f"{MASTER_PATH}/",
        f"{MIRROR_PATH}/",
    ])
    if success:
        log("✓ NAS mirror updated")
        return {"success": True, "message": stdout}
    log(f"✗ NAS mirror failed: {stderr}")
    return {"success": False, "error": stderr}

def sync_from_machine(machine_name: str, ip: str) -> dict:
    """Sync knowledge from a specific machine to master"""
    log(f"Syncing from {machine_name} ({ip})...")
    
    # SSH into machine and push to master
    cmd = [
        "ssh", f"scott@{ip}",
        f"cd ~/knowledge/nas-knowledge && git add . && "
        f"git commit -m 'Sync from {machine_name}' && "
        f"git push origin main 2>/dev/null || true"
    ]
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        log(f"✓ Successfully synced from {machine_name}")
        return {"success": True, "machine": machine_name, "message": stdout}
    else:
        log(f"✗ Failed to sync from {machine_name}: {stderr}")
        return {"success": False, "machine": machine_name, "error": stderr}

def full_sync() -> dict:
    """Run full knowledge sync from x1-370 master"""
    log("=== Starting Full Knowledge Sync ===")
    
    results = []
    
    # Step 1: Pull all branches
    log("Step 1: Fetching from all machines...")
    for machine_name, ip in MACHINES:
        success, stdout, stderr = run_command(
            ["ssh", f"scott@{ip}", "cd ~/knowledge/nas-knowledge && git fetch origin main"],
            cwd=MASTER_PATH
        )
        results.append({"machine": machine_name, "success": success})
    
    # Step 2: Merge branches
    log("Step 2: Merging branches...")
    success, stdout, stderr = run_command(
        ["git", "pull", "--rebase", "origin", "main"],
        cwd=MASTER_PATH
    )
    results.append({"machine": "master", "success": success})
    
    # Step 3: Check for conflicts
    log("Step 3: Checking for conflicts...")
    success, stdout, stderr = run_command(
        ["git", "status", "--porcelain"],
        cwd=MASTER_PATH
    )
    
    if "???" in stdout or "UU" in stdout:
        log("⚠ Conflicts detected! Manual resolution required.")
        results.append({"machine": "conflict_check", "has_conflicts": True})
    else:
        log("✓ No conflicts detected")
        results.append({"machine": "conflict_check", "has_conflicts": False})
    
    # Step 4: Run Neo4j sync
    log("Step 4: Running Neo4j indexing...")
    success, stdout, stderr = run_command(
        ["python3", "-m", "knowledge_map", "sync_vault_to_neo4j"],
        cwd=AUTO_INGEST_PATH
    )
    
    if success:
        log("✓ Neo4j indexing complete")
        results.append({"machine": "neo4j", "success": True})
    else:
        log(f"✗ Neo4j indexing failed: {stderr}")
        results.append({"machine": "neo4j", "success": False, "error": stderr})
    
    # Step 5: Push to all machines
    log("Step 5: Pushing updates to all machines...")
    for machine_name, ip in MACHINES:
        success, stdout, stderr = run_command(
            ["ssh", f"scott@{ip}", "cd ~/knowledge/nas-knowledge && git pull origin main"],
            cwd=MASTER_PATH
        )
        results.append({"machine": machine_name, "success": success})
    
    # Step 6: Mirror the vault to the NAS durable layer
    log("Step 6: Mirroring vault to NAS...")
    mirror_result = mirror_to_nas()
    results.append({"machine": "nas_mirror", "success": mirror_result.get("success", False)})
    
    log("=== Knowledge Sync Complete ===")
    return {"results": results}

def neo4j_index(dry_run: bool = False) -> dict:
    """Run Neo4j indexing on local vault"""
    log(f"{'Dry run - ' if dry_run else ''}Running Neo4j indexing...")
    
    cmd = [
        "python3", "-m", "knowledge_map", 
        "sync_vault_to_neo4j",
        "--config", f"{AUTO_INGEST_PATH}/config.yaml"
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    success, stdout, stderr = run_command(cmd, cwd=AUTO_INGEST_PATH)
    
    return {
        "success": success,
        "output": stdout,
        "error": stderr if not success else None
    }

def list_conflicts() -> dict:
    """List git merge conflicts in knowledge base"""
    log("Checking for merge conflicts...")
    
    success, stdout, stderr = run_command(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=MASTER_PATH
    )
    
    if success and stdout.strip():
        files = [f.strip() for f in stdout.split("\n") if f.strip()]
        return {
            "has_conflicts": True,
            "conflicted_files": files,
            "count": len(files)
        }
    else:
        return {"has_conflicts": False, "message": "No conflicts detected"}

def main():
    """Main entry point for slash command handler"""
    if len(sys.argv) < 2:
        print("Usage: knowledge_sync_handler.py [command] [args]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "sync":
        # Full sync from master to all machines
        result = full_sync()
        print(f"Sync complete. Results: {len(result['results'])} operations")
        
    elif command == "from-machine":
        # Sync from specific machine (requires IP argument)
        if len(sys.argv) < 3:
            print("Usage: knowledge_sync_handler.py from-machine <machine_name> <ip>")
            sys.exit(1)
        
        machine_name = sys.argv[2]
        ip = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Find IP from MACHINES list if not provided
        if not ip:
            for name, addr in MACHINES:
                if name == machine_name:
                    ip = addr
                    break
        
        if ip:
            result = sync_from_machine(machine_name, ip)
            print(f"Sync result: {result}")
        else:
            print(f"Machine '{machine_name}' not found in MACHINES list")
    
    elif command == "neo4j":
        # Run Neo4j indexing
        dry_run = "--dry-run" in sys.argv
        result = neo4j_index(dry_run=dry_run)
        print(f"Neo4j sync: {'success' if result['success'] else 'failed'}")
        
    elif command == "conflicts":
        # List conflicts
        result = list_conflicts()
        if result["has_conflicts"]:
            print(f"Found {result['count']} conflicted files:")
            for f in result["conflicted_files"]:
                print(f"  - {f}")
        else:
            print("No conflicts detected")
    
    elif command == "help":
        print("""Knowledge Sync Commands:
  sync              - Full sync from master to all machines
  from-machine      - Sync from specific machine (requires name + IP)
  neo4j             - Run Neo4j indexing on local vault
  conflicts         - List git merge conflicts
  help              - Show this help message""")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
