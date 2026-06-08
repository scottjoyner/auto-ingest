#!/usr/bin/env python3
import subprocess

def check_macos_load(host: str) -> float:
    """Check macOS load average."""
    
    # Use a simpler approach - just get the first number from uptime
    cmd = f"ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key {host} 'uptime'"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=10
    )
    
    if result.stdout.strip():
        # Parse uptime output: "19:46  up 3 days,  5:08, 4 users, load averages: 2.09 1.91 1.69"
        import re
        match = re.search(r'load averages:\s*([0-9.]+)', result.stdout)
        if match:
            return float(match.group(1))
    
    return None

# Test it
print("Testing macOS load check...")
macbook_load = check_macos_load("scottjoyner@100.85.64.117")
print(f"Macbook load: {macbook_load}")
