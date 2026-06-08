#!/usr/bin/env python3
"""
Agent Health Check Script

Quick check of all fleet agents' status (load, SSH reachability).
Useful for verifying agent availability before delegating tasks.
"""

import subprocess
from datetime import datetime


def check_agent(host: str) -> dict:
    """Check if agent is reachable and get load."""
    
    try:
        result = subprocess.run(
            f"ssh -o ConnectTimeout=3 -i ~/.ssh/hermes-agent-key "
            f"-o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes "
            f"{host} 'cat /proc/loadavg | cut -d\" \" -f1' 2>&1",
            shell=True, capture_output=True, text=True, timeout=5
        )
        
        output = result.stdout.strip()
        if output and "No such file" not in output and "Permission denied" not in output:
            load = float(output)
            return {"status": "online", "load": load}
    except Exception as e:
        pass
    
    # Try macOS
    try:
        result = subprocess.run(
            f"ssh -o ConnectTimeout=3 -i ~/.ssh/hermes-agent-key "
            f"-o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes "
            f"{host} 'uptime' 2>&1",
            shell=True, capture_output=True, text=True, timeout=5
        )
        
        if result.stdout.strip():
            import re
            match = re.search(r'load averages:\s*([0-9.]+)', result.stdout)
            if match:
                load = float(match.group(1))
                return {"status": "online", "load": load}
    except Exception as e:
        pass
    
    return {"status": "offline"}


def main():
    """Check all fleet agents."""
    
    print(f"🔍 Agent Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    agents = [
        ("x1-370", "scottjoyner@100.64.43.123"),
        ("deathstar-XPS-8920", "deathstar@100.78.106.121"),
        ("destroyer", "scott@100.81.57.77"),
        ("macbook-air", "scottjoyner@100.85.64.117")
    ]
    
    for name, host in agents:
        result = check_agent(host)
        
        if result["status"] == "online":
            load = result["load"]
            status_icon = "✅ IDLE" if load < 1.0 else "⏳ BUSY"
            print(f"{name:25} {host:30} {status_icon:12} (load: {load:.2f})")
        else:
            print(f"{name:25} {host:30} {'❌ OFFLINE':12}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
