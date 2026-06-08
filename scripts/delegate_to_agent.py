#!/usr/bin/env python3
"""
Agent Delegation Script

Send tasks to idle agents based on their current load.
Automatically selects the best available agent for the task.
"""

import subprocess
import json
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


def find_best_agent() -> tuple[str, str] | None:
    """Find the best agent for task delegation (lowest load)."""
    
    agents = [
        ("x1-370", "scottjoyner@100.64.43.123"),
        ("deathstar-XPS-8920", "deathstar@100.78.106.121"),
        ("destroyer", "scott@100.81.57.77"),
        ("macbook-air", "scottjoyner@100.85.64.117")
    ]
    
    available = []
    
    for name, host in agents:
        result = check_agent(host)
        
        if result["status"] == "online" and result["load"] < 2.0:
            available.append((name, host, result["load"]))
    
    if not available:
        return None
    
    # Sort by load (lowest first)
    available.sort(key=lambda x: x[2])
    
    # Return best agent
    return available[0][0], available[0][1]


def send_task_to_agent(agent_name: str, host: str, task: str):
    """Send a task to an agent via SSH."""
    
    # Create a temporary file with the task
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"Task: {task}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        temp_file = f.name
    
    # Send via SSH (this would need to be adapted for your agent setup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = (
        f"ssh -o BatchMode=yes -i ~/.ssh/hermes-agent-key {host} "
        f"'cat > /tmp/agent_task_{timestamp}.txt'"
    )

    with open(temp_file, 'r') as task_file:
        subprocess.run(cmd, shell=True, input=task_file.read(), text=True)


def main():
    """Main delegation logic."""
    
    print(f"🚀 Agent Delegation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check all agents
    agents = [
        ("x1-370", "scottjoyner@100.64.43.123"),
        ("deathstar-XPS-8920", "deathstar@100.78.106.121"),
        ("destroyer", "scott@100.81.57.77"),
        ("macbook-air", "scottjoyner@100.85.64.117")
    ]
    
    print("\nAgent Status:")
    for name, host in agents:
        result = check_agent(host)
        
        if result["status"] == "online":
            load = result["load"]
            status_icon = "✅ IDLE" if load < 1.0 else "⏳ BUSY"
            print(f"  {name:25} (load: {load:.2f}) {status_icon}")
        else:
            print(f"  {name:25} ❌ OFFLINE")
    
    # Find best agent
    best = find_best_agent()
    
    if best:
        best_name, best_host = best
        best_status = check_agent(best_host)
        best_load = best_status.get("load", 0.0)

        print(f"\n🎯 Best agent for delegation: {best_name}")
        print(f"   Host: {best_host}")
        print(f"   Load: {best_load:.2f} (lowest)")
    else:
        print("\n⚠️  No agents available (all busy or offline)")


if __name__ == "__main__":
    main()
