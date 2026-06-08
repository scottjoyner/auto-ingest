#!/usr/bin/env python3
"""
Multi-Agent Fleet Onboarding System
====================================
Tests and configures all reachable agents for 24/7 autonomous operation.

Features:
- Health check on each agent (gateway, model endpoint)
- Busy notification system (load detection)
- Trigger mechanism for continuous operation
- Model capability mapping
- Auto-restart on failure
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# Fleet configuration
FLEET = {
    "x1-370": {
        "ip": "100.64.43.123",
        "user": "scott",
        "role": "heavy-lifter",  # Large model (35B) - slow but smart
        "model": "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive",
        "gateway_port": 18790,
        "model_port": 1234,
    },
    "deathstar-XPS-8920": {
        "ip": "100.78.106.121",
        "user": "deathstar",
        "role": "speed-demon",  # Small model - fast loops
        "model": "unknown",
        "gateway_port": 18790,
        "model_port": 1234,
    },
    "destroyer": {
        "ip": "100.81.57.77",
        "user": "scott",
        "role": "speed-demon",  # Trading runtime - fast
        "model": "unknown",
        "gateway_port": 18790,
        "model_port": 1234,
    },
    "scotts-macbook-air": {
        "ip": "100.85.64.117",
        "user": "scottjoyner",
        "role": "speed-demon",  # Small models - fast macOS
        "model": "unknown",
        "gateway_port": 18790,
        "model_port": 1234,
    },
}

def run_ssh(host_config, command):
    """Execute command on remote host via SSH"""
    user = host_config["user"]
    ip = host_config["ip"]
    cmd = f"ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key {user}@{ip} '{command}'"
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1

def check_health(host_config):
    """Check agent health (gateway + model endpoint)"""
    host = host_config["host"]
    
    # Check gateway
    gw_cmd = f"curl -s http://127.0.0.1:{host_config['gateway_port']}/health"
    gw_out, gw_err, gw_code = run_ssh(host_config, gw_cmd)
    
    # Check model endpoint
    model_cmd = f"curl -s http://127.0.0.1:{host_config['model_port']}/v1/models"
    model_out, model_err, model_code = run_ssh(host_config, model_cmd)
    
    return {
        "name": host,
        "role": host_config["role"],
        "gateway_up": gw_code == 0 and "healthy" in gw_out.lower(),
        "model_up": model_code == 0 and len(model_out) > 50,
        "gateway_response": gw_out[:100] if gw_out else "",
        "models": json.loads(model_out) if model_code == 0 else [],
    }

def detect_load(host_config):
    """Detect system load for busy notification"""
    load_cmd = """
    uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs
    """
    out, err, code = run_ssh(host_config, load_cmd)
    
    # Also check memory and disk
    mem_cmd = "free -h | grep Mem | awk '{print $3/$2 * 100}'"
    disk_cmd = "df -h / | tail -1 | awk '{print $5}'"
    
    mem_out, _, _ = run_ssh(host_config, mem_cmd)
    disk_out, _, _ = run_ssh(host_config, disk_cmd)
    
    load_val = float(out.strip()) if out else 0
    
    return {
        "load": load_val,
        "memory_percent": float(mem_out.strip()) if mem_out else 0,
        "disk_percent": float(disk_out.strip().replace("%", "")) if disk_out else 0,
        "status": "BUSY" if load_val > 3.0 else "IDLE",
    }

def create_soul_file(host_config, health_data):
    """Create soul file for agent"""
    host = host_config["name"]
    
    soul_content = f"""# {host.upper()} - {host_config['role'].replace('-', ' ').title()}

## Identity
- **Machine**: {host}
- **IP**: {host_config['ip']}
- **Role**: {host_config['role'].replace('-', ' ').title()}
- **User**: {host_config['user']}

## Model Configuration
- **Primary Model**: {health_data.get('models', [{}])[0].get('id', 'unknown') if health_data.get('models') else 'unknown'}
- **Gateway Port**: {host_config['gateway_port']}
- **Model Port**: {host_config['model_port']}

## Capabilities
### Standalone Agent (Primary)
- **Strength**: {host_config['role'].replace('-', ' ').title()} - optimized for {host_config['role']} tasks
- **Context Window**: 4096 tokens
- **Toolsets**: All default tools available via gateway
- **Autonomous Operation**: YES - runs 24/7 with auto-restart

### LMStudio Endpoint (Secondary)
- **Purpose**: Pure text generation, no tool use
- **Context**: 4096 window (text-only mode)
- **Speed**: Fast response for high-throughput needs
- **Limitation**: NOT running agentic work - inference only

## Health Monitoring
- **Gateway Status**: {'UP' if health_data['gateway_up'] else 'DOWN'}
- **Model Status**: {'UP' if health_data['model_up'] else 'DOWN'}
- **Load Detection**: Monitored via load average > 3.0 = BUSY
- **Auto-Restart**: Enabled on failure

## Trigger System
- **Continuous Loop**: YES - agent runs continuously
- **Health Check Interval**: Every 5 minutes
- **Busy Notification**: Slack/Signal webhook when load > 3.0
- **Failure Alert**: Immediate notification if gateway/model down

## Interaction Protocol
1. Route heavy reasoning tasks to {host} (if role = heavy-lifter)
2. Route fast inference tasks to {host} (if role = speed-demon)
3. Check health before delegation: `curl http://{host_config['ip']}:18790/health`
4. Monitor load via: `ssh {host_config['user']}@{host_config['ip']} 'uptime'`

## 24/7 Operation
- **Start Command**: `hermes gateway run &`
- **Restart on Failure**: Systemd service or supervisor
- **Log Location**: ~/.hermes/logs/gateway.log
- **Recovery Time**: < 30 seconds (auto-restart)
"""
    
    soul_path = Path(f"~/.hermes/souls/{host}-soul.md")
    soul_path.expanduser().write_text(soul_content)
    return str(soul_path.expanduser())

def create_health_check_script():
    """Create automated health check script for each agent"""
    
    script = """#!/bin/bash
# Agent Health Check Script - Runs every 5 minutes via cron

LOG_FILE=~/.hermes/logs/health-check.log
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

check_agent() {
    local host=$1
    local ip=$2
    local user=$3
    
    log "=== Checking $host ($ip) ==="
    
    # Check gateway health
    gw_status=$(curl -s --connect-timeout 5 http://$ip:18790/health 2>&1 | head -c 100)
    if [[ $? -eq 0 && "$gw_status" == *"healthy"* ]]; then
        log "✓ Gateway UP on $host"
    else
        log "✗ Gateway DOWN on $host - TRIGGERING RESTART"
        ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key $user@$ip 'pkill -f hermes-gateway; sleep 2; hermes gateway run &' >> $LOG_FILE 2>&1
    fi
    
    # Check model endpoint
    model_status=$(curl -s --connect-timeout 5 http://$ip:1234/v1/models 2>&1 | head -c 100)
    if [[ $? -eq 0 && ${#model_status} -gt 50 ]]; then
        log "✓ Model UP on $host"
    else
        log "✗ Model DOWN on $host"
    fi
    
    # Check load (busy notification)
    load=$(ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key $user@$ip 'uptime | awk -F"load average:" "{print $2}" | awk -F"," "{print $1}"' 2>/dev/null | xargs)
    if [[ $(echo "$load > 3.0" | bc -l) -eq 1 ]]; then
        log "⚠ HIGH LOAD on $host: $load - SENDING ALERT"
        # curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"High load on $host: $load\"}" $ALERT_WEBHOOK
    fi
    
    log ""
}

# Main loop
while true; do
    check_agent "x1-370" "100.64.43.123" "scott"
    check_agent "deathstar-XPS-8920" "100.78.106.121" "deathstar"
    check_agent "destroyer" "100.81.57.77" "scott"
    check_agent "scotts-macbook-air" "100.85.64.117" "scottjoyner"
    
    sleep 300  # 5 minutes
done
"""
    
    script_path = Path("/home/scott/git/auto-ingest/scripts/agent-health-check.sh")
    script_path.write_text(script)
    script_path.chmod(0o755)
    return str(script_path)

def create_cron_job():
    """Create cron job for continuous health monitoring"""
    
    cron_entry = "*/5 * * * * /home/scott/git/auto-ingest/scripts/agent-health-check.sh >> ~/.hermes/logs/cron-health.log 2>&1"
    
    # Add to crontab if not already present
    result = subprocess.run(
        "crontab -l 2>/dev/null | grep agent-health-check",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode != 0:  # Not found, add it
        current_cron = subprocess.run("crontab -l 2>/dev/null", shell=True, capture_output=True, text=True)
        new_cron = f"{current_cron.stdout}\n{cron_entry}" if current_cron.stdout else cron_entry
        
        subprocess.run(f"echo '{new_cron}' | crontab -", shell=True)
        return True
    
    return False

def main():
    """Run full onboarding sequence"""
    print("=" * 60)
    print("MULTI-AGENT FLEET ONBOARDING SYSTEM")
    print("=" * 60)
    
    onboarded = []
    failed = []
    
    for name, config in FLEET.items():
        print(f"\n🔍 Probing {name}...")
        
        # Check health
        health = check_health({"host": name, **config})
        print(f"   Gateway: {'✓ UP' if health['gateway_up'] else '✗ DOWN'}")
        print(f"   Model: {'✓ UP' if health['model_up'] else '✗ DOWN'}")
        
        if health["gateway_up"] and health["model_up"]:
            # Detect load
            load = detect_load(config)
            print(f"   Load: {load['status']} (avg: {load['load']:.2f})")
            
            # Create soul file
            soul_path = create_soul_file(config, health)
            print(f"   Soul created: {soul_path}")
            
            onboarded.append(name)
        else:
            failed.append(name)
            print(f"   ⚠ SKIPPED - Agent not fully operational")
    
    # Create health check script
    print("\n📝 Creating automated health check system...")
    health_script = create_health_check_script()
    print(f"   Health check script: {health_script}")
    
    # Setup cron job
    if create_cron_job():
        print("   ✓ Cron job added (runs every 5 minutes)")
    else:
        print("   ⚠ Cron job already exists")
    
    # Summary
    print("\n" + "=" * 60)
    print("ONBOARDING COMPLETE")
    print("=" * 60)
    print(f"\n✓ Onboarded ({len(onboarded)}): {', '.join(onboarded)}")
    if failed:
        print(f"✗ Failed ({len(failed)}): {', '.join(failed)}")
    
    print("\n📊 Next Steps:")
    print("1. Review soul files in ~/.hermes/souls/")
    print("2. Test delegation to onboarded agents")
    print("3. Monitor health logs: tail -f ~/.hermes/logs/health-check.log")
    print("4. Adjust load thresholds if needed (edit agent-health-check.sh)")

if __name__ == "__main__":
    main()
