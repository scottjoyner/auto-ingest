#!/bin/bash
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
        # curl -X POST -H 'Content-type: application/json' --data "{"text":"High load on $host: $load"}" $ALERT_WEBHOOK
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
