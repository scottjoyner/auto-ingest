#!/usr/bin/env python3
"""
Idle Agent Knowledge Harvesting System

When agents are idle (load < 1.0), they query the assistx Neo4j database for:
- Sophia voice logs that haven't been processed
- Interesting conversations/memories to add to knowledge graph  
- Unprocessed transcriptions or insights

This turns idle compute into productive knowledge extraction.

The main knowledge graph is on x1-370, so all agents query there.
"""

import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path


def check_agent_load(host: str) -> float:
    """Check if agent is idle (load < 1.0)."""
    
    print(f"   Checking {host}...")  # DEBUG
    
    # Try Linux first, fall back to macOS
    try:
        result = subprocess.run(
            f"ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes {host} 'cat /proc/loadavg | cut -d\" \" -f1' 2>&1",
            shell=True, capture_output=True, text=True, timeout=10
        )
        # Check if it's actually a number (not an error message)
        output = result.stdout.strip()
        print(f"   Linux check: {repr(output)}")  # DEBUG
        if output and not "No such file" in output and "Permission denied" not in output:
            load_avg = float(output)
            return load_avg
    except Exception as e:
        print(f"   Linux check exception: {e}")  # DEBUG
    
    # Try macOS (uptime command with regex parsing)
    try:
        cmd = f"ssh -o ConnectTimeout=5 -i ~/.ssh/hermes-agent-key -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes {host} 'uptime'"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        print(f"   macOS check stdout: {repr(result.stdout)}")  # DEBUG
        if result.stdout.strip():
            import re
            match = re.search(r'load averages:\s*([0-9.]+)', result.stdout)
            if match:
                load_val = float(match.group(1))
                print(f"   macOS load detected: {load_val}")  # DEBUG
                return load_val
    except Exception as e:
        print(f"   macOS check exception: {e}")  # DEBUG
    
    print(f"⚠️  Load check failed for {host}")
    return 999.0


def query_neo4j_from_x1_370(query: str) -> list[dict]:
    """Query the main Neo4j database on x1-370 (central knowledge graph)."""
    
    # Write Python script to temporary file instead of inline -c to avoid quoting issues
    python_script = f'''from neo4j import GraphDatabase
import json
import os

uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
user = os.environ.get('NEO4J_USER', 'neo4j')
passwd = os.environ.get('NEO4J_PASSWORD') or os.environ.get('NEO4J_PASSWORD_DEFAULT') or 'knowledge_graph_2026'

driver = GraphDatabase.driver(uri, auth=(user, passwd))
with driver.session() as session:
    result = session.run("""{query}""")
    records = [record.data() for record in result]
    print(json.dumps(records, default=str))

driver.close()
'''
    
    try:
        # Write script to x1-370
        import base64
        encoded = base64.b64encode(python_script.encode()).decode()
        
        write_cmd = f"ssh -o ConnectTimeout=10 -i ~/.ssh/hermes-agent-key scott@x1-370 \"echo '{encoded}' | base64 -d > /tmp/query_neo4j_temp.py\""
        subprocess.run(write_cmd, shell=True, capture_output=True, timeout=30)
        
        # Run the script from the neo4j directory
        run_cmd = f"ssh -o ConnectTimeout=10 -i ~/.ssh/hermes-agent-key scott@x1-370 \"cd /media/scott/S/neo4j && python3 /tmp/query_neo4j_temp.py\""
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"⚠️  Neo4j query failed: {e}")
    return []


def get_sophia_voice_logs(hours_back: int = 24) -> list[dict]:
    """Query Sophia voice logs from the past N hours."""
    
    # Note: The actual label is SophiaVoiceUiEvent with properties: score, event_type, user_id, session_id, accepted, source, id, payload_json, ts_ms (milliseconds!)
    # ts_ms is in milliseconds, so we need to convert to seconds for datetime comparison
    query = f"""
    MATCH (e:SophiaVoiceUiEvent)
    WHERE e.ts_ms >= (datetime().epochMillis - ({hours_back} * 60 * 60 * 1000))
    WITH e ORDER BY e.ts_ms DESC LIMIT 100
    RETURN 
      e.session_id as session,
      e.payload_json as transcript,
      e.event_type as intent,
      e.ts_ms as timestamp,
      e.score as confidence,
      CASE WHEN e.accepted = true THEN 'processed' ELSE 'unprocessed' END as status
    """
    
    return query_neo4j_from_x1_370(query)


def get_unprocessed_transcriptions() -> list[dict]:
    """Get transcriptions that haven't been added to knowledge graph."""
    
    # Note: Using actual Transcription properties: id, text, source_json, embedding, created_at
    query = """
    MATCH (t:Transcription)
    WHERE t.embedding IS NULL OR t.source_json IS NULL
    WITH t ORDER BY t.created_at DESC LIMIT 50
    RETURN 
      t.id as id,
      t.text as content,
      t.source_json as source,
      t.created_at as timestamp,
      CASE WHEN t.embedding IS NOT NULL THEN 'processed' ELSE 'unprocessed' END as status
    """
    
    return query_neo4j_from_x1_370(query)


def get_interesting_memories(min_confidence: float = 0.5) -> list[dict]:
    """Get high-confidence memories that might be worth reviewing."""
    
    # Note: Using VoiceprintVersion as the memory source with properties: device_id, created_at, session_id, active, threshold, source, version_id, user_id, sample_count, embedding
    query = f"""
    MATCH (m:VoiceprintVersion)
    WHERE m.threshold >= {min_confidence} 
      AND m.created_at >= datetime() - duration('P7D')
    WITH m ORDER BY m.threshold DESC LIMIT 25
    RETURN 
      m.version_id as id,
      m.samples_json as content,
      m.source as source,
      m.threshold as confidence,
      m.created_at as timestamp,
      labels(m) as tags
    """
    
    return query_neo4j_from_x1_370(query)


def get_conversation_highlights() -> list[dict]:
    """Get conversation highlights from Sophia logs."""
    
    # Note: Using SophiaVoiceUiEvent directly since there are no Frame nodes in the database
    query = """
        MATCH (e:SophiaVoiceUiEvent)
        WHERE e.ts_ms >= datetime().epochMillis - (24 * 60 * 60 * 1000)
          AND e.accepted = false
        WITH e ORDER BY e.ts_ms DESC LIMIT 50
        RETURN 
            e.session_id as session,
            e.event_type as intent,
            e.payload_json as transcript,
            e.score as confidence,
            e.ts_ms as timestamp,
            'unprocessed' as status
    """
    
    return query_neo4j_from_x1_370(query)


def extract_knowledge_insights(voice_logs: list[dict], memories: list[dict]) -> dict:
    """Extract actionable insights from voice logs and memories."""
    
    insights = {
        "timestamp": datetime.now().isoformat(),
        "total_events": len(voice_logs),
        "processed_count": sum(1 for e in voice_logs if e.get("status") == "processed"),
        "unprocessed_count": sum(1 for e in voice_logs if e.get("status") == "unprocessed"),
        "high_confidence_memories": len(memories),
        "topics_found": [],
        "action_items": [],
        "memories_to_add": []
    }
    
    # Extract topics from transcripts
    all_transcripts = [e.get("transcript", "") for e in voice_logs if e.get("transcript")]
    topic_keywords = ["project", "task", "meeting", "idea", "plan", "decision", "action"]
    
    for keyword in topic_keywords:
        count = sum(1 for t in all_transcripts if keyword.lower() in t.lower())
        if count > 0:
            insights["topics_found"].append(f"{keyword}: {count}")
    
    # Extract action items from unprocessed logs
    for log in voice_logs:
        transcript = log.get("transcript", "").lower()
        if any(word in transcript for word in ["todo", "remember", "add to knowledge", "note"]):
            insights["action_items"].append({
                "source": log.get("session"),
                "content": log.get("transcript")[:200],
                "confidence": log.get("confidence", 0)
            })
    
    # Prepare memories for knowledge graph addition
    for memory in memories:
        content = memory.get("content") or ""
        insights["memories_to_add"].append({
            "id": memory.get("id"),
            "content": content[:500],
            "confidence": memory.get("confidence"),
            "source": memory.get("source"),
            "tags": memory.get("tags", [])
        })
    
    return insights


def save_harvest_report(insights: dict, host: str) -> Path:
    """Save the harvesting report to a file."""
    
    reports_dir = Path.home() / ".hermes" / "knowledge-harvest"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"{host}_harvest_{timestamp}.json"
    
    with open(report_file, "w") as f:
        json.dump(insights, f, indent=2, default=str)
    
    return report_file


def main():
    """Main harvesting loop."""
    
    # Fleet agents to check (query central Neo4j on x1-370)
    fleet = {
        "x1-370": ("scott", "100.64.43.123"),
        "deathstar-XPS-8920": ("deathstar", "100.78.106.121"),
        "destroyer": ("scott", "100.81.57.77"),
        "scotts-macbook-air": ("scottjoyner", "100.85.64.117")
    }
    
    print(f"🎯 Starting knowledge harvesting at {datetime.now().isoformat()}")
    print("=" * 60)
    
    all_insights = []
    
    for agent_name, (user, ip) in fleet.items():
        print(f"\n🔍 Checking {agent_name} ({ip})...")
        
        # Check load - pass full user@host format
        host = f"{user}@{ip}"
        load = check_agent_load(host)
        print(f"   Load: {load:.2f}")
        
        if load < 1.0:
            print(f"   ✅ IDLE - Harvesting knowledge from central Neo4j...")
            
            # Query central Neo4j on x1-370
            voice_logs = get_sophia_voice_logs(hours_back=24)
            memories = get_interesting_memories()
            conversations = get_conversation_highlights()
            
            print(f"   Found {len(voice_logs)} Sophia events")
            print(f"   Found {len(memories)} high-confidence memories")
            print(f"   Found {len(conversations)} highlighted conversations")
            
            # Extract insights
            insights = extract_knowledge_insights(voice_logs, memories)
            insights["agent"] = agent_name
            insights["load_at_harvest"] = load
            
            all_insights.append(insights)
            
            # Save report
            report_file = save_harvest_report(insights, agent_name)
            print(f"   📄 Report saved: {report_file}")
        else:
            print(f"   ⏳ BUSY (load > 1.0) - Skipping")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 HARVESTING COMPLETE")
    print(f"   Agents harvested: {len(all_insights)}")
    for insights in all_insights:
        agent = insights["agent"]
        print(f"   • {agent}: {insights['total_events']} events, "
              f"{insights['high_confidence_memories']} memories, "
              f"{insights['action_items']} action items")
    
    # Save summary
    summary_file = Path.home() / ".hermes" / "knowledge-harvest" / "harvest_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "agents_harvested": len(all_insights),
            "insights": all_insights
        }, f, indent=2, default=str)
    
    print(f"   Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
