#!/usr/bin/env python3
"""
Idle Agent Knowledge Harvesting System

When agents are idle (load < 1.0), they query the knowledge graph for:
- Sophia voice logs that haven't been processed
- Interesting conversations/memories to add to knowledge graph
- Unprocessed transcriptions or insights

This turns idle compute into productive knowledge extraction.

The knowledge graph is reachable directly via neo4j Bolt (auto_ingest_config),
so queries run locally on xwing — no SSH hop into x1-370 required.
"""

import re
import subprocess
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the repo root importable so auto_ingest_config resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from neo4j import GraphDatabase
    _NEO = True
except ImportError:
    _NEO = False

# --- direct Bolt access to the knowledge graph (no SSH) --------------------
_cfg = None
_drv = None


def _driver():
    global _cfg, _drv
    if not _NEO:
        raise RuntimeError("neo4j driver not available")
    if _drv is None:
        from auto_ingest_config import get_neo4j_config
        _cfg = get_neo4j_config()
        _drv = GraphDatabase.driver(_cfg["uri"], auth=(_cfg["user"], _cfg["password"]))
    return _drv


def query_neo4j(query: str, **params) -> list:
    """Run a read query against the knowledge graph and return list of dicts."""
    with _driver().session(database=(_cfg or {}).get("db") or "neo4j") as sess:
        return [r.data() for r in sess.run(query, **params)]


def close_neo4j():
    global _drv
    if _drv is not None:
        _drv.close()
        _drv = None


# --- fleet idle detection (SSH loadavg) ------------------------------------
def check_agent_load(host: str) -> float:
    """Check if agent is idle (load < 1.0). Returns 999 if it can't be determined."""
    print(f"   Checking {host}...")  # DEBUG

    # Linux: read loadavg first 1-min field
    try:
        result = subprocess.run(
            f"ssh -o ConnectTimeout=5 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "
            f"-o BatchMode=yes {host} 'cat /proc/loadavg | cut -d\" \" -f1' 2>&1",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        print(f"   Linux check: {repr(output)}")  # DEBUG
        if output and "Permission denied" not in output and "Connection refused" not in output:
            # stdout may contain SSH 'Warning: ...' lines before the number; grab the
            # last whitespace-delimited token that parses as a float.
            for tok in reversed(output.replace("\n", " ").split()):
                try:
                    return float(tok)
                except ValueError:
                    continue
    except Exception as e:
        print(f"   Linux check exception: {e}")  # DEBUG

    # macOS fallback (uptime)
    try:
        cmd = f"ssh -o ConnectTimeout=5 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no " \
              f"-o BatchMode=yes {host} 'uptime'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(f"   macOS check stdout: {repr(result.stdout)}")  # DEBUG
        if result.stdout.strip():
            match = re.search(r'load averages:\s*([0-9.]+)', result.stdout)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"   macOS check exception: {e}")  # DEBUG

    print(f"⚠️  Load check failed for {host}")
    return 999.0


# --- knowledge-graph queries ------------------------------------------------
def get_sophia_voice_logs(hours_back: int = 24) -> list:
    """Sophia voice UI events from the past N hours (ts_ms is epoch millis)."""
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
    return query_neo4j(query)


def get_unprocessed_transcriptions() -> list:
    """Transcriptions that are missing an embedding or source_json (not yet fully processed)."""
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
    return query_neo4j(query)


def get_interesting_memories(min_confidence: float = 0.5) -> list:
    """High-confidence VoiceprintVersion memories from the past 7 days."""
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
    return query_neo4j(query)


def get_conversation_highlights() -> list:
    """Sophia voice events from the last 24h that were not accepted."""
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
    return query_neo4j(query)


def extract_knowledge_insights(voice_logs: list, memories: list) -> dict:
    """Extract actionable insights from voice logs and memories."""
    insights = {
        "timestamp": datetime.now().isoformat(),
        "total_events": len(voice_logs),
        "processed_count": sum(1 for e in voice_logs if e.get("status") == "processed"),
        "unprocessed_count": sum(1 for e in voice_logs if e.get("status") == "unprocessed"),
        "high_confidence_memories": len(memories),
        "topics_found": [],
        "action_items": [],
        "memories_to_add": [],
    }

    all_transcripts = [e.get("transcript", "") for e in voice_logs if e.get("transcript")]
    topic_keywords = ["project", "task", "meeting", "idea", "plan", "decision", "action"]

    for keyword in topic_keywords:
        count = sum(1 for t in all_transcripts if keyword.lower() in t.lower())
        if count > 0:
            insights["topics_found"].append(f"{keyword}: {count}")

    for log in voice_logs:
        transcript = log.get("transcript", "").lower()
        if any(word in transcript for word in ["todo", "remember", "add to knowledge", "note"]):
            insights["action_items"].append({
                "source": log.get("session"),
                "content": log.get("transcript")[:200],
                "confidence": log.get("confidence", 0),
            })

    for memory in memories:
        content = memory.get("content") or ""
        insights["memories_to_add"].append({
            "id": memory.get("id"),
            "content": content[:500],
            "confidence": memory.get("confidence"),
            "source": memory.get("source"),
            "tags": memory.get("tags", []),
        })

    return insights


def save_harvest_report(insights: dict, host: str) -> Path:
    reports_dir = Path.home() / ".hermes" / "knowledge-harvest"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"{host}_harvest_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(insights, f, indent=2, default=str)
    return report_file


def main():
    # Fleet agents to check for idle load. The KG query itself runs locally via Bolt.
    fleet = {
        "x1-370": ("scott", "100.64.43.123"),
        "deathstar-XPS-8920": ("deathstar", "100.78.106.121"),
        "destroyer": ("scott", "100.81.57.77"),
        "scotts-macbook-air": ("scottjoyner", "100.85.64.117"),
    }

    print(f"🎯 Starting knowledge harvesting at {datetime.now().isoformat()}")
    print("=" * 60)

    all_insights = []

    for agent_name, (user, ip) in fleet.items():
        print(f"\n🔍 Checking {agent_name} ({ip})...")
        host = f"{user}@{ip}"
        load = check_agent_load(host)
        print(f"   Load: {load:.2f}")

        if load < 1.0:
            print(f"   ✅ IDLE - Harvesting knowledge from KG (Bolt)...")
            voice_logs = get_sophia_voice_logs(hours_back=24)
            memories = get_interesting_memories()
            conversations = get_conversation_highlights()

            print(f"   Found {len(voice_logs)} Sophia events")
            print(f"   Found {len(memories)} high-confidence memories")
            print(f"   Found {len(conversations)} highlighted conversations")

            insights = extract_knowledge_insights(voice_logs, memories)
            insights["agent"] = agent_name
            insights["load_at_harvest"] = load

            all_insights.append(insights)
            report_file = save_harvest_report(insights, agent_name)
            print(f"   📄 Report saved: {report_file}")
        else:
            print(f"   ⏳ BUSY (load > 1.0) - Skipping")

    print("\n" + "=" * 60)
    print("🎉 HARVESTING COMPLETE")
    print(f"   Agents harvested: {len(all_insights)}")
    for insights in all_insights:
        agent = insights["agent"]
        print(f"   • {agent}: {insights['total_events']} events, "
              f"{insights['high_confidence_memories']} memories, "
              f"{len(insights['action_items'])} action items")

    harvest_dir = Path.home() / ".hermes" / "knowledge-harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    summary_file = harvest_dir / "harvest_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "agents_harvested": len(all_insights),
            "insights": all_insights,
        }, f, indent=2, default=str)

    print(f"   Summary saved: {summary_file}")
    close_neo4j()


if __name__ == "__main__":
    main()
