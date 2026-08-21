#!/usr/bin/env python3
"""Lightweight health monitor for the dashcam vision pipeline.

Samples KG counts and writes a single PipelineHealth node (name:'dashcam_vision')
every 30s so the pipeline state is queryable from the graph itself. Also counts
source clips across BOTH dashcam archives (8TB_2025 + 8TBHDD).

Run via the dashcam-monitor systemd service (auto-start on boot).
"""
import os
import time
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026")
BASES = ["/mnt/8TB_2025/fileserver/dashcam", "/mnt/8TBHDD/fileserver/dashcam"]
HEALTH = "dashcam_vision"

_driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PWD))


def count_disk_clips():
    total = 0
    per_mount = {}
    for b in BASES:
        n = 0
        if os.path.isdir(b):
            for root, _dirs, files in os.walk(b):
                n += sum(1 for f in files if f.endswith(".MP4"))
        per_mount[b] = n
        total += n
    return total, per_mount


def main():
    last_disk_scan = 0
    disk_total = 0
    disk_per = {}
    while True:
        now = time.time()
        if now - last_disk_scan > 300:  # re-scan disk every 5 min (walk is slow)
            disk_total, disk_per = count_disk_clips()
            last_disk_scan = now
        with _driver.session() as s:
            total_frames = s.run("MATCH (f:DashcamFrame) RETURN count(f) AS n").single()["n"]
            embedded = s.run("MATCH (f:DashcamFrame) WHERE f.emb_e5_large IS NOT NULL RETURN count(f) AS n").single()["n"]
            processed_clips = s.run("MATCH (c:DashcamClip {processed:true}) RETURN count(c) AS n").single()["n"]
            failed_clips = s.run("MATCH (c:DashcamClip {failed:true}) RETURN count(c) AS n").single()["n"]
            clip_nodes = s.run("MATCH (c:DashcamClip) RETURN count(c) AS n").single()["n"]
            s.run(
                """
                MERGE (h:PipelineHealth {name:$name})
                SET h.total_frames=$tf, h.embedded=$em, h.processed_clips=$pc,
                    h.failed_clips=$fc, h.clip_nodes=$cn, h.clips_on_disk=$cod,
                    h.clips_on_disk_breakdown=$bd, h.updated=datetime()
                """,
                name=HEALTH, tf=total_frames, em=embedded, pc=processed_clips,
                fc=failed_clips, cn=clip_nodes, cod=disk_total, bd=str(disk_per),
            )
            print(f"[{time.strftime('%H:%M:%S')}] frames={total_frames} "
                  f"embedded={embedded} processed={processed_clips} "
                  f"failed={failed_clips} on_disk={disk_total}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
