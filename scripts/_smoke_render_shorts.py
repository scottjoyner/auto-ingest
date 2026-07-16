"""Smoke-test the full render loop with one real highway clip + a spoken-discussion script.

Self-contained: resolves a local clip path manually because get_fileserver_path
does not remap /mnt/8TB_2025 -> /media/scott/SSD_4TB on this host.
"""
from __future__ import annotations
import os
from pathlib import Path

from neo4j import GraphDatabase
from auto_ingest_config import get_neo4j_password
from auto_ingest.shorts import curator, models

ROOT = "/media/scott/SSD_4TB/fileserver"  # this host's mount


def localize(graph_path: str) -> str:
    return graph_path.replace("/mnt/8TB_2025/fileserver", ROOT).replace(
        "/mnt/8TB_2026/fileserver", ROOT)


def main() -> None:
    d = GraphDatabase.driver("bolt://localhost:7687",
                             auth=("neo4j", get_neo4j_password()))
    try:
        # Spoken discussion about LLMs (W-56 linkage).
        clips = curator.discusses_topic(
            d, "large_language_models", min_score=0.65, min_text_len=40, limit=6)
        assert clips, "no discussion clips"

        # One resolvable highway clip.
        with d.session() as s:
            row = s.run(
                """
                MATCH (f:Frame)-[:BELONGS_TO]->(c:DashcamClip)
                WHERE c.view = 'F' AND f.mph > 55
                  AND c.path CONTAINS '/fileserver/dashcam/2025/01/'
                RETURN c.key AS ck, c.path AS p, max(f.mph) AS mph
                LIMIT 1
                """
            ).single()
        clip_path = localize(row["p"])
        assert os.path.exists(clip_path), clip_path

        # Build a 1-shot short: the spoken hook as a caption over the B-roll.
        hook = clips[0].text.strip()
        brief = curator.brief_from_discussions("large_language_models", clips)
        shot = models.Shot(clip_key=row["ck"], fr_path=clip_path,
                            t_sec=0.0, dur=8.0, mph=row["mph"])
        cues = [
            models.Cue(0.0, 2.5, brief.title, kind="hook"),
            models.Cue(2.5, 5.5, hook[:80], kind="point"),
            models.Cue(5.5, 8.0, "— your own words, not a paper", kind="source"),
        ]
        short = models.PlannedShort(
            id="smoke_llm", brief_topic="large_language_models",
            title=brief.title, cues=cues, shots=[shot])
        plan = models.Plan(topic="large_language_models", brief=brief,
                           shorts=[short])

        out = Path("./shorts_out_smoke")
        from auto_ingest.shorts import render
        path = render.render_short(short, out, overwrite=True)
        print("RENDERED ->", path, "exists=", Path(path).exists(),
              "size=", Path(path).stat().st_size)
    finally:
        d.close()


if __name__ == "__main__":
    main()
