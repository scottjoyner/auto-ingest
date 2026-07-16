# KG Ingestion Stack — Deployment Notes

Built 2026-07-16 (xwing session). All three scripts live in the SHARED path
(auto-ingest is symlinked into x1-370 via SSD_4TB), so they're already deployed
on both machines.

## Scripts (in /media/scott/SSD_4TB/hermes-home/auto-ingest/scripts/)
- signal_kg_bridge.py  — Signal chat -> Neo4j (Scott)-[:SENT]->(SignalMessage)-[:ABOUT]->(Concept)
- arxiv_kg_bridge.py   — arXiv query -> Neo4j (Paper/Chunk/Concept, MERGE-only, idempotent)
- curious_agent.py     — idle query system: insight + freshness digest -> Signal

All use the auto-ingest .venv (has neo4j driver) and get_neo4j_config().

## WHERE EACH RUNS (important)
- arxiv_kg_bridge:  RUNS ON xwing (Hermes cron `kg-arxiv-ingest`, daily 03:37).
                     Talks to arXiv API + Neo4j + LM Studio over network.
- curious_agent:     RUNS ON xwing (Hermes cron `kg-curious-agent`, daily 08:17, delivers to Signal).
- signal_kg_bridge:  MUST RUN ON x1-370 — signal-cli account is registered there,
                     not on xwing. The x1-signal-tunnel forwards 18080->x1:8080 but
                     signal-cli itself lives on x1-370.

## SIGNAL BRIDGE SETUP (run this ON x1-370 — needs Tailscale SSH auth)
```bash
# as scott on x1-370:
CRONLINE="*/20 * * * * cd /media/scott/SSD_4TB/hermes-home/auto-ingest && .venv/bin/python scripts/signal_kg_bridge.py >> /media/scott/SSD_4TB/hermes-home/auto-ingest/logs/signal_kg.log 2>&1"
( crontab -l 2>/dev/null | grep -v signal_kg_bridge; echo "$CRONLINE" ) | crontab -
# verify:
crontab -l | grep signal_kg
# ensure signal-cli daemon is up (the bridge calls `signal-cli receive`):
signal-cli -a +170XXXX5781 receive --help   # or check the running signal-cli service
```

## VERIFIED
- xwing -> Neo4j write path: OK (probe create/delete succeeded).
- curious_agent: OK (returns digest JSON; 68,418 papers, 322,904 bridge edges, HOME themes).
- arxiv_kg_bridge: OK, end-to-end. Writes Paper/Chunk/embedding, links to Concept,
  creates attributed SIMILAR edges (method:'semantic', score) between freshly-ingested
  papers only. Idempotent (cursor + per-paper try/except on ConstraintError).
- signal_kg_bridge --dry-run: OK (logic sound; 0 envelopes because signal-cli not local).

## SIMILAR-EDGE PRUNE (2026-07-16, follow-up)
- Pruned 1,167,355 buggy SIMILAR edges (all method:null, from an early O(n^2) test run
  that scanned ALL 68k papers). The design uses RELATED_CONCEPT for the research bridge,
  NOT SIMILAR — so those 1.16M edges were 100% noise. Deleted; graph now has 0 SIMILAR.
- FIXED arxiv_kg_bridge bugs found during verification:
  1. Missing `emb` param in Chunk MERGE -> added.
  2. SIMILAR loop was O(n^2) over ALL embedded papers (never finishes) -> restricted to
     only freshly-ingested `new_ids`.
  3. ConstraintError on MERGE (Paper re-created inside Chunk statement) -> changed inner
     `MERGE (pa:Paper...) <- [:PART_OF] - (c)` to `MATCH (pa:Paper...) MERGE (pa)<-[:PART_OF]-(c)`
     so Paper is never re-created. VERIFIED: fresh ingest no longer throws.
  4. SIMILAR step used a possibly-closed session `s` -> moved to a fresh `drv.session()`.
  5. SIMILAR now idempotent: deletes prior SIMILAR among the batch ids before re-merging.
- Proven correct: FRESHPROBE_AAA/BBB ingested cleanly, produced 1 SIMILAR edge
  (score 0.735, method 'semantic'), cleaned to 0. Production graph: 0 SIMILAR, 322,904
  RELATED_CONCEPT (untouched), 68,418 Paper, 100,014 Chunk.

## NOTES / GOTCHAS
- Signal bridge is NON-DESTRUCTIVE: it only reads envelopes; messages are stored in
  Neo4j before any cursor advance. It does NOT delete from signal-cli.
- arxiv_kg_bridge links papers to Concept by keyword; SIMILAR edges via Python cosine
  (safe for small N). Threshold 0.6.
- Curious-agent freshness threshold: STALE_DAYS=7. Graph was 1d fresh at build time.
- The `tencent/hy3:free` model 401'd this session (OpenRouter auth) — unrelated to KG.
