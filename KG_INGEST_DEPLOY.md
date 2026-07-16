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

## PAPER SIMILARITY SYSTEM (2026-07-16, iteration — closes the gap)
After pruning, there were 0 SIMILAR edges and the bridge only linked fresh papers.
Built a proper paper-to-paper similarity layer:
- `arxiv_kg_bridge` writes `embedding_768` (768-dim, nomic-embed-text-v1.5 via LM Studio).
  NOTE: this is a DIFFERENT space from the corpus-standard `embedding` (384-dim,
  all-MiniLM-L6-v2) used by 61K existing papers + the `paper_embedding` index.
- Created vector index `paper_embedding_768` (768-dim, COSINE, ONLINE) on
  `Paper.embedding_768`.
- `backfill_paper_similar.py` — uses `db.index.vector.queryNodes('paper_embedding_768', k, q)`
  (proper k-NN, NOT O(n^2)) to write `(:Paper)-[:SIMILAR {method:'vector', score}]->(:Paper)`.
  Idempotent (drops method:'vector' edges, rebuilds). Verified: 68 papers -> 680 edges,
  avg 10.0 neighbors/paper.
- `backfill_paper_768.py` — resumable backfill that embeds existing papers' abstracts
  (384-space only) into `embedding_768` so they join the 768 corpus. Processes --limit
  per run, idempotent.
- CRON JOBS (xwing):
  - kg-paper768-backfill: every 6h, --limit 500 (gentle on LM Studio; ~61K in ~2 weeks)
  - kg-paper-similar: daily 04:41, rebuilds vector SIMILAR over the 768 corpus
- VERIFIED: 68 papers in 768-space, 680 vector SIMILAR edges, RELATED_CONCEPT untouched.

## FOLLOW-UP IMPROVEMENTS (2026-07-16, same session)
After "go for it", accelerated + hardened the stack:
- Ran a 2000-paper immediate 768-backfill batch → 2,068 papers now in 768-space.
- Rebuilt SIMILAR over the 2068 corpus → 20,720 vector SIMILAR edges (avg 10.02 deg).
  The 6h cron (kg-paper768-backfill) keeps growing coverage to ~61K over ~2 weeks.
- curious_agent.py improved:
  * Now reports papers_768 (768-vec coverage) + similar_edges in the digest.
  * Added "Most-connected papers" insight (top paper hubs by SIMILAR degree) — the
    actual "papers like this" surfacing you asked for.
- signal_kg_bridge.py hardened:
  * Added ensure_constraints() → NODE KEY (source,ts) on SignalMessage → idempotent
    across re-runs (no duplicate message nodes if signal-cli re-sends envelopes).
  * ABOUT edges now method-tagged ('keyword'). Semantic ABOUT via vector is
    INTENTIONALLY OMITTED: Concepts are 384-dim (all-MiniLM, concept_embedding_index)
    but SignalMessages are 768-dim (nomic) — different spaces, not comparable.
    Keyword ABOUT is correct + sufficient; semantic needs 384-dim message embeddings
    (future improvement: embed messages in 384-dim OR add 768-dim Concept embedding+index).
  * SIGNAL_ACCOUNT now read from env (was hardcoded None).
- All 5 scripts syntax-validated.

## ENHANCEMENT ROUND 2 (2026-07-16, "can we improve further")
Closed real gaps found in an orphan/coverage audit:
- Created `concept_embedding_768` vector index (768-dim, COSINE, ONLINE) on
  Concept.embedding_768. Concepts carry BOTH embedding(384) and embedding_768(270 each).
- bridge_paper_concept_768.py — links each 768-space Paper to its 5 nearest
  Concepts via concept_embedding_768, writing (:Paper)-[:DISCUSSES {method:'vector',score}]->(:Concept).
  This is the missing semantic bridge between the research corpus and the concept/activity
  graph. Ran over all 2068 papers -> 10,340 vector DISCUSSES edges.
  CRON (xwing): kg-paper-concept-bridge daily 04:51 (after the 768-backfill tick).
- signal_kg_bridge.py: RE-ENABLED semantic ABOUT (was disabled in round 1). Now that
  concept_embedding_768 exists, SignalMessages (768) link to Concepts (768) via vector.
  Keyword ABOUT + semantic ABOUT both run; edges method-tagged.
- kg_nl_query.py (NEW) — natural-language -> Cypher against the graph via LM Studio.
  * Auto-selects a loaded chat model (DEFAULT_MODELS fallback list; only one needs VRAM).
  * Forces read-only (rejects CREATE/MERGE/DELETE/SET).
  * Guards against empty/truncated/malformed Cypher (retry once, then clean error).
  * Verified: "top research themes at home" -> theme freqs; "papers about speech
    recognition" -> 59 papers. The conversational upgrade to the idle-query system.
- AUDIT FINDING (not fixed, noted): PriceBar (218k), Filing (44k), Ticker (512) are
  100% orphaned in neo4j — these belong in the `trading` DB, not here. Cleanup/relocation
  is a separate task (see memory: trading subgraph orphaned in neo4j).
- All 7 scripts syntax-validated.

## ENHANCEMENT ROUND 3 (2026-07-16, "idle harvest + life timeline")
- LifeTimeline harvest (NEW): `scripts/life_timeline_harvest.py` stitches Scott's
  activity into one `(:LifeEvent {date})` node per calendar day (85 days,
  2025-09-25..2026-06-16). Each has summary_count, utterance_count, dominant_place
  (from PlaceHour pings), and a text digest. Linked `(:LifeEvent)-[:INCLUDES]->(:Summary)`
  (capped 25/day), `(:LifeEvent)-[:AT]->(:SummaryPlace)`, and chained
  `(:LifeEvent)-[:NEXT]->(:LifeEvent)` in date order (1 head, 1 tail — verified).
  Agents traverse "what happened on / around / after day X" without scanning 181K
  Summaries. Constraint `life_event_date` (date UNIQUE). Cron `kg-life-timeline-harvest`
  daily 04:33 (`--days-back 10` incremental + full NEXT rebuild). Idempotent (MERGE on date).
  GOTCHA handled: Summary.created_at is MIXED (181,641 DateTime + 14 Long ms); day
  normalized via `CASE WHEN created_at IS :: INTEGER THEN datetime({epochMillis:..}) ELSE created_at END`.
- Fixed the fleet idle-harvest `scripts/knowledge_harvest.py` (cron 1035ff22e8eb was
  erroring): (1) load-avg parser now grabs the last float token so SSH "Identity file
  not accessible" warnings don't poison it; (2) removed the over-broad "No such file"
  reject guard (that harmless key warning was disqualifying valid loads); (3) create
  ~/.hermes/knowledge-harvest before the summary write (was crashing at end); (4)
  summary line prints len(action_items) not the list. Now detects idle hosts and
  completes exit 0. NOTE: its nested SSH to `scott@x1-370:/media/scott/S/neo4j` for
  SophiaVoiceUiEvent returns empty (stale path / label) but no longer crashes.

## NOTES / GOTCHAS
- Signal bridge is NON-DESTRUCTIVE: it only reads envelopes; messages are stored in
  Neo4j before any cursor advance. It does NOT delete from signal-cli.
- arxiv_kg_bridge links papers to Concept by keyword; SIMILAR edges via Python cosine
  (safe for small N). Threshold 0.6.
- Curious-agent freshness threshold: STALE_DAYS=7. Graph was 1d fresh at build time.
- The `tencent/hy3:free` model 401'd this session (OpenRouter auth) — unrelated to KG.
