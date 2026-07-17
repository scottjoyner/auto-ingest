# PLAN: Personal iPhone Media Recall + Broader Pipeline Hardening

Status: PLAN (not yet built — awaiting sign-off)
Date: 2026-07-17
Owner: Scott

## 0. Scope boundary (decided)

- **iPhone media is PERSONAL, not content.** It is NOT used as research-short B-roll.
  Research-shorts stay papers + dashcam (unchanged).
- The iPhone work delivers a **personal memory-recall** feature: embed photos/videos,
  link them to trips / phonelogs / places / GPS in Neo4j, and let Scott query his own
  history ("photos from the Seattle trip where we talked about embeddings").
- **Both embedding modes available**: image-only CLIP (already present, `ViT-B/32` 512d)
  AND text+image CLIP (so natural-language + concept queries work). Flexibility.
- **Full recall feature**: storage + linking + a usable search query layer (CLI).
- **Plus broader hardening** of the research-shorts pipeline and remaining gap items.

## 1. Personal recall architecture

```
iPhone media (Nextcloud/local)
   └─ ingest_media.py  (already ingests pictures+movies, EXIF GPS, CLIP embed)
        MediaFile { sha256, kind, date, gps_lat, gps_lon, embedding(512), embedding_text(768?), path, thumb }

Linking layer (NEW):  auto_ingest/personal/link_media.py
   (MediaFile)-[:AT_PLACE]->(SummaryPlace)        via GPS haversine (template: enrich_phonelog_place.py)
   (MediaFile)-[:DURING]->(Trip)                   via GPS proximity to Trip LocationEvents
                                                  (MediaFile has no epoch_ms → GPS-spatial join, not temporal)
   (MediaFile)-[:NEAR {meters}]->(PhoneLog)        via gps_lat/lon proximity to PhoneLog.loc

Embedding layer (NEW/EXTEND):  auto_ingest/personal/embed.py
   - reuse _clip_model() (open_clip ViT-B/32, 512d) for image-only
   - add text+image CLIP (e.g. open_clip ViT-B/32 also encodes text → 512d text emb,
     OR a 768/1024 model) for (MediaFile)-[:DESCRIBED_BY]->(Concept) semantic links
   - store both: m.embedding (image), m.text_embedding (text query side)

Recall query layer (NEW):  auto_ingest/personal/recall.py  +  bin/auto-ingest recall ...
   - similar-media  : ANN over MediaFile.embedding (template: vector_search.similar-frames)
   - recall         : "show media during <trip/place> where <text query>" →
                      text emb → top-K MediaFile via vector index, filtered by AT_PLACE/DURING/NEAR
   - geo-media      : radius search by gps (template: vector_search.geo-frames)
```

### 1.1 Required graph changes
- **Add `MediaFile.captured_at` / `epoch_millis`** during ingest (`ingest_media.py` `image_meta`/`video_meta`)
  so future temporal joins are possible. Date-only `date` is insufficient for `DURING` Trip.
  (Non-blocking for v1; v1 uses GPS-spatial join.)
- **Create vector index** `media_embedding_index` on `MediaFile.embedding` (cosine)
  via `vector_search._create_vector_index` (reusable). Also `media_text_embedding_index`.
- **Relationships** use established vocabulary: `AT_PLACE`, `DURING`, `NEAR` (mirrors
  `IN_TRIP`, `BELONGS_TO`, `AT_PLACE`, `NEAR` already in graph).

### 1.2 Embedding model decision
- Image-only: reuse existing `_clip_model()` (open_clip ViT-B/32, 512d) — zero new deps.
- Text+image: CLIP text encoding shares the SAME ViT-B/32 space (512d), so a text query
  embeds to the same space as the image emb → no extra model needed for cross-modal.
  This satisfies "both available" with ONE model: image emb for `similar-media`,
  text emb (same model) for `recall` NL queries. (Upgrade to a larger CLIP later if wanted.)

### 1.3 Slideshow reels (personal, separate from content)
- `build_slideshows` currently groups by capture DATE only. Improve to:
  - group by **Trip / Place** (using new `AT_PLACE`/`DURING` links) instead of raw date,
  - optional embedding-based dedup/quality ordering (drop near-duplicate frames),
  - keep ffmpeg Ken-Burns + xfade, but select better cover frame.
- This is the "slideshows are bad" fix — now it's a personal-reel generator, not content.

## 2. Implementation steps (proposed build order)

### Phase A — Personal recall (the iPhone work)
- [A1] `auto_ingest/personal/embed.py`: wrap CLIP — `embed_image()` (image 512d, reuse),
      `embed_text()` (text 512d, same model), `ensure_media_indexes()` (idempotent
      CREATE INDEX IF NOT EXISTS for MediaFile.embedding + text_embedding).
- [A2] `auto_ingest/personal/link_media.py`: GPS haversine linker
      MediaFile→SummaryPlace (template enrich_phonelog_place.py), MediaFile→Trip
      (proximity to Trip LocationEvents), MediaFile→PhoneLog (NEAR). Idempotent MERGE,
      bounded/index-backed, uses `db_retry.with_driver`. Resumable (cursor/checkpoint).
- [A3] `auto_ingest/personal/recall.py` + `bin/auto-ingest recall` subcommands:
      `similar-media --sha/--path`, `recall --text "..." [--place/--trip/--since/--until]`,
      `geo-media --lat --lon --radius`. Mirror vector_search.py CLI style.
- [A4] Wire `ingest_media.py` to: populate `captured_at`/`epoch_millis`, run embed +
      link as post-ingest steps (behind flags to keep ingest fast), record
      `MediaFile.embedding`/`text_embedding` + `linked_at`.
- [A5] Improve `build_slideshows` to group by Trip/Place + embedding dedup.
- [A6] Tests: `tests/test_personal_embed.py`, `test_personal_link.py`, `test_recall.py`
      (synthetic MediaFile/Trip/Place fixtures, no live 21M-node graph required for unit).

### Phase B — Broader hardening (research-shorts + gaps)
- [B1] Fix "five parallel short generators" inconsistency: document which is canonical
      (`compose.py` via `render_short`), mark legacy (`tiktok_shorts.py`, `shorts_builder.py`,
      `smart_shorts.py`, `highway_montoge.py`, `generate_short_1.py`) DEPRECATED in
      `docs/PLAN_orchestration.md`; add imports/usages lint gate.
- [B2] Talking-head: keep `make_talking_head()` stub but harden the persona degrade path
      and add a real ONNX Wav2Lip/MuseTalk loader probe (CPU-safe) so dropping in the
      model on the Strix Halo host "just works" (no code change per brand_spec).
- [B3] Address remaining unaddressed GAP_ANALYSIS items not yet done (review doc,
      pick P1/P2 leftovers: e.g. metrics loop completion, schedule persona_source
      wiring already done — verify, O-G* watchdog already done).
- [B4] Migrate `knowledge_sync_handler.py` hardcoded IPs to config (leftover from G2 scope).
- [B5] Re-run ruff (`ruff check auto_ingest tests`) + both test suites; confirm green.
- [B6] Commit in logical chunks; push after sign-off.

## 3. Constraints honored
- Neo4j ~21M nodes → all graph access index-backed, bounded, paged; no full-graph aggs.
- Writes via `db_retry.with_driver` (OOM/TransientError safe).
- Publishing HELD (no OAuth/live). Personal recall is local query only — no upload.
- AMD-only: CLIP runs on CPU (fine for embed; no GPU needed). No CUDA deps.
- New code ruff-clean; pre-existing debt dirs untouched.

## 4. Open questions (non-blocking, will default if unanswered)
- Text+image: same ViT-B/32 space (recommended, 1 model) vs larger CLIP (better but
  heavier). → default: same ViT-B/32.
- Recall NL queries: pure vector top-K + graph filter (recommended) vs LLM-rewrite
  of query into Cypher. → default: vector + filter (no LLM dependency).
