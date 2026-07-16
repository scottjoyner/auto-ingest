# archive/ — preserved dead / orphan / duplicate / version-sprawl scripts

These files were moved here during the W-43 remediation (LLD §3.4) instead of
being deleted, to preserve history. They are **not** imported by `bin/auto-ingest`
or the `auto_ingest` package and are not part of the supported pipeline.

## Why each was moved (2026-07-16)

### Version sprawl — `postprocess_audio*`
- `postprocess_audio.v2.py`, `postprocess_audio.v3.py`, `postprocess_audiov4.py`
  — superseded variants of `postprocess_audio.py`. Kept the base `postprocess_audio.py`
  as canonical; these three are abandoned iterations (no importers).

### Version sprawl — `transcriber1-6*`
- `transcriber2.py` … `transcriber6.py` — duplicate/iterative transcribers.
  Kept `transcriber.py` as the canonical base. None of 2–6 are referenced anywhere.

### `yolo_vehicle_detction*` — differ only by a commented regex
- `yolo_vehicle_detction_F.py`, `yolo_vehicle_detction_R.py` — per-letter regex
  variants of `yolo_vehicle_detction.py`. Kept `yolo_vehicle_detction.py` +
  `yolo_vehicle_detction_2.py`. (Note: `yolo_vehicle_detction_2.py` is the
  preferred detector used by the dashcam pipeline.)

### `dashcam_yolo_embeddings*` — keep `_2` / `_ents`
- `dashcam_yolo_embeddings.py` (base, unreferenced) — the canonical detector is
  now `auto_ingest/dashcam/yolo_embeddings.py` (moved from `dashcam_yolo_embeddings_2.py`).
  `dashcam_yolo_embeddings_ents.py` retained at repo root as a complementary variant.

### `ingest_transcriptions.py` — older, unreferenced
- Superseded by `auto_ingest/ingest/transcripts.py` (was `ingest_transcriptsv5_3.py`).
  The run scripts (`runall.sh`, `run_all_optimized.sh`, `vector_search.sh`) were
  repointed at the packaged module.

### `diarize_and_transcribe copy.py` — literal copy
- A byte-duplicate of `diarize_and_transcribe.py` (note the space in the name).
  Renamed to `diarize_and_transcribe_copy.py` on move to avoid the awkward space.

### `metata_scraper_iterator.py` — typo duplicate
- Misspelled duplicate of `metadata_scraper_iterator.py`. Archived; the correctly
  spelled module remains at repo root.

## `quality_api/` — source lost (UNRECOVERABLE)

The `quality_api/` directory was committed (and present) with **only** `.pyc`
bytecode — no `.py` source for `__init__`, `config`, `app`, `embeddings`,
`neo4j_client`, or `services/transcript_service`. Python 3.12 `.pyc` files embed
a hash of the source path + size and cannot be reliably decompiled to working
source. No script in the repo imports `quality_api` (grep found zero references),
so it is not on any import path.

- Moved to `archive/quality_api_pyc_only/` for preservation.
- **Resolution:** treated as unrecoverable. If the source resurfaces, restore it
  under `auto_ingest/` (or a new `quality/` subpackage) and add tests; do not
  re-add a `.pyc`-only package.

_See LLD §3.4 W-44._


