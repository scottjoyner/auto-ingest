# Proposed Unified `run` CLI — Design

Goal: **one command, same on every Tailscale machine**, that (a) detects host + hardware, (b) shows where preprocessing is, and (c) dispatches ingest/diarize/link/sync with best-fit backend.

## Command surface
```
auto-ingest                        # alias: run
  status [--source dashcam|audio|bodycam] [--key <k>]   # preprocessing manifest view
  caps                                              # probe + print hardware/backend profile
  ingest      [--day YYYY-MM-DD] [--limit N] [--dry-run]
  diarize     [--day YYYY-MM-DD] [--backend auto]
  link-speakers [--anchor-me] [--global-prefilter]
  sync        [--to-vault|--to-neo4j|--both]
  whoami      [--rebuild] [--corpus ~/me]           # speaker "me" anchor tool
  claim       <key>            # claim a job for this host
  import      <clean.json|csv> # load cleaned intermediate via dynamic tx
```

## `auto-ingest status` output (example)
```
HOST: destroyer (AMD ROCm)   NEO4J: 100.64.43.123  VAULT: NAS4/shared-knowledge
SOURCE: dashcam  pending=12, transcribed=3, diarized=1, linked=0, graph=0
KEY: dashcam/2026-07-14/CLT_1423_F
  copied       ✓ 10:01
  transcribed  ✓ 10:05 (whisper medium / ROCm)
  diarized     ✓ 10:09 (pyannote, 2 spk)
  embedded     … pending
  linked       … pending
  graph_written … pending
  owner: (unclaimed)
```

## How it works
- **Host detect:** `config.yaml` `machine_paths[].hostname_pattern` match → `fileserver_root`, `neo4j_uri`, mount map. Same logic already in `auto_ingest_config.py`.
- **Caps detect (one-time, cached to `~/.hermes/state/caps.json`):**
  - Apple Silicon → MLX (whisper-mlx, mlx-ecapa)
  - `nvidia-smi` present → CUDA
  - `rocminfo` present → ROCm
  - else → ONNX Runtime (CPU, quantized)
  - Each task (transcribe / diarize / embed / yolo) mapped to a backend per host.
- **Manifest:** stored as JSON under `<fileserver_root>/.ingest/manifest/` AND mirrored as `IngestJob` nodes in Neo4j (idempotent `MERGE`). `status` reads it; no full rescans needed.
- **No hardcoded secrets:** reads `.env` / `config.yaml`; password comes from `NEO4J_PASSWORD` only.
- **Claim/import:** `claim` does a conditional `SET owner=$host` (guarded by `owner=""`); `import` runs `UNWIND $rows AS r MERGE ...` batches (dynamic transactions, resumable).

## Why this fixes the gaps
- G3 (one command) — single entrypoint, host-agnostic.
- G4 (step visibility) — `status` + manifest.
- G5/G6 (clean intermediates + claim) — `import` + `claim`.
- G7 (backend routing) — `caps`.
- G10 (password drift) — single source of password.
