# Content OS

Content OS is a local-first, file-based workflow for capturing ideas, routing them, building compact writer briefs, drafting in a creator voice, verifying quality, requiring human approval, preparing scheduler handoff, and learning from feedback.

It is intentionally **approval-gated**. The CLI does not scrape social platforms, does not bypass platform rules, and does not auto-publish. External inputs should come from user-provided files, approved APIs, exports, or manual notes. Postiz support is an export/handoff layer only unless an approved draft integration is explicitly added later.

## Folder structure

`content-os init` creates:

```text
content-os/
  strategy/                 positioning, audience, pillars, source watchlist
  voice/                    voice profile and master avoid-slop rules
  stores/                   inbox, workboard, ideas, hooks, proof, feedback, winners, losers
  runs/active/              one active content object per run folder
  runs/archive/             archived or example runs
  modules/writer/           writer skill, references, templates
  workflows/                lifecycle docs and checklists
  config.yaml               score and feedback thresholds
```

Each run folder contains `content-object.yaml`, `idea.md`, `brief.md`, `draft-package.md`, `verification.md`, `review.md`, `scheduler-handoff.md`, and `feedback.md` as they are created.

See [docs/architecture.md](docs/architecture.md) for the implementation diagram, lifecycle gate diagram, module responsibilities, and autoingest integration flow.

## Lifecycle

`captured → idea_review → brief_ready → drafting → verification → human_review → approved → scheduler_ready → scheduled → published → feedback_24h → feedback_72h → learned → archived`

Backward movement requires force in commands that expose it and is recorded in review notes when relevant.

## Routes

- `ORIGINAL`: user notes, proof, strategy, voice, personal knowledge, and prior content.
- `REPURPOSE`: user-owned content converted into a new format.
- `REWRITE`: user-supplied external signal rewritten through the creator viewpoint; attribution metadata must be preserved.
- `RESEARCH_IDEATE`: topic exploration that feeds candidate ideas instead of final posts.

## CLI usage

```bash
content-os init
content-os capture --title "Raw idea" --text "A note from today" --route auto
content-os new-run --title "Bookmarkable content systems" --route ORIGINAL --format x_thread
content-os brief 2026-05-bookmarkable-content-systems
content-os draft 2026-05-bookmarkable-content-systems
# Edit draft-package.md to replace TODOs with proof-backed copy before verifying/approving.
content-os verify 2026-05-bookmarkable-content-systems
content-os approve 2026-05-bookmarkable-content-systems
content-os scheduler-handoff 2026-05-bookmarkable-content-systems --platform x
content-os postiz-export 2026-05-bookmarkable-content-systems --out ./postiz_payload.json
content-os feedback 2026-05-bookmarkable-content-systems --views 10000 --bookmarks 600 --likes 300 --replies 40 --reposts 80
```

Use `content-os capture --no-run` when you only want to append to `stores/inbox.md`. Use `content-os status` for active work and `content-os doctor` to validate structure, config, environment, and YAML/Markdown health.

## LLM configuration

The system works without an LLM; `draft` creates a structured placeholder with TODOs. To use an OpenAI-compatible chat endpoint, set:

```bash
CONTENT_OS_LLM_BASE_URL=https://api.openai.com/v1
CONTENT_OS_LLM_API_KEY=...
CONTENT_OS_LLM_MODEL=...
CONTENT_OS_LLM_TEMPERATURE=0.4
CONTENT_OS_LLM_MAX_TOKENS=1200
```

### LM Studio example

```bash
export CONTENT_OS_LLM_BASE_URL=http://localhost:1234/v1
export CONTENT_OS_LLM_API_KEY=lm-studio
export CONTENT_OS_LLM_MODEL=local-model
content-os draft <RUN_ID>
```

## Postiz handoff flow

After verification passes and a human approves, `scheduler-handoff` creates approved copy, platform notes, attribution, UTM fields, and a Postiz-ready JSON block. `postiz-export` writes a JSON payload with `requires_manual_review=true`. This project does not auto-publish.

## Reviewing voice over time

Put proven examples in `stores/proof/` and strong openings in `stores/hooks/`. Feedback stores winner/loser lessons and proposes changes in `feedback.md`; it never overwrites `voice/master-avoid-slop.md` automatically.

## Privacy and compliance

Do not scrape X/Twitter or other platforms. Use manual inputs, user-owned content, exports, and approved APIs. Always review generated drafts and scheduler payloads before publishing.

## Autoingest repo integration

This repository already contains local media/transcript/metadata tooling. Content OS can now adapt those approved local outputs into proof and run folders without scraping anything:

```bash
content-os scan-inputs ./transcripts --out ./content-os/stores/proof/input-scan.md
content-os ingest-source ./transcripts/interview_segments.json --title "Lessons from the ingest pipeline" --route REPURPOSE --format linkedin
content-os ingest-source ./Dashcam_Research.pdf --title "Dashcam research notes" --no-run
```

Supported local input families include Markdown/text notes, transcript exports (`.txt`, `.srt`, `.vtt`, Whisper-style JSON segments), CSV/JSON metadata exports, PDFs when `pypdf` is installed, and binary media manifests. Binary media is never transcribed automatically by Content OS; use this repo's approved local transcription/export scripts first, then ingest the resulting files.

Dynamic source rules live in `content-os/integrations/source-adapters.json`. Add a rule there to support a new extension or JSON shape without changing Python code, then run `content-os list-adapters` to confirm the rule is active. `content-os doctor` validates the adapter registry and reports malformed JSON, invalid extensions, invalid routes, and invalid formats. Ingested sources are tracked by SHA-256 in `stores/proof/source-manifest.json` so proof can be traced back to local inputs.

`RESEARCH_IDEATE` routes write candidate material into `stores/ideas/` so research scans do not become publishable drafts without a separate human-reviewed run.
