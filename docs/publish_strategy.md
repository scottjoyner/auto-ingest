# Publishing Strategy — AI/ML Research Shorts

Status: **PLANNING ONLY.** No accounts created, no OAuth, no uploads. This doc
captures the strategy to execute *after* accounts exist and credentials are
authorized. Everything here is designed to stay fully detached from personal
accounts.

## Niche

**AI/ML research shorts** — bite-sized, cinematic explainers built from the
local knowledge graph (papers, concepts) and your own discussion clips. Brand
voice: "the part of your own research nobody cites" — curious, slightly
contrarian, never clickbait-empty. The pipeline already produces this; the
content pillars are:

| Pillar            | Source in graph                | Example angle                          |
|-------------------|--------------------------------|---------------------------------------|
| Concept explainer| Concept + Paper                | "You've been using X wrong"           |
| Paper bust        | Top paper Chunk                | "This paper quietly fixed X"          |
| Myth vs Fact      | misconception + paper correction| "Most people assume X. Research says Y"|
| Your own words    | DiscussionClip utterances      | real spoken lines, time-aligned B-roll|
| Series            | multi-part Topic              | "Part N of M" curiosity arc           |

## Accounts (fully separate from personal identity)

Create **three independent accounts** from a clean, isolated environment (NOT
your personal phone/app/browser) using a single dedicated "brand" email + one
virtual/SMS-receiving number. This keeps them off your personal device ID, IP,
and credentials.

| Platform  | Account type                        | Verification needed        | Isolation notes                              |
|-----------|-------------------------------------|----------------------------|----------------------------------------------|
| YouTube   | Brand Account (custom name)         | email (non-Gmail OK)       | Create under a NEW Google account made with the brand email. Not your personal Google. |
| TikTok    | Standard account                    | unique email + phone number| One number = one account. Use the virtual number. Create in an isolated browser profile. |
| Instagram | Standard account (business/profile)| unique email + phone       | Business profile for insights. Separate email. |

**Hard rules to avoid linkage / bans:**
- One unique email AND one unique phone number per account (TikTok rejects reuse).
- Never log into these from your personal device/browser/IP if avoidable; use a
  dedicated browser profile or a separate machine/anti-detect environment.
- Do NOT cross-interact (no liking/commenting between the three accounts).
- Enable 2FA on each (authenticator app) immediately after creation.
- Keep recovery credentials in a shared secrets manager, not in the repo.

## Handles / branding (proposed — confirm before creating)

- Channel name stem: `Research in the Fast Lane` (or `Paper in 30 Seconds`).
- Handles: `@researchfastlane` (IG/TikTok), `ResearchInTheFastLane` (YT).
- Avatar: high-contrast topic pill on dark; Banner (YT): dashcam highway still.
- Bio: "AI/ML papers, explained before you scroll. Built from a real research
  graph + my own drives. New here — part of the series."

## Cadence

**2–3 posts/day per platform → 6–9 posts/day total.** Spread across the day to
match each platform's peak windows:

| Platform  | Posts/day | Suggested local times (window)     |
|-----------|-----------|------------------------------------|
| TikTok    | 2–3       | 11:00, 17:00, 20:00                |
| YouTube   | 2–3       | 12:00, 18:00, 21:00                |
| Instagram | 2–3       | 12:30, 19:00, 21:30                |

## Warm-up (first 14 days)

New accounts should NOT immediately post 9/day. Ramp to avoid shadowbans:
- Days 1–3: 1 post/day total (one platform test).
- Days 4–7: 1 post/day per platform (3/day).
- Days 8–14: 2 posts/day per platform (6/day).
- Day 15+: full 2–3/day per platform (6–9/day).

## Content rotation

Rotate pillars so the feed isn't monotonous. See `publish_schedule.json` for the
machine-readable 7-day rotation mapping topics → days → platforms. Topics
available from the pipeline: `large_language_models`, `diffusion_models`,
`graph_neural_networks`, `reinforcement_learning`, `computer_vision`,
`retrieval_augmented_generation`, `robotics`, `multi_agent_systems`,
`knowledge_graph`, plus `montage`/`highlights`/`trip_story` as B-roll filler.

## Future work (post-auth backlog)

1. Acquire the brand email + virtual number; create the 3 accounts in isolation.
2. Run `publish auth {youtube|tiktok|instagram}` to bootstrap tokens.
3. Wire `scheduling.py` to emit a daily upload list from `publish_schedule.json`
   + available rendered shorts (currently `scan_rendered` + `queue_all_on_disk`).
4. First live upload: ONE platform (YouTube) at 1 post, verify it lands, then
   expand per the warm-up curve.
5. Add analytics hooks (YT Analytics API / TikTok/IG insights) once live.

## Risks / guardrails

- TikTok device-fingerprint linking is the biggest ban risk — isolation matters.
- Phone-number reuse is blocked by TikTok; budget for a rental number.
- Don't post identical caption across platforms verbatim (platforms downrank
  cross-posted dupes) — the uploader already varies titles/hashtags per platform.
- Keep personal accounts completely out of this flow.

## Branding & assets

See [`docs/brand/`](brand/):
- [`brand_spec.md`](brand/brand_spec.md) — palette, typography, handles, bios, voice.
- [`account_runbook.md`](brand/account_runbook.md) — step-by-step account creation in isolation (pre-auth).
- [`avatar.png`](brand/avatar.png) + [`banner_youtube.png`](brand/banner_youtube.png) — generated by
  [`scripts/gen_brand_assets.py`](../scripts/gen_brand_assets.py) (reproducible; matches the video-thumbnail palette).
- [`captions.md`](brand/captions.md) — per-platform caption blueprints + voice rules.
- [`brand_manifest.json`](brand/brand_manifest.json) — machine-readable handles/bios/assets; checked by
  `python3 -m auto_ingest.shorts.cli publish brand-check`.

Verify assets/handles/bios any time with:
```
python3 -m auto_ingest.shorts.cli publish brand-check
```

## On-screen personality (persona)

The shorts get a real "YouTuber" presence via `auto_ingest/shorts/persona.py`:
the owner-voice clone (XTTS-v2, from the `scott` voiceprint) narrates the full
script, and a face-cam is composited over the B-roll. By default the face-cam
is a **stylized brand avatar** (the monogram card); when a photo/short video of
the owner is configured (`PERSONA_SOURCE=photo|video`, `PERSONA_FACE_PATH=...`)
on a GPU host, an **ONNX talking-head** (Wav2Lip / MuseTalk ONNX via
`onnxruntime`) drives it from the cloned audio. We run on **AMD** (Strix Halo
Ryzen AI 395 Radeon 8060S, or fallback RX 480) — NOT CUDA — so the engine is
ONNX `onnxruntime` with a Vulkan (Mesa RADV) or ROCm execution provider. The
pipeline is already wired so real cloning activates later with no code change —
only the source asset + an AMD box with Vulkan/ROCm drivers. Render with
`render_short(..., persona=True)` (or `persona="photo"`/`"video"`).
