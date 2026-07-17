# Brand & Asset Spec — "Research in the Fast Lane"

Visual identity for the AI/ML research-shorts accounts. Designed to be
reproducible from code (see `scripts/gen_brand_assets.py`) so the avatar,
banner, and video thumbnails share one palette and feel. No personal identity
in any asset.

## Palette

| Token            | Hex        | RGB             | Use                              |
|------------------|-----------|-----------------|----------------------------------|
| Ink (bg)         | `#0B0E14` | (11,14,20)      | Avatar/banner background         |
| Pill cyan (brand)| `#00C8FF` | (0,200,255)     | Pill, accents, handle highlight  |
| Paper white      | `#FFFFFF` | (255,255,255)   | Primary text                     |
| Stroke black     | `#000000` | (0,0,0)         | Text outline for contrast        |
| Accent amber     | `#FFC400` | (255,196,0)     | Hook words / "payoff" highlights |

These match the video thumbnail style (`thumbnail.py`: cyan pill + white hook
text + black stroke on a darkened highway frame), so a viewer sees one brand
from feed → thumbnail → video.

## Typography

- Font: DejaVu Sans Bold (same as thumbnails) for consistency; falls back to
  system bold.
- Avatar: channel initial or a single glyph, large, centered, white on ink.
- Keep all text uppercase or Title Case; no lowercase body copy in logos.

## Handles (proposed — confirm before creating accounts)

| Platform  | Handle                          |
|-----------|---------------------------------|
| YouTube   | `ResearchInTheFastLane`         |
| TikTok    | `@researchfastlane`             |
| Instagram | `@researchfastlane`             |

Channel display name: **Research in the Fast Lane**.

## Assets to produce

### 1. Avatar (profile picture)
- Square 1000×1000 (renders at 320 on IG, 200 on YT/TikTok).
- Ink background, cyan ring, centered white monogram **"R"** (or a road/lane
  glyph). No photo of a person — keeps it detached from personal identity.
- Generator: `scripts/gen_brand_assets.py --avatar`.

### 2. YouTube banner
- 2560×1440 (safe area 1546×423 centered; text must sit inside 1235×338).
- Ink background, a desaturated highway still at low opacity, cyan channel name
  + tagline "AI/ML papers, explained before you scroll." in the safe area.
- Generator: `scripts/gen_brand_assets.py --banner`.

### 3. TikTok / Instagram profile photo
- Reuse the avatar (square). IG optionally crops to circle — keep the monogram
  centered with padding.

### 4. Bio / description (per platform)
- **YouTube**: "AI/ML research, explained before you scroll. Built from a real
  research graph + my own drives. Part of a series — new here."
- **TikTok**: "papers in 30 seconds 🧠 | AI/ML explainers from a real research
  graph | part of the series"
- **Instagram**: "AI/ML, bite-sized. Research shorts from a real knowledge
  graph. 🧠 Link in bio for the series."

### 5. Video thumbnail (already automated)
- `thumbnail.py` produces the 9:16 cover + 16:9 YouTube variant with the cyan
  pill + hook text. This IS the brand's on-platform visual signature — keep it
  consistent across all three platforms.

## Voice / caption style
- Hook-first, curious, slightly contrarian ("You've been using X wrong").
- Never fake urgency; the curiosity gap does the work.
- End cards / captions carry the branded payoff line
  ("Part of your own research. Not a paper.").

## On-screen personality (persona)
- `auto_ingest/shorts/persona.py` layers a face/voice presence on top of the
  cinematic B-roll so it reads as a real "YouTuber", not just captions.
- VOICE: the owner-voice clone already exists (`tts.py`, XTTS-v2 from the
  `Speaker{user_id:'scott'}` voiceprint) and narrates the full script.
- FACE: configurable via `PersonaConfig` / env (`PERSONA_SOURCE` =
  `stylized` | `photo` | `video`, `PERSONA_FACE_PATH`). When a real photo or
  short video of the owner is configured on a GPU host, an **ONNX talking-head**
  (Wav2Lip / MuseTalk ONNX via `onnxruntime`) drives it from the cloned audio
  on AMD — NOT the CUDA-only SadTalker/Wav2Lip. Until then — and by default — a
  STYLIZED brand avatar (the monogram card) is the on-screen personality. The
  pipeline is already wired so dropping in `photo`/`video` later just works, no
  code change.
- The face-cam composites in a corner (default bottom-right) over the B-roll;
  every stage degrades gracefully (no TTS -> silent; no face model -> stylized
  avatar).

### AMD deployment targets (talking-head GPU)
We have two AMD boxes, no NVIDIA — the engine is ONNX `onnxruntime` with a
Vulkan or ROCm execution provider, prioritized:
- **Strix Halo (Ryzen AI 395, Radeon 8060S, RDNA 3.5)** — preferred. Mesa RADV
  gives a Vulkan EP out of the box; ROCm (gfx115x) is an option for a speed bump
  and the XDNA NPU can accelerate some ONNX nodes. Unified memory helps.
- **AMD RX 480 (Polaris / gfx803)** — ROCm dropped gfx803 years ago, so the
  realistic route is **Vulkan via Mesa AMDGPU** (no modern ROCm). Slower and
  fiddly; use only if the Strix Halo isn't available. The current dev host is a
  Ryzen AI 9 HX 370 + Radeon 890M (same RDNA 3.5 family, smaller NPU) — a good
  proxy for validating the Vulkan path before moving to the 395.

## File layout (generated into the repo)
```
docs/brand/
  avatar.png          # 1000x1000
  banner_youtube.png  # 2560x1440
  brand_spec.md       # this file
scripts/gen_brand_assets.py   # reproducible generator
```

Regenerate anytime with `python3 scripts/gen_brand_assets.py --all`.
Assets are committed so the brand is version-controlled and reviewable.
