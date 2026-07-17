# Caption Templates — AI/ML Research Shorts

Per-platform caption blueprints. The uploader already varies titles/hashtags
per platform (`uploader._title_for` / `_hashtags` / `_description_for`); these
templates define the human-facing copy style and the reusable blocks so every
post stays on-brand. Hook text is pulled from the short's hook cue or brief.

## Shared blocks

```
HOOK      = first sentence of the short (curiosity gap; <= 9 words preferred)
SERIES    = "Part N of M — " (only for multi-part)
CTA        = "Watch the full research breakdown and tell me what to cover next."
BRAND     = "Part of the research-shorts series — bite-sized paper & concept breakdowns."
```

## YouTube Shorts

Title: `<Keyword-first topic> in 30 Seconds` or `<Hook as title>`
Description:
```
{HOOK}

{SERIES if any}This is part of a real research graph + my own drives — not a
paper behind a paywall. {CTA}

{BRAND}

#<topic> #ai #research #machinelearning #shorts
```
Notes: YouTube reads the first 1–2 lines in the preview, so lead with the hook.
Pin the first comment with the source paper link when available.

## TikTok

Caption (bio/body):
```
{HOOK} {SERIES if any}

{payoff one-liner, e.g. "the twist is in the last 3 seconds"}

#<topic> #learnontiktok #ai #machinelearning #fyp
```
Notes: TikTok rewards the first 3 seconds — the hook IS the on-screen caption.
Keep hashtags to 4–5; `#fyp` optional. No link in caption (IG/TikTok downrank
outbound links).

## Instagram Reels

Caption:
```
{HOOK} {SERIES if any}

{BRAND}
Save this for your next deep-dive. {CTA}

#<topic> #ai #machinelearning #research #reels #explainer
```
Notes: IG values saves/shares — the CTA asks for a save. The 9:16 thumbnail is
the cover; pick a frame with the hook visible.

## Voice / style rules
- Hook-first, curious, slightly contrarian ("You've been using X wrong").
- No fake urgency, no all-caps spam, no "follow for more" as the only CTA.
- Vary the hook across a series so Part 1/2/3 don't repeat the cold-open.
- Keep hashtags platform-appropriate (YouTube #shorts, TikTok #learnontiktok,
  IG #reels) — do NOT copy the exact same caption verbatim across platforms
  (cross-posted dupes get downranked).
- End card / payoff line stays consistent: "Part of your own research. Not a paper."
