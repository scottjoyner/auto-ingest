"""Topic-typed hook bank + content-format cue helpers.

Implements the highest-leverage editorial strategies from the content review:
  * Topic-Typed Hook Bank   - first 1.5s hook varies by topic type
                             (paper / concept / debate / opinion / utterance).
  * Curiosity-gap payoff     - a reserved `payoff` cue at the end, threaded to
                             the branded end card, with a mid-video `callback`.
  * Myth vs Fact             - `myth`/`fact` cue kinds for shareable splits.
  * Series wrapper           - `series` intro lockup + Part N/M badge.

These are pure functions over the Brief/Cue model; no Neo4j required, so they
are unit-testable without the graph.
"""
from __future__ import annotations

from typing import List, Optional

from auto_ingest.shorts.models import Brief, Cue

# Hook templates keyed by topic type. `{}` is filled with the paper/concept
# name or a short claim fragment. Kept <=7 words where possible for punch.
_HOOK_TEMPLATES: dict = {
    "paper": [
        "This paper broke my brain: {}",
        "A 2024 paper quietly fixed {}",
        "Researchers just settled {}",
        "The paper everyone cites on {} is wrong",
    ],
    "concept": [
        "You've been using {} wrong",
        "{} explained before you scroll",
        "Stop confusing {} with the easy version",
        "The one idea behind {}",
    ],
    "debate": [
        "Everyone argues about {}. They're both wrong.",
        "The {} debate has a third answer",
        "Two camps hate each other over {}",
        "What both sides of {} miss",
    ],
    "opinion": [
        "Unpopular take on {}:",
        "Hot take at 70mph: {}",
        "My real opinion on {}",
        "Nobody says this about {}",
    ],
    "utterance": [
        "In my own words: {}",
        "What I actually said about {}",
        "Off the cuff on {}:",
        "No paper, just me on {}",
    ],
}

# Fallback when topic type is unknown.
_DEFAULT_TEMPLATES = [
    "Here's the real story on {}",
    "{} in 30 seconds",
    "What you missed about {}",
]


def topic_type_of(brief: Brief) -> str:
    """Infer a topic type from brief metadata (tags / topic / title)."""
    t = (brief.topic or "").lower()
    tags = {str(x).lower() for x in (brief.tags or [])}
    blob = f"{t} {' '.join(tags)}"
    if "debate" in blob or "vs" in blob:
        return "debate"
    if "opinion" in blob or "hot take" in blob or "take" in blob:
        return "opinion"
    if "paper" in blob or "arxiv" in blob or "2024" in blob or "study" in blob:
        return "paper"
    if "utterance" in tags or t == "":
        return "utterance"
    return "concept"


def hook_for(brief: Brief, *, variant: int = 0, topic_type: Optional[str] = None) -> str:
    """Pick a hook for the topic type, rotating by ``variant``.

    ``variant`` lets each short in a batch open with a different line so the
    same topic doesn't repeat its cold-open across Part 1/2/3.
    """
    ttype = topic_type or topic_type_of(brief)
    name = (brief.title or brief.topic or "this").strip()
    # Prefer the curated hook as the name fragment for paper/concept types.
    frag = name
    templates = _HOOK_TEMPLATES.get(ttype, _DEFAULT_TEMPLATES)
    tmpl = templates[variant % len(templates)]
    return tmpl.format(frag)


def build_hook_cues(brief: Brief, duration: float, *,
                    short_index: int = 0, total_parts: int = 1,
                    topic_type: Optional[str] = None) -> List[Cue]:
    """Build the opening hook cue(s) for one short.

    Returns a list with the hook (always) plus a mid-video callback pill when
    the short is part of a multi-part series (curiosity-gap device).
    """
    ttype = topic_type or topic_type_of(brief)
    hook_text = hook_for(brief, variant=short_index, topic_type=ttype)
    lead = 0.4
    cues: List[Cue] = [
        Cue(start=round(lead, 2), end=round(lead + 3.0, 2),
            text=hook_text, kind="hook"),
    ]
    if total_parts > 1:
        # Callback pill ~60% through: promise the payoff at the end card.
        cb_t = round(lead + (duration - lead) * 0.6, 2)
        cues.append(Cue(
            start=cb_t, end=round(cb_t + 2.5, 2),
            text=f"Part {short_index + 1} of {total_parts} — payoff at the end",
            kind="callback"))
    return cues


def build_payoff_cue(brief: Brief, duration: float) -> Cue:
    """Reserved end cue threaded to the branded end card (curiosity reward)."""
    lead = 0.4
    start = round(max(lead, duration - 3.2), 2)
    return Cue(
        start=start, end=round(min(duration, start + 3.0), 2),
        text="Part of your own research. Not a paper.",
        kind="payoff")


def curiosity_reveal(question: str, answer: str, duration: float, *,
                     start: Optional[float] = None) -> List[Cue]:
    """Curiosity-gap device: pose a question, then karaoke-reveal the answer.

    Returns two cues: a `question` cue (tease, no highlight) and a `reveal`
    cue (the payoff, karaoke-highlighted word-by-word so the answer 'unfolds'
    on screen). Used mid-video to hold retention before the end payoff.
    """
    lead = 0.4
    if start is None:
        start = round(lead + (duration - lead) * 0.45, 2)
    q_end = round(start + 2.2, 2)
    a_end = round(min(duration - 0.4, start + 5.0), 2)
    return [
        Cue(start=start, end=q_end, text=question, kind="question"),
        Cue(start=q_end, end=a_end, text=answer, kind="reveal"),
    ]


def build_series_intro(brief: Brief, total_parts: int, short_index: int) -> Cue:
    """Series lockup cue: branded 'Part N of M' opener badge.

    Placed at the very top of a multi-part short so a returning viewer knows
    where they are in the run. Single-part topics don't use this.
    """
    lead = 0.0
    label = (brief.title or brief.topic or "Series").strip()
    return Cue(
        start=round(lead, 2), end=round(lead + 2.6, 2),
        text=f"{label} — Part {short_index + 1} of {total_parts}",
        kind="series",
    )


def build_series_outro(brief: Brief, total_parts: int, short_index: int,
                        duration: float = 12.0) -> Cue:
    """Series outro cue: points to the next part (or wraps the run).

    For the final part this becomes a 'series complete' wrap; earlier parts
    tease the next installment to drive follow-through. Placed near the end
    (after the payoff) to avoid colliding with the opening hook.
    """
    start = round(max(0.4, duration - 2.6), 2)
    if short_index + 1 >= total_parts:
        text = f"Series complete — {brief.title or brief.topic}"
    else:
        text = f"Next: Part {short_index + 2} of {total_parts}"
    return Cue(
        start=start, end=round(min(duration, start + 2.4), 2),
        text=text, kind="series",
    )


def myth_fact_pair(claim: str, correction: str) -> List[Cue]:
    """Two back-to-back cues: a red 'myth' then a green 'fact' (shareable)."""
    return [
        Cue(start=0.4, end=3.2, text=f"Myth: {claim}", kind="myth"),
        Cue(start=3.4, end=6.4, text=f"Fact: {correction}", kind="fact"),
    ]
