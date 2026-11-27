"""Hermetic contract pin for ``content_os.core.make_payload`` (r32).

``make_payload`` is the single pure builder behind both the scheduler handoff
markdown and the Postiz export JSON.  It is the last gate that stamps the
"manual review only / never auto-publish" safety fields before a run's copy
leaves the content OS.  No existing test (on any branch) or prior review pinned
its output contract; these cases do so without touching the filesystem, DB,
network, clock or any live state.
"""

from __future__ import annotations

from content_os import core
from content_os.models import ContentObject

# The exact eight keys the handoff/export payload is contractually allowed to
# carry.  Any new key is an intentional API change and must be pinned here.
EXPECTED_KEYS = {
    "requires_manual_review",
    "approval_status",
    "platform",
    "title",
    "content",
    "attribution",
    "utm",
    "postiz_note",
}
EXPECTED_NOTE = "Create a draft/queue item only; do not auto-publish."


def obj(**overrides):
    """Build a real ContentObject with deterministic, minimal defaults."""
    kwargs = {
        "id": "2026-11-r32",
        "title": "A Test Idea",
        "slug": "a-test-idea",
        "state": "approved",
        "human_approved": True,
        "attribution": {"source_url": "https://example.test/post"},
    }
    kwargs.update(overrides)
    return ContentObject(**kwargs)


def test_exact_top_level_key_set():
    payload = core.make_payload(obj(), "final copy", "linkedin")
    assert set(payload) == EXPECTED_KEYS


def test_requires_manual_review_true_when_approved():
    payload = core.make_payload(obj(human_approved=True), "final copy", "x_post")
    assert payload["requires_manual_review"] is True


def test_requires_manual_review_true_when_not_approved():
    payload = core.make_payload(obj(human_approved=False), "final copy", "x_post")
    assert payload["requires_manual_review"] is True


def test_approval_status_approved_when_human_approved():
    payload = core.make_payload(obj(human_approved=True), "final copy", "x_post")
    assert payload["approval_status"] == "approved"


def test_approval_status_not_approved_when_unapproved():
    payload = core.make_payload(obj(human_approved=False), "final copy", "x_post")
    assert payload["approval_status"] == "not_approved"


def test_approval_status_pins_truthy_non_bool_behavior():
    # The builder uses a plain truthiness ternary; pin the real behavior so a
    # future "fix" to strict-bool handling is a deliberate, reviewed change.
    payload = core.make_payload(obj(human_approved="yes"), "final copy", "x_post")
    assert payload["approval_status"] == "approved"


def test_platform_echoed_at_top_level():
    payload = core.make_payload(obj(), "final copy", "newsletter")
    assert payload["platform"] == "newsletter"


def test_title_passthrough_verbatim_unicode():
    title = "Café — Déjà vu 日本語"
    payload = core.make_payload(obj(title=title), "final copy", "custom")
    assert payload["title"] == title


def test_content_is_exactly_the_draft_text():
    draft = "line one\n\nline two\n"
    payload = core.make_payload(obj(), draft, "blog")
    assert payload["content"] == draft


def test_content_empty_string_preserved():
    payload = core.make_payload(obj(), "", "x_post")
    assert payload["content"] == ""


def test_attribution_preserved():
    attribution = {"source_url": "https://example.test/a", "author": "Ada", "n": 3}
    payload = core.make_payload(obj(attribution=attribution), "copy", "linkedin")
    assert payload["attribution"] == attribution


def test_attribution_empty_dict_preserved():
    payload = core.make_payload(obj(attribution={}), "copy", "linkedin")
    assert payload["attribution"] == {}


def test_utm_exact_key_set():
    payload = core.make_payload(obj(), "copy", "x_thread")
    assert set(payload["utm"]) == {"campaign", "source", "medium"}


def test_utm_campaign_comes_from_slug_not_id_or_title():
    payload = core.make_payload(
        obj(id="2026-11-r32", title="A Different Title", slug="the-slug"),
        "copy",
        "x_post",
    )
    assert payload["utm"]["campaign"] == "the-slug"


def test_utm_source_comes_from_platform():
    payload = core.make_payload(obj(), "copy", "instagram")
    assert payload["utm"]["source"] == "instagram"
    assert payload["utm"]["source"] == payload["platform"]


def test_utm_medium_is_social():
    payload = core.make_payload(obj(), "copy", "custom")
    assert payload["utm"]["medium"] == "social"


def test_postiz_note_is_exact_do_not_auto_publish_string():
    payload = core.make_payload(obj(), "copy", "x_post")
    assert payload["postiz_note"] == EXPECTED_NOTE
    assert "do not auto-publish" in payload["postiz_note"]


def test_state_does_not_change_payload():
    # State gating is the caller's job (scheduler_handoff/postiz_export); the
    # payload itself is a pure function of approval flag + copy + platform.
    approved = core.make_payload(obj(state="approved"), "copy", "x_post")
    published = core.make_payload(obj(state="published"), "copy", "x_post")
    assert approved == published


def test_object_is_not_mutated():
    thing = obj(human_approved=False)
    before = thing.attribution.copy()
    core.make_payload(thing, "copy", "x_post")
    assert thing.human_approved is False
    assert thing.attribution == before


def test_id_and_internal_fields_do_not_leak_as_top_level_keys():
    payload = core.make_payload(obj(), "copy", "x_post")
    for leaked in ("id", "slug", "state", "created_at", "updated_at", "route"):
        assert leaked not in payload


def test_deterministic_repeated_calls():
    thing = obj()
    first = core.make_payload(thing, "copy", "linkedin")
    second = core.make_payload(thing, "copy", "linkedin")
    assert first == second
