"""Hermetic contract tests for ``auto_ingest.shorts.uploader._hook_for`` (r46).

``_hook_for`` chooses the single best hook string for a queued short before the
upload adapters render titles and descriptions. It is a pure function over the
in-memory ``QueueItem`` dataclass: no file, clock, credential, network or DB is
read. These tests pin the documented precedence --

    hook cue  ->  brief hook  ->  title  ->  topic  ->  key

-- together with the whitespace trimming and underscore-to-space normalisation
applied to the fallback fields. No fixture file, live service or generated state
is involved; every input is a literal ``QueueItem``.

This surface had no direct test and no review-report coverage before this round.
"""

from auto_ingest.shorts import uploader


def _item(
    *,
    key="k",
    topic="topic_val",
    title="Title",
    platform="youtube_shorts",
    brief_hook=None,
    hook_cue=None,
):
    """Build a minimal in-memory queue item; no disk path is ever touched."""
    return uploader.QueueItem(
        key=key,
        topic=topic,
        title=title,
        out_path="/nonexistent/out.mp4",
        platform=platform,
        brief_hook=brief_hook,
        hook_cue=hook_cue,
    )


def test_hook_cue_wins_over_brief_title_and_topic():
    item = _item(brief_hook="Brief hook", title="My title", topic="my_topic", hook_cue="Cue hook")
    assert uploader._hook_for(item) == "Cue hook"


def test_hook_cue_is_stripped():
    item = _item(hook_cue="  Cue hook  ", brief_hook="Brief hook")
    assert uploader._hook_for(item) == "Cue hook"


def test_hook_cue_wins_even_with_no_title_or_key():
    item = _item(key="", title="", topic="my_topic", brief_hook="Brief hook", hook_cue="Cue hook")
    assert uploader._hook_for(item) == "Cue hook"


def test_brief_hook_used_when_no_hook_cue():
    item = _item(brief_hook="Brief hook", title="My title", topic="my_topic")
    assert uploader._hook_for(item) == "Brief hook"


def test_brief_hook_is_stripped():
    item = _item(brief_hook="  Brief hook  ", title="My title")
    assert uploader._hook_for(item) == "Brief hook"


def test_title_used_when_no_hook_fields():
    item = _item(title="My title", topic="my_topic", key="k")
    assert uploader._hook_for(item) == "My title"


def test_title_underscores_become_spaces_and_edges_are_stripped():
    item = _item(title="  some_cool_name  ", topic="my_topic", key="k")
    assert uploader._hook_for(item) == "some cool name"


def test_topic_used_when_title_and_key_are_absent():
    item = _item(key="", title="", topic="large_language")
    assert uploader._hook_for(item) == "large language"


def test_topic_with_spaces_is_preserved_when_key_is_absent():
    item = _item(key="", title="", topic="my research topic")
    assert uploader._hook_for(item) == "my research topic"


def test_key_is_last_resort_with_underscores_normalised():
    item = _item(key="some_key", topic="my_topic", title="")
    assert uploader._hook_for(item) == "some key"


def test_fully_empty_item_yields_empty_hook():
    item = _item(key="", topic="", title="")
    assert uploader._hook_for(item) == ""
