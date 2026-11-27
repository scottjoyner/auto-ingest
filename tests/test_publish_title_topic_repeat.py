"""Offline regression test: YouTube titles must not repeat the topic.

`uploader._title_for("youtube", ...)` builds a keyword-front title of the form
``"<Topic>: <Title>"``. The comment on that branch says the topic must not be
prepended when the title already contains it. The pristine implementation tests
the containment the wrong way round (``base in topic`` instead of
``topic in base``), so a title that already names the topic gets it prepended a
second time, while a title that is only a *substring* of the topic loses the
topic prefix entirely.

Pure string assembly only: no network, no credentials, no filesystem writes,
no Neo4j, no model.
"""
from auto_ingest.shorts import uploader


def _item(topic, title, platform="youtube_shorts"):
    return uploader.QueueItem(
        key="abc123", topic=topic, title=title,
        out_path="/tmp/does_not_exist.mp4", platform=platform,
    )


def _yt(topic, title):
    return uploader._title_for("youtube", _item(topic, title))


def test_title_that_already_names_topic_is_not_prefixed_again():
    """``"<Topic> Explained"`` must stay as-is, not become ``"<Topic>: <Topic> Explained"``."""
    got = _yt("large_language_models", "Large Language Models Explained Simply")
    assert got == "Large Language Models Explained Simply"
    assert got.lower().count("large language models") == 1


def test_title_that_ends_with_topic_is_not_prefixed_again():
    """Mixed-case, underscore-bearing topic names are normalised before the check."""
    got = _yt("diffusion_models", "A Primer on diffusion_models")
    assert got == "A Primer on diffusion models"
    assert got.lower().count("diffusion models") == 1


def test_title_that_is_only_a_substring_of_topic_keeps_the_topic_prefix():
    """A short title such as ``language`` does not contain the full topic."""
    got = _yt("large_language_models", "language")
    assert got == "Large Language Models: language"


def test_unrelated_title_still_gets_topic_prefix():
    """Control: the ordinary, well-formed path is unchanged by the fix."""
    assert _yt("large_language_models", "Cool Paper Title") == \
        "Large Language Models: Cool Paper Title"


def test_title_equal_to_topic_is_not_prefixed_again():
    """Control: exact equality already produced the right answer on the base."""
    assert _yt("diffusion_models", "Diffusion Models") == "Diffusion Models"


def test_empty_title_keeps_its_existing_key_fallback():
    """Control: an empty title still falls back to ``item.key`` (pre-existing behaviour)."""
    assert _yt("diffusion_models", "") == "Diffusion Models: abc123"


def test_title_override_is_not_changed_by_the_containment_fix():
    """Control: A/B title overrides bypass the YouTube prefix branch entirely."""
    it = _item("large_language_models", "ignored")
    assert uploader._title_for("youtube", it, title_override="My Override") == "My Override"


def test_non_youtube_platforms_are_unchanged():
    """Control: the fix touches only the YouTube keyword-front branch."""
    it = _item("large_language_models", "Large Language Models Explained")
    assert uploader._title_for("tiktok", it) == "large language models explained"
    assert uploader._title_for("instagram", it) == \
        "Large Language Models Explained 🔖 #largelanguagemodels"
