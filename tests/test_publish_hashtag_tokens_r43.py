"""R43 — publish-queue hashtag tokens must be well formed (offline, pure).

``_hashtags`` feeds both ``_description_for`` and ``plan_report``'s would-be
uploaded caption. A blank topic currently emits a bare ``#`` and a multi-word
topic emits a hashtag containing spaces; neither is a valid hashtag. These
tests pin the token contract with in-memory QueueItems only: no network, no
credentials, no Neo4j, no media and no live queue file.
"""

from auto_ingest.shorts import uploader

PLATFORMS = ("youtube_shorts", "tiktok", "instagram")


def _item(topic, *, platform="youtube_shorts", title="Cool Paper Title"):
    return uploader.QueueItem(
        key="r43-key",
        topic=topic,
        title=title,
        out_path="/nonexistent/r43.mp4",
        platform=platform,
    )


def _tokens(tags):
    return tags.split()


def test_blank_topic_never_emits_a_bare_hash():
    for platform in PLATFORMS:
        tags = uploader._hashtags(platform, _item("", platform=platform))
        assert "#" not in _tokens(tags), (platform, tags)


def test_multiword_topic_becomes_one_clean_token():
    for platform in PLATFORMS:
        tags = uploader._hashtags(platform, _item("large language models", platform=platform))
        tokens = _tokens(tags)
        assert "#largelanguagemodels" in tokens, (platform, tags)
        assert all("#" in t and " " not in t for t in tokens), (platform, tags)


def test_underscore_topic_still_normalizes_to_one_token():
    tags = uploader._hashtags("youtube", _item("diffusion_models"))
    assert "#diffusionmodels" in _tokens(tags)
    assert "#shorts" in _tokens(tags)


def test_description_has_no_malformed_hashtag():
    for platform in PLATFORMS:
        desc = uploader._description_for(platform, _item("ai research", platform=platform))
        assert "#" not in desc.split(), (platform, desc)
        assert "#ai research" not in desc, (platform, desc)


def test_common_platform_tags_are_preserved():
    yt = uploader._hashtags("youtube", _item("", platform="youtube"))
    tk = uploader._hashtags("tiktok", _item("", platform="tiktok"))
    ig = uploader._hashtags("instagram", _item("", platform="instagram"))
    assert "#research" in _tokens(yt) and "#shorts" in _tokens(yt)
    assert "#ai" in _tokens(tk) and "#research" in _tokens(tk)
    assert "learnontiktok" in ig and "savethis" in ig
