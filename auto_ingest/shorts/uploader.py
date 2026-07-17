"""Real upload adapters for narrated shorts (YouTube / TikTok / Instagram).

This module turns the local publish queue (``publish_queue.jsonl``) into actual
uploads. It is deliberately *credential-gated*: each platform adapter only
activates when its secrets are present in the environment, and the default run
mode is ``dry_run`` (stage + report, no network calls).

No secrets are read from the repo. Expect these env vars per platform:

  YouTube Shorts
    YT_CLIENT_SECRET_JSON  path to an OAuth client_secret.json (installed app)
    YT_TOKEN_JSON          path to a cached OAuth token (created on first auth)
    YT_PRIVACY             "private" | "unlisted" | "public"  (default unlisted)

  TikTok (Content Posting API)
    TIKTOK_ACCESS_TOKEN    long-lived access token from a TikTok developer app
    TIKTOK_OPEN_ID         account open_id (some flows)

  Instagram Reels (Graph API)
    IG_USER_ID             Instagram Business/Creator user id
    IG_ACCESS_TOKEN        long-lived Graph API token

Adapters that lack credentials log a warning and are skipped, so a partial
setup (e.g. YouTube only) still works end-to-end.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

log = logging.getLogger("shorts.uploader")

QUEUE_PATH = Path(os.environ.get(
    "SHORTS_PUBLISH_QUEUE",
    "/media/scott/NAS5/fileserver/dashcam_timelapse/_mix_run/publish_queue.jsonl",
))


@dataclass
class QueueItem:
    key: str
    topic: str
    title: str
    out_path: str
    platform: str
    queued_at: Optional[str] = None
    brief_hook: Optional[str] = None
    hook_cue: Optional[str] = None
    thumbnail_path: Optional[str] = None


def load_queue() -> List[QueueItem]:
    if not QUEUE_PATH.exists():
        return []
    items: List[QueueItem] = []
    for line in QUEUE_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        items.append(QueueItem(
            key=d.get("key", ""), topic=d.get("topic", ""),
            title=d.get("title", ""), out_path=d.get("out_path", ""),
            platform=d.get("platform", "youtube_shorts"),
            queued_at=d.get("queued_at"),
            brief_hook=d.get("brief_hook"),
            hook_cue=d.get("hook_cue"),
            thumbnail_path=d.get("thumbnail_path")))
    return items


def enrich_item_from_plan(item: QueueItem, plans_dir: Path) -> QueueItem:
    """Best-effort: pull brief hook + hook cue + thumbnail from a matching Plan.

    Looks for a Plan JSON whose shorts include ``item.key``; copies its
    ``brief.hook``, the first ``kind=='hook'`` cue text, and the short's
    ``thumbnail_path`` onto the item. No-ops (returns item unchanged) if no
    plan matches or the dir is absent. Pure disk reads — no network/DB.
    """
    plans_dir = Path(plans_dir)
    if not plans_dir.exists():
        return item
    from auto_ingest.shorts.models import Plan
    for plan_json in plans_dir.rglob("*.json"):
        try:
            plan = Plan.load(plan_json)
        except Exception:
            continue
        for s in plan.shorts:
            if s.id != item.key and item.key not in (s.out_path or ""):
                continue
            if item.brief_hook is None:
                item.brief_hook = plan.brief.hook or None
            if item.hook_cue is None:
                for c in s.cues:
                    if c.kind == "hook":
                        item.hook_cue = c.text
                        break
            if item.thumbnail_path is None:
                item.thumbnail_path = s.thumbnail_path
            return item
    return item


def _hook_for(item: QueueItem) -> str:
    """Best hook available: hook cue -> brief hook -> title -> topic."""
    hook = (item.hook_cue or item.brief_hook or "").strip()
    if hook:
        return hook
    base = (item.title or item.key).replace("_", " ").strip()
    topic = item.topic.replace("_", " ").strip()
    return base or topic or item.key


def _published_marker(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".published.json")


def already_published(item: QueueItem) -> bool:
    return _published_marker(Path(item.out_path)).exists()


def _record_published(item: QueueItem, url: str = "") -> None:
    rec = {"key": item.key, "platform": item.platform,
           "url": url, "at": time.time()}
    _published_marker(Path(item.out_path)).write_text(json.dumps(rec))


def _title_for(platform: str, item: QueueItem, *,
               title_override: Optional[str] = None) -> str:
    """Platform-native title templates (keyword-front YT, lowercase TikTok).

    ``title_override`` lets an A/B-test variant supply its own title without
    disturbing the standard platform templating for other callers.
    """
    if title_override:
        # Still apply light platform shaping so the override fits the platform.
        ov = title_override.strip()
        if platform == "tiktok":
            return ov.lower()
        return ov
    base = (item.title or item.key).replace("_", " ").strip()
    topic = item.topic.replace("_", " ")
    if platform.startswith("youtube"):
        # Keyword-front, informative. Avoid repeating the topic if the title
        # already contains it.
        if base and base.lower() not in topic.lower():
            return f"{topic.title()}: {base}"
        return base or topic.title()
    if platform == "tiktok":
        # Casual, lowercase curiosity hook.
        return (base or topic).lower()
    if platform == "instagram":
        return f"{base or topic} 🔖 #{topic.replace(' ', '')}"
    return base or topic


def _hashtags(platform: str, item: QueueItem) -> str:
    topic = item.topic.replace("_", "")
    common = ["ai", "research", "machinelearning", "shorts"]
    if platform == "tiktok":
        return " ".join(f"#{t}" for t in [topic] + common[:3])
    if platform == "instagram":
        return " ".join(f"#{t}" for t in [topic, "ai", "research", "learnontiktok", "savethis"])
    return " ".join(f"#{t}" for t in [topic] + common)


def _description_for(platform: str, item: QueueItem) -> str:
    """Full per-platform description: hook + CTA + hashtags + series line.

    No network/secrets; pure string assembly from the item's metadata.
    """
    hook = _hook_for(item)
    base = (item.title or item.key).replace("_", " ").strip()
    hashtags = _hashtags(platform, item)
    cta = ("Watch the full research breakdown and let me know what to cover next. "
           "Part of the research-shorts series — bite-sized paper & concept "
           "breakdowns from the vault.")
    if platform.startswith("youtube"):
        return (f"{hook}\n\n{base}\n\n{cta}\n\n{hashtags}")
    if platform == "tiktok":
        return (f"{hook}\n\n{cta}\n\n{hashtags}")
    if platform == "instagram":
        return (f"{hook}\n\n{base}\n\n{cta}\n\n{hashtags}")
    return f"{hook}\n\n{cta}\n\n{hashtags}"


def _thumbnail_for(item: QueueItem, platform: str) -> Optional[Path]:
    """Return a usable thumbnail path for platforms that support covers.

    Prioritizes an explicit ``item.thumbnail_path`` (the 9:16 cover). For
    platforms that prefer 16:9 (YouTube), prefer the ``*_16x9.*`` variant if it
    exists. If no thumbnail is present, generate one on the fly via
    :func:`make_thumbnail` (best-effort; returns ``None`` on any failure).
    """
    platforms_with_thumbs = ("youtube", "instagram")
    if platform not in platforms_with_thumbs:
        return None
    out = Path(item.out_path)
    if not out.exists():
        return None
    gen_dir = out.parent
    base = gen_dir / f"{out.stem}.thumb"
    yt_path = base.with_name(base.stem + "_16x9.jpg")
    ig_path = base.with_suffix(".jpg")
    explicit = Path(item.thumbnail_path) if item.thumbnail_path else None
    if platform.startswith("youtube"):
        if explicit:
            yt_variant = explicit.with_name(
                explicit.stem + "_16x9" + explicit.suffix)
            if yt_variant.exists():
                return yt_variant
        if yt_path.exists():
            return yt_path
    else:  # instagram: 9:16 cover
        if explicit and explicit.exists():
            return explicit
        if ig_path.exists():
            return ig_path
    # Generate on the fly (best-effort).
    try:
        from auto_ingest.shorts import thumbnail as thumb_mod
        topic = item.topic.replace("_", " ")
        title = (item.title or item.key).replace("_", " ")
        made = thumb_mod.make_thumbnail(out, base, title=title, topic=topic)
        if platform.startswith("youtube"):
            yt_variant = made.with_name(made.stem + "_16x9" + made.suffix)
            return yt_variant if yt_variant.exists() else made
        return made
    except Exception as e:  # pragma: no cover - environment dependent
        log.warning("Thumbnail generation failed for %s: %s", item.key, e)
        return None


def _thumbnail_for_variant(item: QueueItem, platform: str,
                           thumbnail_path: Optional[Union[str, Path]]) -> Optional[Path]:
    """Return a specific A/B thumbnail path (a variant's cover) for a platform.

    Reuses :func:`_thumbnail_for`'s platform logic but prefers the explicit
    ``thumbnail_path`` (e.g. an A/B variant's ``.v1.jpg`` cover). For YouTube it
    prefers the matching ``*_16x9.*`` sibling when present. Falls back to
    :func:`_thumbnail_for` if the path is missing.
    """
    platforms_with_thumbs = ("youtube", "youtube_shorts", "instagram")
    if platform not in platforms_with_thumbs or thumbnail_path is None:
        return _thumbnail_for(item, platform)
    explicit = Path(thumbnail_path)
    if platform.startswith("youtube"):
        yt_variant = explicit.with_name(explicit.stem + "_16x9" + explicit.suffix)
        if yt_variant.exists():
            return yt_variant
    if explicit.exists():
        return explicit
    return _thumbnail_for(item, platform)


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

class BaseAdapter:
    name = "base"
    needs: List[str] = []

    def available(self) -> bool:
        return all(os.environ.get(k) for k in self.needs)

    def upload(self, item: QueueItem, *, dry_run: bool = True) -> Optional[str]:
        raise NotImplementedError


class YouTubeShortsAdapter(BaseAdapter):
    name = "youtube_shorts"
    needs = ["YT_CLIENT_SECRET_JSON", "YT_TOKEN_JSON"]

    def upload(self, item: QueueItem, *, dry_run: bool = True) -> Optional[str]:
        if dry_run:
            log.info("[dry-run] youtube_shorts <- %s (%s)", item.key, item.out_path)
            return None
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        from auto_ingest.shorts import yt_auth

        creds = yt_auth.load_credentials(os.environ.get("YT_TOKEN_JSON"))
        if creds is None:
            raise RuntimeError("No YouTube token; run: publish auth youtube")
        yt = build("youtube", "v3", credentials=creds)
        thumb = _thumbnail_for(item, "youtube")
        body = {
            "snippet": {
                "title": _title_for("youtube", item),
                "description": _description_for("youtube", item),
                "tags": ["shorts", item.topic, "research", "ai"],
                "categoryId": "28",
            },
            "status": {"privacyStatus": os.environ.get("YT_PRIVACY", "unlisted")},
        }
        media = MediaFileUpload(item.out_path, chunksize=-1, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = req.execute()
        if thumb is not None:
            yt.thumbnails().set(videoId=resp["id"], media_body=str(thumb)).execute()
        url = f"https://youtu.be/{resp['id']}"
        log.info("Uploaded %s -> %s", item.key, url)
        return url


class TikTokAdapter(BaseAdapter):
    name = "tiktok"
    needs = ["TIKTOK_ACCESS_TOKEN"]

    def upload(self, item: QueueItem, *, dry_run: bool = True) -> Optional[str]:
        if dry_run:
            log.info("[dry-run] tiktok <- %s (%s)", item.key, item.out_path)
            return None
        import requests

        from auto_ingest.shorts import tiktok_auth

        tok = tiktok_auth.load_token()
        if tok is None:
            raise RuntimeError("No TikTok token; run: publish auth tiktok")
        token = tok.get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        # 1) create a video post container
        init = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers=headers,
            json={"post_info": {"title": _title_for("tiktok", item),
                                "privacy_level": "SELF_ONLY",
                                "disable_comment": False, "auto_add_music": True},
                  "source_info": {"source": "FILE",
                                  "video_size": Path(item.out_path).stat().st_size,
                                  "chunk_size": Path(item.out_path).stat().st_size,
                                  "filename": Path(item.out_path).name}},
            timeout=60)
        init.raise_for_status()
        publish_id = init.json()["data"]["publish_id"]
        # TikTok processes async; poll status until done (best-effort).
        for _ in range(20):
            st = requests.get(
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                headers=headers, params={"publish_id": publish_id}, timeout=60)
            d = st.json().get("data", {})
            if d.get("status") in ("PUBLISH_COMPLETE", "COMPLETE"):
                log.info("TikTok published %s (%s)", item.key, d.get("share_url"))
                return d.get("share_url") or publish_id
            if d.get("status") == "FAILED":
                raise RuntimeError(f"TikTok publish failed: {d}")
            import time
            time.sleep(5)
        log.info("TikTok publish_id %s for %s (still processing)", publish_id, item.key)
        return publish_id


class InstagramReelsAdapter(BaseAdapter):
    name = "instagram"
    needs = ["IG_USER_ID", "IG_ACCESS_TOKEN"]

    def upload(self, item: QueueItem, *, dry_run: bool = True) -> Optional[str]:
        if dry_run:
            log.info("[dry-run] instagram <- %s (%s)", item.key, item.out_path)
            return None
        import requests

        from auto_ingest.shorts import instagram_auth

        tok = instagram_auth.load_token()
        if tok is None:
            raise RuntimeError("No Instagram token; run: publish auth instagram")
        uid = tok.get("ig_user_id") or os.environ.get("IG_USER_ID")
        token = tok.get("access_token") or os.environ.get("IG_ACCESS_TOKEN")
        if not uid or not token:
            raise RuntimeError("Instagram token missing ig_user_id/access_token")
        base = f"https://graph.facebook.com/v21.0/{uid}"
        caption = (f"{_title_for('instagram', item)}\n\n"
                   f"{_description_for('instagram', item)}")
        # 1) object-staged upload (single request for <128MB)
        with open(item.out_path, "rb") as fh:
            files = {"source": fh}
            r = requests.post(f"{base}/media",
                              data={"upload_type": "multipart",
                                    "caption": caption,
                                    "media_type": "REELS"},
                              files=files, timeout=120)
        r.raise_for_status()
        container = r.json().get("id")
        # 2) publish the container
        pub = requests.post(f"{base}/media_publish",
                            data={"creation_id": container, "access_token": token},
                            timeout=60)
        pub.raise_for_status()
        log.info("Instagram reel published (container %s) for %s", container, item.key)
        return container


ADAPTERS = {
    "youtube_shorts": YouTubeShortsAdapter,
    "youtube": YouTubeShortsAdapter,
    "tiktok": TikTokAdapter,
    "instagram": InstagramReelsAdapter,
}


def plan_report(item: QueueItem, *, plans_dir: Optional[Path] = None,
                truncate: int = 240) -> dict:
    """Assemble the exact payload that *would* be uploaded for ``item``.

    Pure / offline: pulls hook + thumbnail (best-effort, see
    :func:`_thumbnail_for`) and renders title/description/hashtags. Never reads
    secrets or touches the network. Returns a dict the caller can print.
    """
    if plans_dir is not None:
        item = enrich_item_from_plan(item, Path(plans_dir))
    platform = item.platform
    # Apply the active A/B variant (if any) for this short/platform. Returns a
    # title override + thumbnail override; both may be None.
    title_override, thumb_override = _apply_ab_variant(item)
    thumb = _thumbnail_for_variant(item, platform, thumb_override) \
        if thumb_override else _thumbnail_for(item, platform)
    desc = _description_for(platform, item)
    title = _title_for(platform, item, title_override=title_override)
    hashtags = _hashtags(platform, item)
    # The actual post body as it would be submitted (title + description +
    # hashtags joined) — platform-specific (TikTok/IG lead with caption, YT
    # uses title + description separately).
    caption = f"{title}\n\n{desc}\n\n{hashtags}".strip()
    return {
        "key": item.key,
        "platform": platform,
        "file": item.out_path,
        "title": title,
        "description": desc[:truncate] + ("…" if len(desc) > truncate else ""),
        "hashtags": hashtags,
        "caption": caption[:truncate] + ("…" if len(caption) > truncate else ""),
        "thumbnail": str(thumb) if thumb is not None else None,
        "ab_variant": None if title_override is None and thumb_override is None
                     else _active_variant_index(item),
        # TikTok/IG have no separate thumbnail field in this pipeline; note it.
        "uses_thumbnail": platform in ("youtube", "youtube_shorts", "instagram"),
        "file_exists": Path(item.out_path).exists(),
    }


def _apply_ab_variant(item: QueueItem) -> tuple:
    """Return ``(title_override, thumbnail_path)`` for the active A/B variant.

    Looks up a stored :class:`VariantPlan` for ``(item.key, item.platform)` and
    uses :func:`abtest.assign_variant` to pick the active thumbnail + title. If
    no plan exists, returns ``(None, None)`` (standard platform rendering).
    """
    try:
        from auto_ingest.shorts import abtest
    except Exception:
        return (None, None)
    plans = abtest.load_variants(short_id=item.key, platform=item.platform)
    if not plans:
        return (None, None)
    plan = plans[0]
    thumb, title = abtest.assign_variant(item, plan)
    return (title, thumb)


def _active_variant_index(item: QueueItem) -> Optional[int]:
    try:
        from auto_ingest.shorts import abtest
    except Exception:
        return None
    plans = abtest.load_variants(short_id=item.key, platform=item.platform)
    return plans[0].active_variant if plans else None


def validate_queue(platforms: Optional[List[str]] = None) -> List[dict]:
    """Check queue integrity: out_path existence + thumbnail presence.

    Returns a list of issue dicts (empty == all good). Does not modify state.
    """
    items = load_queue()
    if platforms:
        items = [i for i in items if i.platform in platforms]
    issues: List[dict] = []
    for item in items:
        if not Path(item.out_path).exists():
            issues.append({"key": item.key, "platform": item.platform,
                           "issue": "missing_file", "path": item.out_path})
            continue
        # Only platforms that actually use a cover should be flagged for a
        # missing thumbnail (TikTok has no thumbnail field in this pipeline).
        if item.platform in ("youtube", "youtube_shorts", "instagram"):
            if _thumbnail_for(item, item.platform) is None:
                issues.append({"key": item.key, "platform": item.platform,
                               "issue": "missing_thumbnail", "path": item.out_path})
    return issues


def process_queue(platforms: Optional[List[str]] = None, *, dry_run: bool = True,
                  only_platforms: Optional[List[str]] = None,
                  plans_dir: Optional[Path] = None) -> int:
    """Upload every queued item (credential-gated) or report what would upload.

    In ``dry_run`` (default) no network/secrets are touched and every queued
    item is *reported* via :func:`plan_report` regardless of whether its
    credentials are configured. In live mode (``dry_run=False``) each item is
    skipped with a clear message when its secrets/tokens are absent, and the
    run never raises a traceback for missing credentials.
    """
    items = load_queue()
    if only_platforms:
        items = [i for i in items if i.platform in only_platforms]
    if platforms:
        items = [i for i in items if i.platform in platforms]

    if not dry_run:
        # Defense in depth: never perform a real upload without explicit opt-in
        # (publishing is held until accounts/OAuth are deliberately configured).
        from auto_ingest.shorts.publish_guard import require_live_mode
        require_live_mode(where="process_queue")

    attempted = 0
    for item in items:
        adapter_cls = ADAPTERS.get(item.platform)
        if adapter_cls is None:
            log.warning("No adapter for platform %s (key %s)", item.platform, item.key)
            continue
        adapter = adapter_cls()

        if dry_run:
            rep = plan_report(item, plans_dir=plans_dir)
            log.info("[dry-run] %s <- %s | title=%r thumb=%s exists=%s",
                     rep["platform"], rep["key"], rep["title"],
                     rep["thumbnail"], rep["file_exists"])
            if not rep["file_exists"]:
                log.warning("  ! file missing: %s", rep["file"])
            attempted += 1
            continue

        if not adapter.available():
            log.warning("Credentials not configured for %s (missing %s). "
                        "Run `publish auth %s` or set the env vars; skipping %s.",
                        item.platform, adapter.needs, adapter.name, item.key)
            continue
        # Apply the active A/B variant (title + thumbnail override) before
        # upload, so the winning/active variant is what actually posts.
        title_override, thumb_override = _apply_ab_variant(item)
        if title_override:
            item.title = title_override
        if thumb_override:
            item.thumbnail_path = str(thumb_override)
        if already_published(item):
            log.info("Already published %s/%s; skip", item.platform, item.key)
            continue
        if not Path(item.out_path).exists():
            log.warning("Missing file %s; skip", item.out_path)
            continue
        try:
            url = adapter.upload(item, dry_run=False)
        except Exception as e:  # graceful: missing creds/tokens -> message, no crash
            log.error("Upload failed for %s/%s: %s", item.platform, item.key, e)
            continue
        attempted += 1
        if url is not None:
            _record_published(item, url)
    return attempted
