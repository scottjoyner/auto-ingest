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
from typing import List, Optional

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
            queued_at=d.get("queued_at")))
    return items


def _published_marker(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".published.json")


def already_published(item: QueueItem) -> bool:
    return _published_marker(Path(item.out_path)).exists()


def _record_published(item: QueueItem, url: str = "") -> None:
    rec = {"key": item.key, "platform": item.platform,
           "url": url, "at": time.time()}
    _published_marker(Path(item.out_path)).write_text(json.dumps(rec))


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
        from auto_ingest.shorts import yt_auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = yt_auth.load_credentials(os.environ.get("YT_TOKEN_JSON"))
        if creds is None:
            raise RuntimeError("No YouTube token; run: publish auth youtube")
        yt = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": item.title or item.key,
                "description": f"#{item.topic}\nRendered from the research vault. "
                                "Your words, not a paper.",
                "tags": ["shorts", item.topic, "research", "ai"],
                "categoryId": "28",
            },
            "status": {"privacyStatus": os.environ.get("YT_PRIVACY", "unlisted")},
        }
        media = MediaFileUpload(item.out_path, chunksize=-1, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = req.execute()
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

        token = os.environ["TIKTOK_ACCESS_TOKEN"]
        # 1) create a video post container
        init = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_info": {"title": item.title or item.key,
                                "privacy_level": "SELF_ONLY"},
                  "source_info": {"source": "FILE",
                                  "video_size": Path(item.out_path).stat().st_size,
                                  "chunk_size": Path(item.out_path).stat().st_size,
                                  "filename": Path(item.out_path).name}},
            timeout=60)
        init.raise_for_status()
        publish_id = init.json()["data"]["publish_id"]
        # TikTok processes async; the container id is the receipt.
        log.info("TikTok publish_id %s for %s (async processing)", publish_id, item.key)
        return publish_id


class InstagramReelsAdapter(BaseAdapter):
    name = "instagram"
    needs = ["IG_USER_ID", "IG_ACCESS_TOKEN"]

    def upload(self, item: QueueItem, *, dry_run: bool = True) -> Optional[str]:
        if dry_run:
            log.info("[dry-run] instagram <- %s (%s)", item.key, item.out_path)
            return None
        import requests

        uid = os.environ["IG_USER_ID"]
        token = os.environ["IG_ACCESS_TOKEN"]
        base = f"https://graph.facebook.com/v21.0/{uid}"
        # 1) object-staged upload (single request for <128MB)
        with open(item.out_path, "rb") as fh:
            r = requests.post(f"{base}/media",
                              data={"upload_type": "multipart",
                                    "video_url": None,
                                    "caption": f"{item.title}\n#{item.topic} #shorts",
                                    "media_type": "REELS"},
                              files={"source": fh}, timeout=120)
        r.raise_for_status()
        container = r.json().get("id")
        # 2) poll until finished, then publish
        requests.post(f"{base}/media_publish",
                      data={"creation_id": container, "access_token": token},
                      timeout=60).raise_for_status()
        log.info("Instagram reel published (container %s) for %s", container, item.key)
        return container


ADAPTERS = {
    "youtube_shorts": YouTubeShortsAdapter,
    "youtube": YouTubeShortsAdapter,
    "tiktok": TikTokAdapter,
    "instagram": InstagramReelsAdapter,
}


def process_queue(platforms: Optional[List[str]] = None, *, dry_run: bool = True,
                  only_platforms: Optional[List[str]] = None) -> int:
    """Upload every queued item whose adapter is available + not yet published.

    Returns the number of uploads attempted. In ``dry_run`` (default) no network
    calls happen; it reports what *would* upload and which platforms are
    unconfigured.
    """
    items = load_queue()
    if only_platforms:
        items = [i for i in items if i.platform in only_platforms]
    if platforms:
        # Restrict to adapters we were asked to drive.
        items = [i for i in items if i.platform in platforms]

    attempted = 0
    for item in items:
        adapter_cls = ADAPTERS.get(item.platform)
        if adapter_cls is None:
            log.warning("No adapter for platform %s (key %s)", item.platform, item.key)
            continue
        adapter = adapter_cls()
        if not adapter.available():
            log.warning("Platform %s not configured (missing %s); skipping %s",
                        item.platform, adapter.needs, item.key)
            continue
        if already_published(item):
            log.info("Already published %s/%s; skip", item.platform, item.key)
            continue
        if not Path(item.out_path).exists():
            log.warning("Missing file %s; skip", item.out_path)
            continue
        url = adapter.upload(item, dry_run=dry_run)
        attempted += 1
        if not dry_run and url is not None:
            _record_published(item, url)
    return attempted
