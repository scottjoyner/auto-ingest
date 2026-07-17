"""TikTok Content Posting API OAuth bootstrap for narrated-shorts publishing.

TikTok requires a developer app (https://developers.tiktok.com) with the
"Content Posting API" product enabled. For individual creators this is
self-serve; for businesses it may need app review. The app's redirect URI must
include the loopback we use below (http://localhost:<port>/callback).

Run once to authenticate:

    python -m auto_ingest.shorts.cli publish auth tiktok

Opens a browser sign-in; on approval we exchange the code for an access +
refresh token, saved to $TIKTOK_TOKEN_JSON (default
~/.config/auto-ingest/tiktok_token.json). The access token is short-lived; the
uploader refreshes it via the refresh token automatically.

Client secret: $TIKTOK_CLIENT_SECRET_JSON (default
~/.config/auto-ingest/tiktok_client_secret.json), a JSON of
{"client_key": "...", "client_secret": "..."}.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Optional

log = logging.getLogger("shorts.tiktok_auth")

TOKEN_DEFAULT = Path(os.environ.get(
    "TIKTOK_TOKEN_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "tiktok_token.json")))
SECRET_DEFAULT = Path(os.environ.get(
    "TIKTOK_CLIENT_SECRET_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "tiktok_client_secret.json")))

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
SCOPES = "video.upload"


def _secret() -> dict:
    p = Path(os.environ.get("TIKTOK_CLIENT_SECRET_JSON", str(SECRET_DEFAULT)))
    if not p.exists():
        raise FileNotFoundError(
            f"Missing TikTok client secret at {p}. Create a TikTok developer app "
            "with the Content Posting API, add its client_key/client_secret there, "
            "or set TIKTOK_CLIENT_SECRET_JSON.")
    return json.loads(p.read_text())


def _exchange(code: str, client_key: str, client_secret: str, redirect_uri: str) -> dict:
    import requests
    r = requests.post(TOKEN_URL, data={
        "client_key": client_key,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }, timeout=60)
    r.raise_for_status()
    return r.json().get("data", {})


def bootstrap_token(headless: bool = False) -> Path:
    secret = _secret()
    ck, cs = secret["client_key"], secret["client_secret"]
    TOKEN_DEFAULT.parent.mkdir(parents=True, exist_ok=True)

    redirect_uri = "http://localhost:8091/callback"
    params = {
        "client_key": ck,
        "scope": SCOPES,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": "authed",
    }
    url = AUTH_URL + "?" + urllib.parse.urlencode(params)

    if headless:
        print("\nOpen this URL in a browser and paste the 'code' from the redirect:\n")
        print(url + "\n")
        code = input("code: ").strip()
        data = _exchange(code, ck, cs, redirect_uri)
    else:
        import http.server
        import threading

        got = threading.Event()
        received: dict = {}

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                q = urllib.parse.urlparse(self.path).query
                received["code"] = urllib.parse.parse_qs(q).get("code", [""])[0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"TikTok authorized. You can close this tab.")
                got.set()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("localhost", 8091), _H)
        threading.Thread(target=srv.handle_request, daemon=True).start()
        print(f"\nOpen this URL to sign in to TikTok:\n\n{url}\n")
        got.wait(300)
        srv.server_close()
        code = received.get("code", "")
        if not code:
            raise RuntimeError("No authorization code received from TikTok.")
        data = _exchange(code, ck, cs, redirect_uri)

    TOKEN_DEFAULT.write_text(json.dumps({
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "expires_in": data.get("expires_in"),
        "open_id": data.get("open_id"),
        "client_key": ck,
        "client_secret": cs,
    }, indent=2))
    log.info("Saved TikTok token -> %s", TOKEN_DEFAULT)
    print(f"TikTok token saved: {TOKEN_DEFAULT}")
    return TOKEN_DEFAULT


def load_token() -> Optional[dict]:
    """Load + refresh the TikTok token. Returns the token dict or None."""
    import requests
    p = Path(os.environ.get("TIKTOK_TOKEN_JSON", str(TOKEN_DEFAULT)))
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not d.get("refresh_token"):
        return d
    # Refresh if near expiry (or always refresh for simplicity).
    try:
        r = requests.post(TOKEN_URL, data={
            "client_key": d.get("client_key"),
            "client_secret": d.get("client_secret"),
            "grant_type": "refresh_token",
            "refresh_token": d["refresh_token"],
        }, timeout=60)
        r.raise_for_status()
        nd = r.json().get("data", {})
        d.update({k: nd[k] for k in ("access_token", "refresh_token", "expires_in") if k in nd})
        p.write_text(json.dumps(d, indent=2))
    except Exception as e:  # refresh is best-effort
        log.warning("TikTok token refresh failed: %s", e)
    return d


def safe_to_run() -> bool:
    """True only when live publishing is explicitly opted in and this platform
    has credentials present (see ``publish_guard.safe_to_run``)."""
    from auto_ingest.shorts.publish_guard import safe_to_run as _safe
    return _safe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap_token()
