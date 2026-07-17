"""Instagram (Graph API) OAuth bootstrap for Reels publishing.

Requires an Instagram Business or Creator account connected to a Facebook
Page you manage. The flow: Facebook Login -> short-lived user token ->
long-lived token -> find the IG business user id via /me/accounts.

Run once to authenticate:

    python -m auto_ingest.shorts.cli publish auth instagram

Opens a browser sign-in; on approval we save IG_USER_ID + IG_ACCESS_TOKEN
(long-lived) to $IG_TOKEN_JSON (default
~/.config/auto-ingest/ig_token.json). The uploader reads those, or the raw
env vars IG_USER_ID / IG_ACCESS_TOKEN.

Client secret: $IG_CLIENT_SECRET_JSON (default
~/.config/auto-ingest/ig_client_secret.json), a JSON of
{"app_id": "...", "app_secret": "...", "redirect_uri": "http://localhost:8092/callback"}.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Optional

log = logging.getLogger("shorts.ig_auth")

TOKEN_DEFAULT = Path(os.environ.get(
    "IG_TOKEN_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "ig_token.json")))
SECRET_DEFAULT = Path(os.environ.get(
    "IG_CLIENT_SECRET_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "ig_client_secret.json")))

FB_AUTH = "https://www.facebook.com/v21.0/dialog/oauth"
FB_TOKEN = "https://graph.facebook.com/v21.0/oauth/access_token"
GRAPH = "https://graph.facebook.com/v21.0"
SCOPES = "instagram_content_publish,instagram_basic,pages_show_list"


def _secret() -> dict:
    p = Path(os.environ.get("IG_CLIENT_SECRET_JSON", str(SECRET_DEFAULT)))
    if not p.exists():
        raise FileNotFoundError(
            f"Missing Instagram client secret at {p}. Create a Meta app (Instagram "
            "Basic Display or Business) and save its app_id/app_secret/redirect_uri "
            "there, or set IG_CLIENT_SECRET_JSON.")
    return json.loads(p.read_text())


def bootstrap_token(headless: bool = False) -> Path:
    secret = _secret()
    app_id = secret["app_id"]
    app_secret = secret["app_secret"]
    redirect_uri = secret.get("redirect_uri", "http://localhost:8092/callback")
    TOKEN_DEFAULT.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "response_type": "code",
        "state": "ig",
    }
    url = FB_AUTH + "?" + urllib.parse.urlencode(params)

    if headless:
        print("\nOpen this URL in a browser and paste the 'code':\n")
        print(url + "\n")
        code = input("code: ").strip()
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
                self.wfile.write(b"Instagram authorized. Close this tab.")
                got.set()

            def log_message(self, *a):
                pass

        port = int(urllib.parse.urlparse(redirect_uri).port or 8092)
        srv = http.server.HTTPServer(("localhost", port), _H)
        threading.Thread(target=srv.handle_request, daemon=True).start()
        print(f"\nOpen this URL to sign in to Instagram/Facebook:\n\n{url}\n")
        got.wait(300)
        srv.server_close()
        code = received.get("code", "")
        if not code:
            raise RuntimeError("No authorization code received from Facebook.")

    import requests
    # 1) short-lived user token
    r = requests.get(FB_TOKEN, params={
        "client_id": app_id, "redirect_uri": redirect_uri,
        "client_secret": app_secret, "code": code}, timeout=60)
    r.raise_for_status()
    short = r.json()["access_token"]

    # 2) exchange for long-lived token
    r = requests.get(FB_TOKEN, params={
        "grant_type": "fb_exchange_token",
        "client_id": app_id, "client_secret": app_secret,
        "fb_exchange_token": short}, timeout=60)
    r.raise_for_status()
    long_tok = r.json()["access_token"]

    # 3) find the IG business user id (page-instagram)
    ig_user_id = None
    accts = requests.get(f"{GRAPH}/me/accounts", params={
        "fields": "instagram_business_account,username", "access_token": long_tok},
        timeout=60).json().get("data", [])
    for pg in accts:
        if pg.get("instagram_business_account"):
            ig_user_id = pg["instagram_business_account"]["id"]
            break

    TOKEN_DEFAULT.write_text(json.dumps({
        "ig_user_id": ig_user_id,
        "access_token": long_tok,
    }, indent=2))
    log.info("Saved Instagram token -> %s (ig_user_id=%s)", TOKEN_DEFAULT, ig_user_id)
    print(f"Instagram token saved: {TOKEN_DEFAULT} (ig_user_id={ig_user_id})")
    return TOKEN_DEFAULT


def load_token() -> Optional[dict]:
    p = Path(os.environ.get("IG_TOKEN_JSON", str(TOKEN_DEFAULT)))
    if not p.exists():
        return None
    # Long-lived IG tokens last ~60 days; refresh via appsecret_proof if needed.
    return json.loads(p.read_text())


def safe_to_run() -> bool:
    """True only when live publishing is explicitly opted in and this platform
    has credentials present (see ``publish_guard.safe_to_run``)."""
    from auto_ingest.shorts.publish_guard import safe_to_run as _safe
    return _safe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap_token()
