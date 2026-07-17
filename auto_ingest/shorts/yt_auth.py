"""YouTube OAuth token bootstrap for narrated-shorts publishing.

Run once (per machine) to authenticate the channel:

    python -m auto_ingest.shorts.cli publish auth youtube

This opens a browser to Google's sign-in, you pick the channel, and the
resulting token is saved to ``$YT_TOKEN_JSON`` (default
``~/.config/auto-ingest/yt_token.json``). The uploader reads that token on
every publish. No secrets are committed; the client_secret.json comes from
your Google Cloud OAuth app and is referenced via ``$YT_CLIENT_SECRET_JSON``.

Scopes: youtube.upload (needed to push Shorts/long-form).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("shorts.yt_auth")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_DEFAULT = Path(os.environ.get(
    "YT_TOKEN_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "yt_token.json")))
SECRET_DEFAULT = Path(os.environ.get(
    "YT_CLIENT_SECRET_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "yt_client_secret.json")))


def _client_secret_path() -> Path:
    p = Path(os.environ.get("YT_CLIENT_SECRET_JSON", str(SECRET_DEFAULT)))
    return p


def bootstrap_token(secret_path: Optional[Path] = None,
                    token_path: Optional[Path] = None,
                    *,
                    headless: bool = False) -> Path:
    """Run the OAuth sign-in flow and persist the token. Returns token path.

    ``headless=False`` opens a local browser via run_local_server (the normal
    "click the sign-in link" experience). On a machine with no browser, pass
    ``headless=True`` to get a manual authorization-URL + code-paste fallback.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = Path(secret_path) if secret_path else _client_secret_path()
    token = Path(token_path) if token_path else TOKEN_DEFAULT
    if not secret.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secret at {secret}. Create a 'Desktop' "
            "OAuth client in Google Cloud Console (enabled YouTube Data API v3) "
            "and save it there, or set YT_CLIENT_SECRET_JSON.")

    token.parent.mkdir(parents=True, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    if headless:
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url, _ = flow.authorization_url(prompt="consent")
        print("\nOpen this URL in a browser and paste the code:\n")
        print(auth_url + "\n")
        code = input("Authorization code: ").strip()
        flow.fetch_token(code=code)
        creds = flow.credentials
    else:
        creds = flow.run_local_server(port=0, prompt="consent")

    token.write_text(json.dumps({
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }, indent=2))
    log.info("Saved YouTube token -> %s", token)
    print(f"Token saved: {token}")
    return token


def load_credentials(token_path: Optional[Path] = None):
    """Load saved creds, refreshing if expired (requires refresh_token)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token = Path(token_path) if token_path else TOKEN_DEFAULT
    if not token.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token.write_text(json.dumps({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }, indent=2))
    return creds


def safe_to_run() -> bool:
    """True only when live publishing is explicitly opted in and this platform
    has credentials present (see ``publish_guard.safe_to_run``)."""
    from auto_ingest.shorts.publish_guard import safe_to_run as _safe
    return _safe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap_token()
