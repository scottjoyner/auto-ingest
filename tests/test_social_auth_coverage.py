from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_instagram_secret_headless_bootstrap_load_and_guard(monkeypatch, tmp_path):
    from auto_ingest.shorts import instagram_auth as ig

    secret = tmp_path / "ig-secret.json"
    secret.write_text(json.dumps({"app_id": "id", "app_secret": "sec", "redirect_uri": "http://localhost:8092/callback"}))
    token = tmp_path / "ig-token.json"
    monkeypatch.setenv("IG_CLIENT_SECRET_JSON", str(secret))
    monkeypatch.setenv("IG_TOKEN_JSON", str(token))
    monkeypatch.setattr(ig, "TOKEN_DEFAULT", token)
    assert ig._secret()["app_id"] == "id"
    monkeypatch.setenv("IG_CLIENT_SECRET_JSON", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        ig._secret()
    monkeypatch.setenv("IG_CLIENT_SECRET_JSON", str(secret))
    monkeypatch.setattr("builtins.input", lambda _p="": "code")

    class Resp:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): return None
        def json(self): return self.payload

    replies = iter([
        Resp({"access_token": "short"}),
        Resp({"access_token": "long"}),
        Resp({"data": [{"instagram_business_account": {"id": "ig1"}}]}),
    ])
    requests = ModuleType("requests")
    requests.get = lambda *a, **kw: next(replies)
    monkeypatch.setitem(sys.modules, "requests", requests)
    assert ig.bootstrap_token(headless=True) == token
    assert ig.load_token()["ig_user_id"] == "ig1"
    token.unlink()
    assert ig.load_token() is None
    monkeypatch.setattr("auto_ingest.shorts.publish_guard.safe_to_run", lambda: True)
    assert ig.safe_to_run() is True


def test_tiktok_exchange_headless_bootstrap_refresh_and_guard(monkeypatch, tmp_path):
    from auto_ingest.shorts import tiktok_auth as tt

    secret = tmp_path / "tt-secret.json"
    secret.write_text(json.dumps({"client_key": "ck", "client_secret": "cs"}))
    token = tmp_path / "tt-token.json"
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET_JSON", str(secret))
    monkeypatch.setenv("TIKTOK_TOKEN_JSON", str(token))
    monkeypatch.setattr(tt, "TOKEN_DEFAULT", token)
    monkeypatch.setattr("builtins.input", lambda _p="": "code")

    class Resp:
        def raise_for_status(self): return None
        def json(self): return {"data": {"access_token": "a", "refresh_token": "r", "expires_in": 10, "open_id": "o"}}

    requests = ModuleType("requests")
    requests.post = lambda *a, **kw: Resp()
    monkeypatch.setitem(sys.modules, "requests", requests)
    assert tt._exchange("c", "ck", "cs", "u")["access_token"] == "a"
    assert tt.bootstrap_token(headless=True) == token
    assert tt.load_token()["access_token"] == "a"
    token.write_text(json.dumps({"access_token": "old"}))
    assert tt.load_token()["access_token"] == "old"
    token.unlink()
    assert tt.load_token() is None
    monkeypatch.setattr("auto_ingest.shorts.publish_guard.safe_to_run", lambda: True)
    assert tt.safe_to_run() is True


def test_youtube_bootstrap_load_refresh_and_guard(monkeypatch, tmp_path):
    from auto_ingest.shorts import yt_auth as yt

    secret = tmp_path / "yt-secret.json"
    secret.write_text("{}")
    token = tmp_path / "yt-token.json"
    creds = SimpleNamespace(
        token="tok", refresh_token="ref", token_uri="uri", client_id="id",
        client_secret="sec", scopes=["scope"], expired=True,
        refresh=lambda _req: None,
    )

    class Flow:
        redirect_uri = None
        credentials = creds
        @classmethod
        def from_client_secrets_file(cls, *_a): return cls()
        def authorization_url(self, **_kw): return ("http://auth", None)
        def fetch_token(self, **_kw): return None
        def run_local_server(self, **_kw): return self.credentials

    flow_mod = ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = Flow
    package = ModuleType("google_auth_oauthlib")
    package.flow = flow_mod
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", package)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_mod)
    monkeypatch.setattr("builtins.input", lambda _p="": "code")
    assert yt.bootstrap_token(secret, token, headless=True) == token
    with pytest.raises(FileNotFoundError):
        yt.bootstrap_token(tmp_path / "missing", token)

    request_mod = ModuleType("google.auth.transport.requests")
    request_mod.Request = lambda: object()
    credentials_mod = ModuleType("google.oauth2.credentials")
    credentials_mod.Credentials = SimpleNamespace(from_authorized_user_file=lambda *_a: creds)
    for name in ("google", "google.auth", "google.auth.transport", "google.oauth2"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", request_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_mod)
    assert yt.load_credentials(token) is creds
    token.unlink()
    assert yt.load_credentials(token) is None
    monkeypatch.setattr("auto_ingest.shorts.publish_guard.safe_to_run", lambda: True)
    assert yt.safe_to_run() is True
