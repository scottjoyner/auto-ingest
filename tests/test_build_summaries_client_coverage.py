from __future__ import annotations

import json

import pytest

from auto_ingest.content import build_summaries as b


class Response:
    def __init__(self, status=200, payload=None, raw=None):
        self.status = status
        self.payload = payload
        self.raw = raw

    def read(self):
        if self.raw is not None:
            return self.raw
        return json.dumps(self.payload).encode()


class Connection:
    responses = []
    calls = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def getresponse(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_client_url_parsing_http_and_https():
    client = b.OllamaClient("http://example:1234/base/")
    assert (client._is_https, client._host, client._port, client._base_path) == (False, "example", 1234, "/base")
    client = b.OllamaClient("https://example")
    assert client._is_https is True and client._port == 443 and client._base_path == ""
    client = b.OllamaClient("")
    assert client._host == "127.0.0.1" and client._port == 11434


def test_generate_success_response_and_text_fallback(monkeypatch):
    Connection.responses = [Response(payload={"response": "ok"}), Response(payload={"text": "fallback"})]
    Connection.calls = []
    monkeypatch.setattr(b.http.client, "HTTPConnection", Connection)
    client = b.OllamaClient("http://host:9/root")
    assert client.generate("m", "p", options={"x": 1}) == "ok"
    assert client.generate("m", "p") == "fallback"
    args, kwargs = Connection.calls[0]
    assert args[0] == "POST" and args[1] == "/root/api/generate"
    assert json.loads(kwargs["body"])["options"] == {"x": 1}


def test_generate_retries_http_missing_output_and_exception(monkeypatch):
    slept = []
    monkeypatch.setattr(b.time, "sleep", lambda n: slept.append(n))
    Connection.responses = [Response(status=500, raw=b"bad"), Response(payload={}), Response(payload={"response": "yes"})]
    monkeypatch.setattr(b.http.client, "HTTPConnection", Connection)
    assert b.OllamaClient("http://h").generate("m", "p", retries=3) == "yes"
    assert slept == [2, 4]

    Connection.responses = [RuntimeError("network"), RuntimeError("network")]
    with pytest.raises(RuntimeError, match="after retries"):
        b.OllamaClient("http://h").generate("m", "p", retries=2)


def test_generate_https_connection(monkeypatch):
    Connection.responses = [Response(payload={"response": "secure"})]
    monkeypatch.setattr(b.http.client, "HTTPSConnection", Connection)
    assert b.OllamaClient("https://h/base").generate("m", "p") == "secure"
