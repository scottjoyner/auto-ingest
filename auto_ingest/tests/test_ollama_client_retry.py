"""Focused, I/O-free tests for the local Ollama client's bounded retry path.

Every socket operation is replaced by an in-memory connection stand-in and
``time.sleep`` is stubbed, so no Ollama server is contacted and the suite stays
fast even when the retry backoff would normally sleep for seconds.
"""
from __future__ import annotations

from unittest import mock

import pytest

from auto_ingest.content import build_summaries


class _Response:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body


def _connection_factory(outcomes, seen):
    """Build a connection class that replays ``outcomes`` in order.

    Each outcome is either an exception to raise from ``getresponse`` or a
    ``(status, body)`` pair. The connection's host/port/timeout are recorded in
    ``seen`` so the test can assert the configured timeout actually reached I/O.
    """
    state = {"n": 0}

    class _Connection:
        def __init__(self, host, port, timeout):
            seen.append((host, port, timeout))

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            outcome = outcomes[state["n"]]
            state["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return _Response(*outcome)

        def close(self):
            return None

    return _Connection


def test_retry_succeeds_and_preserves_bounded_timeout():
    seen = []
    outcomes = [OSError("connection refused"), (200, b'{"response":"ok"}')]
    client = build_summaries.OllamaClient("http://127.0.0.1:11434")

    with mock.patch.object(build_summaries.http.client, "HTTPConnection",
                           _connection_factory(outcomes, seen)), \
         mock.patch.object(build_summaries.time, "sleep") as sleep:
        out = client.generate("gemma3:4b", "hello", retries=3, timeout=2.5)

    assert out == "ok"
    assert seen == [("127.0.0.1", 11434, 2.5), ("127.0.0.1", 11434, 2.5)]
    sleep.assert_called_once_with(2)


def test_exhausted_retries_report_endpoint_model_and_last_error():
    seen = []
    outcomes = [OSError("connection refused")] * 3
    client = build_summaries.OllamaClient("http://127.0.0.1:11434")

    with mock.patch.object(build_summaries.http.client, "HTTPConnection",
                           _connection_factory(outcomes, seen)), \
         mock.patch.object(build_summaries.time, "sleep") as sleep:
        with pytest.raises(RuntimeError) as exc:
            client.generate("gemma3:4b", "hello", retries=3, timeout=1.0)

    msg = str(exc.value)
    assert "127.0.0.1:11434/api/generate" in msg
    assert "gemma3:4b" in msg
    assert "3 attempt(s)" in msg
    assert "OSError: connection refused" in msg
    # Two failures between three attempts get a backoff; the final one does not.
    assert sleep.call_count == 2


def test_http_error_detail_survives_exhausted_retries():
    seen = []
    outcomes = [(503, b"model loading")] * 2
    client = build_summaries.OllamaClient("http://ollama.local:11434")

    with mock.patch.object(build_summaries.http.client, "HTTPConnection",
                           _connection_factory(outcomes, seen)), \
         mock.patch.object(build_summaries.time, "sleep") as sleep:
        with pytest.raises(RuntimeError) as exc:
            client.generate("gemma3:4b", "hello", retries=2, timeout=1.0)

    msg = str(exc.value)
    assert "HTTP 503" in msg
    assert "model loading" in msg
    assert "ollama.local:11434/api/generate" in msg
    assert sleep.call_count == 1
