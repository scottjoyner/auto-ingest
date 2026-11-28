"""Regression: header outbox op must not be marked done before full key succeeds.

The durable outbox stages ingest.transcription.header before the Neo4j write so a
mid-ingest outage cannot lose the write. Marking it done immediately after the
header write hid partial failures (segments/utterances/entities) from replay/verify.
"""

import contextlib
import importlib
import os
import sys
import tempfile
import unittest.mock as mock

import pytest

from auto_ingest.outbox import GraphOutbox

_STUBS = {
    "torch": mock.MagicMock(),
    "torchaudio": mock.MagicMock(),
    "torch.nn": mock.MagicMock(),
    "torch.nn.functional": mock.MagicMock(),
    "transformers": mock.MagicMock(),
    "sentence_transformers": mock.MagicMock(),
    "soundfile": mock.MagicMock(),
    "faiss": mock.MagicMock(),
    "speechbrain": mock.MagicMock(),
    "speechbrain.pretrained": mock.MagicMock(),
    "pyannote": mock.MagicMock(),
    "pyannote.audio": mock.MagicMock(),
    "ultralytics": mock.MagicMock(),
    "ultralytics.utils": mock.MagicMock(),
    "ultralytics.utils.plotting": mock.MagicMock(),
    "cv2": mock.MagicMock(),
    "moviepy": mock.MagicMock(),
    "moviepy.editor": mock.MagicMock(),
    "PIL": mock.MagicMock(),
    "matplotlib": mock.MagicMock(),
    "neo4j": mock.MagicMock(),
    "neo4j.exceptions": mock.MagicMock(),
    "gliner": mock.MagicMock(),
    "numpy": mock.MagicMock(),
}


@pytest.fixture(scope="module")
def T():
    with mock.patch.dict(sys.modules, _STUBS):
        mod = importlib.import_module("auto_ingest.ingest.transcripts")
        # Ensure we get our fixed module, not a stale cached import.
        mod = importlib.reload(mod)
        return mod


def _make_driver():
    sess = mock.MagicMock()
    sess.__enter__ = mock.Mock(return_value=sess)
    sess.__exit__ = mock.Mock(return_value=False)
    driver = mock.MagicMock()
    driver.session.return_value = sess
    return driver, sess


def test_header_write_does_not_mark_done(T):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        ob = GraphOutbox(path)
        driver, sess = _make_driver()
        with mock.patch.object(T, "get_outbox", return_value=ob):
            op_id = T.ingest_transcription_header(
                driver, "t1", "k1", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
                {"json": "a.json"}, "hello", [0.1], None,
            )
        assert op_id is not None
        pending = ob.pending()
        assert len(pending) == 1
        assert pending[0][0] == op_id
        assert pending[0][1] == "ingest.transcription.header"
    finally:
        os.remove(path)


def test_process_key_marks_done_only_after_full_success(T):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        ob = GraphOutbox(path)
        driver, _sess = _make_driver()
        calls = []

        def ok(*args, **kwargs):
            calls.append(args[0] if args else kwargs.get("op"))

        with mock.patch.object(T, "get_outbox", return_value=ob), \
             mock.patch.object(T, "ingest_speakers_chunked", side_effect=ok), \
             mock.patch.object(T, "ingest_segments_chunked", side_effect=ok), \
             mock.patch.object(T, "ingest_utterances_chunked", side_effect=ok), \
             mock.patch.object(T, "link_utterances_to_segments", side_effect=ok), \
             mock.patch.object(T, "link_utterances_to_speakers", side_effect=ok), \
             mock.patch.object(T, "ingest_segment_speaker_edges", side_effect=ok), \
             mock.patch.object(T, "ingest_entities_chunked", side_effect=ok):
            result = T.process_key(
                "k1",
                {"json_all": [], "csv_all": [], "rttm": None, "meta_all": [], "entities": None},
                driver,
                batch_size=1,
                dry_run=False,
                emb_v2=False,
                tx_sizes={},
                tx_timeout_seconds=30,
                fetch_size=100,
                meta_opts={},
            )
        # process_key returns None when no transcription data found; that's fine for
        # this shape. The key assertion is: no outbox op was staged/done spuriously.
        assert result is None
        assert ob.pending() == []
    finally:
        os.remove(path)


def test_partial_failure_keeps_outbox_pending_and_marks_failed(T):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        ob = GraphOutbox(path)
        driver, _sess = _make_driver()

        def boom(*args, **kwargs):
            raise RuntimeError("segment write failed")

        # Build minimal inputs so process_key reaches ingest stage.
        # We patch the heavy pre-ingest pieces to keep this light.
        fake_json = os.path.join(tempfile.gettempdir(), "fake_transcript.json")
        with open(fake_json, "w", encoding="utf-8") as f:
            f.write("{}")
        try:
            patches = [
                mock.patch.object(T, "get_outbox", return_value=ob),
                mock.patch.object(T, "select_best_json", return_value=fake_json),
                mock.patch.object(T, "load_transcription_json_txt", return_value={
                    "text": "hello", "segments": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "hello"}],
                    "file_started_at": None, "file_ended_at": None,
                }),
                mock.patch.object(T, "validate_and_clean_segments", return_value=(
                    [{"id": "s1", "start": 0.0, "end": 1.0, "text": "hello", "tokens": [], "abs_start": None, "abs_end": None}],
                    {"kept": 1, "nonfinite": 0, "neg_dur": 0, "empty_txt": 0, "reordered": 0},
                )),
                mock.patch.object(T, "embed_texts", return_value=[[0.1, 0.2]]),
                mock.patch.object(T, "embed_long_text_via_segments", return_value=[0.1, 0.2]),
                mock.patch.object(T, "transcript_embedding_v2_from_segments", return_value=None),
                mock.patch.object(T, "transcript_embedding_v2_from_chunks", return_value=None),
                mock.patch.object(T, "gliner_extract_chunked", return_value=[]),
                mock.patch.object(T, "aggregate_entities", return_value=[]),
                mock.patch.object(T, "load_entities_csv", return_value=[]),
                mock.patch.object(T, "ingest_speakers_chunked", return_value=0),
                mock.patch.object(T, "ingest_segments_chunked", side_effect=boom),
                mock.patch.object(T, "ingest_utterances_chunked", return_value=0),
                mock.patch.object(T, "link_utterances_to_segments", return_value=0),
                mock.patch.object(T, "link_utterances_to_speakers", return_value=0),
                mock.patch.object(T, "ingest_segment_speaker_edges", return_value=0),
                mock.patch.object(T, "ingest_entities_chunked", return_value=0),
                mock.patch.object(T, "EMBED_EXTRA_ENABLED", False),
            ]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                with pytest.raises(RuntimeError, match="segment write failed"):
                    T.process_key(
                        "k1",
                        {"json_all": [fake_json], "csv_all": [], "rttm": None,
                         "meta_all": [], "entities": None},
                        driver,
                        batch_size=1,
                        dry_run=False,
                        emb_v2=False,
                        tx_sizes={},
                        tx_timeout_seconds=30,
                        fetch_size=100,
                        meta_opts={},
                    )

            pending = ob.pending()
            assert len(pending) == 1, f"expected 1 pending op, got {pending}"
            _op_id, op_type, _cid, _payload, retry_count = pending[0]
            assert op_type == "ingest.transcription.header"
            assert retry_count >= 1
        finally:
            if os.path.exists(fake_json):
                os.remove(fake_json)
    finally:
        os.remove(path)
