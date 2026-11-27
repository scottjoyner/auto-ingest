"""Hermetic regression tests for audio-base path containment.

``auto_ingest.ingest.transcripts.is_in_audio_base`` is the first sort key of
``select_best_json``: a path wrongly classified as in-base outranks a genuine
in-base transcript. The base implementation compares ``startswith`` on the two
resolved *strings*, which also accepts a sibling directory whose name merely
begins with the base name (``.../audio_backup/x.json`` vs ``.../audio``).

The module is imported with the same ML-stack stubs used by the existing key
tests, so no model, live database, network or real media path is touched.
"""
from __future__ import annotations

import importlib
import sys
import unittest.mock as mock

import pytest

_STUBS = {
    "torch": mock.MagicMock(),
    "torchaudio": mock.MagicMock(),
    "torch.nn": mock.MagicMock(),
    "torch.nn.functional": mock.MagicMock(),
    "transformers": mock.MagicMock(),
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
}


@pytest.fixture(scope="module")
def T():
    with mock.patch.dict(sys.modules, _STUBS):
        return importlib.import_module("auto_ingest.ingest.transcripts")


@pytest.fixture
def base(T, tmp_path, monkeypatch):
    """A synthetic AUDIO_BASE under tmp_path; no live path is consulted."""
    audio = tmp_path / "audio"
    audio.mkdir()
    monkeypatch.setattr(T, "AUDIO_BASE", audio, raising=True)
    return audio


def test_sibling_with_prefix_is_not_in_audio_base(T, base, tmp_path):
    """`.../audio_backup` must not count as inside `.../audio`."""
    sibling = tmp_path / "audio_backup"
    sibling.mkdir()
    p = sibling / "2025_0202_171732_large-v3_transcription.json"
    p.write_text("{}", encoding="utf-8")
    assert T.is_in_audio_base(str(p)) is False


def test_real_in_base_file_is_still_in_audio_base(T, base):
    """Positive control: a real child must not be rejected by the fix."""
    p = base / "2025_0202_171732_large-v3_transcription.json"
    p.write_text("{}", encoding="utf-8")
    assert T.is_in_audio_base(str(p)) is True


def test_symlink_out_of_base_is_not_in_audio_base(T, base, tmp_path):
    """Containment follows the resolved target, not the link's spelling."""
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = base / "linked.json"
    link.symlink_to(outside)
    assert T.is_in_audio_base(str(link)) is False
