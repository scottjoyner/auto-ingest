"""BOM hardening for the remaining transcripts.py CSV readers (round 20261128).

The r36 merge normalized three other CSV readers to utf-8-sig; these three
were left with plain utf-8, so a NAS/Excel BOM stuck to the first header and
silently produced empty/partial parses. Byte digests are taken over raw bytes
and are unaffected by the decode change.
"""
from __future__ import annotations

import hashlib

from auto_ingest.ingest import transcripts as tx

BOM = b"\xef\xbb\xbf"


def _write(path, body: bytes, bom: bool) -> bytes:
    raw = (BOM if bom else b"") + body
    path.write_bytes(raw)
    return raw


def test_transcription_csv_accepts_bom_header(tmp_path):
    p = tmp_path / "transcript.csv"
    _write(p, b"Text,StartTime,EndTime\r\nhello,0.0,1.0\r\n", bom=True)
    got = tx.load_transcription_csv(str(p))
    assert got is not None
    assert got["segments"][0]["text"] == "hello"
    assert got["segments"][0]["start"] == 0.0
    assert "\ufeff" not in got["text"]


def test_transcription_csv_bomless_unchanged(tmp_path):
    body = b"Text,StartTime,EndTime\r\nhello,0.0,1.0\r\n"
    p = tmp_path / "transcript.csv"
    _write(p, body, bom=False)
    got = tx.load_transcription_csv(str(p))
    assert got is not None
    assert got["segments"][0]["text"] == "hello"


def test_entities_csv_accepts_bom_header(tmp_path):
    p = tmp_path / "entities.csv"
    _write(p, b"text,label\r\nfoo,PERSON\r\n", bom=True)
    got = tx.load_entities_csv(str(p))
    assert len(got) == 1
    assert got[0]["text"] == "foo" and got[0]["label"] == "PERSON"


def test_dashcam_metadata_csv_accepts_bom_header(tmp_path):
    p = tmp_path / "metadata.csv"
    raw = _write(p, b"Frame,Lat,Long,MPH\r\n0,40.0,-73.0,55\r\n", bom=True)
    kept, stats = tx.parse_dashcam_metadata_csv(
        str(p), fps=1.0, downsample_sec=1, lon_auto_west=False,
        allow_swap=False, bbox_str="", max_speed_mph=100,
    )
    assert stats["seen"] == 1, "BOM must not hide the Frame header"
    assert len(kept) == 1, "the parsed row must survive BOM normalization"
    # The pinned byte digest is over raw bytes; BOM remains part of the file.
    assert hashlib.sha256(raw).digest() == hashlib.sha256(p.read_bytes()).digest()
