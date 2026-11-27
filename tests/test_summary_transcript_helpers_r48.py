"""Hermetic contract tests for ``build_summaries_ollama_a.py`` input-side helpers.

The module is a pure-stdlib script: no Ollama service, network call, model load,
live transcript tree or production path is touched. Every filesystem input is a
pytest ``tmp_path`` and every helper runs against the unmodified module. These
tests lock in the existing (r48) behaviour; they change no production code.
"""

import csv
from pathlib import Path

import pytest

import build_summaries_ollama_a as m  # repo root is on sys.path via tests/conftest


def _write_csv(path, name, fieldnames, rows):
    csv_path = path / name
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


# ---------------------------------------------------------------------------
# _detect_stem -- recorder-stem regex recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("2024_1229_194226", "2024_1229_194226"),
        ("20250827061344", "20250827061344"),
        ("20250405212855_000029", "20250405212855_000029"),
        ("2024_1229_194226_extra", "2024_1229_194226"),
        ("20250101120000.wav", "20250101120000"),
    ],
)
def test_detect_stem_recognized_layouts(name, expected):
    assert m._detect_stem(Path(name)) == expected


@pytest.mark.parametrize("name", ["random", "12345", "", "recording-final"])
def test_detect_stem_rejects_non_recorder_names(name):
    assert m._detect_stem(Path(name)) is None


# ---------------------------------------------------------------------------
# _read_text_file -- transcript text ingestion
# ---------------------------------------------------------------------------


def test_read_text_file_strips_whitespace(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("  hello world \n", encoding="utf-8")
    assert m._read_text_file(p) == "hello world"


def test_read_text_file_returns_none_when_missing(tmp_path):
    assert m._read_text_file(tmp_path / "missing.txt") is None


def test_read_text_file_replaces_invalid_utf8(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"caf\xe9 ok")
    assert m._read_text_file(p) == "caf\ufffd ok"


# ---------------------------------------------------------------------------
# _read_csv_concat_text -- CSV transcript column preference / concatenation
# ---------------------------------------------------------------------------


def test_read_csv_concat_text_prefers_named_column_in_header_order(tmp_path):
    p = _write_csv(
        tmp_path,
        "pref.csv",
        ["ts", "utterance", "text"],
        [
            {"ts": "t1", "utterance": "  first line  ", "text": "noise"},
            {"ts": "t2", "utterance": "second", "text": "noise2"},
        ],
    )
    # Cells are copied verbatim (no per-cell strip), but the joined text is
    # globally stripped, so the first row's leading spaces are removed.
    assert m._read_csv_concat_text(p) == "first line  \nsecond"


def test_read_csv_concat_text_prefer_column_case_insensitive(tmp_path):
    p = _write_csv(
        tmp_path,
        "upper.csv",
        ["Transcript", "junk"],
        [{"Transcript": "UPPER", "junk": "j"}],
    )
    assert m._read_csv_concat_text(p) == "UPPER"


def test_read_csv_concat_text_missing_cell_falls_back_to_all_strings(tmp_path):
    p = _write_csv(
        tmp_path,
        "fallback.csv",
        ["ts", "utterance", "text"],
        [{"ts": "t3", "utterance": "", "text": "fallback text"}],
    )
    assert m._read_csv_concat_text(p) == "t3 fallback text"


def test_read_csv_concat_text_joins_all_nonempty_strings_without_preference(tmp_path):
    p = _write_csv(
        tmp_path,
        "all.csv",
        ["a", "b", "n"],
        [
            {"a": "one", "b": "", "n": "5"},
            {"a": "", "b": "two", "n": "6"},
        ],
    )
    assert m._read_csv_concat_text(p) == "one 5\ntwo 6"


def test_read_csv_concat_text_empty_and_headerless(tmp_path):
    p = _write_csv(tmp_path, "empty.csv", ["text"], [])
    assert m._read_csv_concat_text(p) is None
    q = tmp_path / "none.csv"
    q.write_text("", encoding="utf-8")
    assert m._read_csv_concat_text(q) is None


# ---------------------------------------------------------------------------
# _choose_best_transcript -- transcript sidecar selection order
# ---------------------------------------------------------------------------


def test_choose_best_transcript_prefers_medium_over_plain_and_csv(tmp_path):
    d = tmp_path / "sel"
    d.mkdir()
    stem = "2024_1229_194226"
    for fn, content in [
        (f"{stem}_medium_transcription.txt", "medium content"),
        (f"{stem}_transcription.txt", "plain content"),
        (f"{stem}_transcription.csv", "ts,text\n1,csv content\n"),
    ]:
        (d / fn).write_text(content, encoding="utf-8")
    best = m._choose_best_transcript(d, stem)
    assert best is not None
    assert best.stem == stem
    assert best.text == "medium content"
    assert best.source_path == d / f"{stem}_medium_transcription.txt"


@pytest.mark.parametrize(
    "filename,content,expected",
    [
        ("2024_1229_194226_transcription.txt", "plain content", "plain content"),
        ("2024_1229_194226_transcription.csv", "ts,text\n1,csv content\n", "csv content"),
        ("2024_1229_194226.json", '{"a": 1}', '{"a": 1}'),
        ("2024_1229_194226.txt", "fallback txt", "fallback txt"),
        ("2024_1229_194226.vtt", "WEBVTT\nline", "WEBVTT\nline"),
    ],
)
def test_choose_best_transcript_fallback_order(tmp_path, filename, content, expected):
    d = tmp_path / "f"
    d.mkdir()
    (d / filename).write_text(content, encoding="utf-8")
    best = m._choose_best_transcript(d, "2024_1229_194226")
    assert best is not None
    assert best.text == expected
    assert best.source_path == d / filename


def test_choose_best_transcript_returns_none_when_absent(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert m._choose_best_transcript(d, "2024_1229_194226") is None


# ---------------------------------------------------------------------------
# _iter_unique_stems -- recursive transcript discovery + dedupe
# ---------------------------------------------------------------------------


def test_iter_unique_stems_recursive_and_deduped(tmp_path):
    scan = tmp_path / "scan"
    (scan / "sub").mkdir(parents=True)
    (scan / "20250101120000_transcription.txt").write_text("a", encoding="utf-8")
    (scan / "sub" / "20250101120000_transcription.csv").write_text(
        "ts,text\n1,b\n", encoding="utf-8"
    )
    (scan / "sub" / "20250102120000_transcription.txt").write_text("c", encoding="utf-8")
    (scan / "ignored.txt").write_text("x", encoding="utf-8")

    got = sorted((stem, str(dp)) for stem, dp in m._iter_unique_stems([scan, scan]))
    assert got == sorted(
        [
            ("20250101120000", str(scan)),
            ("20250101120000", str(scan / "sub")),
            ("20250102120000", str(scan / "sub")),
        ]
    )


def test_iter_unique_stems_ignores_non_recorder_names(tmp_path):
    d = tmp_path / "misc"
    d.mkdir()
    (d / "notes_transcription.txt").write_text("x", encoding="utf-8")
    (d / "2026010_transcription.txt").write_text("x", encoding="utf-8")
    assert m._iter_unique_stems([d]) == []


# ---------------------------------------------------------------------------
# _truncate -- prompt-bound head/tail shaping
# ---------------------------------------------------------------------------


def test_truncate_short_text_unchanged():
    assert m._truncate("abc", 100) == "abc"
    assert m._truncate("abcdef", 6) == "abcdef"


def test_truncate_long_text_head_tail_shape():
    text = "".join(str(i % 10) for i in range(200))
    out = m._truncate(text, 100)
    assert out == text[:70] + "\n...\n" + text[-25:]
    assert len(out) == 100


# ---------------------------------------------------------------------------
# OllamaClient.__init__ -- pure URL validation/defaults (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,host,port,base,is_https",
    [
        ("http://127.0.0.1:11434", "127.0.0.1", 11434, "", False),
        ("https://host:1/base/", "host", 1, "/base", True),
        ("http://myhost", "myhost", 11434, "", False),
    ],
)
def test_ollama_client_parses_http_url_without_network(url, host, port, base, is_https):
    c = m.OllamaClient(url)
    assert (c._host, c._port, c._base_path, c._is_https) == (host, port, base, is_https)


@pytest.mark.parametrize("url", ["ftp://x", "not-a-url", "127.0.0.1:11434"])
def test_ollama_client_rejects_non_http_scheme(url):
    with pytest.raises(ValueError):
        m.OllamaClient(url)
