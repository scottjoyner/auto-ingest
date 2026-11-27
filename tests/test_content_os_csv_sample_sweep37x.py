"""Hermetic regression for CsvAdapter's silent 10-row text truncation (sweep37x).

A CSV export with no recognised text column is turned into SourceDocument.text by
serialising only ``rows[:10]`` while ``metadata['rows_read']`` reports the full
count and no warning is emitted.  Downstream drafting therefore sees a truncated
source that looks complete.  All fixtures are tiny local temp files; no network,
Neo4j, model load or live service is involved.
"""

from __future__ import annotations

from pathlib import Path

from content_os.adapters import CsvAdapter

N_ROWS = 12
VALUES = [f"value-{i:02d}" for i in range(1, N_ROWS + 1)]


def _export(tmp_path: Path, *, header: str, values: list[str]) -> Path:
    path = tmp_path / "export.csv"
    lines = [header] + [f"{i},{v}" for i, v in enumerate(values, 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _no_text_column(tmp_path: Path) -> Path:
    return _export(tmp_path, header="id,value", values=VALUES)


def test_no_text_column_exposes_every_row(tmp_path):
    """DEFECT: rows 11+ silently vanish from the extracted text."""
    doc = CsvAdapter().extract(_no_text_column(tmp_path))
    assert doc.metadata["rows_read"] == N_ROWS  # control: full count is known
    missing = [v for v in VALUES if v not in doc.text]
    assert not missing, f"rows silently omitted from extracted text: {missing}"


def test_extracted_row_count_matches_rows_read(tmp_path):
    """DEFECT: metadata reports 12 rows but text holds only the 10-row sample."""
    doc = CsvAdapter().extract(_no_text_column(tmp_path))
    emitted = [line for line in doc.text.splitlines() if line.strip()]
    assert len(emitted) == doc.metadata["rows_read"]


def test_text_column_csv_still_extracts_all_rows(tmp_path):
    """CONTROL: the recognised-text-column branch already includes every row."""
    path = _export(tmp_path, header="id,text", values=VALUES)
    doc = CsvAdapter().extract(path)
    assert doc.metadata["rows_read"] == N_ROWS
    assert all(v in doc.text for v in VALUES)


def test_header_only_csv_reports_no_rows(tmp_path):
    """CONTROL: an empty export keeps the explicit 'No rows found.' text."""
    path = _export(tmp_path, header="id,value", values=[])
    doc = CsvAdapter().extract(path)
    assert doc.metadata["rows_read"] == 0
    assert doc.text == "No rows found."


def test_preview_metadata_stays_bounded(tmp_path):
    """CONTROL: sample_rows is a bounded preview and remains first-10."""
    doc = CsvAdapter().extract(_no_text_column(tmp_path))
    assert len(doc.metadata["sample_rows"]) == 10
    assert doc.metadata["sample_rows"][0]["value"] == "value-01"
