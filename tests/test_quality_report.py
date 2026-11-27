"""Hermetic tests for the dashcam metadata validator.

``validate_rows`` consumes ``csv.DictReader`` output, so the header is already
gone by the time it sees the first row. These tests pin that contract: row 1 of
the report is row 1 of the *data*, and every data row is checked on its own
merits even when it is the only row in the file.
"""
from __future__ import annotations

import csv
import io

from quality_report import validate_rows


def _rows(csv_text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def test_single_data_row_with_invalid_lat_is_reported():
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,not-a-number,-74.0,1\n"
    )
    issues = validate_rows(rows)
    assert len(issues) == 1
    assert "Row 1" in issues[0]
    assert "not-a-number" in issues[0]


def test_single_data_row_with_invalid_long_is_reported():
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,40.0,oops,1\n"
    )
    issues = validate_rows(rows)
    assert len(issues) == 1
    assert "Row 1" in issues[0]
    assert "oops" in issues[0]


def test_bad_first_row_is_attributed_to_first_row_not_second():
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,not-a-number,-74.0,1\n"
        "clip2,12,40.0,-74.0,61\n"
    )
    issues = validate_rows(rows)
    assert len(issues) == 1
    assert "Row 1" in issues[0]
    assert "Row 2" not in issues[0]


def test_valid_rows_still_produce_no_issues():
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,40.0,-74.0,1\n"
        "clip2,12,40.5,-74.1,61\n"
    )
    assert validate_rows(rows) == []


def test_lat_jump_is_reported_on_the_later_row():
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,40.0,-74.0,1\n"
        "clip2,12,42.0,-74.0,61\n"
    )
    issues = validate_rows(rows)
    assert len(issues) == 1
    assert "Row 2" in issues[0]
    assert "exceeds threshold" in issues[0]


def test_missing_column_is_reported_not_a_crash():
    # DictReader fills absent trailing cells with None. The old float() probe
    # raised TypeError on that value instead of reporting an issue.
    rows = _rows(
        "Key,MPH,Lat,Long,Frame\n"
        "clip1,10,40.0\n"
    )
    issues = validate_rows(rows)
    assert len(issues) == 1
    assert "Row 1" in issues[0]
    assert "Long" in issues[0]


def test_main_reports_bad_single_row_instead_of_claiming_consistent(
    tmp_path, monkeypatch, capsys
):
    import quality_report

    src = tmp_path / "in.csv"
    src.write_text("Key,MPH,Lat,Long,Frame\nclip1,10,bad,-74.0,1\n")
    out = tmp_path / "report.txt"
    monkeypatch.setattr(quality_report, "INPUT_FILE", str(src))
    monkeypatch.setattr(quality_report, "OUTPUT_FILE", str(out))

    quality_report.main()

    printed = capsys.readouterr().out
    assert "No issues found" not in printed
    assert "1 issues" in printed
    report = out.read_text()
    assert "Row 1" in report
    assert "bad" in report
