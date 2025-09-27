#!/usr/bin/env python3
"""CLI for validating telemetry metadata consistency."""

from __future__ import annotations

from pathlib import Path

from quality_api.services.validation_service import (
    QualityValidator,
    QualityValidatorConfig,
    load_csv_rows,
    write_report,
)

INPUT_FILE = "rdx_dashcam_metadata_filled_6.csv"
OUTPUT_FILE = "validation_report.txt"


def main() -> None:
    config = QualityValidatorConfig()
    validator = QualityValidator(config)

    rows = load_csv_rows(INPUT_FILE)
    issues = validator.validate_rows(rows)

    if issues:
        write_report(issues, OUTPUT_FILE)
        print(
            f"Validation complete. Found {len(issues)} issues. "
            f"See '{Path(OUTPUT_FILE).resolve()}' for details."
        )
    else:
        print("No issues found. Data appears consistent with the defined rules.")


if __name__ == "__main__":
    main()
