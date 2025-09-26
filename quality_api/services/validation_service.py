"""Reusable data quality validation utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..config import settings


@dataclass
class QualityValidatorConfig:
    max_mph_change: float = settings.validation.max_mph_change
    max_lat_change: float = settings.validation.max_lat_change
    max_long_change: float = settings.validation.max_long_change
    expected_frame_increment: float = settings.validation.expected_frame_increment
    enforce_mph: bool = settings.validation.enforce_mph
    enforce_frame_increment: bool = settings.validation.enforce_frame_increment


@dataclass
class ValidationIssue:
    row_index: int
    key: str
    frame: str
    messages: List[str]

    def format(self) -> str:
        header = f"Row {self.row_index} (Key={self.key}, Frame={self.frame}):"
        body = "\n  ".join(self.messages)
        return f"{header}\n  {body}"


class QualityValidator:
    """Validate row based telemetry consistency."""

    def __init__(self, config: QualityValidatorConfig | None = None) -> None:
        self.config = config or QualityValidatorConfig()

    def validate_rows(self, rows: Sequence[Dict[str, str]]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        prev_row: Dict[str, str] | None = None

        for index, row in enumerate(rows, start=1):
            if prev_row is None:
                prev_row = row
                continue

            row_issues: List[str] = []

            if self.config.enforce_frame_increment and self.config.expected_frame_increment > 0:
                self._check_frame_increment(
                    row.get("Frame"),
                    prev_row.get("Frame"),
                    self.config.expected_frame_increment,
                    row_issues,
                )

            if self.config.enforce_mph and self.config.max_mph_change > 0:
                self._check_numeric_delta(
                    row.get("MPH"),
                    prev_row.get("MPH"),
                    self.config.max_mph_change,
                    "MPH",
                    row_issues,
                )

            self._check_numeric_delta(
                row.get("Lat"),
                prev_row.get("Lat"),
                self.config.max_lat_change,
                "Lat",
                row_issues,
            )
            self._check_numeric_delta(
                row.get("Long"),
                prev_row.get("Long"),
                self.config.max_long_change,
                "Long",
                row_issues,
            )

            if row_issues:
                issues.append(
                    ValidationIssue(
                        row_index=index,
                        key=row.get("Key", ""),
                        frame=row.get("Frame", ""),
                        messages=row_issues,
                    )
                )

            prev_row = row

        return issues

    @staticmethod
    def _check_numeric_delta(
        current: str | None,
        previous: str | None,
        max_delta: float,
        field_name: str,
        accumulator: List[str],
    ) -> None:
        if current is None or previous is None:
            accumulator.append(
                f"{field_name} is missing (current={current}, prev={previous})"
            )
            return
        try:
            current_value = float(current)
            previous_value = float(previous)
        except ValueError:
            accumulator.append(
                f"{field_name} is not numeric (current={current}, prev={previous})"
            )
            return
        delta = abs(current_value - previous_value)
        if delta > max_delta:
            accumulator.append(
                f"{field_name} changed by {delta}, exceeds threshold of {max_delta}"
            )

    @staticmethod
    def _check_frame_increment(
        current: str | None,
        previous: str | None,
        expected_increment: float,
        accumulator: List[str],
    ) -> None:
        if current is None or previous is None:
            accumulator.append(
                f"Frame missing for increment check (current={current}, prev={previous})"
            )
            return
        try:
            current_value = float(current)
            previous_value = float(previous)
        except ValueError:
            accumulator.append(
                f"Frame value is not numeric (current={current}, prev={previous})"
            )
            return
        delta = current_value - previous_value
        if delta != expected_increment:
            accumulator.append(
                f"Frame should increment by {expected_increment}, got {delta}"
            )


def load_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return list(reader)


def write_report(issues: Iterable[ValidationIssue], output_path: str | Path) -> None:
    issues = list(issues)
    with open(output_path, "w", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(issue.format() + "\n\n")

