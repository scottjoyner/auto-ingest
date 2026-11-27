"""The human `custody status` report must not silently drop fields.

``auto_ingest.custody.report`` documents its human mode as "a fixed-width
rendering of exactly the same fields" as ``--json``. ``status`` defaults to that
human mode, so it is the interface an operator - and a label-keyed log parser -
actually sees. Two lines shared the label ``destination``: the second overwrote
the first in any consumer that keys a ``label  value`` line by its label, and the
logical path, host path and mount observation vanished without a trace.
"""
from __future__ import annotations

from custody_helpers import campaign, evidence, strict_policy

from auto_ingest.custody import build_status
from auto_ingest.custody.report import status_text


def _status_text() -> str:
    # Entirely in-memory: no bundle, no mount, no clock-dependent fields.
    return status_text(build_status(campaign(), evidence(), strict_policy()))


def _top_level_labels(text: str) -> list[str]:
    """Field labels of the fixed-width report (indented details are not fields)."""
    labels = []
    for line in text.splitlines():
        if not line[:1].isalpha():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            labels.append(parts[0])
    return labels


def _label_map(text: str) -> dict[str, str]:
    """Mimic a consumer that keys each `label  value` line by its label."""
    parsed: dict[str, str] = {}
    for label in _top_level_labels(text):
        line = next(ln for ln in text.splitlines()
                    if ln[:1].isalpha() and ln.split(None, 1)[0] == label)
        parsed[label] = line.split(None, 1)[1]
    return parsed


def test_status_text_top_level_labels_are_unique():
    labels = _top_level_labels(_status_text())
    assert len(labels) == len(set(labels)), (
        "duplicate field labels silently hide earlier fields: "
        f"{[label for label in labels if labels.count(label) > 1]}"
    )


def test_destination_facts_survive_a_label_keyed_parse():
    parsed = _label_map(_status_text())
    # The identity/location facts used to be dropped because a second line
    # reused the `destination` label and won the parse.
    assert "logical=" in parsed["destination"]
    assert "host_path=" in parsed["destination"]
    assert "mounted=" in parsed["destination"]
    # The verification facts must remain reachable under their own label.
    assert "verified_files=" in parsed["destination_verification"]
    assert "verification_complete=" in parsed["destination_verification"]
