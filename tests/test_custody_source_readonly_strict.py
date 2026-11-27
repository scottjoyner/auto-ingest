"""Source read_only must be parsed, not inferred from truthiness.

`SourceRef.from_dict` used `bool(raw.get("read_only", False))`. A campaign.json
carrying the string `"false"` (or `0`, `""`, etc.) was loaded as a read-only
source, weakening the `require_read_only_source` release condition and the
declared-vs-observed comparison.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.campaign import SourceRef


def test_string_false_is_not_read_only():
    assert SourceRef.from_dict({"read_only": "false"}).read_only is False
    assert SourceRef.from_dict({"read_only": "0"}).read_only is False
    assert SourceRef.from_dict({"read_only": 0}).read_only is False


@pytest.mark.parametrize("value", [True, "true", "yes", "on", 1])
def test_true_spellings_still_parse(value):
    assert SourceRef.from_dict({"read_only": value}).read_only is True


def test_missing_flag_defaults_to_false():
    assert SourceRef.from_dict({}).read_only is False


@pytest.mark.parametrize("value", ["maybe", "false-ish", [], {}])
def test_unparseable_flags_are_refused(value):
    with pytest.raises(ValueError, match="read_only"):
        SourceRef.from_dict({"read_only": value})
