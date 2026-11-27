"""PERSONA_SOURCE must be interpreted identically by the two canonical readers.

``auto_ingest.shorts.scheduling.default_persona_source()`` is the calendar-side
reader; ``auto_ingest.shorts.persona.PersonaConfig.from_env()`` is the render-side
reader. They are configured by the SAME environment variable, so a value that is
valid for one must be valid for the other. Otherwise the posting calendar plans
one face source while the renderer silently uses another.

Pure/offline: only env reads and two dependency-light modules are involved.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import persona, scheduling


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" photo ", "photo"),          # surrounding whitespace from a shell config
        ("PHOTO", "photo"),            # upper-case from an exported variable
        ("Video\n", "video"),          # trailing newline from a dotenv-style value
        (" Stylized ", "stylized"),    # known value, padded
    ],
)
def test_persona_source_env_is_normalized_consistently(monkeypatch, raw, expected):
    monkeypatch.setenv("PERSONA_SOURCE", raw)
    assert scheduling.default_persona_source() == expected
    # The render-side reader must agree with the calendar-side reader.
    assert persona.PersonaConfig.from_env().source == expected


@pytest.mark.parametrize("raw", ["bogus", "", "   "])
def test_invalid_persona_source_falls_back_to_stylized(monkeypatch, raw):
    """Control: unknown/blank values still degrade to the stylized avatar."""
    monkeypatch.setenv("PERSONA_SOURCE", raw)
    assert persona.PersonaConfig.from_env().source == "stylized"
    assert scheduling.default_persona_source() == "stylized"
