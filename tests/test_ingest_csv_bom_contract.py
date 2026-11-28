"""Static contract: every NAS/Excel CSV decode site uses utf-8-sig.

BOM fixes landed piecemeal across r35/r36/r7 and this round. A runtime test
cannot cover every reader without importing torch/transformers/neo4j, so this
test reads the source text directly and fails closed if any fixed decode site
regresses to plain utf-8. No runtime behaviour change, no heavy imports.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (relative path, decode form, minimum number of sites that must carry it)
SITES = (
    ("auto_ingest/ingest/transcripts.py", 'encoding="utf-8-sig"', 3),
    ("auto_ingest/ingest_write.py", 'encoding="utf-8-sig"', 1),
    ("auto_ingest/ingest_import.py", 'encoding="utf-8-sig"', 1),
    ("auto_ingest/custody/external_inventory.py", 'decode("utf-8-sig")', 1),
)


def test_csv_decode_sites_keep_bom_normalization() -> None:
    for rel, needle, minimum in SITES:
        path = ROOT / rel
        assert path.is_file(), "missing reader: " + rel
        text = path.read_text(encoding="utf-8")
        found = len(re.findall(re.escape(needle), text))
        assert found >= minimum, (
            f"{rel}: expected >= {minimum} {needle!r} sites, found {found}"
        )
        # A plain-utf8 open next to a csv.DictReader is the regression this
        # contract exists to catch; keep at least the normalized form present.
        assert "utf-8-sig" in text, rel + " lost BOM normalization entirely"
