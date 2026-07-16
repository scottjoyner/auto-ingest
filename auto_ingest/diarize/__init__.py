"""Diarize subpackage: global speaker linking.

NOTE (W-50): Sophia owns voice AUTH (authenticated_scott / unknown_speaker
taxonomy in the shared contract). auto-ingest only LINKS local Speaker nodes to
global identities (GlobalSpeaker). It must never make an authentication
decision. See LLD §3.4 / HLD §4 cross-cutting "Identity / auth".
"""

from __future__ import annotations

from .link_global_speakers import main as run_link_global_speakers

__all__ = ["run_link_global_speakers", "link_global_speakers"]
