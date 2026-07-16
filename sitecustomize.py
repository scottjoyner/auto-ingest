"""Runtime compatibility shims for the auto-ingest venv.

Python imports this module automatically when running scripts from this
repository. Keep this limited to import-time compatibility fixes for third-party
packages used by older pipeline stages.
"""

try:
    import numpy as _np
    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan  # pyannote.audio 3.1.x compatibility with NumPy 2.x callers
except Exception:
    pass

try:
    import torchaudio as _torchaudio
    if not hasattr(_torchaudio, "set_audio_backend"):
        # pyannote.audio 3.1.x still calls this legacy API; torchaudio 2.6+
        # removed it because audio backends are dispatcher based.
        setattr(_torchaudio, "set_audio_backend", lambda *args, **kwargs: None)
except Exception:
    pass
