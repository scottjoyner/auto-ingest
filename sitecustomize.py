"""Runtime compatibility shims for the auto-ingest venv.

NOTE: this module is NOT auto-imported by the interpreter (the system
``sitecustomize`` at ``/usr/lib/python3.12/`` shadows it). The functional
copy of these shims is installed into the venv as
``.venv/lib/python3.12/site-packages/auto_ingest_compat.py`` and wired up via
``auto_ingest_compat.pth`` (processed by the ``site`` module at startup).

Keep the two files in sync. The shims fix import-time / load-time
incompatibilities between the pinned ML stack and newer packages installed in
this venv:

  * torchaudio 2.6+ removed ``set_audio_backend``/``get_audio_backend``;
    pyannote.audio 3.1.x still calls them at import time.
  * torchaudio 2.9+ removed the ``torchaudio.backend`` package; pyannote
    imports ``torchaudio.backend.common.AudioMetaData`` at module scope.
  * torchaudio 2.11 removed ``torchaudio.info``; soundfile provides the info.
  * huggingface_hub 0.26+ renamed ``use_auth_token`` -> ``token``; pyannote
    still passes the old name internally.
  * torch 2.6+ defaults ``torch.load(weights_only=True)``; pyannote checkpoints
    pickle arbitrary classes, so we restore pre-2.6 behavior (trusted models).
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
        setattr(_torchaudio, "set_audio_backend", lambda *args, **kwargs: None)
    if not hasattr(_torchaudio, "get_audio_backend"):
        setattr(_torchaudio, "get_audio_backend", lambda *args, **kwargs: "soundfile")
except Exception:
    pass

try:
    import torchaudio as _ta2
    if not hasattr(_ta2, "info"):
        import soundfile as _sf

        def _compat_info(filepath, **kwargs):
            _inf = _sf.info(str(filepath))

            class _Info:
                sample_rate = _inf.samplerate
                num_frames = _inf.frames
                num_channels = _inf.channels
                bits_per_sample = getattr(_inf, "bits_per_sample", 16)
                encoding = _inf.subtype

            return _Info()

        _ta2.info = _compat_info
except Exception:
    pass

try:
    import sys as _sys
    import types as _types
    import torchaudio as _ta
    if not hasattr(_ta, "backend"):
        _backend = _types.ModuleType("torchaudio.backend")
        _backend.common = _types.ModuleType("torchaudio.backend.common")
        try:
            from collections import namedtuple as _nt
            _AudioMetaData = _nt(
                "AudioMetaData",
                "sample_rate num_frames num_channels bits_per_sample encoding",
            )
            _AudioMetaData.__module__ = "torchaudio.backend.common"
        except Exception:
            _AudioMetaData = type(
                "AudioMetaData",
                (),
                {"__init__": lambda self, *a, **k: None},
            )
        _backend.common.AudioMetaData = _AudioMetaData
        _sys.modules["torchaudio.backend"] = _backend
        _sys.modules["torchaudio.backend.common"] = _backend.common
except Exception:
    pass

try:
    import inspect as _inspect
    import huggingface_hub as _hf

    if "use_auth_token" not in _inspect.signature(_hf.hf_hub_download).parameters:
        _orig_download = _hf.hf_hub_download

        def _compat_download(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _orig_download(*args, **kwargs)

        _hf.hf_hub_download = _compat_download
except Exception:
    pass

try:
    import torch as _torch

    try:
        _torch.serialization.add_safe_globals([_torch.torch_version.TorchVersion])
    except Exception:
        pass

    import inspect as _tinspect

    if _tinspect.isfunction(_torch.load):
        _orig_load = _torch.load
        try:
            from torch import serialization as _tserialization
            _orig_load_safer = _tserialization.load
        except Exception:
            _orig_load_safer = None

        def _compat_load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **kwargs):
            if weights_only is None:
                weights_only = False
            return _orig_load(f, map_location, pickle_module, weights_only=weights_only, mmap=mmap, **kwargs)

        _torch.load = _compat_load
        if _orig_load_safer is not None:
            _tserialization.load = _compat_load
except Exception:
    pass
