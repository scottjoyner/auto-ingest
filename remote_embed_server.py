#!/usr/bin/env python3
"""
remote_embed_server.py — GPU speaker-embedding worker for link_global_speakers.

Runs on a machine with an AMD GPU + ROCm torch (e.g. x1-370's R9 9700XT) and
serves ECAPA embeddings over HTTP. The linker on the pipeline host batches
already-clipped snip waveforms and posts them here as raw float32 PCM; this
worker runs one ECAPA forward pass on the GPU and returns 192-d vectors.

Why remote: ECAPA per-snip inference is CPU-bound and batch-flat, so on a
CPU-only host the linker saturates ~1 core per embedding and the rest idle.
A GPU (RDNA4/R9 9700XT) does the same forward pass in milliseconds.

The client clips + gates snips locally (it already holds the audio in RAM)
and sends raw audio inline — no audio-path coupling, so any pipeline host can
use any embed worker regardless of shared mounts.

Endpoints
---------
GET  /health                 backend, device, model
POST /embed                  {sr: 16000, snips: [{idx, audio: <b64 f32>}]}
                             -> {vectors: {idx: [...192 floats...]}}

Run (on the GPU host):
    ./.venv-roc/bin/python3 remote_embed_server.py --host 0.0.0.0 --port 8901
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SR = 16000
ECAPA_NAME = os.getenv("ECAPA_NAME", "speechbrain/spkrec-ecapa-voxceleb")

_model = None
_device = "cpu"


def _load_model():
    global _model, _device
    if _model is not None:
        return _model
    import torch
    if torch.cuda.is_available() or getattr(torch.version, "hip", None):
        _device = "cuda:0"
        torch.set_num_threads(1)
    else:
        _device = "cpu"
    from speechbrain.pretrained import EncoderClassifier
    logging.info(f"Loading ECAPA on {_device}…")
    _model = EncoderClassifier.from_hparams(source=ECAPA_NAME,
                                            run_opts={"device": _device})
    return _model


def _embed(wavs: Dict[int, np.ndarray]) -> Dict[int, List[float]]:
    model = _load_model()
    import torch
    out: Dict[int, List[float]] = {}
    if not wavs:
        return out
    idxs = list(wavs.keys())
    max_len = max(int(w.shape[0]) for w in wavs.values())
    batch = torch.zeros(len(idxs), max_len, dtype=torch.float32, device=_device)
    lens = torch.zeros(len(idxs), dtype=torch.float32, device=_device)
    for i, idx in enumerate(idxs):
        w = torch.from_numpy(np.ascontiguousarray(wavs[idx], dtype=np.float32))
        batch[i, :w.numel()] = w
        lens[i] = w.numel() / max_len
    with torch.inference_mode():
        embs = model.encode_batch(batch, wav_lens=lens).cpu().numpy()
    for i, emb in zip(idxs, embs):
        out[i] = emb.reshape(-1).astype(np.float32).tolist()
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: Any):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "device": _device, "ecapa": ECAPA_NAME})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > (256 << 20):
            self._json(413, {"error": "payload too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        sr = int(payload.get("sr", DEFAULT_SR))
        wavs: Dict[int, np.ndarray] = {}
        for sn in payload.get("snips", []):
            idx = int(sn["idx"])
            raw = base64.b64decode(sn["audio"])
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size:
                wavs[idx] = arr
        t0 = time.time()
        try:
            vecs = _embed(wavs)
        except Exception as e:
            self._json(500, {"error": f"embed failed: {e}"})
            return
        self._json(200, {"vectors": vecs, "elapsed": round(time.time() - t0, 3), "sr": sr})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    logging.info(f"remote-embed server on http://{args.host}:{args.port} "
                 f"(device={_device})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()