#!/usr/bin/env python3
"""
Auto-Ingest Job Trigger API
Lightweight HTTP server that exposes endpoints to enqueue jobs
into the distributed worker queue.

Usage:
    # Enqueue a job
    curl -X POST http://localhost:8765/api/enqueue -H "Content-Type: application/json" \
         -d '{"kind": "all", "scan_roots": "/nas/fileserver/dashcam"}'

    # Check worker queue status
    curl http://localhost:8765/api/status

    # Trigger a one-shot ingest run
    curl -X POST http://localhost:8765/api/run -H "Content-Type: application/json" \
         -d '{"type": "ingest"}'
"""

import http.server
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Path configuration — single source of truth is `.env` (NAS_ROOT / DROP_ROOT),
# which docker-compose.yml injects into the container environment. The defaults
# below MUST match `.env` + `docker-compose.yml` (W-46). The host mount
# `${NAS_ROOT}` (= NAS4) is bound into the container at `/nas`, so DROP_ROOT
# resolves to `/nas/drop` inside the container. Do NOT reintroduce the old
# `/media/scott/NAS1` default — that was a stale path (NAS1 no longer exists).
DROP_ROOT = os.environ.get("DROP_ROOT", "/nas/drop")
NAS_ROOT = os.environ.get("NAS_ROOT", "/media/scott/NAS4")
APP_DIR = "/app"  # mounted as /app in container

class JobTriggerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for job trigger endpoints."""

    def log_message(self, format, *args):
        """Suppress default logging to stderr."""
        pass

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._handle_status()
        elif parsed.path == "/api/health":
            self._send_json(200, {"status": "ok", "drop_root": DROP_ROOT})
        else:
            self._send_json(404, {"error": "not found", "endpoints": [
                "GET /api/status",
                "GET /api/health",
                "POST /api/enqueue",
                "POST /api/run"
            ]})

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if parsed.path == "/api/enqueue":
            self._handle_enqueue(data)
        elif parsed.path == "/api/run":
            self._handle_run(data)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_status(self):
        """Return queue status."""
        drop = Path(DROP_ROOT)
        if not drop.exists():
            self._send_json(200, {"drop_root": str(drop), "exists": False, "message": "drop root not mounted yet"})
            return

        pending = list(drop.glob("*.job"))
        claimed = list((drop / "claimed").glob("*")) if (drop / "claimed").exists() else []
        done = list((drop / "done").glob("*")) if (drop / "done").exists() else []
        failed = list((drop / "failed").glob("*")) if (drop / "failed").exists() else []

        self._send_json(200, {
            "drop_root": str(drop),
            "pending_jobs": len(pending),
            "claimed_jobs": len(claimed),
            "done_jobs": len(done),
            "failed_jobs": len(failed),
            "pending_files": [p.name for p in sorted(pending)[-10:]],
            "failed_files": [f.name for f in sorted(failed)[-10:]]
        })

    def _handle_enqueue(self, data):
        """Create a .job file in the drop queue."""
        kind = data.get("kind", "all")
        scan_roots = data.get("scan_roots")

        # Validate kind
        if kind not in ("audio", "dashcam", "bodycam", "all"):
            self._send_json(400, {"error": f"invalid kind: {kind}. Must be audio|dashcam|bodycam|all"})
            return

        # Create job file via the existing create_job.sh
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{ts}_{kind}.job"
        job_path = Path(DROP_ROOT) / job_name

        # Build job script
        cmd = ""
        if kind == "audio":
            roots = scan_roots or os.environ.get("AUDIO_ROOT", "/nas/S/audio")
            cmd = f'cd /app && SCAN_ROOTS="{roots}" /usr/bin/env bash run_ingest_all.sh'
        elif kind == "dashcam":
            roots = scan_roots or os.environ.get("DASHCAM_ROOT", "/nas/fileserver/dashcam")
            cmd = f'cd /app && SCAN_ROOTS="{roots}" DASHCAM_ROOT="{roots}" /usr/bin/env bash run_ingest_all.sh'
        elif kind == "bodycam":
            roots = scan_roots or os.environ.get("BODYCAM_ROOT", "/nas/fileserver/bodycam")
            cmd = f'cd /app && SCAN_ROOTS="{roots}" /usr/bin/env bash run_ingest_all.sh'
        else:
            cmd = "cd /app && /usr/bin/env bash run_ingest_all.sh"

        job_content = f"#!/usr/bin/env bash\nset -euo pipefail\n{cmd}\n"

        # Ensure drop dirs exist
        drop = Path(DROP_ROOT)
        drop.mkdir(parents=True, exist_ok=True)
        (drop / "claimed").mkdir(exist_ok=True)
        (drop / "done").mkdir(exist_ok=True)
        (drop / "failed").mkdir(exist_ok=True)

        job_path.write_text(job_content)
        job_path.chmod(0o755)

        self._send_json(201, {
            "status": "queued",
            "job_file": str(job_path),
            "kind": kind,
            "message": "Job created. Worker will pick it up automatically."
        })

    def _handle_run(self, data):
        """Trigger an immediate one-shot run."""
        run_type = data.get("type", "ingest")

        if run_type == "ingest":
            # Run ingest directly (not via job queue)
            result = subprocess.run(
                ["bash", "-c", f"cd {APP_DIR} && SCAN_ROOTS='{os.environ.get('SCAN_ROOTS', '')}' /app/run_ingest_all.sh"],
                capture_output=True, text=True, timeout=3600
            )
            self._send_json(200, {
                "status": "completed" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "return_code": result.returncode
            })
        elif run_type == "sync":
            result = subprocess.run(
                ["bash", "-c", f"cd {APP_DIR} && /app/deploy/sync_from_legacy_drop.sh"],
                capture_output=True, text=True, timeout=600
            )
            self._send_json(200, {
                "status": "completed" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "return_code": result.returncode
            })
        else:
            self._send_json(400, {"error": f"unknown run type: {run_type}"})


def main():
    host = os.environ.get("JOB_API_HOST", "0.0.0.0")
    port = int(os.environ.get("JOB_API_PORT", "8765"))
    server = http.server.HTTPServer((host, port), JobTriggerHandler)
    print(f"Job Trigger API listening on {host}:{port}")
    print(f"Drop root: {DROP_ROOT}")
    print("Endpoints:")
    print("  GET  /api/status  - Queue status")
    print("  GET  /api/health  - Health check")
    print("  POST /api/enqueue - Enqueue a job (body: {kind, scan_roots})")
    print("  POST /api/run     - One-shot run (body: {type: ingest|sync})")
    server.serve_forever()


if __name__ == "__main__":
    main()
