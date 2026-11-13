"""Synthetic crash-lock acceptance probe for draft PR #26.

Runs no ingestion, model, scheduler or graph calls. The inherited-fd variant
is constructed only in memory; it is not a production patch. A child that
closes the inherited descriptor demonstrates the remaining custody limit.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def probe(source, inherit_lock, close_child_fds=False):
    with tempfile.TemporaryDirectory(prefix="hermes-lock-probe-") as tmp:
        root = Path(tmp)
        checkout = root / "checkout"
        scripts = checkout / "scripts"
        scripts.mkdir(parents=True)
        marker = root / "child.pid"
        child = scripts / "arxiv_kg_bridge.py"
        child.write_text(
            "import os,time\nfrom pathlib import Path\n"
            + ("os.closerange(3,1024)\n" if close_child_fds else "")
            + f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            + "time.sleep(10)\n"
        )
        (scripts / "embedding_client.py").write_text("# synthetic placeholder\n")
        registry = checkout / "embedding_endpoints.json"
        registry.write_text(json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "embedding_dimensions": 768,
        }))
        if inherit_lock:
            assert source.count("close_fds=True,") == 1
            source = source.replace(
                "close_fds=True,", "close_fds=True,\n                pass_fds=(lock.fileno(),),"
            )
        (root / "supervisor.py").write_text(source)
        kwargs = {
            "checkout": checkout,
            "python_executable": Path(sys.executable).resolve(),
            "reviewed_source_sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
            "reviewed_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "lock_file": root / "job.lock",
            "timeout_seconds": 5,
        }
        expressions = ", ".join(
            f"{k}=Path({str(v)!r})" if isinstance(v, Path) else f"{k}={v!r}"
            for k, v in kwargs.items()
        )
        runner = root / "runner.py"
        runner.write_text(
            "from pathlib import Path\nimport supervisor\n"
            + f"supervisor.supervise({expressions})\n"
        )
        parent = subprocess.Popen(
            [sys.executable, str(runner)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        child_pid = None
        try:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                if parent.poll() is not None:
                    raise RuntimeError("supervisor exited before child marker")
                time.sleep(0.02)
            if not marker.exists():
                raise RuntimeError("synthetic launch timeout")
            child_pid = int(marker.read_text())
            parent.kill()
            parent.wait(timeout=2)
            child_state = Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            child_alive = child_state != "Z"
            with (root / "job.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    successor_admitted = True
                except BlockingIOError:
                    successor_admitted = False
            return {
                "variant": "inherited_fd" if inherit_lock else "current_PR26",
                "child_closes_inherited_fds": close_child_fds,
                "child_alive_after_supervisor_kill": child_alive,
                "successor_can_acquire_lock": successor_admitted,
                "overlap_gate": "FAIL" if child_alive and successor_admitted else "PASS",
            }
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2)
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    stat = Path(f"/proc/{child_pid}/stat")
                    if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                        break
                    time.sleep(0.02)
                else:
                    raise RuntimeError("synthetic cleanup unconfirmed")


def run(source):
    results = [
        probe(source, False),
        probe(source, True),
        probe(source, True, True),
    ]
    assert [r["overlap_gate"] for r in results] == ["FAIL", "PASS", "FAIL"]
    assert all(r["child_alive_after_supervisor_kill"] for r in results)
    return {
        "schema": "hermes-crash-lock-probe-v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "results": results,
        "scope": "synthetic_single_host_only",
        "production_changed": False,
        "can_deploy": False,
        "conclusion": "Inherited lock fd alone is insufficient for arbitrary child custody.",
    }


if __name__ == "__main__":
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "hermes_arxiv_supervisor_experiment.py").read_text()
    print(json.dumps(run(source), sort_keys=True))
