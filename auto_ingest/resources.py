"""Host resource snapshots and admission policy for background ingest work."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    load1: float
    memory_available_mb: int
    disk_free_gb: float

    @property
    def load_per_cpu(self) -> float:
        return self.load1 / max(self.cpu_count, 1)


@dataclass(frozen=True)
class ResourcePolicy:
    max_load_per_cpu: float = 0.60
    min_memory_available_mb: int = 2048
    min_disk_free_gb: float = 20.0


def _memory_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def snapshot(path: str | Path = ".") -> ResourceSnapshot:
    cpu = os.cpu_count() or 1
    try:
        load1 = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load1 = 0.0
    usage = shutil.disk_usage(Path(path))
    return ResourceSnapshot(
        cpu_count=cpu,
        load1=load1,
        memory_available_mb=_memory_available_mb(),
        disk_free_gb=usage.free / (1024 ** 3),
    )


def admission(snapshot_: ResourceSnapshot, policy: ResourcePolicy) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if snapshot_.load_per_cpu >= policy.max_load_per_cpu:
        reasons.append(
            f"load_per_cpu={snapshot_.load_per_cpu:.2f} >= {policy.max_load_per_cpu:.2f}"
        )
    if snapshot_.memory_available_mb < policy.min_memory_available_mb:
        reasons.append(
            f"memory_available_mb={snapshot_.memory_available_mb} < {policy.min_memory_available_mb}"
        )
    if snapshot_.disk_free_gb < policy.min_disk_free_gb:
        reasons.append(
            f"disk_free_gb={snapshot_.disk_free_gb:.1f} < {policy.min_disk_free_gb:.1f}"
        )
    return not reasons, reasons
