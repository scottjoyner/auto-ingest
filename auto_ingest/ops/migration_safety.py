"""Shared safety primitives for production migrations and backfills.

Keep this module dependency-free so destructive scripts can import it before
opening network/database connections. Invalid parameters should fail closed.
"""
from __future__ import annotations


class SafetyViolation(ValueError):
    """Raised when an operation violates an explicit production safety bound."""


def validate_batch_size(batch_size: int, *, max_batch_size: int) -> int:
    """Validate a bounded transaction size before work starts."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise SafetyViolation("batch size must be an integer")
    if batch_size < 1:
        raise SafetyViolation("batch size must be at least 1")
    if max_batch_size < 1:
        raise SafetyViolation("max batch size must be at least 1")
    if batch_size > max_batch_size:
        raise SafetyViolation(
            f"batch size {batch_size:,} exceeds safety cap {max_batch_size:,}"
        )
    return batch_size


def batches_required(total_items: int, batch_size: int) -> int:
    """Return the number of bounded batches required, validating both inputs."""
    if isinstance(total_items, bool) or not isinstance(total_items, int):
        raise SafetyViolation("total_items must be an integer")
    if total_items < 0:
        raise SafetyViolation("total_items cannot be negative")
    if batch_size < 1:
        raise SafetyViolation("batch_size must be positive")
    return (total_items + batch_size - 1) // batch_size


def preflight_summary(
    *,
    operation: str,
    total_candidates: int,
    eligible_candidates: int,
    batch_size: int,
    max_batch_size: int,
    dry_run: bool,
) -> dict:
    """Build a deterministic preflight record suitable for logs or JSON output."""
    if not operation.strip():
        raise SafetyViolation("operation name must be non-empty")
    validate_batch_size(batch_size, max_batch_size=max_batch_size)
    if total_candidates < 0 or eligible_candidates < 0:
        raise SafetyViolation("candidate counts cannot be negative")
    if eligible_candidates > total_candidates:
        raise SafetyViolation("eligible candidates cannot exceed total candidates")
    return {
        "operation": operation,
        "dry_run": bool(dry_run),
        "total_candidates": total_candidates,
        "eligible_candidates": eligible_candidates,
        "ineligible_candidates": total_candidates - eligible_candidates,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "batches_required": batches_required(eligible_candidates, batch_size),
    }
