"""Resolved-configuration snapshots and append-only workflow records.

The record layer improves reproducibility without changing any algorithm.  Each
heavy command stores the exact configuration seen by the workflow and appends a
small JSON record under the current output ``logs`` directory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_resolved_config(cfg: Mapping[str, Any], logs_dir: str | Path, stage: str) -> Path:
    """Save the exact resolved configuration used by one workflow invocation."""
    try:
        import yaml
    except ImportError as exc:  # PyYAML is a formal project dependency.
        raise RuntimeError("PyYAML is required to save resolved configuration snapshots.") from exc
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"resolved_config_{stage}_{_timestamp()}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, allow_unicode=True, sort_keys=False)
    return path


def append_run_record(
    logs_dir: str | Path,
    *,
    stage: str,
    status: str,
    command: str,
    resolved_config: Optional[str | Path] = None,
    primary_artifact: Optional[str | Path] = None,
    message: str = "",
) -> Path:
    """Append one machine-readable workflow record to ``workflow_runs.jsonl``."""
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "workflow_runs.jsonl"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        "command": command,
        "resolved_config": str(resolved_config) if resolved_config else None,
        "primary_artifact": str(primary_artifact) if primary_artifact else None,
        "message": message,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path
