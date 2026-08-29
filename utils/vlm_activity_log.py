"""
Rolling VLM activity log.

Stored as a JSON array at <data_dir>/vlm_activity.json, capped at _MAX_ENTRIES.
Each entry: {ts, backend, model_id, operation, profile_id}.

Used for:
  - Model history in the Modal backend tab (model IDs used per backend)
  - Idle-timeout tracking (last activity timestamp)
  - Lightweight per-session analytics
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 500


def _log_path(data_dir: str) -> Path:
    return Path(data_dir) / "vlm_activity.json"


def record(
    data_dir: str,
    backend: str,
    model_id: str,
    operation: str,
    profile_id: str = "",
) -> None:
    """Append one activity entry.  Best-effort — errors are logged, not raised."""
    entry = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "backend":    backend,
        "model_id":   model_id,
        "operation":  operation,
        "profile_id": profile_id,
    }
    try:
        path = _log_path(data_dir)
        entries: list[dict] = []
        if path.exists():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(entries, list):
                    entries = []
            except Exception:
                entries = []
        entries.append(entry)
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("vlm_activity_log.record failed: %s", exc)


def recent(data_dir: str, n: int = 100) -> list[dict]:
    """Return the most recent n entries, newest first."""
    try:
        path = _log_path(data_dir)
        if not path.exists():
            return []
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            return []
        return list(reversed(entries[-n:]))
    except Exception:
        return []


def model_history(data_dir: str, backend: str) -> list[str]:
    """Return deduplicated model IDs for the given backend, most recent first."""
    seen: set[str] = set()
    result: list[str] = []
    for entry in recent(data_dir, n=_MAX_ENTRIES):
        if entry.get("backend") == backend:
            mid = entry.get("model_id", "")
            if mid and mid not in seen:
                seen.add(mid)
                result.append(mid)
    return result


def last_activity_ts(data_dir: str, backend: str | None = None) -> str | None:
    """Return ISO timestamp of the most recent entry, optionally filtered by backend."""
    for entry in recent(data_dir, n=1 if backend is None else _MAX_ENTRIES):
        if backend is None or entry.get("backend") == backend:
            return entry.get("ts")
    return None
