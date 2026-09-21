"""Pure logic for the node-output auto-tracking feature.

Extracted so it can be tested without mocking ComfyUI internals. The
monkeypatching glue that uses these functions against live node classes,
the PromptServer on_prompt hook, and the runtime value store all live in
extension.py — this module only knows about plain dicts and strings.
"""
from __future__ import annotations

import re

# A tracked node's title is "<marker> <label>", e.g. "🐾 Video Shift". The legacy
# "[track: Label]" form is still recognised so old workflows and Run History entries
# keep working; the frontend rewrites legacy titles to the marker form on load.
# Keep TRACK_MARKER in sync with js/utils/run_tracker.js and js/ui/run_history.js.
TRACK_MARKER = "\U0001F43E"  # 🐾
TRACK_RE = re.compile(r"\[track:\s*([^\]]*)\]")

# Caps a single captured value's length. Without this, tracking a node whose
# input is a large object (e.g. a whole loaded dict/profile, not a simple
# string) turns str(value) into an unreadable multi-KB blob in Run History —
# and since these values ship over the wire on every /run_tracker/runs poll,
# capping at the source also keeps that payload small.
MAX_CAPTURE_VALUE_LEN = 300


def get_track_label(title: str | None) -> str | None:
    """Extract a node's track label from its title, or None if it isn't tracked.

    Marker form: the title starts with TRACK_MARKER and the label is the rest,
    trimmed ("🐾 Video Shift" -> "Video Shift"). Legacy form: `[track: Label]`
    anywhere in the title.
    """
    if not title:
        return None
    stripped = title.strip()
    if stripped.startswith(TRACK_MARKER):
        label = stripped[len(TRACK_MARKER):].strip()
        return label or None
    m = TRACK_RE.search(title)
    return m.group(1).strip() if m else None


def extract_tracked_nodes(prompt: dict) -> dict[str, str]:
    """Scan an API-format prompt dict for tracked nodes (marker or legacy `[track: Label]` titles).

    Returns {node_id: label} for every node whose `_meta.title` carries the tag.
    """
    tracked: dict[str, str] = {}
    if not isinstance(prompt, dict):
        return tracked
    for node_id, node_def in prompt.items():
        if not isinstance(node_def, dict):
            continue
        title = (node_def.get("_meta") or {}).get("title", "")
        label = get_track_label(title)
        if label:
            tracked[node_id] = label
    return tracked


def stringify_capture_values(values: dict) -> dict[str, str]:
    """Stringify non-blank values only — mirrors RunMetaCapture's own filtering.

    Values longer than MAX_CAPTURE_VALUE_LEN are truncated with a suffix
    noting the real length, so a non-scalar input (a whole dict/object, not a
    simple string) can't turn into an unreadable blob in Run History.
    """
    out: dict[str, str] = {}
    if not values:
        return out
    for key, v in values.items():
        if v is None:
            continue
        try:
            s = str(v)
        except Exception:
            continue
        if not s.strip():
            continue
        if len(s) > MAX_CAPTURE_VALUE_LEN:
            s = s[:MAX_CAPTURE_VALUE_LEN] + f"… ({len(s)} chars total)"
        out[key] = s
    return out
