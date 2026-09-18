"""Pure logic for the node-output auto-tracking feature.

Extracted so it can be tested without mocking ComfyUI internals. The
monkeypatching glue that uses these functions against live node classes,
the PromptServer on_prompt hook, and the runtime value store all live in
extension.py — this module only knows about plain dicts and strings.
"""
from __future__ import annotations

import re

TRACK_RE = re.compile(r"\[track:\s*([^\]]*)\]")

# Caps a single captured value's length. Without this, tracking a node whose
# input is a large object (e.g. a whole loaded dict/profile, not a simple
# string) turns str(value) into an unreadable multi-KB blob in Run History —
# and since these values ship over the wire on every /run_tracker/runs poll,
# capping at the source also keeps that payload small.
MAX_CAPTURE_VALUE_LEN = 300


def get_track_label(title: str | None) -> str | None:
    """Extract the label from a `[track: Label]` tag in a node title, or None."""
    if not title:
        return None
    m = TRACK_RE.search(title)
    return m.group(1).strip() if m else None


def extract_tracked_nodes(prompt: dict) -> dict[str, str]:
    """Scan an API-format prompt dict for `[track: Label]`-tagged nodes.

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
