"""Readable Run History rows for the Prompt Composition Loader.

Pure stdlib (no ComfyUI, no sibling utils imports). Turns the raw scene_cast
dict — which the auto-tracker would otherwise stringify into an unreadable
300-char blob — into one short row per fact: subject/bundle, reference images,
reference video, audio, and a compact LoRA list in the same style as
LoraStackBuilder's "Enabled Summary". The assembled prompt and the H3 ref
plan are deliberately not included.
"""
from __future__ import annotations

import os

MAX_ROW_LEN = 600
_DIALOGUE_LEN = 120


def _num(v) -> str:
    try:
        return f"{float(v):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(v)


def _cap(s: str, limit: int = MAX_ROW_LEN) -> str:
    return s if len(s) <= limit else s[:limit] + "…"


def _timing(start, duration) -> str:
    parts = []
    if start:
        parts.append(f"start {_num(start)}s")
    if duration:
        parts.append(f"{_num(duration)}s")
    return f" ({', '.join(parts)})" if parts else ""


def _file_name(item) -> str:
    if isinstance(item, dict):
        item = item.get("file", "")
    return str(item or "")


def _selected_images(files: list, selection) -> list[str]:
    """Same rules as _resolve_cast_media: None = all, list of indices, or a legacy single int."""
    if selection is None:
        return [_file_name(f) for f in files]
    if isinstance(selection, list):
        return [_file_name(files[i]) for i in selection if isinstance(i, int) and 0 <= i < len(files)]
    try:
        idx = int(selection)
    except (TypeError, ValueError):
        return [_file_name(f) for f in files]
    return [_file_name(files[idx])] if 0 <= idx < len(files) else []


def _audio_row(audio: dict, visual: dict) -> str:
    source = audio.get("source", "none")
    if source == "file":
        name = audio.get("file", "")
        return f"{name}{_timing(audio.get('start_time'), audio.get('duration'))}" if name else "(no file set)"
    if source == "extract_from_video":
        name = audio.get("video_file", "")
        return f"from {name}{_timing(audio.get('start_time'), audio.get('duration'))}" if name else "(no video set)"
    if source == "extract_from_visual":
        name = visual.get("file", "")
        return f"from reference video{f' {name}' if name else ''}"
    return "(none configured)"


def summarize_scene_cast(scene_cast, bundle_lookup) -> dict[str, str]:
    """Rows describing each cast entry. bundle_lookup(bundle_id) -> bundle dict | None."""
    rows: dict[str, str] = {}
    entries = scene_cast.get("entries", []) if isinstance(scene_cast, dict) else []
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            continue
        subject = entry.get("subject_id") or "?"
        bundle_id = entry.get("bundle_id") or ""
        mode = entry.get("visual_mode") or "images"
        use_audio = bool(entry.get("use_audio"))
        head = f"{bundle_id or '(no bundle)'} · {mode}" + (" · +audio" if use_audio else "")
        if entry.get("retention"):
            head += f" · {entry['retention']}"
        rows[f"Cast {i}: {subject}"] = _cap(head)

        if entry.get("source_profile_id") or entry.get("source_subject_id"):
            rows[f"Cast {i} source"] = _cap(f"{entry.get('source_profile_id', '')}: {entry.get('source_subject_id', '')}")

        bundle = bundle_lookup(bundle_id) if bundle_id else None
        if bundle_id and bundle is None:
            rows[f"Cast {i} bundle"] = f"(bundle '{bundle_id}' not found)"
        elif bundle is not None:
            visual = bundle.get("visual") or {}
            audio = bundle.get("audio") or {}
            if mode != "video":
                images = [n for n in _selected_images(list(visual.get("files") or []), entry.get("image_selection")) if n]
                if images:
                    rows[f"Cast {i} images"] = _cap(", ".join(images))
            if mode in ("video", "both") and visual.get("file"):
                rows[f"Cast {i} video"] = _cap(f"{visual['file']}{_timing(visual.get('start_time'), visual.get('duration'))}")
            if use_audio:
                rows[f"Cast {i} audio"] = _cap(_audio_row(audio, visual))

        dialogue = str(entry.get("dialogue") or "").strip()
        if dialogue:
            rows[f"Cast {i} dialogue"] = dialogue if len(dialogue) <= _DIALOGUE_LEN else dialogue[:_DIALOGUE_LEN] + "…"
    return rows


def summarize_loras(loras) -> str:
    """One 'name weight (target)' line per enabled LoRA; '' when there are none."""
    lines = []
    for e in loras or []:
        if not isinstance(e, dict) or e.get("enabled", True) is False:
            continue
        raw = e.get("name") or e.get("lora") or ""
        if not raw:
            continue
        name = os.path.splitext(os.path.basename(str(raw)))[0][:48]
        weight = e.get("weight", e.get("strength_model", 1.0))
        line = f"{name} {_num(weight)}"
        if e.get("target"):
            line += f" ({e['target']})"
        lines.append(line)
    return "\n".join(lines)
