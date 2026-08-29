"""Resolve %libber_name:key% notation in strings.

This module is intentionally dependency-free (stdlib only) so it can be
imported by both extension.py and tests without any ComfyUI context.

Registry shape: {libber_name: {key: value_string}}
The caller (extension.py) builds this from LibberStateManager instances.
"""

import re
import random


_TOKEN_RE = re.compile(r"%([^:%\s]+):([^%]+)%")


def resolve_libber_refs(text: str, registry: dict) -> str:
    """Resolve ``%name:key%`` and ``%name:*%`` tokens in *text*.

    ``%name:key%``  → the string value stored under ``key`` in libber ``name``
    ``%name:*%``    → value for a randomly-chosen key from libber ``name``

    Tokens whose libber or key cannot be found are left unchanged so they
    remain visible in the output rather than silently disappearing.
    """
    if not text or "%" not in text:
        return text

    def _sub(m: re.Match) -> str:
        lib_name = m.group(1)
        key = m.group(2).strip()
        lib: dict | None = registry.get(lib_name)
        if not lib:
            return m.group(0)
        if key == "*":
            keys = list(lib.keys())
            if not keys:
                return m.group(0)
            key = random.choice(keys)
        value = lib.get(key)
        if value is None:
            return m.group(0)
        return str(value)

    return _TOKEN_RE.sub(_sub, text)


def extract_libber_names(text: str) -> list[str]:
    """Return the unique libber names referenced in *text* (order preserved)."""
    seen: set[str] = set()
    names: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def apply_slot_dialogue(
    slot_dialogue: dict,
    registry: dict,
    shot_id: str,
) -> tuple:
    """Resolve dialogue tokens and split into scene_instance fields and shot patches.

    ``slot_dialogue``  – {slot_key: raw_dialogue_text}
    ``registry``       – {libber_name: {key: value}} passed to resolve_libber_refs
    ``shot_id``        – id of the single shot (used as the key in dialogue_map)

    Returns ``(dialogue_map, shot_patch)`` where:
    - ``dialogue_map`` is ``{shot_id: resolved_text}`` for normal-speech entries
      (last non-silent, non-sounds entry wins — one speaker per shot)
    - ``shot_patch`` is a dict with any/all of:
        ``"dialogue"``    → ``{"speaker_slot": slot}``  (for the winning speech entry)
        ``"sound_events"``→ semicolon-joined sound descriptions from ``[sounds]`` entries
    """
    dialogue_map: dict = {}
    shot_patch: dict = {}

    for slot, raw in slot_dialogue.items():
        resolved = resolve_libber_refs(raw, registry)
        if not resolved:
            continue
        if resolved.startswith("[silent]"):
            continue
        if resolved.startswith("[sounds]"):
            desc = resolved[len("[sounds]"):].strip()
            if desc:
                existing = shot_patch.get("sound_events", "") or ""
                shot_patch["sound_events"] = (existing + "; " + desc).lstrip("; ")
        else:
            dialogue_map[shot_id] = resolved
            shot_patch["dialogue"] = {"speaker_slot": slot}

    return dialogue_map, shot_patch
