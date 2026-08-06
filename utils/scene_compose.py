"""Pure logic for the Scene Composition system.

No ComfyUI dependencies and no cross-imports from other utils modules so
this can be loaded standalone via import_test_module() in CI.

All inputs are plain Python dicts/lists.  Extension.py converts
SceneTemplate objects to dicts before calling these functions.
"""
from __future__ import annotations

import copy


# ── Internal helpers ───────────────────────────────────────────────────────────

def _placeholder_shots(shots: list) -> list:
    """Return shots that have a placeholder dialogue entry, in order."""
    return [
        s for s in shots
        if isinstance(s.get("dialogue"), dict) and s["dialogue"].get("placeholder")
    ]


# ── Core compose ───────────────────────────────────────────────────────────────

def compose_scene(
    template: dict,
    slot_assignments: dict[str, dict | None],
    dialogue: list[str],
    outfit_overrides: dict[str, str],
) -> dict:
    """Build a SCENE_INSTANCE dict from a template dict + subject assignments.

    Args:
        template:         template.to_dict() from a SceneTemplate object.
        slot_assignments: {slot_id: subject_dict | None}.  None means unassigned.
        dialogue:         positional list — dialogue[0] fills the first placeholder
                          shot in shot order, dialogue[1] fills the second, etc.
        outfit_overrides: {slot_id: outfit_string}.  Empty strings are ignored.

    Returns a SCENE_INSTANCE dict.
    """
    shots = template.get("shots", [])
    placeholders = _placeholder_shots(shots)

    # Map positional dialogue strings to placeholder shot IDs
    dialogue_map: dict[str, str] = {}
    for i, shot in enumerate(placeholders):
        text = dialogue[i].strip() if i < len(dialogue) else ""
        if text:
            dialogue_map[shot["id"]] = text

    # Filter None assignments and make a deep copy
    assigned = {k: copy.deepcopy(v) for k, v in slot_assignments.items() if v is not None}

    # Filter blank outfit overrides
    overrides = {k: v.strip() for k, v in outfit_overrides.items() if v and v.strip()}

    return {
        "template_id": template.get("id", ""),
        "template_name": template.get("name", ""),
        "template": copy.deepcopy(template),
        "slot_assignments": assigned,
        "dialogue": dialogue_map,
        "outfit_overrides": overrides,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_scene(template: dict, slot_assignments: dict[str, dict | None]) -> list[str]:
    """Return a list of warning strings.  Empty list means validation passed."""
    warnings: list[str] = []
    slots = template.get("slots", {})

    for slot_id in sorted(slots):
        spec = slots[slot_id]
        subject = slot_assignments.get(slot_id)
        if subject is None:
            role = spec.get("role", "")
            role_str = f" ({role})" if role else ""
            warnings.append(f"Slot {slot_id}{role_str} is not assigned")
            continue
        if spec.get("needs_voice") and not subject.get("voice", {}).get("audio_reference_file"):
            warnings.append(f"Slot {slot_id}: voice reference recommended but not set in profile")
        if spec.get("needs_character_sheet") and not subject.get("character_sheet_images"):
            warnings.append(f"Slot {slot_id}: character sheet images recommended but not set in profile")

    return warnings


# ── Summary formatting ─────────────────────────────────────────────────────────

def format_scene_summary(
    scene_instance: dict,
    validation_warnings: list[str] | None = None,
) -> str:
    """Return a human-readable summary of a composed scene."""
    lines = [f"Scene: {scene_instance.get('template_name', 'Unknown')}"]

    assignments = scene_instance.get("slot_assignments", {})
    overrides = scene_instance.get("outfit_overrides", {})
    dialogue = scene_instance.get("dialogue", {})
    template_data = scene_instance.get("template", {})

    # Slots
    if assignments:
        lines.append("Slots:")
        for slot_id in sorted(assignments):
            subject = assignments[slot_id]
            name = subject.get("name", "(unnamed)")
            concept = subject.get("concept_id", "")
            concept_str = f" ({concept})" if concept else ""
            sheets = subject.get("character_sheet_images", [])
            audio_file = subject.get("voice", {}).get("audio_reference_file", "")
            media: list[str] = []
            if sheets:
                n = len(sheets)
                media.append(f"{n} sheet{'s' if n != 1 else ''}")
            if audio_file:
                media.append("voice")
            media_str = f" [{', '.join(media)}]" if media else ""
            lines.append(f"  {slot_id} → {name}{concept_str}{media_str}")
    else:
        lines.append("Slots: (none assigned)")

    # Dialogue
    if dialogue:
        n = len(dialogue)
        lines.append(f"Dialogue ({n} placeholder{'s' if n != 1 else ''}):")
        shots_by_id = {s["id"]: s for s in template_data.get("shots", [])}
        for shot_id, text in dialogue.items():
            shot = shots_by_id.get(shot_id, {})
            dlg = shot.get("dialogue") or {}
            speaker = dlg.get("speaker_slot", "?")
            preview = (text[:70] + "…") if len(text) > 70 else text
            lines.append(f"  {shot_id} ({speaker}): \"{preview}\"")

    # Outfit overrides
    if overrides:
        lines.append("Outfit overrides:")
        for slot_id, outfit in sorted(overrides.items()):
            subject = assignments.get(slot_id, {})
            default = subject.get("appearance", {}).get("default_outfit", "")
            from_str = f" (overrides: {default})" if default else ""
            lines.append(f"  {slot_id}: {outfit}{from_str}")

    # Validation
    warnings = validation_warnings or []
    if warnings:
        lines.append("Validation warnings:")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("Validation: OK")

    return "\n".join(lines)
