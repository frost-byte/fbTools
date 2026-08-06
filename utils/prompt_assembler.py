"""Pure prompt assembly logic for the Scene Composition Engine.

All functions accept plain Python dicts (no ComfyUI or torch dependencies).
Extension.py handles actual tensor/audio loading and passes dicts here.

Each model type has its own assembly function:
  h3_ref2va  — MiniMax H3 structured 6-section brief with reference labels
  h3_fl2va   — MiniMax H3 free-language with shot structure, no reference labels
  wan22      — Wan 2.2 production-direction block
  bernini    — BerniniR production-direction block
  ltx23      — LTX 2.3 simple descriptive
  flux2      — Flux 2 simple descriptive
  krea2      — Krea 2 simple descriptive
  qwen       — Qwen Image simple descriptive
"""
from __future__ import annotations

import re

MODEL_TYPES = [
    "h3_ref2va",
    "h3_fl2va",
    "wan22",
    "bernini",
    "ltx23",
    "flux2",
    "krea2",
    "qwen",
]

_PRODUCTION_MODELS = {"wan22", "bernini"}
_SIMPLE_MODELS = {"ltx23", "flux2", "krea2", "qwen"}


# ── Reference map ──────────────────────────────────────────────────────────────

def _build_ref_map(scene_instance: dict) -> dict[str, dict]:
    """Assign reference numbers (Subject N, Picture N, Audio N) per slot.

    Numbering convention (independent per type, per spec):
      Subject 1, 2, … in slot order.
      Picture 1, 2, 3, … in slot order (continuous across all subjects).
      Audio 1, 2, … in slot order for slots that have audio files.

    Returns a dict keyed by slot_id with typing info for each assigned slot.
    Slots not present in slot_assignments are absent from the result.
    """
    assignments = scene_instance.get("slot_assignments", {})
    outfit_overrides = scene_instance.get("outfit_overrides", {})
    ordered_slots = sorted(assignments.keys())

    ref_map: dict[str, dict] = {}
    subject_counter = 1
    picture_counter = 1
    audio_counter = 1

    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        appearance = subject.get("appearance", {})
        voice = subject.get("voice", {})
        sheets = subject.get("character_sheet_images", [])
        audio_file = voice.get("audio_reference_file", "")
        outfit = outfit_overrides.get(slot_id) or appearance.get("default_outfit", "")

        picture_nums = list(range(picture_counter, picture_counter + len(sheets)))
        picture_counter += len(sheets)

        audio_num: int | None = None
        if audio_file:
            audio_num = audio_counter
            audio_counter += 1

        ref_map[slot_id] = {
            "subject_id": subject.get("subject_id", ""),
            "name": subject.get("name", slot_id),
            "appearance_summary": appearance.get("summary", ""),
            "face": appearance.get("face", ""),
            "hair": appearance.get("hair", ""),
            "body": appearance.get("body", ""),
            "default_outfit": appearance.get("default_outfit", ""),
            "outfit": outfit,
            "voice_description": voice.get("description", ""),
            "audio_file": audio_file,
            "language": voice.get("language", "en-us") or "en-us",
            "character_sheet_images": list(sheets),
            "concept_id": subject.get("concept_id", ""),
            "subject_num": subject_counter,
            "speaker_id": f"S{subject_counter}",
            "subject_label": f"<Subject {subject_counter}>",
            "picture_nums": picture_nums,
            "audio_num": audio_num,
        }
        subject_counter += 1

    return ref_map


# ── Placeholder replacement ────────────────────────────────────────────────────

def _replace_h3(text: str, ref_map: dict, seen: set) -> str:
    """Replace {A}/{B}/{C}/{D} with H3 subject labels, tracking first appearance.

    First appearance: "<Subject 1> (Name — appearance_summary)"
    Subsequent:       "<Subject 1> (Name)"
    """
    def _sub(match: re.Match) -> str:
        slot_id = match.group(1)
        info = ref_map.get(slot_id)
        if not info:
            return match.group(0)
        label = info["subject_label"]
        name = info["name"]
        if slot_id not in seen:
            seen.add(slot_id)
            appearance = info.get("appearance_summary", "")
            if appearance:
                return f"{label} ({name} — {appearance})"
            return f"{label} ({name})"
        return f"{label} ({name})"

    return re.sub(r"\{([A-D])\}", _sub, text)


def _replace_named(text: str, ref_map: dict) -> str:
    """Replace {A}/{B}/{C}/{D} with subject names (for non-H3 formats)."""
    def _sub(match: re.Match) -> str:
        slot_id = match.group(1)
        info = ref_map.get(slot_id)
        return info["name"] if info else match.group(0)

    return re.sub(r"\{([A-D])\}", _sub, text)


# ── H3 Ref2VA ─────────────────────────────────────────────────────────────────

def _assemble_h3_ref2va(scene_instance: dict, ref_map: dict) -> str:
    template = scene_instance.get("template", {})
    dialogue_map = scene_instance.get("dialogue", {})
    ordered_slots = sorted(ref_map.keys())
    sections: list[str] = []

    # ── subject_definitions ────────────────────────────────────────────────────
    sd: list[str] = []

    # All subjects first
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        name, label = info["name"], info["subject_label"]
        summary = info["appearance_summary"]
        sd.append(f"{label} [{name}]: {summary}")
        if info["face"]:
            sd.append(f"  Face: {info['face']}")
        if info["hair"]:
            sd.append(f"  Hair: {info['hair']}")
        if info["body"]:
            sd.append(f"  Body: {info['body']}")
        if info["outfit"]:
            sd.append(f"  Outfit: {info['outfit']}")

    # All pictures
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if not info["picture_nums"]:
            continue
        name = info["name"]
        for i, pic_num in enumerate(info["picture_nums"]):
            note = "primary identity reference" if i == 0 else "additional reference"
            sd.append(
                f"<Picture {pic_num}>: character sheet for {name} — {note}. "
                "Preserve: face, hair, build."
            )

    # All audio
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["audio_num"] is None:
            continue
        name = info["name"]
        voice = info["voice_description"] or f"{name}'s voice"
        sd.append(
            f"<Audio {info['audio_num']}>: voice timbre reference for {name} "
            f"({info['speaker_id']}) — {voice}. Language: {info['language']}."
        )

    sections.append("subject_definitions:\n" + "\n".join(sd))

    # ── summary ────────────────────────────────────────────────────────────────
    has_sheets = any(info["picture_nums"] for info in ref_map.values())
    has_audio = any(info["audio_num"] is not None for info in ref_map.values())
    task_parts = []
    if has_sheets:
        task_parts.append("reference generation")
    if has_audio:
        task_parts.append("audio reference")
    task_str = " + ".join(task_parts) if task_parts else "video generation"

    pic_refs: list[str] = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if not info["picture_nums"]:
            continue
        nums = info["picture_nums"]
        if len(nums) == 1:
            pic_str = f"<Picture {nums[0]}>"
        else:
            pic_str = f"<Picture {nums[0]}>-<Picture {nums[-1]}>"
        pic_refs.append(f"{info['name']} ({pic_str})")

    audio_refs: list[str] = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["audio_num"] is not None:
            audio_refs.append(f"{info['name']} (<Audio {info['audio_num']}>)")

    template_name = scene_instance.get("template_name", "scene")
    summary_body_parts: list[str] = []
    if pic_refs:
        summary_body_parts.append(
            f"Using character sheet images of {', '.join(pic_refs)} to generate a {template_name}."
        )
    if audio_refs:
        summary_body_parts.append(f"Voice references: {', '.join(audio_refs)}.")

    sections.append(
        f"summary:\nThis is a {task_str} task. " + " ".join(summary_body_parts)
    )

    # ── retention_analysis ─────────────────────────────────────────────────────
    ra: list[str] = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        char_parts = [info["appearance_summary"]] if info["appearance_summary"] else []
        if info["outfit"]:
            char_parts.append(f"outfit: {info['outfit']}")
        char_str = ", ".join(char_parts)
        ra.append(f"{info['subject_label']} ({info['name']}): fully_preserved | {char_str}")

    for slot_id in ordered_slots:
        for pic_num in ref_map[slot_id]["picture_nums"]:
            ra.append(f"<Picture {pic_num}>: reference")

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["audio_num"] is not None:
            ra.append(f"<Audio {info['audio_num']}>: reference (voice timbre only, not fully_copy)")

    sections.append("retention_analysis:\n" + "\n".join(ra))

    # ── detailed_description ───────────────────────────────────────────────────
    seen_slots: set[str] = set()
    dd: list[str] = []

    style = template.get("style", "")
    env_summary = template.get("environment", {}).get("summary", "")
    opening: list[str] = []
    if style:
        opening.append(style[0].upper() + style[1:] if style else style)
    if env_summary:
        opening.append(f"Set in {env_summary}")
    if opening:
        dd.append(". ".join(opening) + ".")

    for i, shot in enumerate(template.get("shots", [])):
        shot_id = shot.get("id", f"shot_{i + 1}")
        timestamp = shot.get("timestamp")
        header = f"[Shot {i + 1}]" + (f" {timestamp}" if timestamp else "")
        dd.append("")
        dd.append(header)

        camera = _replace_h3(shot.get("camera", ""), ref_map, seen_slots)
        action = _replace_h3(shot.get("action", ""), ref_map, seen_slots)
        if camera:
            s = camera.rstrip(".")
            dd.append(s + ".")
        if action:
            dd.append(action)

        dialogue_entry = shot.get("dialogue")
        if isinstance(dialogue_entry, dict):
            speaker_slot = dialogue_entry.get("speaker_slot", "")
            text = dialogue_map.get(shot_id, "")
            if not text and not dialogue_entry.get("placeholder"):
                text = dialogue_entry.get("default_text") or ""
            if text:
                lang = ref_map.get(speaker_slot, {}).get("language", "en-us") or "en-us"
                dd.append(f"<d>[{lang}] {text}</d>")

        sound_events = shot.get("sound_events")
        if sound_events:
            dd.append(f"[{sound_events}]")

    sections.append("detailed_description:\n" + "\n".join(dd))

    # ── overall_soundscape ─────────────────────────────────────────────────────
    soundscape = template.get("overall_soundscape", "")
    if soundscape:
        sections.append(f"overall_soundscape:\n{soundscape}")

    # ── non_diegetic_music ─────────────────────────────────────────────────────
    music = template.get("non_diegetic_music", "")
    if music:
        sections.append(f"non_diegetic_music:\n{music}")

    return "\n\n".join(sections)


# ── H3 FL2VA ──────────────────────────────────────────────────────────────────

def _assemble_h3_fl2va(scene_instance: dict, ref_map: dict) -> str:
    template = scene_instance.get("template", {})
    dialogue_map = scene_instance.get("dialogue", {})
    ordered_slots = sorted(ref_map.keys())
    lines: list[str] = []

    # Opening
    style = template.get("style", "")
    env_summary = template.get("environment", {}).get("summary", "")
    intros = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        summary = info["appearance_summary"]
        intros.append(f"{info['name']} ({summary})" if summary else info["name"])

    opening: list[str] = []
    if style:
        opening.append(style[0].upper() + style[1:] if style else style)
    if env_summary:
        opening.append(f"Set in {env_summary}")
    if intros:
        opening.append(", ".join(intros))
    if opening:
        lines.append(". ".join(opening) + ".")

    # Shots
    for i, shot in enumerate(template.get("shots", [])):
        shot_id = shot.get("id", f"shot_{i + 1}")
        timestamp = shot.get("timestamp")
        header = f"[Shot {i + 1}]" + (f" {timestamp}" if timestamp else "")
        lines.append("")
        lines.append(header)

        camera = _replace_named(shot.get("camera", ""), ref_map)
        action = _replace_named(shot.get("action", ""), ref_map)
        if camera:
            lines.append(camera.rstrip(".") + ".")
        if action:
            lines.append(action)

        dialogue_entry = shot.get("dialogue")
        if isinstance(dialogue_entry, dict):
            speaker_slot = dialogue_entry.get("speaker_slot", "")
            text = dialogue_map.get(shot_id, "")
            if not text and not dialogue_entry.get("placeholder"):
                text = dialogue_entry.get("default_text") or ""
            if text:
                lang = ref_map.get(speaker_slot, {}).get("language", "en-us") or "en-us"
                lines.append(f"<d>[{lang}] {text}</d>")

        sound_events = shot.get("sound_events")
        if sound_events:
            lines.append(f"[{sound_events}]")

    soundscape = template.get("overall_soundscape", "")
    if soundscape and soundscape.upper() != "N/A":
        lines.append(f"\nSoundscape: {soundscape}")

    return "\n".join(lines)


# ── Production direction (Wan 2.2 / BerniniR) ─────────────────────────────────

def _assemble_production(scene_instance: dict, ref_map: dict) -> str:
    template = scene_instance.get("template", {})
    dialogue_map = scene_instance.get("dialogue", {})
    ordered_slots = sorted(ref_map.keys())
    lines: list[str] = []

    has_sheets = any(info["picture_nums"] for info in ref_map.values())
    task = "reference image generation" if has_sheets else "video generation"
    template_name = scene_instance.get("template_name", "scene")
    lines.append(f"[{task}] {template_name}.")

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        summary = info["appearance_summary"]
        outfit = info["outfit"]
        desc_parts = [summary] if summary else []
        if outfit:
            desc_parts.append(f"wearing {outfit}")
        desc = ", ".join(desc_parts) if desc_parts else info["name"]
        lines.append(f"The reference image defines {info['name']}: {desc}.")

    env = template.get("environment", {})
    env_summary = env.get("summary", "")
    env_lighting = env.get("lighting", "")
    if env_summary:
        preserve_parts = [env_summary]
        if env_lighting:
            preserve_parts.append(env_lighting)
        lines.append(f"Preserve: {'. '.join(preserve_parts)}.")

    shot_descs: list[str] = []
    for i, shot in enumerate(template.get("shots", [])):
        shot_id = shot.get("id", f"shot_{i + 1}")
        action = _replace_named(shot.get("action", ""), ref_map)
        text = ""
        dlg = shot.get("dialogue")
        if isinstance(dlg, dict):
            text = dialogue_map.get(shot_id, "")
            if not text and not dlg.get("placeholder"):
                text = dlg.get("default_text") or ""
        s = f"Shot {i + 1}: {action}"
        if text:
            s += f' "{text}"'
        sound = shot.get("sound_events")
        if sound:
            s += f" [{sound}]"
        shot_descs.append(s)
    if shot_descs:
        lines.append(" ".join(shot_descs))

    style = template.get("style", "")
    if style:
        lines.append(f"{style}.")

    return "\n".join(lines)


# ── Simple descriptive (LTX 2.3 / Flux 2 / Krea 2 / Qwen) ───────────────────

def _assemble_simple(scene_instance: dict, ref_map: dict) -> str:
    template = scene_instance.get("template", {})
    dialogue_map = scene_instance.get("dialogue", {})
    ordered_slots = sorted(ref_map.keys())
    parts: list[str] = []

    subject_parts: list[str] = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        summary = info["appearance_summary"]
        outfit = info["outfit"]
        desc = f"{info['name']}, {summary}" if summary else info["name"]
        if outfit:
            desc += f", {outfit}"
        subject_parts.append(desc)
    if subject_parts:
        parts.append("; ".join(subject_parts))

    env_summary = template.get("environment", {}).get("summary", "")
    if env_summary:
        parts.append(f"Setting: {env_summary}")

    shot_parts: list[str] = []
    for i, shot in enumerate(template.get("shots", [])):
        shot_id = shot.get("id", f"shot_{i + 1}")
        action = _replace_named(shot.get("action", ""), ref_map)
        text = ""
        dlg = shot.get("dialogue")
        if isinstance(dlg, dict):
            text = dialogue_map.get(shot_id, "")
            if not text and not dlg.get("placeholder"):
                text = dlg.get("default_text") or ""
        shot_text = action
        if text:
            shot_text += f' "{text}"'
        if shot_text:
            shot_parts.append(shot_text)
    if shot_parts:
        parts.append(". ".join(shot_parts))

    style = template.get("style", "")
    if style:
        parts.append(style)

    return ". ".join(p.rstrip(".") for p in parts if p) + "."


# ── Assembly report ────────────────────────────────────────────────────────────

def _build_assembly_report(scene_instance: dict, ref_map: dict, model_type: str) -> str:
    ordered_slots = sorted(ref_map.keys())
    total_images = sum(len(info["picture_nums"]) for info in ref_map.values())
    audio_slots = [s for s in ordered_slots if ref_map[s]["audio_num"] is not None]
    audio_files = [ref_map[s]["audio_file"] for s in audio_slots]
    concept_ids = [info["concept_id"] for info in ref_map.values() if info["concept_id"]]

    lines = [
        f"Scene: {scene_instance.get('template_name', 'Unknown')} ({len(ref_map)} speaker{'s' if len(ref_map) != 1 else ''})",
        f"Model: {model_type}",
        "Subjects:",
    ]

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        nums = info["picture_nums"]
        audio = info["audio_num"]
        label = info["subject_id"] or info["concept_id"] or "—"
        pic_str = (
            f"sheets: {len(nums)} image{'s' if len(nums) != 1 else ''}"
            if nums else "no sheets"
        )
        audio_str = f"voice: yes" if audio is not None else "voice: no"
        lines.append(
            f"  Slot {slot_id} → {info['name']} ({label}) as {info['speaker_id']} [{audio_str}, {pic_str}]"
        )

    dialogue_map = scene_instance.get("dialogue", {})
    if dialogue_map:
        lines.append("Dialogue:")
        for shot_id, text in dialogue_map.items():
            preview = (text[:60] + "…") if len(text) > 60 else text
            lines.append(f'  {shot_id}: "{preview}"')

    lines.append("Reference media:")
    lines.append(f"  Images: {total_images} total")
    audio_detail = (
        f"{', '.join(audio_files)}" if audio_files else "none"
    )
    lines.append(
        f"  Audio: {len(audio_files)} file{'s' if len(audio_files) != 1 else ''}"
        + (f" ({audio_detail})" if audio_files else "")
    )

    if concept_ids:
        lines.append(f"Concepts for LoRA resolution:")
        for cid in concept_ids:
            lines.append(f"  {cid} ({model_type})")

    overrides = scene_instance.get("outfit_overrides", {})
    if overrides:
        lines.append("Outfit overrides:")
        for slot_id in sorted(overrides):
            outfit = overrides[slot_id]
            default = ref_map.get(slot_id, {}).get("default_outfit", "")
            lines.append(
                f"  Slot {slot_id}: {outfit}"
                + (f" (overrides default {default})" if default else "")
            )
    else:
        lines.append("Outfit overrides: none")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def assemble_prompt(scene_instance: dict, model_type: str) -> dict:
    """Assemble a model-specific prompt from a SCENE_INSTANCE dict.

    Args:
        scene_instance: dict from compose_scene() / SceneCompose node.
        model_type:     one of MODEL_TYPES.

    Returns a dict with:
        prompt:                str — assembled prompt text
        concept_ids:           list[str] — concept IDs from assigned subjects
        reference_image_order: list[tuple[str, str]] — (slot_id, filename) in emit order
        audio_slots:           list[str] — slot IDs with audio files, in slot order
        assembly_report:       str — human-readable summary
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Valid: {MODEL_TYPES}")

    ref_map = _build_ref_map(scene_instance)

    if model_type == "h3_ref2va":
        prompt = _assemble_h3_ref2va(scene_instance, ref_map)
    elif model_type == "h3_fl2va":
        prompt = _assemble_h3_fl2va(scene_instance, ref_map)
    elif model_type in _PRODUCTION_MODELS:
        prompt = _assemble_production(scene_instance, ref_map)
    else:
        prompt = _assemble_simple(scene_instance, ref_map)

    ordered_slots = sorted(ref_map.keys())

    concept_ids = [
        ref_map[s]["concept_id"]
        for s in ordered_slots
        if ref_map[s]["concept_id"]
    ]

    reference_image_order = [
        (slot_id, fname)
        for slot_id in ordered_slots
        for fname in ref_map[slot_id]["character_sheet_images"]
    ]

    audio_slots = [s for s in ordered_slots if ref_map[s]["audio_file"]]

    report = _build_assembly_report(scene_instance, ref_map, model_type)

    return {
        "prompt": prompt,
        "concept_ids": concept_ids,
        "reference_image_order": reference_image_order,
        "audio_slots": audio_slots,
        "assembly_report": report,
    }
