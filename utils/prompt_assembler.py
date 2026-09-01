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

# ── Speech duration estimation ─────────────────────────────────────────────────

def estimate_speech_duration(
    text: str,
    chars_per_second: float = 13.0,
    inter_word_pause: float = 0.07,
) -> float:
    """Estimate speaking duration of text in seconds.

    Combines character count (phonemic density) with word count (inter-word
    pauses) and punctuation offsets. chars_per_second maps to pace:
      slow ≈ 10, normal ≈ 13, fast ≈ 16.
    """
    words = text.split()
    if not words:
        return 0.0
    char_count = sum(len(w) for w in words)
    word_count = len(words)
    commas = text.count(",") + text.count(";")
    sentence_ends = len(re.findall(r"[.!?]", text))
    return (
        char_count / chars_per_second
        + word_count * inter_word_pause
        + commas * 0.20
        + sentence_ends * 0.35
    )


_PACE_PHRASES: dict[str, str] = {
    "slow": "speaking slowly and deliberately",
    "fast": "speaking quickly",
}

_PACE_CHARS_PER_SEC: dict[str, float] = {
    "slow": 10.0,
    "normal": 13.0,
    "fast": 16.0,
}

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


# ── H3 Ref2VA validation helpers ───────────────────────────────────────────────

def validate_h3_refs_pre(references: list) -> list:
    """Check H3 Ref2VA structural invariants before any media is loaded.

    Returns a list of error strings (empty list = all OK).  Called by
    CompositionToH3Conditioning.execute() before touching the filesystem.

    Rules (§1 of the H3 Ref2VA spec):
      • ≤ 3 standalone audio references
      • audio must be paired with at least one image or video
      • ≤ 12 total counted references (image + video + audio + soundtrack_audio)
      • computed trim_to must be ≥ 2 s when present
    """
    standalone_audio = [r for r in references if r.get("modality") == "audio"]
    visual           = [r for r in references if r.get("modality") in ("image", "video")]
    all_counted      = [r for r in references
                        if r.get("modality") in ("image", "video", "audio", "soundtrack_audio")]

    errors = []
    if len(standalone_audio) > 3:
        errors.append(
            f"{len(standalone_audio)} standalone audio references exceed the limit of 3."
        )
    if standalone_audio and not visual:
        errors.append(
            "Audio references must accompany at least one image or video "
            "(H3 Ref2VA requires audio to be paired with a visual reference)."
        )
    if len(all_counted) > 12:
        errors.append(
            f"{len(all_counted)} total reference files exceed the combined limit of 12."
        )
    for ref in standalone_audio:
        trim = ref.get("trim_to")
        if trim is not None and trim < 2.0:
            aord = ref.get("audio_ordinal", "?")
            errors.append(
                f"<Audio {aord}>: computed trim_to={trim:.2f}s is below the 2s minimum. "
                f"The dialogue line is too short — lengthen it or set a manual minimum."
            )
    return errors


def validate_h3_audio_clip(actual_dur: float, aord, basename: str) -> str | None:
    """Validate a single loaded audio clip's duration after trimming.

    Returns an error string, or None if the clip is within the 2–15 s range.
    """
    if actual_dur < 2.0:
        return (
            f"<Audio {aord}> ({basename}): loaded duration {actual_dur:.2f}s "
            f"is below the 2s minimum required by H3 Ref2VA. "
            f"Use a longer source clip or reduce start_time/trim."
        )
    if actual_dur > 15.0:
        return (
            f"<Audio {aord}> ({basename}): loaded duration {actual_dur:.2f}s "
            f"exceeds the 15s maximum. Set a shorter duration or trim_to."
        )
    return None


def validate_h3_audio_total(durations: list) -> str | None:
    """Validate the sum of all loaded audio clip durations.

    Returns an error string if the total exceeds 15 s, otherwise None.
    """
    total = sum(durations)
    if total > 15.0:
        return (
            f"Total audio duration {total:.2f}s across "
            f"{len(durations)} audio reference(s) "
            f"exceeds the 15s limit."
        )
    return None


# ── Formatting helpers ─────────────────────────────────────────────────────────

_LANG_LABELS: dict[str, str] = {
    "en-us": "American English",
    "en-gb": "British English",
    "ja":    "Japanese",
    "ko":    "Korean",
    "zh":    "Mandarin Chinese",
    "zh-tw": "Taiwanese Mandarin",
    "es":    "Spanish",
    "fr":    "French",
    "de":    "German",
    "it":    "Italian",
    "pt":    "Portuguese",
    "ru":    "Russian",
    "ar":    "Arabic",
    "hi":    "Hindi",
    "no":    "Norwegian",
}


def _lang_label(code: str) -> str:
    """Return a natural-language name for a BCP-47 code, falling back to the code itself."""
    return _LANG_LABELS.get((code or "").lower().strip(), code or "English")


def _possessive(info: dict) -> str:
    """Return the possessive form for a subject based on its pronoun_style.

    masculine → his
    feminine  → her
    object    → its
    location  → the {short_name}'s   (falls back to "its" when short_name absent)
    neutral   → their  (default)
    """
    style = info.get("pronoun_style", "neutral")
    if style == "masculine":
        return "his"
    if style == "feminine":
        return "her"
    if style == "object":
        return "its"
    if style == "location":
        short = info.get("short_name", "").strip()
        return f"the {short}'s" if short else "its"
    return "their"


def _join_labels(labels: list[str]) -> str:
    """Oxford-comma join for reference labels: '<Pic 1>, <Pic 2>, and <Pic 3>'."""
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


def _join_details(parts: list[str]) -> str:
    """Oxford-comma join for appearance detail phrases."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


# ── Reference map ──────────────────────────────────────────────────────────────

def _build_ref_map(
    scene_instance: dict,
    video_entries: list[dict] | None = None,
) -> dict[str, dict]:
    """Assign reference numbers (Subject N, Picture N, Video N, Audio N) per slot.

    Numbering convention (independent per type, per spec):
      Subject 1, 2, … in slot order.
      Picture 1, 2, 3, … in slot order (continuous across all subjects).
      Video 1, 2, … in slot order for slots matched by subject_id in video_entries.
      Audio 1, 2, … mirrors native ref_items order:
        first all soundtrack audios (extract_from_visual, in slot order),
        then all standalone audio files (in slot order).

    video_entries is an optional list of {subject_id, video_file, audio_source, …}
    dicts from cast resolution.  Matching is by subject_id identity.

    Returns a dict keyed by slot_id with typing info for each assigned slot.
    Slots not present in slot_assignments are absent from the result.
    """
    assignments = scene_instance.get("slot_assignments", {})
    outfit_overrides = scene_instance.get("outfit_overrides", {})
    retention_markers = scene_instance.get("retention_markers", {})
    ordered_slots = sorted(assignments.keys())

    # Build subject_id → video_entry lookup.
    # Entries from source profiles carry subject_ids (list) so all co-sourced
    # subjects map to the same video_entry object, giving them a shared <Video N>.
    video_lookup: dict[str, dict] = {}
    for ve in (video_entries or []):
        sid_list = ve.get("subject_ids")
        if sid_list:
            for sid in sid_list:
                if sid and sid not in video_lookup:
                    video_lookup[sid] = ve
        else:
            sid = ve.get("subject_id", "")
            if sid and sid not in video_lookup:
                video_lookup[sid] = ve

    # Track which video_entries already have a video_num so co-sourced subjects
    # share the same ordinal rather than each incrementing the counter.
    _video_entry_num: dict[int, int] = {}  # id(ve) → video_num

    # Pre-assign audio ordinals in native ref_items order so <Audio N> in the
    # prompt matches what MiniMaxH3ReferenceToVideo assigns:
    #   1. soundtrack audios (extract_from_visual, in slot order)
    #   2. standalone audio files (voice.audio_reference_file, in slot order)
    _audio_ctr = 1
    pre_soundtrack_nums: dict[str, int] = {}   # subject_id → ordinal
    pre_standalone_nums: dict[str, int] = {}   # slot_id    → ordinal
    for slot_id in ordered_slots:
        subj = assignments.get(slot_id)
        if subj is None:
            continue
        sid = subj.get("subject_id", "")
        ve = video_lookup.get(sid) if sid else None
        if ve is not None and ve.get("audio_source") == "extract_from_visual":
            pre_soundtrack_nums[sid] = _audio_ctr
            _audio_ctr += 1
    for slot_id in ordered_slots:
        subj = assignments.get(slot_id)
        if subj is None:
            continue
        if subj.get("voice", {}).get("audio_reference_file", ""):
            pre_standalone_nums[slot_id] = _audio_ctr
            _audio_ctr += 1

    # Pre-assign speaker IDs using speaking order, not subject order.
    # Audio-bearing subjects get IDs first; then any slot that appears as
    # speaker_slot in a shot's dialogue entry (dialogue-only speakers with no
    # audio file still need an (Sx) marker so H3 knows who is speaking).
    _speaker_ctr = 1
    pre_speaker_ids: dict[str, str] = {}  # slot_id → "S{n}"
    for slot_id in ordered_slots:
        subj = assignments.get(slot_id)
        if subj is None:
            continue
        sid = subj.get("subject_id", "")
        ve = video_lookup.get(sid) if sid else None
        has_audio = (
            (ve is not None and ve.get("audio_source") == "extract_from_visual")
            or bool(subj.get("voice", {}).get("audio_reference_file", ""))
        )
        if has_audio:
            pre_speaker_ids[slot_id] = f"S{_speaker_ctr}"
            _speaker_ctr += 1
    # Extend to dialogue-only speakers (no audio reference) so (Sx) appears in
    # subject_definitions even when there is no <Audio N> reference.
    # Only consider shots that have actual resolved text in dialogue_map (not
    # just a placeholder marker without text), so (Sx) only appears when
    # real dialogue is present.
    _dialogue_map = scene_instance.get("dialogue", {})
    for _shot in scene_instance.get("template", {}).get("shots", []):
        if _shot.get("id", "") not in _dialogue_map:
            continue
        _dlg = _shot.get("dialogue") or {}
        _spk = _dlg.get("speaker_slot") or _dlg.get("speaker", "")
        if _spk and _spk in ordered_slots and _spk not in pre_speaker_ids:
            pre_speaker_ids[_spk] = f"S{_speaker_ctr}"
            _speaker_ctr += 1

    ref_map: dict[str, dict] = {}
    picture_counter = 1
    video_counter = 1

    # Pre-assign Subject N numbers with bundle subjects first.
    # Processing order (audio ordinals, picture ordinals) stays alphabetical so
    # it matches _build_h3_refplan().  Only the visible Subject label uses this
    # bundle-first numbering.  Replaced source slots get no Subject label.
    _pre_subject_nums: dict[str, int] = {}
    _sub_ctr = 1
    for _sid in ordered_slots:  # bundles (attribute_transfer) first
        if assignments.get(_sid, {}).get("_cast_retention", "fully_preserved") == "attribute_transfer":
            _pre_subject_nums[_sid] = _sub_ctr
            _sub_ctr += 1
    for _sid in ordered_slots:  # retained source subjects next
        _m = assignments.get(_sid, {}).get("_cast_retention", "fully_preserved")
        if _m not in ("attribute_transfer", "replaced"):
            _pre_subject_nums[_sid] = _sub_ctr
            _sub_ctr += 1
    # Replaced source slots intentionally omitted — they get no Subject N label.

    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        appearance = subject.get("appearance", {})
        voice = subject.get("voice", {})
        raw_sheets = subject.get("character_sheet_images", [])
        # Normalize to {file, role} dicts (supports legacy plain strings)
        sheets: list[dict] = [
            e if isinstance(e, dict) else {"file": e, "role": "character sheet"}
            for e in raw_sheets if e
        ]
        audio_file = voice.get("audio_reference_file", "")
        outfit = outfit_overrides.get(slot_id) or appearance.get("default_outfit", "")
        subject_id = subject.get("subject_id", "")

        picture_nums = list(range(picture_counter, picture_counter + len(sheets)))
        # Per-image entries carry (picture_num, file, role) for role-line emission
        character_sheet_entries = [
            {"picture_num": picture_counter + i, "file": s["file"], "role": s.get("role", "character sheet")}
            for i, s in enumerate(sheets)
        ]
        picture_counter += len(sheets)

        ve = video_lookup.get(subject_id) if subject_id else None
        video_num: int | None = None
        video_file: str = ""
        soundtrack_num: int | None = None
        soundtrack_retention: str = "timbre"
        soundtrack_role: str = ""
        if ve:
            is_audio_only = ve.get("audio_only", False)
            if not is_audio_only:
                ve_key = id(ve)
                if ve_key in _video_entry_num:
                    video_num = _video_entry_num[ve_key]
                else:
                    video_num = video_counter
                    _video_entry_num[ve_key] = video_num
                    video_counter += 1
            video_file = ve.get("video_file", "")
            if ve.get("audio_source") == "extract_from_visual":
                soundtrack_num = pre_soundtrack_nums.get(subject_id)
                soundtrack_retention = ve.get("audio_retention", "timbre")
                soundtrack_role = ve.get("audio_role", "")

        audio_num: int | None = None
        if audio_file:
            audio_num = pre_standalone_nums.get(slot_id)

        # Precompute inline appearance phrase for first-appearance injection in shots.
        # Always uses the definite article ("the …") since we're always describing a
        # specific individual, regardless of whether a visual reference is present.
        _ap_sum = appearance.get("summary", "")
        _ap_body = (_ap_sum[0].lower() + _ap_sum[1:]).rstrip(". ") if _ap_sum else ""
        _ap_body = re.sub(r"^an? ", "the ", _ap_body, count=1) if _ap_body else _ap_body
        _ap_detail = [f for f in (
            appearance.get("hair", ""), appearance.get("face", ""), appearance.get("body", "")
        ) if f]
        if outfit:
            _ap_detail.append(f"wearing {outfit}")
        _ap_detail_str = f", with {_join_details(_ap_detail)}" if _ap_detail else ""
        appearance_phrase = f"{_ap_body}{_ap_detail_str}"

        subject_num = _pre_subject_nums.get(slot_id, 0)
        subject_label = f"<Subject {subject_num}>" if subject_num else ""

        ref_map[slot_id] = {
            "subject_id": subject_id,
            "name": subject.get("name", slot_id),
            "appearance_summary": appearance.get("summary", ""),
            "appearance_phrase": appearance_phrase,
            "face": appearance.get("face", ""),
            "hair": appearance.get("hair", ""),
            "body": appearance.get("body", ""),
            "default_outfit": appearance.get("default_outfit", ""),
            "outfit": outfit,
            "voice_description": voice.get("description", ""),
            "audio_file": audio_file,
            "audio_retention": voice.get("audio_retention", "timbre"),
            "audio_role": voice.get("audio_role", ""),
            "language": voice.get("language", "en-us") or "en-us",
            "character_sheet_images": [s["file"] for s in sheets],
            "character_sheet_entries": character_sheet_entries,
            "concept_id": subject.get("concept_id", ""),
            "subject_num": subject_num,
            "speaker_id": pre_speaker_ids.get(slot_id, f"S{subject_num or 1}"),
            "subject_label": subject_label,
            "picture_nums": picture_nums,
            "video_num": video_num,
            "video_file": video_file,
            "audio_num": audio_num,
            "soundtrack_num": soundtrack_num,
            "soundtrack_retention": soundtrack_retention,
            "soundtrack_role": soundtrack_role,
            "retention_marker": (
                retention_markers.get(slot_id)
                or subject.get("_cast_retention", "fully_preserved")
            ),
            "transfer_to_slot": subject.get("_transfer_to_slot", ""),
            "pronoun_style":    subject.get("_pronoun_style", "neutral"),
            "short_name":       subject.get("_short_name", ""),
        }

    return ref_map


# ── H3 reference plan ─────────────────────────────────────────────────────────

def _build_h3_refplan(
    scene_instance: dict,
    video_entries: list[dict] | None = None,
    slot_trim_to: dict[str, float] | None = None,
) -> dict:
    """Build the FBTOOLS_H3_REFPLAN ordered reference descriptor bundle.

    Produces a references list in the exact order the native
    MiniMaxH3ReferenceToVideo node builds its ref_items, so that both the
    assembled prompt and the terminal node derive identical label ordinals:

        images (slot order, one entry per sheet)
        per video (slot order):
            soundtrack_audio entry (if audio_source == "extract_from_visual")
            video entry
        standalone audio (slot order, from voice.audio_reference_file)

    video_entries items must carry:
        subject_id, video_file, load_params,
        audio_source ("extract_from_visual" | "none"),
        audio_path, audio_start_time, audio_duration,
        audio_retention ("timbre" | "reuse" | "style"), audio_role

    Subject dicts (slot_assignments values) must carry voice fields:
        audio_reference_file, audio_start_time, audio_duration,
        audio_retention, audio_role
    (populated by apply_cast_to_subjects when use_audio is True).

    Returns {"references": [...]} — caller merges additional fields (prompt,
    model_type, etc.) before attaching to the node output.
    """
    assignments = scene_instance.get("slot_assignments", {})
    ordered_slots = sorted(assignments.keys())

    # Build subject_id → video_entry lookup (slot/cast order; first match wins)
    video_lookup: dict[str, dict] = {}
    for ve in (video_entries or []):
        sid = ve.get("subject_id", "")
        if sid and sid not in video_lookup:
            video_lookup[sid] = ve

    references: list[dict] = []
    picture_counter = 1
    video_counter   = 1
    audio_counter   = 1

    # Pass 1: collect image entries (all images before any audio/video)
    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        sheets = subject.get("character_sheet_images", [])
        subject_id = subject.get("subject_id", "")
        for entry in sheets:
            file_path = entry.get("file", "") if isinstance(entry, dict) else entry
            if not file_path:
                continue
            references.append({
                "modality":       "image",
                "picture_ordinal": picture_counter,
                "subject_id":     subject_id,
                "slot_id":        slot_id,
                "path":           file_path,
            })
            picture_counter += 1

    # Pass 2: per-video — soundtrack audio (if any) immediately before its video
    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        subject_id = subject.get("subject_id", "")
        ve = video_lookup.get(subject_id) if subject_id else None
        if ve is None:
            continue

        has_soundtrack = ve.get("audio_source") == "extract_from_visual"
        is_audio_only  = ve.get("audio_only", False)
        this_video_ordinal = video_counter

        if has_soundtrack:
            if is_audio_only:
                # No video is emitted for this slot (images mode with extracted audio).
                # Treat as standalone audio — pairing with a nonexistent <Video N> is wrong.
                references.append({
                    "modality":      "audio",
                    "audio_ordinal": audio_counter,
                    "subject_id":    subject_id,
                    "slot_id":       slot_id,
                    "path":          ve.get("audio_path", ve.get("video_file", "")),
                    "start_time":    ve.get("audio_start_time", 0.0),
                    "duration":      ve.get("audio_duration", 0.0),
                    "retention":     ve.get("audio_retention", "timbre"),
                    "role":          ve.get("audio_role", ""),
                    "audio_cache":   ve.get("audio_cache", ""),
                })
            else:
                references.append({
                    "modality":      "soundtrack_audio",
                    "audio_ordinal": audio_counter,
                    "video_ordinal": this_video_ordinal,
                    "subject_id":    subject_id,
                    "slot_id":       slot_id,
                    "path":          ve.get("audio_path", ve.get("video_file", "")),
                    "start_time":    ve.get("audio_start_time", 0.0),
                    "duration":      ve.get("audio_duration", 0.0),
                    "retention":     ve.get("audio_retention", "timbre"),
                    "role":          ve.get("audio_role", ""),
                    "audio_cache":   ve.get("audio_cache", ""),
                })
            audio_counter += 1

        # audio_only entries provide only an audio reference — no visual video
        # item is emitted, and the video counter is not incremented.
        if not is_audio_only:
            references.append({
                "modality":      "video",
                "video_ordinal": this_video_ordinal,
                "subject_id":    subject_id,
                "slot_id":       slot_id,
                "path":          ve.get("video_file", ""),
                "load_params":   ve.get("load_params", {}),
            })
            video_counter += 1

    # Pass 3: standalone audio entries (voice.audio_reference_file)
    _trim_map = slot_trim_to or {}
    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        voice = subject.get("voice", {})
        audio_file = voice.get("audio_reference_file", "")
        if not audio_file:
            continue
        subject_id = subject.get("subject_id", "")
        references.append({
            "modality":      "audio",
            "audio_ordinal": audio_counter,
            "subject_id":    subject_id,
            "slot_id":       slot_id,
            "path":          audio_file,
            "start_time":    voice.get("audio_start_time", 0.0),
            "duration":      voice.get("audio_duration", 0.0),
            "retention":     voice.get("audio_retention", "timbre"),
            "role":          voice.get("audio_role", ""),
            "trim_to":       _trim_map.get(slot_id),
            "audio_cache":   voice.get("audio_cache", ""),
        })
        audio_counter += 1

    return {"references": references}


# ── Placeholder replacement ────────────────────────────────────────────────────

def _replace_h3(
    text: str,
    ref_map: dict,
    speaking_slots: "set[str] | frozenset[str]" = frozenset(),
    seen_globally: "set[str] | None" = None,
) -> str:
    """Replace {A}/{B}/{C}/{D} with H3 subject labels.

    First appearance (slot_id not yet in seen_globally):
        "<Subject N>, the appearance phrase…"   — non-speaking
        "<Subject N> (SN), the appearance phrase…" — speaking
    Subsequent appearances:
        "<Subject N>"        — non-speaking
        "<Subject N> (SN)"  — speaking

    Pass a mutable set as seen_globally and share it across all calls in one shot
    sequence so first-appearance detection works across camera + action lines.
    """
    def _sub(match: re.Match) -> str:
        slot_id = match.group(1)
        info = ref_map.get(slot_id)
        if not info:
            return match.group(0)
        # Redirect replaced source slots to their bundle replacement in descriptions.
        # effective_slot tracks the bundle slot for seen_globally so first-appearance
        # detection is consistent across multiple {A} references in the same shot.
        effective_slot = slot_id
        if info.get("retention_marker") == "replaced" and info.get("transfer_to_slot"):
            redirect_target = info["transfer_to_slot"]   # bundle slot, e.g. "E"
            repl_info = ref_map.get(redirect_target)
            if repl_info:
                info = repl_info
                effective_slot = redirect_target
        label = info["subject_label"]
        # A replaced source slot refers to the same entity as its bundle replacement,
        # so treat it as speaking if the bundle slot is the active speaker.
        is_speaking = slot_id in speaking_slots or effective_slot in speaking_slots
        base = f"{label} ({info['speaker_id']})" if is_speaking else label
        if seen_globally is not None and effective_slot not in seen_globally:
            seen_globally.add(effective_slot)
            ap = info.get("appearance_phrase", "")
            if ap:
                return f"{base}, {ap}"
        return base

    return re.sub(r"\{([A-H])\}", _sub, text)


def _replace_named(text: str, ref_map: dict) -> str:
    """Replace {A}/{B}/{C}/{D} with subject names (for non-H3 formats)."""
    def _sub(match: re.Match) -> str:
        slot_id = match.group(1)
        info = ref_map.get(slot_id)
        return info["name"] if info else match.group(0)

    return re.sub(r"\{([A-D])\}", _sub, text)


# ── H3 Ref2VA ─────────────────────────────────────────────────────────────────

# Human-readable descriptions for each character sheet role used in <Picture N> lines.
# All descriptions end with "do not use as scene composition" so H3 treats them as
# appearance-only references rather than spatial layout anchors.
_SHEET_ROLE_H3: dict[str, str] = {
    "character sheet":  "a character reference sheet (appearance reference; do not use as scene composition)",
    "portrait":         "a frontal portrait (facial likeness reference only; do not use as scene composition)",
    "side profile":     "a side-profile view (silhouette and facial structure reference; do not use as scene composition)",
    "full body":        "a full-body reference (costume and proportion reference; do not use as scene composition)",
    "costume detail":   "a costume detail reference (texture and accessory reference; do not use as scene composition)",
    "reference":        "an appearance reference image (do not use as scene composition)",
}

# Natural-language inline descriptions for character sheet roles, used inside
# subject_definitions sentences: "appearance comes from <Picture N>, <desc>."
# Single-sheet only — multi-sheet references omit the inline description.
_SHEET_ROLE_INLINE: dict[str, str] = {
    "character sheet":  "a character reference sheet showing the figure from multiple angles",
    "portrait":         "a frontal portrait (facial likeness reference)",
    "side profile":     "a side-profile view (silhouette and structure reference)",
    "full body":        "a full-body reference (costume and proportion reference)",
    "costume detail":   "a costume detail reference",
    "reference":        "an appearance reference image",
}


def _assemble_h3_ref2va(scene_instance: dict, ref_map: dict) -> str:
    template = scene_instance.get("template", {})
    dialogue_map = scene_instance.get("dialogue", {})
    ordered_slots = sorted(ref_map.keys())
    sections: list[str] = []

    # Resolve task flags early — needed in both subject_definitions and summary
    user_flags: list[str] = scene_instance.get("task_flags") or []
    active_flags: set[str] = set(user_flags)

    # Slots that are the active speaker in a shot with real dialogue text — used
    # to inject (Sx) into subject_definitions when there is no <Audio N>.
    # Placeholder slots without resolved dialogue_map entries are excluded so
    # templates with placeholder markers don't add (Sx) when no text is provided.
    _dlg_speaking_slots: set[str] = set()
    for _shot in template.get("shots", []):
        if _shot.get("id", "") not in dialogue_map:
            continue
        _dlg = _shot.get("dialogue") or {}
        _spk = _dlg.get("speaker_slot") or _dlg.get("speaker", "")
        if _spk:
            _dlg_speaking_slots.add(_spk)

    # Build video_num → [subject_label, …] map for shared-video role lines.
    # Only include subjects with a defined Subject label (excludes replaced source slots).
    _vnum_to_labels: dict[int, list[str]] = {}
    for _sid in ordered_slots:
        _info = ref_map[_sid]
        _vn = _info.get("video_num")
        if _vn is not None and _info.get("subject_label"):
            _vnum_to_labels.setdefault(_vn, []).append(_info["subject_label"])

    # Identify which video numbers are source/motion-donor videos vs. pure bundle
    # appearance references. A video is a source unless EVERY associated slot is
    # attribute_transfer (a bundle replacement whose video is only a visual ref).
    # This prevents "is the source video being edited" from appearing on bundle
    # reference videos; simple editing scenes (all slots retained/replaced) still
    # mark their single video as a source correctly.
    _vnum_all_markers: dict[int, list[str]] = {}
    for _sid in ordered_slots:
        _info = ref_map[_sid]
        _vn = _info.get("video_num")
        if _vn is not None:
            _vnum_all_markers.setdefault(_vn, []).append(_info.get("retention_marker") or "")
    _vnum_is_source: set[int] = {
        _vn for _vn, _markers in _vnum_all_markers.items()
        if any(m != "attribute_transfer" for m in _markers)
    }

    # ── subject_definitions ────────────────────────────────────────────────────
    # Format (per MiniMax H3 Ref2VA spec):
    #   Subject lines:  "<Subject N> is [summary] whose appearance comes from <Picture N>…"
    #                   (pictures cited inside Subject line — no standalone <Picture N> entries
    #                    for character references per spec)
    #   Video lines:    "<Video N> is the visual identity reference / source video being edited…"
    #   Audio lines:    "<Audio N> is the voice-timbre reference for <Subject N> (SN)…"
    #
    # Attribute-transfer (video editing replacement) rules:
    #   • Replaced source slot (retention_marker="replaced"): no Subject line emitted.
    #     Their video line still appears so H3 knows the motion source.
    #   • Bundle replacement slot (retention_marker="attribute_transfer"): Subject line
    #     appears with a motion-source sentence referencing the source video.

    # Emit Subject lines in display order: bundle (attribute_transfer) subjects first,
    # then retained source subjects.
    _sd_display_slots = (
        [s for s in ordered_slots if ref_map[s].get("retention_marker") == "attribute_transfer"]
        + [s for s in ordered_slots
           if ref_map[s].get("retention_marker") not in ("attribute_transfer", "replaced")
           and bool(ref_map[s].get("subject_label"))]
    )

    sd: list[str] = []
    video_sd_lines: list[str] = []
    video_sd_emitted: set[int] = set()  # guard against duplicate <Video N> lines
    audio_sd_lines: list[str] = []

    for slot_id in _sd_display_slots:
        info = ref_map[slot_id]
        label = info["subject_label"]
        # Append (Sx) speaker ID to the label when the slot has dialogue so H3
        # can associate the voice with the correct subject definition, even when
        # there is no <Audio N> reference (audio-only subjects already get (Sx)
        # appended via the <Audio N> line; this covers dialogue-only speakers).
        if slot_id in _dlg_speaking_slots and info.get("speaker_id") and label:
            if info.get("audio_num") is None and info.get("soundtrack_num") is None:
                label = f"{label} ({info['speaker_id']})"
        summary = info["appearance_summary"] or info["name"]

        # Build reference anchor using spec-compliant phrasing:
        #   pictures → "whose appearance comes from <Picture N>" (appearance reference, cited inline)
        #   video    → "from <Video N>" (visual identity)
        #   both     → "whose appearance comes from <Picture N> and whose motion comes from <Video N>"
        vid_ref = f"<Video {info['video_num']}>" if info["video_num"] is not None else ""
        pic_ref_str = _join_labels([f"<Picture {p}>" for p in info["picture_nums"]]) if info["picture_nums"] else ""

        cs_entries = info.get("character_sheet_entries", [])
        if len(cs_entries) == 1:
            _cs_role = cs_entries[0].get("role", "character sheet")
            _cs_desc = _SHEET_ROLE_INLINE.get(_cs_role, "")
            pic_role_suffix = f", {_cs_desc}" if _cs_desc else ""
        else:
            pic_role_suffix = ""

        if vid_ref and pic_ref_str:
            ref_anchor = f" whose appearance comes from {pic_ref_str}{pic_role_suffix} and whose motion comes from {vid_ref}"
        elif pic_ref_str:
            ref_anchor = f" whose appearance comes from {pic_ref_str}{pic_role_suffix}"
        elif vid_ref:
            ref_anchor = f" from {vid_ref}"
        else:
            ref_anchor = ""

        # Appearance details as flowing prose (from separate structured fields)
        detail_parts = [
            info[f] for f in ("hair", "face", "body", "outfit") if info.get(f)
        ]
        detail_phrase = f", with {_join_details(detail_parts)}" if detail_parts else ""

        # Prose descriptions start with lowercase after "is"; proper names keep
        # their capitalisation.  Heuristic: if appearance_summary is populated
        # treat it as prose (lowercase first letter); if it's empty and we fell
        # back to the subject's display name, preserve case.
        _ap_sum = info["appearance_summary"]
        if _ap_sum:
            summary_body = (_ap_sum[0].lower() + _ap_sum[1:]).rstrip(". ")
        else:
            summary_body = info["name"].rstrip(". ") if info["name"] else ""
        # When a specific reference is cited, the subject is definite — replace
        # leading indefinite article ("a "/"an ") with "the ".
        if ref_anchor and summary_body:
            summary_body = re.sub(r"^an? ", "the ", summary_body, count=1)

        # Attribute-transfer bundle subject: add a sentence explaining that their
        # pose/motion/position come from the source subject they replace.
        motion_clause = ""
        if info.get("retention_marker") == "attribute_transfer":
            src_slot_id = info.get("transfer_to_slot", "")
            src_info = ref_map.get(src_slot_id)
            if src_info:
                src_vnum = src_info.get("video_num")
                src_name = src_info.get("name") or src_info.get("appearance_summary") or "the replaced subject"
                if src_vnum is not None:
                    motion_clause = (
                        f" Their pose, movement, and screen position in the scene "
                        f"match those of {src_name} in <Video {src_vnum}>."
                    )
                else:
                    motion_clause = (
                        f" Their pose, movement, and screen position in the scene "
                        f"match those of {src_name}."
                    )

        sd.append(f"{label} is {summary_body}{ref_anchor}{detail_phrase}.{motion_clause}")

    # Video/audio sd_lines iterate all slots (including replaced) so every <Video N>
    # and <Audio N> gets a role line.
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        label = info["subject_label"] or info["name"]  # fallback name for replaced slots

        # Standalone video role line (task-flag-aware); emitted for ALL slots including
        # transfer sources so H3 knows what each <Video N> is.
        if info["video_num"] is not None:
            vnum = info["video_num"]
            if vnum not in video_sd_emitted:
                video_sd_emitted.add(vnum)
                if "video continuation" in active_flags:
                    video_role = "is the continuation starting point for the target video"
                elif "video editing" in active_flags and vnum in _vnum_is_source:
                    video_role = "is the source video being edited"
                else:
                    co_labels = _vnum_to_labels.get(vnum, [label])
                    targets = _join_labels(co_labels) if co_labels else label
                    video_role = f"is the visual identity reference for {targets}"
                video_sd_lines.append(f"<Video {vnum}> {video_role}")

        # Soundtrack audio line (extract_from_visual — audio from the video)
        if info["soundtrack_num"] is not None:
            snd_num = info["soundtrack_num"]
            snd_ret = info["soundtrack_retention"]
            snd_role = info["soundtrack_role"]
            vnum = info["video_num"]
            voice_desc = info["voice_description"] or f"a spoken {_lang_label(info['language'])} vocal layer"
            spk = f"{label} ({info['speaker_id']})"
            if snd_role:
                desc = f"{snd_role} for {spk}"
            elif snd_ret == "reuse" and vnum is not None:
                desc = f"the audio track from <Video {vnum}>, reproduced verbatim"
            elif snd_ret == "style":
                desc = (f"the audio style and rhythm reference for {spk}, "
                        f"containing {voice_desc}")
            else:
                desc = (f"the voice-timbre reference for {spk}, "
                        f"containing {voice_desc}, without copying the original signal")
            audio_sd_lines.append(f"<Audio {snd_num}> is {desc}.")

        # Standalone audio line (voice.audio_reference_file)
        if info["audio_num"] is not None:
            aud_num = info["audio_num"]
            aud_ret = info["audio_retention"]
            aud_role = info["audio_role"]
            voice_desc = info["voice_description"] or f"a spoken {_lang_label(info['language'])} vocal layer"
            spk = f"{label} ({info['speaker_id']})"
            if aud_role:
                desc = f"{aud_role} for {spk}"
            elif aud_ret == "reuse":
                desc = f"the audio for {spk}, {voice_desc}, reproduced verbatim"
            elif aud_ret == "style":
                desc = (f"the audio style and rhythm reference for {spk}, "
                        f"containing {voice_desc}")
            else:
                desc = (f"the voice-timbre reference for {spk}, "
                        f"containing {voice_desc}, without copying the original signal")
            audio_sd_lines.append(f"<Audio {aud_num}> is {desc}.")

    sd.extend(video_sd_lines)
    sd.extend(audio_sd_lines)
    sections.append("subject_definitions:\n" + "\n".join(sd))

    # ── summary ────────────────────────────────────────────────────────────────
    has_refs = any(info["picture_nums"] or info["video_num"] is not None for info in ref_map.values())
    has_audio = any(
        info["audio_num"] is not None or info["soundtrack_num"] is not None
        for info in ref_map.values()
    )

    # Task tag: merge user-provided flags with auto-detected ones.
    # user_flags supply intent-based tasks (video editing, video continuation, etc.)
    # that cannot be inferred from references alone.  Auto-detection always adds
    # the reference-type tasks that ARE inferrable, regardless of user_flags.
    _TASK_ORDER = [
        "video continuation",
        "video editing",
        "keyframe completion",
        "reference generation",
        "audio reference",
        "audio reuse",
    ]

    def _sort_tasks(flags: list[str]) -> list[str]:
        known = [f for f in _TASK_ORDER if f in flags]
        unknown = [f for f in flags if f not in _TASK_ORDER]
        return known + unknown

    # Auto-detect reference tasks from what's actually present in ref_map.
    # Intent-based flags (video editing / continuation) explain why a video is present
    # without implying "reference generation" — so only pictures unconditionally trigger
    # "reference generation" when an intent mode is already active.
    _user_set = set(user_flags)
    _intent_modes = {"video continuation", "video editing", "keyframe completion"}
    _is_intent_mode = bool(_intent_modes & _user_set)

    auto_tasks: list[str] = []
    _pics_exist = any(bool(info["picture_nums"]) for info in ref_map.values())
    _vids_exist = any(info["video_num"] is not None for info in ref_map.values())
    if _pics_exist or (_vids_exist and not _is_intent_mode):
        auto_tasks.append("reference generation")

    # Differentiate audio task by retention: timbre/style → "audio reference"; reuse → "audio reuse".
    # If the user already specified any audio task flag, skip auto-detection for audio entirely.
    _audio_flags = {"audio reference", "audio reuse"}
    if not (_audio_flags & _user_set):
        _has_timbre_audio = any(
            (info["soundtrack_num"] is not None and info["soundtrack_retention"] != "reuse")
            or (info["audio_num"] is not None and info["audio_retention"] != "reuse")
            for info in ref_map.values()
        )
        _has_reuse_audio = any(
            (info["soundtrack_num"] is not None and info["soundtrack_retention"] == "reuse")
            or (info["audio_num"] is not None and info["audio_retention"] == "reuse")
            for info in ref_map.values()
        )
        if _has_timbre_audio:
            auto_tasks.append("audio reference")
        if _has_reuse_audio:
            auto_tasks.append("audio reuse")

    # Merge: user_flags lead (they carry intent tasks); auto_tasks fill in reference tasks.
    merged_flags = user_flags + [t for t in auto_tasks if t not in _user_set]
    task_tag = "[" + " + ".join(_sort_tasks(merged_flags)) + "]" if merged_flags else "[video generation]"

    # Build bare {A}→<Subject N> map for the summary narrative (no names/appearance)
    slot_to_label = {slot_id: ref_map[slot_id]["subject_label"] for slot_id in ordered_slots}

    def _bare(text: str) -> str:
        """Replace {A}/{B}/etc. with bare <Subject N> labels."""
        for sid, lbl in slot_to_label.items():
            text = text.replace("{" + sid + "}", lbl)
        return text

    shots = template.get("shots", [])
    scene_synopsis: str = scene_instance.get("scene_synopsis", "").strip()
    narrative: list[str] = []

    # video editing tasks must open with a fixed sentence naming the source video
    if "video editing" in active_flags:
        first_video = next(
            (info["video_num"] for info in ref_map.values() if info["video_num"] is not None),
            None,
        )
        if first_video is not None:
            narrative.append(f"The target video is an edited version of <Video {first_video}>.")

    if scene_synopsis:
        # User-supplied synopsis: replace {A}/{B}/… with bare <Subject N> labels.
        # Prepend "The target video shows" only when not already a video-editing opener.
        replaced = _bare(scene_synopsis).rstrip(".")
        if "video editing" not in active_flags:
            narrative.append(f"The target video shows {replaced}.")
        else:
            # video editing already opened above; synopsis is an additional description
            narrative.append(f"{replaced[0].upper() + replaced[1:]}.")
    elif shots:
        # Fallback: build narrative from shot action fields
        if "video editing" not in active_flags:
            first_action = shots[0].get("action", "").strip()
            if first_action:
                replaced = _bare(first_action).rstrip(".")
                narrative.append(f"The target video shows {replaced}.")
        for shot in (shots[1:] if "video editing" not in active_flags else shots):
            action = shot.get("action", "").strip()
            if action:
                replaced = _bare(action)
                if replaced and not replaced[0].isupper():
                    replaced = replaced[0].upper() + replaced[1:]
                if not replaced.endswith("."):
                    replaced += "."
                narrative.append(replaced)

    # Audio reference closing sentence (soundtracks first, then standalone)
    audio_voice_parts: list[str] = []
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["soundtrack_num"] is not None:
            snd_ret = info["soundtrack_retention"]
            lbl = info["subject_label"]
            if snd_ret == "reuse":
                phrase = f"<Audio {info['soundtrack_num']}> as the audio to reproduce verbatim for {lbl}"
            elif snd_ret == "style":
                phrase = f"<Audio {info['soundtrack_num']}> as the audio style reference for {lbl}"
            else:
                phrase = f"<Audio {info['soundtrack_num']}> as the voice-timbre reference for {lbl}"
            audio_voice_parts.append(phrase)
        if info["audio_num"] is not None:
            aud_ret = info["audio_retention"]
            lbl = info["subject_label"]
            if aud_ret == "reuse":
                phrase = f"<Audio {info['audio_num']}> as the audio to reproduce verbatim for {lbl}"
            elif aud_ret == "style":
                phrase = f"<Audio {info['audio_num']}> as the audio style reference for {lbl}"
            else:
                phrase = f"<Audio {info['audio_num']}> as the voice-timbre reference for {lbl}"
            audio_voice_parts.append(phrase)
    if audio_voice_parts:
        narrative.append("The scene uses " + " and ".join(audio_voice_parts) + ".")

    narrative_str = " ".join(narrative) if narrative else "Video generation scene."
    sections.append(f"summary:\n{task_tag} {narrative_str}")

    # ── retention_analysis ─────────────────────────────────────────────────────
    ra: list[str] = []

    # Determine which shots each subject appears in — check for {slot_id} in
    # action/camera text and dialogue speaker matching the slot key.
    shots_list = template.get("shots", [])
    slot_appearances: dict[str, list[str]] = {slot_id: [] for slot_id in ordered_slots}
    for shot_idx, shot in enumerate(shots_list):
        shot_label = f"[Shot {shot_idx + 1}]"
        action_text = shot.get("action", "")
        camera_text = shot.get("camera", "")
        dlg = shot.get("dialogue") or {}
        dlg_speaker = dlg.get("speaker") or dlg.get("speaker_slot", "")
        for slot_id in ordered_slots:
            marker = "{" + slot_id + "}"
            if (marker in action_text or marker in camera_text
                    or dlg_speaker == slot_id):
                if shot_label not in slot_appearances[slot_id]:
                    slot_appearances[slot_id].append(shot_label)

    # Subject entries in display order: bundle (attribute_transfer) first, then retained.
    # Replaced source slots are skipped — they have no Subject definition.
    _ra_display_slots = (
        [s for s in ordered_slots if ref_map[s].get("retention_marker") == "attribute_transfer"]
        + [s for s in ordered_slots
           if ref_map[s].get("retention_marker") not in ("attribute_transfer", "replaced")
           and bool(ref_map[s].get("subject_label"))]
    )
    for slot_id in _ra_display_slots:
        info = ref_map[slot_id]
        label = info["subject_label"]

        appears_list = slot_appearances[slot_id]
        appears_clause = (
            "appears in " + ", ".join(appears_list) if appears_list else "appears throughout"
        )

        subj_retention = info.get("retention_marker", "fully_preserved")

        if subj_retention == "attribute_transfer":
            # Bundle subject replacing a source profile subject.
            src_slot_id = info.get("transfer_to_slot", "")
            src_info = ref_map.get(src_slot_id)
            src_name = src_info.get("name", "the replaced subject") if src_info else "the replaced subject"
            src_vnum = src_info.get("video_num") if src_info else None
            if src_vnum is not None:
                preserve_desc = (
                    f"{info['name']}'s appearance overrides that of {src_name} in the source video. "
                    f"Their pose, movement, and screen position match those of {src_name} in <Video {src_vnum}>"
                )
            else:
                preserve_desc = (
                    f"{info['name']}'s appearance overrides that of {src_name} in the source video"
                )
        else:
            # Build the same appearance phrase used in subject_definitions (minus the ref anchor).
            summary = info["appearance_summary"]
            summary_body = (summary[0].lower() + summary[1:]).rstrip(". ") if summary else ""
            has_ref = info["video_num"] is not None or bool(info["picture_nums"])
            if has_ref and summary_body:
                summary_body = re.sub(r"^an? ", "the ", summary_body, count=1)
            detail_parts = [f for f in (info.get("hair", ""), info.get("face", ""), info.get("body", "")) if f]
            if info["outfit"]:
                detail_parts.append(f"wearing {info['outfit']}")
            detail_phrase = f", with {_join_details(detail_parts)}" if detail_parts else ""
            preserve_desc = f"{summary_body}{detail_phrase}" if summary_body else "appearance retained"

        ra_retention = "fully_preserved" if subj_retention == "attribute_transfer" else subj_retention
        ra.append(f"{label} ({appears_clause}): {ra_retention} - {preserve_desc}.")

    # Video entries: "<Video N> (role): <status> - ..."
    # Deduplicate by video_num — co-sourced subjects share one video line.
    _ra_video_emitted: set[int] = set()
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["video_num"] is not None:
            vnum = info["video_num"]
            if vnum in _ra_video_emitted:
                continue
            _ra_video_emitted.add(vnum)
            co_labels = _vnum_to_labels.get(vnum, [])
            targets = _join_labels(co_labels) if co_labels else "the target subject"
            if "video continuation" in active_flags:
                role_clause  = "continuation starting point"
                video_status = "fully_preserved"
                preserve_desc = (
                    f"<Video {vnum}> is the continuation starting point; the target video "
                    f"begins where <Video {vnum}> ends, maintaining consistent motion and scene state"
                )
            elif "video editing" in active_flags and vnum in _vnum_is_source:
                # Only motion is carried over; the original subject's appearance is replaced.
                role_clause  = "motion and gestures"
                video_status = "partially_preserved"
                # Bundle slots (attribute_transfer) with transfer_to_slot pointing to a slot
                # that has this video_num are the replacement subjects.
                replacement_pairs: list[tuple[str, str]] = []   # (original_name, replacement_label)
                retained_originals: list[str] = []
                for _sid in ordered_slots:
                    _inf = ref_map[_sid]
                    if _inf.get("retention_marker") == "attribute_transfer":
                        _src_slot = _inf.get("transfer_to_slot", "")
                        _src_inf = ref_map.get(_src_slot)
                        if _src_inf and _src_inf.get("video_num") == vnum:
                            replacement_pairs.append((_src_inf["name"], _inf["subject_label"]))
                    elif _inf.get("video_num") == vnum and _inf.get("retention_marker") != "replaced":
                        retained_originals.append(_inf["subject_label"])
                if replacement_pairs:
                    replacements_str = _join_labels([r for _, r in replacement_pairs])
                    originals_str    = _join_labels([o for o, _ in replacement_pairs])
                    preserve_desc = (
                        f"the actions, gestures, head movements, hand timing, and camera framing "
                        f"from <Video {vnum}> are reproduced exactly by {replacements_str}, "
                        f"without copying the visual appearance of {originals_str}"
                    )
                    if retained_originals:
                        retained_str = _join_labels(retained_originals)
                        n = len(retained_originals)
                        preserve_desc += (
                            f"; {retained_str} appear{'s' if n == 1 else ''} "
                            f"as {'themselves' if n > 1 else 'themselves'} from <Video {vnum}>"
                        )
                else:
                    preserve_desc = (
                        f"the actions, gestures, head movements, hand timing, and camera framing "
                        f"from <Video {vnum}> are reproduced exactly by {targets}, without copying "
                        f"the visual appearance of the original subjects in <Video {vnum}>"
                    )
            else:
                role_clause  = f"visual identity of {targets}"
                video_status = "fully_preserved"
                _plural = len(co_labels) > 1
                _whose  = "their" if _plural else "the subject's"
                _who    = "the people" if _plural else "the person"
                preserve_desc = (
                    f"<Video {vnum}> defines the visual identity of {targets}; "
                    f"{_whose} appearance in the target video "
                    f"must fully match {_who} shown in <Video {vnum}>"
                )
            ra.append(f"<Video {vnum}> ({role_clause}): {video_status} - {preserve_desc}.")

    # Picture entries: "<Picture N>: fully_preserved - ..."
    # Each character-sheet image fully defines the replacement subject's visual appearance.
    _ra_pic_emitted: set[int] = set()
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        for pnum in info["picture_nums"]:
            if pnum in _ra_pic_emitted:
                continue
            _ra_pic_emitted.add(pnum)
            subj_label = info["subject_label"]
            ra.append(
                f"<Picture {pnum}>: fully_preserved - "
                f"{subj_label}'s facial features, hair, and clothing."
            )

    # Audio entries — format matches H3 spec: "fully_copy" or "reference - ..."
    def _audio_ra_line(
        aud_num: int,
        retention: str,
        subj_label: str,
        is_soundtrack: bool = False,
        video_num: int | None = None,
    ) -> str:
        ref_tag = f"<Audio {aud_num}>"
        if retention == "reuse":
            if is_soundtrack and video_num is not None:
                return (
                    f"{ref_tag}: fully_copy - {ref_tag} is the audio track from "
                    f"<Video {video_num}>, reproduced verbatim as the target video's "
                    f"complete final audio track."
                )
            return (
                f"{ref_tag}: fully_copy - {ref_tag} is reused 1:1 as the "
                f"target video's complete final audio track."
            )
        if retention == "style":
            return (
                f"{ref_tag}: reference - the target audio follows {ref_tag}'s "
                f"rhythm, pace, and tonal style without copying the original signal."
            )
        return (
            f"{ref_tag}: reference - the target speaker follows {ref_tag}'s "
            f"voice timbre and measured delivery without copying the original signal."
        )

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["soundtrack_num"] is not None:
            ra.append(_audio_ra_line(
                info["soundtrack_num"], info["soundtrack_retention"], info["subject_label"],
                is_soundtrack=True, video_num=info["video_num"],
            ))

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["audio_num"] is not None:
            ra.append(_audio_ra_line(
                info["audio_num"], info["audio_retention"], info["subject_label"],
            ))

    sections.append("retention_analysis:\n" + "\n".join(ra))

    # ── detailed_description ───────────────────────────────────────────────────
    use_dialogue_tags: bool = scene_instance.get("dialogue_tags", False)
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

    # Video-editing preamble: opening quality directive followed by one sentence per
    # bundle replacement pair, identifying the source subject by appearance so the
    # model can locate them in the video, then naming the bundle subject replacing them.
    if "video editing" in active_flags:
        _bun_slots = [
            s for s in ordered_slots
            if ref_map[s].get("retention_marker") == "attribute_transfer"
        ]
        if _bun_slots:
            dd.append("")
            dd.append(
                "The target video is a photorealistic, seamless identity-replacement edit "
                "with strong temporal consistency."
            )
            for _s in _bun_slots:
                _bun_info = ref_map[_s]
                _src_slot_id = _bun_info.get("transfer_to_slot", "")
                _src_info = ref_map.get(_src_slot_id)
                if not _src_info:
                    continue
                _src_vnum = _src_info.get("video_num")
                _src_raw = _src_info.get("appearance_summary", "") or _src_info.get("name", "")
                _src_lower = (_src_raw[0].lower() + _src_raw[1:]).rstrip(". ") if _src_raw else ""
                # Strip leading article so we can wrap uniformly in "The …"
                _src_desc = re.sub(r"^(?:an? |the )", "", _src_lower).strip() if _src_lower else "replaced subject"
                _bun_label = _bun_info["subject_label"]
                if _src_vnum is not None:
                    dd.append(
                        f"The {_src_desc} in <Video {_src_vnum}> is completely replaced by {_bun_label}."
                    )
                else:
                    dd.append(f"The {_src_desc} is completely replaced by {_bun_label}.")

    seen_globally: set[str] = set()

    for i, shot in enumerate(template.get("shots", [])):
        shot_id = shot.get("id", f"shot_{i + 1}")
        timestamp = shot.get("timestamp")
        header = f"[Shot {i + 1}]" + (f" {timestamp}" if timestamp else "")
        dd.append("")
        dd.append(header)

        dialogue_entry = shot.get("dialogue")
        speaker_slot = ""
        text = ""
        if isinstance(dialogue_entry, dict):
            speaker_slot = dialogue_entry.get("speaker_slot", "")
            text = dialogue_map.get(shot_id, "")
            if not text and not dialogue_entry.get("placeholder"):
                text = dialogue_entry.get("default_text") or ""
        speaking_slots = frozenset([speaker_slot]) if speaker_slot else frozenset()

        camera = _replace_h3(shot.get("camera", ""), ref_map, speaking_slots, seen_globally)
        action = _replace_h3(shot.get("action", ""), ref_map, speaking_slots, seen_globally)
        if camera:
            dd.append(camera.rstrip(".") + ".")
        if action:
            pace_phrase = ""
            if text and isinstance(dialogue_entry, dict):
                pace = (dialogue_entry.get("speech_pace") or "normal").strip()
                pace_phrase = _PACE_PHRASES.get(pace, "")
            if pace_phrase:
                dd.append(action.rstrip(". ") + f", {pace_phrase}.")
            else:
                dd.append(action)

        if text:
            _spk_info = ref_map.get(speaker_slot, {})
            lang = _lang_label(_spk_info.get("language", "en-us") or "en-us")
            _spk_label = _spk_info.get("subject_label", "")
            _spk_id    = _spk_info.get("speaker_id", "")
            _spk_prefix = f"{_spk_label} ({_spk_id}) says: " if _spk_label and _spk_id else ""
            if use_dialogue_tags:
                dd.append(f"{_spk_prefix}<d>[{lang}] {text}</d>")
            else:
                dd.append(f'{_spk_prefix}"[{lang}] {text}"')

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
    use_dialogue_tags: bool = scene_instance.get("dialogue_tags", False)
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

        dialogue_entry = shot.get("dialogue")
        speaker_slot = ""
        text = ""
        if isinstance(dialogue_entry, dict):
            speaker_slot = dialogue_entry.get("speaker_slot", "")
            text = dialogue_map.get(shot_id, "")
            if not text and not dialogue_entry.get("placeholder"):
                text = dialogue_entry.get("default_text") or ""

        camera = _replace_named(shot.get("camera", ""), ref_map)
        action = _replace_named(shot.get("action", ""), ref_map)
        if camera:
            lines.append(camera.rstrip(".") + ".")
        if action:
            pace_phrase = ""
            if text and isinstance(dialogue_entry, dict):
                pace = (dialogue_entry.get("speech_pace") or "normal").strip()
                pace_phrase = _PACE_PHRASES.get(pace, "")
            if pace_phrase:
                lines.append(action.rstrip(". ") + f", {pace_phrase}.")
            else:
                lines.append(action)

        if text:
            lang = _lang_label(ref_map.get(speaker_slot, {}).get("language", "en-us") or "en-us")
            if use_dialogue_tags:
                lines.append(f"<d>[{lang}] {text}</d>")
            else:
                lines.append(f'"[{lang}] {text}"')

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
    video_slots = [s for s in ordered_slots if ref_map[s].get("video_num") is not None]
    video_files = [ref_map[s]["video_file"] for s in video_slots]
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
    video_detail = f"{', '.join(video_files)}" if video_files else "none"
    lines.append(
        f"  Video: {len(video_files)} file{'s' if len(video_files) != 1 else ''}"
        + (f" ({video_detail})" if video_files else "")
    )
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

def assemble_prompt(
    scene_instance: dict,
    model_type: str,
    video_entries: list[dict] | None = None,
) -> dict:
    """Assemble a model-specific prompt from a SCENE_INSTANCE dict.

    Args:
        scene_instance: dict from compose_scene() / SceneCompose node.
        model_type:     one of MODEL_TYPES.
        video_entries:  optional list of {subject_id, video_file} dicts from cast
                        resolution.  Each matching subject gets a <Video N> label
                        in H3 formats.  None or empty → no video references.

    Returns a dict with:
        prompt:                str — assembled prompt text
        concept_ids:           list[str] — concept IDs from assigned subjects
        reference_image_order: list[tuple[str, str]] — (slot_id, filename) in emit order
        video_slots:           list[str] — slot IDs with video refs, in slot order
        audio_slots:           list[str] — slot IDs with audio files, in slot order
        assembly_report:       str — human-readable summary
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Valid: {MODEL_TYPES}")

    ref_map = _build_ref_map(scene_instance, video_entries)

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

    video_slots = [s for s in ordered_slots if ref_map[s].get("video_num") is not None]
    audio_slots = [s for s in ordered_slots if ref_map[s]["audio_file"]]

    report = _build_assembly_report(scene_instance, ref_map, model_type)

    return {
        "prompt": prompt,
        "concept_ids": concept_ids,
        "reference_image_order": reference_image_order,
        "video_slots": video_slots,
        "audio_slots": audio_slots,
        "assembly_report": report,
    }


# ── Prompt Composition adapter ─────────────────────────────────────────────────

def assemble_composition(
    composition: dict,
    resolved_subjects: dict[str, dict],
    resolved_background: dict | None,
    model_type: str,
    video_entries: list[dict] | None = None,
    resolved_outfits: dict[str, dict] | None = None,
) -> dict:
    """Assemble a prompt from a PromptComposition dict.

    Converts composition format (S1/S2 slot keys, shots list, background dict)
    into the scene_instance format expected by assemble_prompt(), then delegates.

    Args:
        composition:        composition dict (from prompt_compositions.py)
        resolved_subjects:  {slot_key: subject_dict} already resolved by
                            prompt_compositions.resolve_subjects()
        resolved_background: background dict or None
        model_type:         one of MODEL_TYPES

    Returns same dict as assemble_prompt().
    """
    # Map S1→A, S2→B, … in stable order
    slot_keys = list(composition.get("subjects", {}).keys())
    slot_map = {sk: chr(ord("A") + i) for i, sk in enumerate(slot_keys)}

    slot_assignments = {}
    for sk, subject in resolved_subjects.items():
        letter = slot_map.get(sk)
        if letter:
            slot_assignments[letter] = subject

    # Apply composition-level slot descriptors: per-slot appearance summary override.
    # These sit on top of the profile's stored appearance.summary without modifying it.
    import copy as _copy
    slot_descriptors   = composition.get("slot_descriptors",   {})
    appearance_overrides = composition.get("appearance_overrides", {})
    for sk in slot_keys:
        letter = slot_map.get(sk)
        if not letter or letter not in slot_assignments:
            continue
        desc = slot_descriptors.get(sk, "").strip()
        field_overrides = appearance_overrides.get(sk) or {}
        if desc or field_overrides:
            slot_assignments[letter] = _copy.deepcopy(slot_assignments[letter])
            app = slot_assignments[letter].setdefault("appearance", {})
            if desc:
                app["summary"] = desc
            for field, value in field_overrides.items():
                if isinstance(value, str) and value.strip():
                    app[field] = value.strip()

    # Running letter index for extra slots (background + outfit references).
    _next_letter_idx = len(slot_keys)

    # Background as visual reference: inject as an additional <Subject N> slot.
    # Uses {BG} shortcut in shot action/camera fields.
    bg_letter = None
    if composition.get("background_as_reference") and resolved_background:
        ref_images = resolved_background.get("reference_images", [])
        if ref_images:
            bg_letter = chr(ord("A") + _next_letter_idx)
            _next_letter_idx += 1
            slot_map["BG"] = bg_letter
            bg_desc = resolved_background.get("description", "")
            bg_lighting = resolved_background.get("lighting", "")
            bg_appearance = (bg_desc.rstrip(". ") + ". " + bg_lighting).strip(". ") if bg_lighting else bg_desc
            slot_assignments[bg_letter] = {
                "name": resolved_background.get("name", "Background"),
                "appearance": {"summary": bg_appearance},
                "voice": {},
                "character_sheet_images": [
                    r if isinstance(r, dict) else {"file": r, "role": "scene reference"}
                    for r in ref_images
                ],
                "concept_id": "",
                "subject_id": "",
            }

    # Outfit reference subjects: each assigned outfit whose reference_images contain
    # at least one entry with use_as_reference=True becomes its own <Subject N> slot.
    # Slots iterate in subject order (S1 → S2 …) so Fit_1 always maps to the
    # earliest slot that has a reference outfit.  Text-only outfits (no flagged
    # images) contribute their description to outfit_overrides instead.
    fit_counter = 1
    if resolved_outfits:
        for sk in slot_keys:
            outfit = resolved_outfits.get(sk)
            if not outfit:
                continue
            ref_images = [
                r if isinstance(r, dict) else {"file": r, "role": "costume detail"}
                for r in outfit.get("reference_images", [])
                if isinstance(r, dict) and r.get("use_as_reference")
            ]
            if not ref_images:
                # Text-only: feed description into outfit_overrides unless a manual
                # override already exists for this slot.
                letter = slot_map.get(sk)
                if letter and outfit.get("description") and letter not in outfit_overrides:
                    outfit_overrides[letter] = outfit["description"]
                continue
            fit_key = f"Fit_{fit_counter}"
            fit_letter = chr(ord("A") + _next_letter_idx)
            _next_letter_idx += 1
            slot_map[fit_key] = fit_letter
            slot_assignments[fit_letter] = {
                "name": outfit.get("name", fit_key),
                "appearance": {"summary": outfit.get("description", "")},
                "voice": {},
                "character_sheet_images": ref_images,
                "concept_id": "",
                "subject_id": "",
            }
            fit_counter += 1

    outfit_overrides = {}
    for sk, override in composition.get("outfit_overrides", {}).items():
        letter = slot_map.get(sk)
        if letter and override:
            outfit_overrides[letter] = override

    # Build dialogue map keyed by the shot's own ID so the lookup in
    # assemble_prompt (which uses the template shot's id) finds the right text.
    dialogue: dict[str, str] = {}
    for shot in composition.get("shots", []):
        d = shot.get("dialogue")
        if d and d.get("text"):
            dialogue[shot["id"]] = d["text"]

    # Build a virtual template from the composition's shots and background
    background = resolved_background or {}
    style = composition.get("style", "")

    virtual_template = {
        "id":          composition.get("id", ""),
        "name":        composition.get("name", ""),
        "description": "",
        "slots": {
            **{
                slot_map[sk]: {"role": sk, "needs_voice": True, "needs_character_sheet": True}
                for sk in slot_keys if sk in slot_map
            },
            **(
                {bg_letter: {"role": "background", "needs_voice": False, "needs_character_sheet": True}}
                if bg_letter else {}
            ),
            **{
                slot_map[fk]: {"role": fk, "needs_voice": False, "needs_character_sheet": True}
                for fk in slot_map if fk.startswith("Fit_")
            },
        },
        "environment": {
            "summary":  background.get("description", ""),
            "lighting": background.get("lighting", ""),
        },
        "style": style,
        "shots": _composition_shots_to_template(composition.get("shots", []), slot_map),
        "overall_soundscape": (
            composition.get("overall_soundscape")
            or background.get("soundscape", "")
        ),
        "non_diegetic_music": composition.get("non_diegetic_music", "N/A"),
    }

    scene_instance: dict = {
        "template_id":      composition.get("id", ""),
        "template_name":    composition.get("name", ""),
        "template":         virtual_template,
        "slot_assignments": slot_assignments,
        "dialogue":         dialogue,
        "outfit_overrides": outfit_overrides,
        "dialogue_tags":    bool(composition.get("use_dialogue_tags", False)),
        "scene_synopsis":   _remap_slots(composition.get("scene_synopsis", ""), slot_map),
    }

    # Pass user-configured task flags into scene_instance for h3_ref2va
    composition_flags = composition.get("task_flags") or []
    if composition_flags:
        scene_instance["task_flags"] = composition_flags

    return assemble_prompt(scene_instance, model_type, video_entries)


def _remap_slots(text: str, slot_map: dict[str, str]) -> str:
    """Replace {S1}/{S2}/… composition slot keys with {A}/{B}/… template slot letters."""
    for sk, letter in slot_map.items():
        text = text.replace(f"{{{sk}}}", f"{{{letter}}}")
    return text


def _composition_shots_to_template(shots: list[dict], slot_map: dict[str, str]) -> list[dict]:
    """Convert composition shot dicts to the template shot format."""
    result = []
    for i, shot in enumerate(shots, 1):
        dlg = shot.get("dialogue") or {}
        has_dialogue = bool(dlg.get("text"))
        action = _remap_slots(shot.get("action", ""), slot_map)
        camera = _remap_slots(shot.get("camera", ""), slot_map)
        # Map the speaker slot key (S1 → A) so the h3 assembler can look up language
        template_dlg = None
        if has_dialogue:
            speaker_sk = dlg.get("speaker", "")
            template_dlg = {
                "placeholder": True,
                "speaker_slot": slot_map.get(speaker_sk, ""),
                "speech_pace": dlg.get("speech_pace") or "normal",
            }
        result.append({
            "id":           shot.get("id", f"shot_{i}"),
            "timestamp":    shot.get("timestamp"),
            "camera":       camera,
            "action":       action,
            "dialogue":     template_dlg,
            "sound_events": shot.get("sound_events"),
        })
    return result
