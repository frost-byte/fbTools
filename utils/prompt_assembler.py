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

    # Build subject_id → video_entry lookup (first match wins)
    video_lookup: dict[str, dict] = {}
    for ve in (video_entries or []):
        sid = ve.get("subject_id", "")
        if sid and sid not in video_lookup:
            video_lookup[sid] = ve

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

    ref_map: dict[str, dict] = {}
    subject_counter = 1
    picture_counter = 1
    video_counter = 1

    for slot_id in ordered_slots:
        subject = assignments.get(slot_id)
        if subject is None:
            continue
        appearance = subject.get("appearance", {})
        voice = subject.get("voice", {})
        sheets = subject.get("character_sheet_images", [])
        audio_file = voice.get("audio_reference_file", "")
        outfit = outfit_overrides.get(slot_id) or appearance.get("default_outfit", "")
        subject_id = subject.get("subject_id", "")

        picture_nums = list(range(picture_counter, picture_counter + len(sheets)))
        picture_counter += len(sheets)

        ve = video_lookup.get(subject_id) if subject_id else None
        video_num: int | None = None
        video_file: str = ""
        soundtrack_num: int | None = None
        soundtrack_retention: str = "timbre"
        soundtrack_role: str = ""
        if ve:
            video_num = video_counter
            video_file = ve.get("video_file", "")
            video_counter += 1
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
            "character_sheet_images": list(sheets),
            "concept_id": subject.get("concept_id", ""),
            "subject_num": subject_counter,
            "speaker_id": f"S{subject_counter}",
            "subject_label": f"<Subject {subject_counter}>",
            "picture_nums": picture_nums,
            "video_num": video_num,
            "video_file": video_file,
            "audio_num": audio_num,
            "soundtrack_num": soundtrack_num,
            "soundtrack_retention": soundtrack_retention,
            "soundtrack_role": soundtrack_role,
            "retention_marker": retention_markers.get(slot_id, "fully_preserved"),
        }
        subject_counter += 1

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
        for path in sheets:
            references.append({
                "modality":       "image",
                "picture_ordinal": picture_counter,
                "subject_id":     subject_id,
                "slot_id":        slot_id,
                "path":           path,
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
        this_video_ordinal = video_counter

        if has_soundtrack:
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

        references.append({
            "modality":     "video",
            "video_ordinal": this_video_ordinal,
            "subject_id":   subject_id,
            "slot_id":      slot_id,
            "path":         ve.get("video_file", ""),
            "load_params":  ve.get("load_params", {}),
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
        label = info["subject_label"]
        base = f"{label} ({info['speaker_id']})" if slot_id in speaking_slots else label
        if seen_globally is not None and slot_id not in seen_globally:
            seen_globally.add(slot_id)
            ap = info.get("appearance_phrase", "")
            if ap:
                return f"{base}, {ap}"
        return base

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

    # Resolve task flags early — needed in both subject_definitions and summary
    user_flags: list[str] = scene_instance.get("task_flags") or []
    active_flags: set[str] = set(user_flags)

    # ── subject_definitions ────────────────────────────────────────────────────
    # Format (per MiniMax best practice):
    #   Subject lines:  "<Subject N> is [summary] from <Video N> / from the
    #                   character sheet(s) contained in <Picture N>, with [details]."
    #   Video lines:    "<Video N> is the continuation starting point…" (task-aware)
    #   Audio lines:    "<Audio N> is the voice-timbre reference for <Subject N> (SN)…"
    sd: list[str] = []
    video_sd_lines: list[str] = []
    audio_sd_lines: list[str] = []

    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        label = info["subject_label"]
        summary = info["appearance_summary"] or info["name"]

        # Build reference anchor — placed AFTER summary per H3 spec:
        #   "<Subject N> is [description] in <Video N>, with [details]."
        #   "<Subject N> is [description] in <Picture N>, with [details]."
        # Both video and picture references use "in" (per official doc examples).
        vid_ref = f"<Video {info['video_num']}>" if info["video_num"] is not None else ""
        pic_ref_str = _join_labels([f"<Picture {p}>" for p in info["picture_nums"]]) if info["picture_nums"] else ""

        if vid_ref and pic_ref_str:
            ref_anchor = f" from {vid_ref} and {pic_ref_str}"
        elif vid_ref:
            ref_anchor = f" from {vid_ref}"
        elif pic_ref_str:
            ref_anchor = f" from {pic_ref_str}"
        else:
            ref_anchor = ""

        # Appearance details as flowing prose (from separate structured fields)
        detail_parts = [
            info[f] for f in ("hair", "face", "body", "outfit") if info.get(f)
        ]
        detail_phrase = f", with {_join_details(detail_parts)}" if detail_parts else ""

        # Summary starts with lowercase after "is"; trailing period stripped since
        # detail_phrase or ref_anchor continues the sentence.
        summary_body = (summary[0].lower() + summary[1:]).rstrip(". ") if summary else ""
        # When a specific reference is cited, the subject is definite — replace a
        # leading indefinite article ("a "/"an ") with "the " to match spec examples
        # ("is the young blonde woman in <Video 1>…").  Leaves "the", names, and
        # article-free summaries untouched.
        if ref_anchor and summary_body:
            summary_body = re.sub(r"^an? ", "the ", summary_body, count=1)
        sd.append(f"{label} is {summary_body}{ref_anchor}{detail_phrase}.")

        # Standalone video role line (task-flag-aware)
        if info["video_num"] is not None:
            vnum = info["video_num"]
            if "video continuation" in active_flags:
                video_role = "is the continuation starting point for the target video"
            elif "video editing" in active_flags:
                video_role = "is the source video being edited"
            else:
                video_role = f"is the visual identity reference for {label}"
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
            elif snd_ret == "reuse":
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

    # User-provided task_flags override auto-detection; otherwise infer from refs.
    # Per MiniMax docs, pictures AND videos that provide character/style/camera guidance
    # both fall under "reference generation".  Intent-based types (video editing,
    # video continuation, keyframe completion, audio reuse) cannot be auto-detected
    # from file presence — users must supply them via task_flags.
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

    if user_flags:
        task_tag = "[" + " + ".join(_sort_tasks(user_flags)) + "]"
    else:
        task_parts: list[str] = []
        if has_refs:
            task_parts.append("reference generation")
        if has_audio:
            task_parts.append("audio reference")
        task_tag = "[" + " + ".join(task_parts) + "]" if task_parts else "[video generation]"

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

    # Subject entries: "<Subject N> (appears in [Shot N], [Shot M]): fully_preserved - ..."
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        label = info["subject_label"]

        appears_list = slot_appearances[slot_id]
        appears_clause = (
            "appears in " + ", ".join(appears_list) if appears_list else "appears throughout"
        )

        # Build the same appearance phrase used in subject_definitions (minus the ref anchor).
        summary = info["appearance_summary"]
        summary_body = (summary[0].lower() + summary[1:]).rstrip(". ") if summary else ""
        # Mirror the definite-article substitution: if the subject has any visual reference
        # the subject_definitions line already says "the …", so match it here.
        has_ref = info["video_num"] is not None or bool(info["picture_nums"])
        if has_ref and summary_body:
            summary_body = re.sub(r"^an? ", "the ", summary_body, count=1)
        detail_parts = [f for f in (info.get("hair", ""), info.get("face", ""), info.get("body", "")) if f]
        if info["outfit"]:
            detail_parts.append(f"wearing {info['outfit']}")
        detail_phrase = f", with {_join_details(detail_parts)}" if detail_parts else ""
        preserve_desc = f"{summary_body}{detail_phrase}" if summary_body else "appearance retained"

        subj_retention = info.get("retention_marker", "fully_preserved")
        ra.append(f"{label} ({appears_clause}): {subj_retention} - {preserve_desc}.")

    # Video entries: "<Video N> (role): fully_preserved - ..."
    for slot_id in ordered_slots:
        info = ref_map[slot_id]
        if info["video_num"] is not None:
            vnum = info["video_num"]
            label = info["subject_label"]
            if "video continuation" in active_flags:
                role_clause = "continuation starting point"
                preserve_desc = (
                    f"<Video {vnum}> is the continuation starting point; the target video "
                    f"begins where <Video {vnum}> ends, maintaining consistent motion and scene state"
                )
            elif "video editing" in active_flags:
                role_clause = "source video for editing"
                preserve_desc = (
                    f"<Video {vnum}> is the source material being edited; "
                    f"the cut structure and pacing serve as the primary reference"
                )
            else:
                role_clause = f"visual identity of {label}"
                preserve_desc = (
                    f"<Video {vnum}> defines the visual identity of {label}; "
                    f"the subject's appearance in the target video must fully match "
                    f"the person shown in <Video {vnum}>"
                )
            ra.append(f"<Video {vnum}> ({role_clause}): fully_preserved - {preserve_desc}.")

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
            lang = _lang_label(ref_map.get(speaker_slot, {}).get("language", "en-us") or "en-us")
            if use_dialogue_tags:
                dd.append(f"<d>[{lang}] {text}</d>")
            else:
                dd.append(f'"[{lang}] {text}"')

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
            slot_map[sk]: {"role": sk, "needs_voice": True, "needs_character_sheet": True}
            for sk in slot_keys if sk in slot_map
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
