"""Tests for utils/source_profile_analysis.py — no VLM or ComfyUI required."""
import json
import os

import pytest
from conftest import import_test_module

spa = import_test_module("utils/source_profile_analysis.py")

build_prompt            = spa.build_prompt
_parse_vlm_json_response = spa._parse_vlm_json_response
load_history            = spa.load_history
history_for_profile     = spa.history_for_profile
append_history_entry    = spa.append_history_entry
PASS_TYPES              = spa.PASS_TYPES
PASS_ENTITY_DEFAULTS    = spa.PASS_ENTITY_DEFAULTS
_JSON_SCHEMA_INSTRUCTION = spa._JSON_SCHEMA_INSTRUCTION


# ── PASS_TYPES ─────────────────────────────────────────────────────────────────

def test_pass_types_contains_expected():
    for pt in ("people", "setting", "soundscape", "objects", "animals", "custom"):
        assert pt in PASS_TYPES


def test_pass_entity_defaults_covers_all_pass_types():
    for pt in PASS_TYPES:
        assert pt in PASS_ENTITY_DEFAULTS


# ── build_prompt ───────────────────────────────────────────────────────────────

def test_build_prompt_people_contains_schema():
    p = build_prompt("people")
    assert "subjects" in p
    assert "entity_type" in p


def test_build_prompt_setting_mentions_location():
    p = build_prompt("setting")
    assert "location" in p.lower()


def test_build_prompt_soundscape_mentions_audio():
    p = build_prompt("soundscape")
    assert "audio" in p.lower() or "sound" in p.lower()


def test_build_prompt_objects_mentions_props():
    p = build_prompt("objects")
    assert "prop" in p.lower() or "object" in p.lower()


def test_build_prompt_animals():
    p = build_prompt("animals")
    assert "animal" in p.lower()


def test_build_prompt_override_replaces_template():
    override = "My custom instruction about the scene."
    p = build_prompt("people", prompt_override=override)
    assert "My custom instruction" in p
    assert "person" not in p.split(_JSON_SCHEMA_INSTRUCTION)[0]


def test_build_prompt_override_appends_schema_if_missing():
    p = build_prompt("people", prompt_override="Find all subjects.")
    assert _JSON_SCHEMA_INSTRUCTION in p


def test_build_prompt_override_does_not_duplicate_schema():
    full = "Find subjects.\n\n" + _JSON_SCHEMA_INSTRUCTION
    p = build_prompt("people", prompt_override=full)
    assert p.count(_JSON_SCHEMA_INSTRUCTION) == 1


def test_build_prompt_unknown_pass_type_returns_fallback():
    p = build_prompt("nonexistent_type")
    assert "subjects" in p


# ── _parse_vlm_json_response ───────────────────────────────────────────────────

def _make_raw(subjects: list) -> str:
    return json.dumps({"subjects": subjects})


def test_parse_clean_json():
    raw = _make_raw([
        {"label": "woman left", "role_description": "A woman standing left.",
         "entity_type": "person", "notes": ""}
    ])
    result = _parse_vlm_json_response(raw, "people")
    assert len(result) == 1
    assert result[0]["label"] == "woman left"
    assert result[0]["entity_type"] == "person"


def test_parse_strips_code_fence():
    inner = _make_raw([{"label": "table", "role_description": "A table.",
                         "entity_type": "object", "notes": ""}])
    raw = f"```json\n{inner}\n```"
    result = _parse_vlm_json_response(raw, "objects")
    assert len(result) == 1
    assert result[0]["label"] == "table"


def test_parse_strips_code_fence_without_language_tag():
    inner = _make_raw([{"label": "cat", "role_description": "A cat.",
                         "entity_type": "animal", "notes": ""}])
    raw = f"```\n{inner}\n```"
    result = _parse_vlm_json_response(raw, "animals")
    assert len(result) == 1


def test_parse_empty_subjects_returns_empty_list():
    raw = json.dumps({"subjects": []})
    assert _parse_vlm_json_response(raw, "people") == []


def test_parse_invalid_json_returns_empty():
    assert _parse_vlm_json_response("not json at all", "people") == []


def test_parse_json_with_leading_prose():
    inner = _make_raw([{"label": "chair", "role_description": "A wooden chair.",
                         "entity_type": "object", "notes": ""}])
    raw = f"Here is the result you asked for:\n{inner}"
    result = _parse_vlm_json_response(raw, "objects")
    assert len(result) == 1


def test_parse_missing_label_skipped():
    raw = _make_raw([
        {"label": "", "role_description": "No label.", "entity_type": "person", "notes": ""},
        {"label": "valid", "role_description": "Has label.", "entity_type": "person", "notes": ""},
    ])
    result = _parse_vlm_json_response(raw, "people")
    assert len(result) == 1
    assert result[0]["label"] == "valid"


def test_parse_invalid_entity_type_replaced_by_default():
    raw = _make_raw([
        {"label": "thing", "role_description": "Something.", "entity_type": "INVALID", "notes": ""}
    ])
    result = _parse_vlm_json_response(raw, "setting")
    assert result[0]["entity_type"] == PASS_ENTITY_DEFAULTS["setting"]


def test_parse_missing_entity_type_uses_pass_default():
    raw = _make_raw([{"label": "sofa", "role_description": "A sofa."}])
    result = _parse_vlm_json_response(raw, "objects")
    assert result[0]["entity_type"] == "object"


def test_parse_missing_notes_defaults_empty_string():
    raw = _make_raw([{"label": "dog", "role_description": "A dog."}])
    result = _parse_vlm_json_response(raw, "animals")
    assert result[0]["notes"] == ""


def test_parse_missing_role_description_defaults_empty_string():
    raw = _make_raw([{"label": "lamp", "entity_type": "object"}])
    result = _parse_vlm_json_response(raw, "objects")
    assert result[0]["role_description"] == ""


def test_parse_multiple_subjects():
    raw = _make_raw([
        {"label": "A", "role_description": "First.", "entity_type": "person", "notes": ""},
        {"label": "B", "role_description": "Second.", "entity_type": "person", "notes": ""},
        {"label": "C", "role_description": "Third.", "entity_type": "person", "notes": ""},
    ])
    assert len(_parse_vlm_json_response(raw, "people")) == 3


def test_parse_non_dict_items_skipped():
    raw = json.dumps({"subjects": [
        "not a dict",
        {"label": "valid", "role_description": "Ok.", "entity_type": "object", "notes": ""},
    ]})
    result = _parse_vlm_json_response(raw)
    assert len(result) == 1


def test_parse_subjects_not_a_list_returns_empty():
    raw = json.dumps({"subjects": "not a list"})
    assert _parse_vlm_json_response(raw) == []


def test_parse_top_level_not_dict_returns_empty():
    assert _parse_vlm_json_response(json.dumps([1, 2, 3])) == []


# ── History ────────────────────────────────────────────────────────────────────

def test_load_history_missing_file_returns_empty(tmp_path):
    assert load_history(str(tmp_path)) == []


def test_append_and_load_history(tmp_path):
    d = str(tmp_path)
    entry = append_history_entry(
        d, "prof1", "video.mp4", "people", "prompt text",
        [{"label": "A", "role_description": "desc", "entity_type": "person", "notes": ""}],
        backup=False,
    )
    assert entry["profile_id"] == "prof1"
    entries = load_history(d)
    assert len(entries) == 1
    assert entries[0]["pass_type"] == "people"
    assert entries[0]["candidates"][0]["label"] == "A"


def test_append_multiple_entries(tmp_path):
    d = str(tmp_path)
    for i in range(3):
        append_history_entry(d, "p1", "v.mp4", "people", "prompt", [], backup=False)
    assert len(load_history(d)) == 3


def test_history_for_profile_filters_by_id(tmp_path):
    d = str(tmp_path)
    append_history_entry(d, "p1", "v.mp4", "people", "p", [], backup=False)
    append_history_entry(d, "p2", "v.mp4", "setting", "p", [], backup=False)
    append_history_entry(d, "p1", "v.mp4", "objects", "p", [], backup=False)

    p1_entries = history_for_profile(d, "p1")
    assert len(p1_entries) == 2
    assert all(e["profile_id"] == "p1" for e in p1_entries)


def test_history_for_profile_newest_first(tmp_path):
    d = str(tmp_path)
    append_history_entry(d, "p1", "v.mp4", "people",  "p", [], backup=False)
    append_history_entry(d, "p1", "v.mp4", "objects", "p", [], backup=False)
    entries = history_for_profile(d, "p1")
    assert entries[0]["pass_type"] == "objects"
    assert entries[1]["pass_type"] == "people"


def test_history_for_unknown_profile_returns_empty(tmp_path):
    d = str(tmp_path)
    append_history_entry(d, "p1", "v.mp4", "people", "p", [], backup=False)
    assert history_for_profile(d, "unknown") == []


def test_append_creates_backup(tmp_path):
    d = str(tmp_path)
    append_history_entry(d, "p1", "v.mp4", "people", "p", [], backup=False)
    append_history_entry(d, "p1", "v.mp4", "setting", "p", [], backup=True)
    bak = os.path.join(d, "source_profile_analysis_history.json.bak")
    assert os.path.exists(bak)


def test_append_entry_has_timestamp(tmp_path):
    entry = append_history_entry(
        str(tmp_path), "p1", "v.mp4", "people", "p", [], backup=False
    )
    assert "T" in entry["timestamp"]


def test_append_does_not_mutate_candidates(tmp_path):
    cands = [{"label": "x", "role_description": "d", "entity_type": "object", "notes": ""}]
    append_history_entry(str(tmp_path), "p1", "v.mp4", "objects", "p", cands, backup=False)
    cands[0]["label"] = "mutated"
    entries = load_history(str(tmp_path))
    assert entries[0]["candidates"][0]["label"] == "x"


# ── _parse_segments_response ───────────────────────────────────────────────────

_parse_segments = spa._parse_segments_response
_parse_clip_desc = spa.parse_clip_description_response


def _seg_json(segments):
    return json.dumps({"segments": segments})


def test_parse_segments_basic():
    raw = _seg_json([
        {"start_time": 0.0, "end_time": 10.0, "label": "Intro", "action": "They walk in"},
        {"start_time": 10.0, "end_time": 20.0, "label": "Chat", "action": "They sit"},
    ])
    segs = _parse_segments(raw)
    assert len(segs) == 2
    assert segs[0]["label"] == "Intro"
    assert segs[1]["start_time"] == 10.0

def test_parse_segments_discards_out_of_order():
    raw = _seg_json([
        {"start_time": 0.0, "end_time": 10.0, "label": "A", "action": ""},
        {"start_time": 5.0, "end_time": 15.0, "label": "B", "action": ""},
    ])
    segs = _parse_segments(raw)
    assert len(segs) == 1
    assert segs[0]["label"] == "A"

def test_parse_segments_discards_zero_length():
    raw = _seg_json([
        {"start_time": 5.0, "end_time": 5.0, "label": "Zero", "action": ""},
        {"start_time": 5.0, "end_time": 15.0, "label": "Real", "action": "ok"},
    ])
    segs = _parse_segments(raw)
    assert len(segs) == 1
    assert segs[0]["label"] == "Real"

def test_parse_segments_fallback_when_empty_with_duration():
    segs = _parse_segments("{}", video_duration=30.0)
    assert len(segs) == 1
    assert segs[0]["start_time"] == 0.0
    assert segs[0]["end_time"] == 30.0

def test_parse_segments_no_fallback_when_no_duration():
    segs = _parse_segments("{}")
    assert segs == []

def test_parse_segments_strips_code_fences():
    inner = json.dumps({"segments": [
        {"start_time": 0.0, "end_time": 5.0, "label": "S", "action": "a"}
    ]})
    raw = f"```json\n{inner}\n```"
    segs = _parse_segments(raw)
    assert len(segs) == 1

def test_parse_segments_rounds_times():
    raw = _seg_json([{"start_time": 0.0, "end_time": 9.9999999, "label": "x", "action": ""}])
    segs = _parse_segments(raw)
    assert segs[0]["end_time"] == 10.0

def test_parse_segments_handles_invalid_json():
    segs = _parse_segments("not json at all", video_duration=5.0)
    assert len(segs) == 1
    assert segs[0]["end_time"] == 5.0

def test_parse_segments_default_label_when_missing():
    raw = _seg_json([{"start_time": 0.0, "end_time": 10.0, "action": "run"}])
    segs = _parse_segments(raw)
    assert "Segment" in segs[0]["label"]


# ── parse_clip_description_response ───────────────────────────────────────────

def test_parse_clip_desc_extracts_action():
    raw = json.dumps({"action": "Two people shake hands."})
    assert _parse_clip_desc(raw) == "Two people shake hands."

def test_parse_clip_desc_strips_fences():
    inner = json.dumps({"action": "A car pulls up."})
    raw = f"```json\n{inner}\n```"
    assert _parse_clip_desc(raw) == "A car pulls up."

def test_parse_clip_desc_fallback_to_raw():
    raw = "They are talking and laughing."
    result = _parse_clip_desc(raw)
    assert result == raw

def test_parse_clip_desc_truncates_long_fallback():
    raw = "x" * 400
    result = _parse_clip_desc(raw)
    assert len(result) <= 300

def test_parse_clip_desc_empty_action_field():
    raw = json.dumps({"action": ""})
    assert _parse_clip_desc(raw) == ""


# ── build_clip_description_prompt ─────────────────────────────────────────────

_build_clip_prompt = spa.build_clip_description_prompt


def test_build_clip_prompt_no_subjects_returns_base_prompt():
    prompt = _build_clip_prompt()
    assert "Examine this video frame" in prompt
    assert "Return ONLY valid JSON" in prompt
    # No subject placeholders injected
    assert "{A}" not in prompt


def test_build_clip_prompt_with_subjects_includes_placeholder_instructions():
    subjects = [("A", "Elena", "woman with auburn hair"), ("B", "Marcus", "tall man in dark jacket")]
    prompt = _build_clip_prompt(subjects=subjects)
    assert "{A} — Elena: woman with auburn hair" in prompt
    assert "{B} — Marcus: tall man in dark jacket" in prompt
    assert "Use {A} / {B} to refer to these subjects" in prompt
    assert "not the subjects' appearance" in prompt


def test_build_clip_prompt_with_subjects_uses_slot_schema():
    subjects = [("A", "Anna", "")]
    prompt = _build_clip_prompt(subjects=subjects)
    # Schema should reference placeholder labels, not generic description
    assert "placeholder labels" in prompt


def test_build_clip_prompt_single_subject_no_slash():
    subjects = [("A", "Sam", "person in red coat")]
    prompt = _build_clip_prompt(subjects=subjects)
    assert "Use {A} to refer to" in prompt
    # No orphaned " / " separator for a single subject
    assert " / {" not in prompt


def test_build_clip_prompt_subject_missing_appearance():
    subjects = [("A", "Person", "")]
    prompt = _build_clip_prompt(subjects=subjects)
    assert "{A} — Person" in prompt


def test_build_clip_prompt_override_includes_subjects():
    override = "Just tell me the vibe."
    subjects = [("A", "Elena", "auburn hair")]
    prompt = _build_clip_prompt(prompt_override=override, subjects=subjects)
    # Override replaces only the base preamble; subjects block is still injected
    assert "Just tell me the vibe" in prompt
    assert "Elena" in prompt
    assert "{A}" in prompt
    assert "action" in prompt  # schema still appended
