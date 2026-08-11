"""Tests for the Prompt Assembly system."""

import pytest
from conftest import import_test_module

pa = import_test_module("utils/prompt_assembler.py")

assemble_prompt = pa.assemble_prompt
MODEL_TYPES = pa.MODEL_TYPES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_subject(
    name, subject_id="", concept_id="",
    sheets=None, audio="",
    summary="", face="", hair="", body="", outfit="",
    language="en-us",
):
    return {
        "subject_id": subject_id or name.lower().replace(" ", "_"),
        "name": name,
        "appearance": {
            "summary": summary or f"a person named {name}",
            "face": face,
            "hair": hair,
            "body": body,
            "default_outfit": outfit,
        },
        "voice": {
            "description": f"{name}'s voice",
            "audio_reference_file": audio,
            "language": language,
        },
        "character_sheet_images": sheets or [],
        "concept_id": concept_id,
    }


def _make_template(
    template_id="cafe_2p",
    name="Café Conversation",
    n_slots=2,
    n_shots=3,
    soundscape="Soft café ambience.",
    music="N/A",
):
    slot_ids = list("ABCD")[:n_slots]
    slots = {
        sid: {"role": f"speaker {sid}", "needs_voice": True, "needs_character_sheet": True}
        for sid in slot_ids
    }
    shots = []
    for i, sid in enumerate(slot_ids):
        shots.append({
            "id": f"shot_{i + 1}",
            "timestamp": None if i == 0 else f"00:0{i * 4}.000",
            "camera": f"Close-up of {{{sid}}}",
            "action": f"{{{sid}}} speaks.",
            "dialogue": {"speaker_slot": sid, "placeholder": True, "default_text": None},
            "sound_events": None,
        })
    if n_shots > n_slots:
        shots.append({
            "id": f"shot_{n_shots}",
            "timestamp": f"00:{n_slots * 4:02d}.000",
            "camera": "Wide shot",
            "action": "Scene ends.",
            "dialogue": None,
            "sound_events": "ambient fade",
        })
    return {
        "version": 1,
        "id": template_id,
        "name": name,
        "description": "A test template",
        "slots": slots,
        "environment": {"summary": "a warm sunlit café", "lighting": "window light from left"},
        "style": "cinematic with shallow depth of field",
        "shots": shots,
        "overall_soundscape": soundscape,
        "non_diegetic_music": music,
    }


def _make_scene(
    template=None,
    slot_A=None, slot_B=None, slot_C=None,
    dialogue=None,
    outfit_overrides=None,
):
    if template is None:
        template = _make_template()
    assignments = {}
    if slot_A is not None:
        assignments["A"] = slot_A
    if slot_B is not None:
        assignments["B"] = slot_B
    if slot_C is not None:
        assignments["C"] = slot_C
    dialogue_map = {}
    for i, text in enumerate(dialogue or []):
        if text.strip():
            dialogue_map[f"shot_{i + 1}"] = text
    return {
        "template_id": template["id"],
        "template_name": template["name"],
        "template": template,
        "slot_assignments": assignments,
        "dialogue": dialogue_map,
        "outfit_overrides": outfit_overrides or {},
    }


# ── MODEL_TYPES ────────────────────────────────────────────────────────────────

def test_model_types_list():
    assert "h3_ref2va" in MODEL_TYPES
    assert "h3_fl2va" in MODEL_TYPES
    assert "wan22" in MODEL_TYPES
    assert "bernini" in MODEL_TYPES
    assert "ltx23" in MODEL_TYPES
    assert "flux2" in MODEL_TYPES
    assert "krea2" in MODEL_TYPES
    assert "qwen" in MODEL_TYPES


def test_invalid_model_type_raises():
    scene = _make_scene(slot_A=_make_subject("Alice"))
    with pytest.raises(ValueError, match="Unknown model_type"):
        assemble_prompt(scene, "bad_model")


# ── Reference map (_build_ref_map via assemble_prompt) ────────────────────────

def test_subject_numbers_in_slot_order():
    alice = _make_subject("Alice", sheets=["a.png"], audio="a.wav")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    prompt = result["prompt"]
    # Subject 1 is Alice (slot A), Subject 2 is Bob (slot B)
    assert "<Subject 1> [Alice]" in prompt
    assert "<Subject 2> [Bob]" in prompt


def test_picture_numbers_continuous_across_slots():
    alice = _make_subject("Alice", sheets=["a1.png", "a2.png"])
    bob = _make_subject("Bob", sheets=["b1.png"])
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    prompt = result["prompt"]
    # Alice gets Pictures 1 and 2, Bob gets Picture 3
    assert "<Picture 1>" in prompt
    assert "<Picture 2>" in prompt
    assert "<Picture 3>" in prompt


def test_audio_numbers_for_slots_with_audio():
    alice = _make_subject("Alice", audio="alice.wav")
    bob = _make_subject("Bob")  # no audio
    carol = _make_subject("Carol")
    carol["voice"]["audio_reference_file"] = "carol.wav"
    template = _make_template(n_slots=3)
    scene = _make_scene(template=template, slot_A=alice, slot_B=bob, slot_C=carol)
    result = assemble_prompt(scene, "h3_ref2va")
    prompt = result["prompt"]
    # Alice gets Audio 1, Carol gets Audio 2 (Bob skipped)
    assert "<Audio 1>" in prompt
    assert "<Audio 2>" in prompt


def test_no_sheets_no_picture_labels():
    alice = _make_subject("Alice")  # no sheets
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "<Picture" not in result["prompt"]


def test_no_audio_no_audio_labels():
    alice = _make_subject("Alice")  # no audio
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "<Audio" not in result["prompt"]


# ── reference_image_order ──────────────────────────────────────────────────────

def test_reference_image_order_slot_a_first():
    alice = _make_subject("Alice", sheets=["a1.png", "a2.png"])
    bob = _make_subject("Bob", sheets=["b1.png"])
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    order = result["reference_image_order"]
    assert order[0] == ("A", "a1.png")
    assert order[1] == ("A", "a2.png")
    assert order[2] == ("B", "b1.png")


def test_reference_image_order_no_sheets():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["reference_image_order"] == []


def test_reference_image_order_same_for_all_models():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    orders = [assemble_prompt(scene, mt)["reference_image_order"] for mt in MODEL_TYPES]
    assert all(o == [("A", "a.png")] for o in orders)


# ── audio_slots ────────────────────────────────────────────────────────────────

def test_audio_slots_returns_slots_with_audio():
    alice = _make_subject("Alice", audio="alice.wav")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["audio_slots"] == ["A"]


def test_audio_slots_both_have_audio():
    alice = _make_subject("Alice", audio="alice.wav")
    bob = _make_subject("Bob", audio="bob.wav")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["audio_slots"] == ["A", "B"]


def test_audio_slots_no_audio():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["audio_slots"] == []


# ── concept_ids ────────────────────────────────────────────────────────────────

def test_concept_ids_extracted_from_subjects():
    alice = _make_subject("Alice", concept_id="char_alice")
    bob = _make_subject("Bob", concept_id="char_bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["concept_ids"] == ["char_alice", "char_bob"]


def test_concept_ids_empty_for_no_concept():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["concept_ids"] == []


def test_concept_ids_skips_blank():
    alice = _make_subject("Alice", concept_id="char_alice")
    bob = _make_subject("Bob", concept_id="")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["concept_ids"] == ["char_alice"]


# ── H3 Ref2VA sections ────────────────────────────────────────────────────────

def test_h3_ref2va_has_six_section_headers():
    alice = _make_subject("Alice", sheets=["a.png"], audio="a.wav")
    bob = _make_subject("Bob", sheets=["b.png"], audio="b.wav")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Hello.", "Hi."])
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "subject_definitions:" in prompt
    assert "summary:" in prompt
    assert "retention_analysis:" in prompt
    assert "detailed_description:" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt


def test_h3_ref2va_subject_definitions_lists_subjects():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<Subject 1> [Alice]: a tall woman with red hair" in prompt


def test_h3_ref2va_subject_definitions_lists_pictures_inline():
    alice = _make_subject("Alice", sheets=["sheet1.png", "sheet2.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # Character sheets are cited inline within the subject block
    assert "Character sheets:" in prompt
    assert "<Picture 1> (primary identity)" in prompt
    assert "<Picture 2> (additional)" in prompt
    # No standalone <Picture N>: entries (each sheet's label appears only inline)
    assert "<Picture 1>: character sheet" not in prompt


def test_h3_ref2va_subject_definitions_lists_audio():
    alice = _make_subject("Alice", audio="alice.wav")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<Audio 1>:" in prompt
    assert "(S1)" in prompt


def test_h3_ref2va_retention_analysis_fully_preserved():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<Subject 1> (Alice): fully_preserved" in prompt


def test_h3_ref2va_retention_analysis_audio_not_fully_copy():
    alice = _make_subject("Alice", audio="a.wav")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "not fully_copy" in prompt


def test_h3_ref2va_dialogue_tags():
    alice = _make_subject("Alice", audio="a.wav", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Hello.", "Hi there."])
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<d>[en-us] Hello.</d>" in prompt
    assert "<d>[en-us] Hi there.</d>" in prompt


def test_h3_ref2va_shot_headers():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "[Shot 1]" in prompt
    assert "[Shot 2]" in prompt


def test_h3_ref2va_first_appearance_includes_description():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # At first mention of Alice in detailed_description, include appearance
    assert "Alice — a tall woman with red hair" in prompt


def test_h3_ref2va_soundscape_from_template():
    alice = _make_subject("Alice")
    template = _make_template(soundscape="Busy street noise.")
    scene = _make_scene(template=template, slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "Busy street noise." in prompt


def test_h3_ref2va_style_in_detailed_description():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "cinematic with shallow depth of field" in prompt.lower()


# ── H3 FL2VA ──────────────────────────────────────────────────────────────────

def test_h3_fl2va_no_subject_definitions_section():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "subject_definitions:" not in prompt


def test_h3_fl2va_no_reference_labels():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "<Subject" not in prompt
    assert "<Picture" not in prompt
    assert "<Audio" not in prompt


def test_h3_fl2va_has_shot_headers():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "[Shot 1]" in prompt
    assert "[Shot 2]" in prompt


def test_h3_fl2va_uses_subject_names():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "Alice" in prompt
    assert "<Subject" not in prompt


def test_h3_fl2va_dialogue_tags():
    alice = _make_subject("Alice", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Good morning.", "Morning."])
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "<d>[en-us] Good morning.</d>" in prompt
    assert "<d>[en-us] Morning.</d>" in prompt


# ── Wan 2.2 / BerniniR ────────────────────────────────────────────────────────

def test_wan22_task_line_reference_generation():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "wan22")["prompt"]
    assert "[reference image generation]" in prompt


def test_wan22_task_line_video_generation_when_no_sheets():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "wan22")["prompt"]
    assert "[video generation]" in prompt


def test_bernini_format_matches_wan22():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    p_wan = assemble_prompt(scene, "wan22")["prompt"]
    p_ber = assemble_prompt(scene, "bernini")["prompt"]
    # Same format
    assert "[reference image generation]" in p_wan
    assert "[reference image generation]" in p_ber


def test_wan22_includes_subject_description():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "wan22")["prompt"]
    assert "a tall woman with red hair" in prompt


def test_wan22_no_h3_labels():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "wan22")["prompt"]
    assert "<Subject" not in prompt
    assert "<Picture" not in prompt
    assert "<Audio" not in prompt


# ── Simple formats ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_type", ["ltx23", "flux2", "krea2", "qwen"])
def test_simple_formats_include_subject_name(model_type):
    alice = _make_subject("Alice", summary="a tall woman")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, model_type)["prompt"]
    assert "Alice" in prompt


@pytest.mark.parametrize("model_type", ["ltx23", "flux2", "krea2", "qwen"])
def test_simple_formats_no_h3_labels(model_type):
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, model_type)["prompt"]
    assert "<Subject" not in prompt
    assert "<Picture" not in prompt


@pytest.mark.parametrize("model_type", ["ltx23", "flux2", "krea2", "qwen"])
def test_simple_formats_include_environment(model_type):
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, model_type)["prompt"]
    assert "café" in prompt.lower()


# ── Outfit overrides ──────────────────────────────────────────────────────────

def test_outfit_override_appears_in_h3_ref2va():
    alice = _make_subject("Alice", outfit="navy blazer")
    scene = _make_scene(slot_A=alice, outfit_overrides={"A": "red dress"})
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "red dress" in prompt


def test_outfit_override_replaces_default_in_report():
    alice = _make_subject("Alice", outfit="navy blazer")
    scene = _make_scene(slot_A=alice, outfit_overrides={"A": "red dress"})
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "red dress" in report
    assert "navy blazer" in report


def test_outfit_override_appears_in_wan22():
    alice = _make_subject("Alice", outfit="navy blazer", summary="a tall woman")
    scene = _make_scene(slot_A=alice, outfit_overrides={"A": "red dress"})
    prompt = assemble_prompt(scene, "wan22")["prompt"]
    assert "red dress" in prompt


# ── Empty / edge cases ────────────────────────────────────────────────────────

def test_empty_scene_no_slots():
    template = _make_template()
    scene = {
        "template_id": template["id"],
        "template_name": template["name"],
        "template": template,
        "slot_assignments": {},
        "dialogue": {},
        "outfit_overrides": {},
    }
    # Should not crash
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["prompt"]
    assert result["concept_ids"] == []
    assert result["reference_image_order"] == []
    assert result["audio_slots"] == []


def test_single_subject_no_media():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "Alice" in result["prompt"]
    assert result["reference_image_order"] == []
    assert result["audio_slots"] == []


def test_no_dialogue_still_assembles():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    result = assemble_prompt(scene, "h3_ref2va")
    assert result["prompt"]
    assert "<d>" not in result["prompt"]


# ── Assembly report ────────────────────────────────────────────────────────────

def test_report_includes_scene_name():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "Café Conversation" in report


def test_report_includes_model_type():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    for mt in MODEL_TYPES:
        report = assemble_prompt(scene, mt)["assembly_report"]
        assert mt in report


def test_report_includes_speaker_id():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "S1" in report
    assert "S2" in report


def test_report_shows_image_count():
    alice = _make_subject("Alice", sheets=["a1.png", "a2.png"])
    bob = _make_subject("Bob", sheets=["b1.png"])
    scene = _make_scene(slot_A=alice, slot_B=bob)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "3 total" in report


def test_report_shows_audio_filenames():
    alice = _make_subject("Alice", audio="alice_voice.wav")
    scene = _make_scene(slot_A=alice)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "alice_voice.wav" in report


def test_report_no_outfit_overrides_says_none():
    alice = _make_subject("Alice")
    scene = _make_scene(slot_A=alice)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "none" in report.lower()


def test_report_shows_outfit_overrides():
    alice = _make_subject("Alice", outfit="navy blazer")
    scene = _make_scene(slot_A=alice, outfit_overrides={"A": "red dress"})
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "red dress" in report
    assert "navy blazer" in report


def test_report_shows_concept_ids():
    alice = _make_subject("Alice", concept_id="char_alice")
    scene = _make_scene(slot_A=alice)
    report = assemble_prompt(scene, "h3_ref2va")["assembly_report"]
    assert "char_alice" in report


# ── Video references ──────────────────────────────────────────────────────────

def _video_entries(*pairs):
    """Build a video_entries list from (subject_id, video_file) pairs."""
    return [{"subject_id": sid, "video_file": vf} for sid, vf in pairs]


def test_video_entries_adds_video_label_in_h3_ref2va():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice_ref.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1>" in prompt


def test_video_label_in_subject_definitions():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1>: reference video for Alice" in prompt


def test_video_in_retention_analysis():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1>: reference (visual identity, not fully_copy)" in prompt


def test_video_in_summary_task_str():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "video reference" in prompt


def test_video_refs_in_summary_body():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "Reference videos:" in prompt
    assert "Alice (<Video 1>)" in prompt


def test_video_numbering_continuous_across_slots():
    alice = _make_subject("Alice", subject_id="char_alice")
    bob = _make_subject("Bob", subject_id="char_bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    ve = _video_entries(("char_alice", "a.mp4"), ("char_bob", "b.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1>" in prompt
    assert "<Video 2>" in prompt


def test_video_slots_in_result():
    alice = _make_subject("Alice", subject_id="char_alice")
    bob = _make_subject("Bob", subject_id="char_bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    ve = _video_entries(("char_alice", "a.mp4"))
    result = assemble_prompt(scene, "h3_ref2va", ve)
    assert result["video_slots"] == ["A"]


def test_no_video_entries_no_video_label():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<Video" not in prompt


def test_video_matched_by_subject_id_not_slot():
    # Bob is in slot A but video_entry matches by subject_id "char_bob"
    bob = _make_subject("Bob", subject_id="char_bob")
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=bob, slot_B=alice)
    ve = _video_entries(("char_bob", "bob.mp4"))
    result = assemble_prompt(scene, "h3_ref2va", ve)
    assert result["video_slots"] == ["A"]  # Bob is in slot A
    assert "<Video 1>" in result["prompt"]


def test_unmatched_video_entry_ignored():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_nobody", "nobody.mp4"))  # no subject matches
    result = assemble_prompt(scene, "h3_ref2va", ve)
    assert result["video_slots"] == []
    assert "<Video" not in result["prompt"]


def test_video_not_shown_in_h3_fl2va():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_fl2va", ve)["prompt"]
    assert "<Video" not in prompt


def test_report_shows_video_files():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    report = assemble_prompt(scene, "h3_ref2va", ve)["assembly_report"]
    assert "alice.mp4" in report
    assert "Video:" in report


def test_video_and_sheets_and_audio_together():
    alice = _make_subject("Alice", subject_id="char_alice", sheets=["a.png"], audio="a.wav")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    result = assemble_prompt(scene, "h3_ref2va", ve)
    prompt = result["prompt"]
    # All three types present
    assert "<Subject 1>" in prompt
    assert "<Picture 1>" in prompt
    assert "<Video 1>" in prompt
    assert "<Audio 1>" in prompt
    # task string includes all three
    assert "reference generation" in prompt
    assert "video reference" in prompt
    assert "audio reference" in prompt
