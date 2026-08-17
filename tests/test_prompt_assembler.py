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
    assert "<Subject 1>" in prompt
    assert "<Subject 2>" in prompt
    # Slot A (Alice) must appear before slot B (Bob) in subject_definitions
    assert prompt.index("<Subject 1>") < prompt.index("<Subject 2>")


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
    # Official format: "<Subject N> is [summary]..."
    assert "<Subject 1> is a tall woman with red hair" in prompt


def test_h3_ref2va_subject_definitions_lists_pictures_inline():
    alice = _make_subject("Alice", sheets=["sheet1.png", "sheet2.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "from <Picture 1> and <Picture 2>" in prompt
    assert "Character sheets:" not in prompt
    assert "(primary identity)" not in prompt


def test_h3_ref2va_indefinite_article_replaced_with_the_when_ref_present():
    alice = _make_subject("Alice", summary="a young woman with red hair", sheets=["s.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "is the young woman with red hair from <Picture 1>" in prompt
    assert "is a young woman" not in prompt


def test_h3_ref2va_an_article_replaced_when_ref_present():
    alice = _make_subject("Alice", summary="an elegant woman", sheets=["s.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "is the elegant woman" in prompt


def test_h3_ref2va_article_unchanged_when_no_ref():
    alice = _make_subject("Alice", summary="a young woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "is a young woman with red hair" in prompt


def test_h3_ref2va_the_summary_unchanged_when_ref_present():
    alice = _make_subject("Alice", summary="the lead character", sheets=["s.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "is the lead character from <Picture 1>" in prompt


def test_h3_ref2va_subject_definitions_lists_audio():
    alice = _make_subject("Alice", audio="alice.wav")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # Official format: "<Audio N> is the voice-timbre reference for <Subject N> (SN)..."
    assert "<Audio 1> is the voice-timbre reference for" in prompt
    assert "(S1)" in prompt


def test_h3_ref2va_retention_analysis_fully_preserved():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # Format: "<Subject 1> (appears in [Shot N]): fully_preserved - ..."
    assert "<Subject 1> (appears" in prompt
    assert "): fully_preserved - " in prompt
    # Must not restate the subject label
    assert "<Subject 1> must" not in prompt


def test_h3_ref2va_retention_analysis_uses_appearance_summary():
    alice = _make_subject("Alice", summary="the young woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "): fully_preserved - the young woman with red hair" in prompt


def test_h3_ref2va_retention_analysis_includes_detail_fields():
    alice = _make_subject("Alice", summary="the young woman", hair="long red hair", body="tall slender frame")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "): fully_preserved - the young woman, with long red hair and tall slender frame" in prompt


def test_h3_ref2va_retention_analysis_article_matches_subject_definitions():
    # With a video reference, "a young female" → "the young female" (mirrors subject_definitions)
    alice = _make_subject("Alice", summary="a young female", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "): fully_preserved - the young female" in prompt


def test_h3_ref2va_retention_analysis_fallback_when_no_summary():
    alice = _make_subject("Alice")
    alice["appearance"] = {}
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "): fully_preserved - appearance retained" in prompt


def test_h3_ref2va_retention_marker_override():
    alice = _make_subject("Alice")
    scene = {**_make_scene(slot_A=alice), "retention_markers": {"A": "partially_preserved"}}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<Subject 1> (appears" in prompt
    assert "): partially_preserved - " in prompt


def test_h3_ref2va_retention_analysis_audio_timbre_has_nocopy():
    alice = _make_subject("Alice", audio="a.wav")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "without copying the original signal" in prompt


def test_h3_ref2va_dialogue_default_quoted():
    alice = _make_subject("Alice", audio="a.wav", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Hello.", "Hi there."])
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # Default: quoted text, no <d> tags
    assert '"[American English] Hello."' in prompt
    assert '"[American English] Hi there."' in prompt
    assert "<d>" not in prompt


def test_h3_ref2va_dialogue_tags_when_enabled():
    alice = _make_subject("Alice", audio="a.wav", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Hello.", "Hi there."])
    scene["dialogue_tags"] = True
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "<d>[American English] Hello.</d>" in prompt
    assert "<d>[American English] Hi there.</d>" in prompt


def test_h3_ref2va_shot_headers():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "[Shot 1]" in prompt
    assert "[Shot 2]" in prompt


def test_h3_ref2va_subject_definitions_contains_description():
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "a tall woman with red hair" in prompt
    assert "Alice — a tall woman" not in prompt  # no inline name—description slug


def test_h3_ref2va_first_appearance_injects_description_inline():
    """First mention of a subject in detailed_description gets a comma-appositive."""
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # "a" → "the" because inline is always definite
    assert "<Subject 1> (S1), the tall woman with red hair" in prompt


def test_h3_ref2va_second_appearance_in_same_shot_is_bare():
    """Second {A} reference within the same camera+action pair is bare — no description."""
    # Template has both camera ("Close-up of {A}") and action ("{A} speaks.")
    # Only the first (camera) should get the description.
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    dd = prompt.split("detailed_description:")[1]
    # Description appears exactly once inline (in camera line of Shot 1)
    assert dd.count("the tall woman with red hair") == 1


def test_h3_ref2va_subsequent_shot_appearance_is_bare():
    """After a subject's first appearance, later shots just use the bare label."""
    alice = _make_subject("Alice", summary="a tall woman with red hair")
    bob = _make_subject("Bob", summary="a young man with dark hair")
    scene = _make_scene(slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    dd = prompt.split("detailed_description:")[1]
    # Each subject's description appears exactly once inline
    assert dd.count("the tall woman with red hair") == 1
    assert dd.count("the young man with dark hair") == 1


def test_h3_ref2va_non_speaking_first_appearance_has_no_speaker_id():
    """Non-speaking first appearance: '<Subject N>, description' — no (SN)."""
    # Bob is in slot B; Shot 1 speaks slot A, Shot 2 speaks slot B.
    # Bob first appears in Shot 1 camera/action (where he's non-speaking), then speaks in Shot 2.
    alice = _make_subject("Alice", summary="a young woman")
    bob = _make_subject("Bob", summary="a young man")
    # Custom template: Shot 1 mentions both A and B but only A speaks
    template = _make_template(n_slots=2, n_shots=2)
    template["shots"][0]["camera"] = "Close-up of {A} and {B}"
    template["shots"][0]["action"] = "{A} speaks to {B}."
    scene = _make_scene(template=template, slot_A=alice, slot_B=bob)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    dd = prompt.split("detailed_description:")[1]
    # B's first appearance is non-speaking → no (S2)
    assert "<Subject 2>, the young man" in dd
    assert "<Subject 2> (S2), the young man" not in dd


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


def test_h3_fl2va_dialogue_default_quoted():
    alice = _make_subject("Alice", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Good morning.", "Morning."])
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert '"[American English] Good morning."' in prompt
    assert '"[American English] Morning."' in prompt
    assert "<d>" not in prompt


def test_h3_fl2va_dialogue_tags_when_enabled():
    alice = _make_subject("Alice", language="en-us")
    bob = _make_subject("Bob")
    scene = _make_scene(slot_A=alice, slot_B=bob, dialogue=["Good morning.", "Morning."])
    scene["dialogue_tags"] = True
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "<d>[American English] Good morning.</d>" in prompt
    assert "<d>[American English] Morning.</d>" in prompt


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
    # Video cited inline after description: "is [desc] from <Video N>"
    assert "from <Video 1>" in prompt
    # Plus standalone role line
    assert "<Video 1> is" in prompt


def test_video_in_retention_analysis():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    # Format: "<Video N> (visual identity of <Subject N>): fully_preserved - ..."
    assert "<Video 1> (visual identity of <Subject 1>): fully_preserved" in prompt


def test_video_in_summary_task_str():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    # Per MiniMax docs, a video providing character guidance auto-detects as
    # "reference generation" — not a separate "video reference" type
    assert "reference generation" in prompt


def test_video_refs_in_summary_body():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    # Video is still numbered and cited in subject_definitions, not the summary body
    assert "<Video 1>" in prompt


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
    # Per MiniMax docs, pictures and videos both fall under "reference generation";
    # voice timbre files fall under "audio reference"
    assert "reference generation" in prompt
    assert "audio reference" in prompt
    # No "video reference" — that is not an official MiniMax task type
    assert "video reference" not in prompt


# ── task_flags user override ──────────────────────────────────────────────────

def test_task_flags_override_auto_detect():
    alice = _make_subject("Alice", subject_id="char_alice")
    ve = _video_entries(("char_alice", "alice.mp4"))
    scene = _make_scene(slot_A=alice)
    # Without override, video auto-detects as "reference generation"
    auto_prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "reference generation" in auto_prompt
    # With user override to "video continuation", bracket tag changes
    scene_with_flags = {**scene, "task_flags": ["video continuation"]}
    override_prompt = assemble_prompt(scene_with_flags, "h3_ref2va", ve)["prompt"]
    assert "video continuation" in override_prompt
    assert "reference generation" not in override_prompt


def test_task_flags_override_empty_falls_back_to_auto():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    scene_with_empty = {**scene, "task_flags": []}
    prompt = assemble_prompt(scene_with_empty, "h3_ref2va")["prompt"]
    assert "reference generation" in prompt


def test_task_flags_multiple_flags_joined_correctly():
    alice = _make_subject("Alice")
    scene = {**_make_scene(slot_A=alice), "task_flags": ["video continuation", "audio reference"]}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "[video continuation + audio reference]" in prompt


def test_task_flags_canonical_order_video_continuation_first():
    """video continuation must appear before reference generation and audio reference."""
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = {**_make_scene(slot_A=alice), "task_flags": ["reference generation", "audio reference", "video continuation"]}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "[video continuation + reference generation + audio reference]" in prompt


def test_subject_definitions_video_uses_from_preposition():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    sd = prompt.split("subject_definitions:")[1].split("\n\n")[0]
    assert "from <Video 1>" in sd
    assert "in <Video 1>" not in sd


def test_subject_definitions_picture_uses_from_preposition():
    alice = _make_subject("Alice", sheets=["a.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    sd = prompt.split("subject_definitions:")[1].split("\n\n")[0]
    assert "from <Picture 1>" in sd
    assert "in <Picture 1>" not in sd


def test_single_picture_uses_from_phrasing():
    alice = _make_subject("Alice", sheets=["sheet.png"])
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "from <Picture 1>" in prompt


def test_video_standalone_line_default_role():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1> is the visual identity reference for <Subject 1>" in prompt


def test_video_standalone_line_continuation_role():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = {**_make_scene(slot_A=alice), "task_flags": ["video continuation"]}
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1> is the continuation starting point for the target video" in prompt


def test_video_standalone_line_editing_role():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = {**_make_scene(slot_A=alice), "task_flags": ["video editing"]}
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "<Video 1> is the source video being edited" in prompt


def test_video_editing_opens_with_edited_version_sentence():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    scene_editing = {**scene, "task_flags": ["video editing"]}
    prompt = assemble_prompt(scene_editing, "h3_ref2va", ve)["prompt"]
    assert "The target video is an edited version of <Video 1>." in prompt
    assert "[video editing]" in prompt


def test_non_editing_tasks_open_with_shows_sentence():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "The target video shows" in prompt
    assert "The target video is an edited version" not in prompt


def test_scene_synopsis_replaces_shot_narrative():
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    scene = {
        **_make_scene(slot_A=alice, slot_B=bob),
        "scene_synopsis": "{A} eating a cookie. {B} enters and lunges towards the cookie.",
    }
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "The target video shows <Subject 1> eating a cookie." in prompt
    assert "<Subject 2> enters and lunges towards the cookie." in prompt


def test_scene_synopsis_no_double_period():
    alice = _make_subject("Alice")
    scene = {**_make_scene(slot_A=alice), "scene_synopsis": "{A} walks into the room."}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "The target video shows <Subject 1> walks into the room." in prompt
    assert "room.." not in prompt


def test_scene_synopsis_fallback_to_shots_when_empty():
    alice = _make_subject("Alice")
    scene = {**_make_scene(slot_A=alice), "scene_synopsis": ""}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    # falls back to shot-action narrative
    assert "The target video shows" in prompt


def test_audio_reuse_flag_appears_in_bracket():
    alice = _make_subject("Alice", audio="alice.wav")
    scene = {**_make_scene(slot_A=alice), "task_flags": ["video editing", "audio reuse"]}
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "[video editing + audio reuse]" in prompt


def test_auto_detect_only_audio_gives_audio_reference():
    alice = _make_subject("Alice", audio="alice.wav")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "audio reference" in prompt
    # No pictures or video → no "reference generation"
    assert "reference generation" not in prompt


def test_soundtrack_audio_subject_definitions_references_subject():
    """extract_from_visual audio must cite the subject, not just the video."""
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = [{"subject_id": "char_alice", "video_file": "alice.mp4",
           "audio_source": "extract_from_visual", "audio_retention": "timbre"}]
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    # Must say "for <Subject 1> (S1)" — not restating the video reference
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)" in prompt


def test_soundtrack_audio_style_subject_definitions_references_subject():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = [{"subject_id": "char_alice", "video_file": "alice.mp4",
           "audio_source": "extract_from_visual", "audio_retention": "style"}]
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "audio style and rhythm reference for <Subject 1> (S1)" in prompt


# ── estimate_speech_duration ───────────────────────────────────────────────────

estimate_speech_duration = pa.estimate_speech_duration


def test_estimate_empty_returns_zero():
    assert estimate_speech_duration("") == 0.0
    assert estimate_speech_duration("   ") == 0.0


def test_estimate_single_word():
    dur = estimate_speech_duration("Hello")
    # 5 chars / 13 + 1 word * 0.07 = 0.455
    assert 0.4 < dur < 0.6


def test_estimate_sentence_with_period():
    dur = estimate_speech_duration("Hello.")
    # 5 chars / 13 + 1 * 0.07 + 0.35 ≈ 0.805
    assert 0.7 < dur < 1.0


def test_estimate_slow_pace_longer():
    text = "How are you today?"
    slow = estimate_speech_duration(text, chars_per_second=10.0)
    normal = estimate_speech_duration(text, chars_per_second=13.0)
    fast = estimate_speech_duration(text, chars_per_second=16.0)
    assert slow > normal > fast


def test_estimate_comma_adds_pause():
    without = estimate_speech_duration("Hello friend")
    with_comma = estimate_speech_duration("Hello, friend")
    assert with_comma > without


def test_estimate_proportional_to_length():
    short = estimate_speech_duration("Hi.")
    long = estimate_speech_duration("Hello, how are you doing today? I hope everything is well.")
    assert long > short * 3


# ── Speech pace injection in prompts ──────────────────────────────────────────

def _make_template_with_pace(pace="normal"):
    """Template with a single shot whose dialogue has a given speech_pace."""
    return {
        "id": "t1", "name": "Test", "description": "",
        "slots": {"A": {}},
        "environment": {"summary": "a room", "lighting": ""},
        "style": "",
        "shots": [{
            "id": "shot_1",
            "timestamp": None,
            "camera": "Close-up on {A}.",
            "action": "{A} turns to face the camera.",
            "dialogue": {
                "placeholder": True,
                "speaker_slot": "A",
                "speech_pace": pace,
            },
            "sound_events": None,
        }],
        "overall_soundscape": "",
        "non_diegetic_music": "",
    }


def _scene_with_dialogue(pace="normal", text="Hello there."):
    tmpl = _make_template_with_pace(pace)
    alice = _make_subject("Alice")
    return {
        "template_id": "t1",
        "template_name": "Test",
        "template": tmpl,
        "slot_assignments": {"A": alice},
        "dialogue": {"shot_1": text},
        "outfit_overrides": {},
    }


def test_normal_pace_no_phrase_injected():
    scene = _scene_with_dialogue(pace="normal", text="Hello.")
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "speaking slowly" not in prompt
    assert "speaking quickly" not in prompt


def test_slow_pace_injects_phrase_h3ref2va():
    scene = _scene_with_dialogue(pace="slow", text="Hello.")
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "speaking slowly and deliberately" in prompt


def test_fast_pace_injects_phrase_h3ref2va():
    scene = _scene_with_dialogue(pace="fast", text="Hello.")
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "speaking quickly" in prompt


def test_slow_pace_injects_phrase_h3fl2va():
    scene = _scene_with_dialogue(pace="slow", text="Hello.")
    prompt = assemble_prompt(scene, "h3_fl2va")["prompt"]
    assert "speaking slowly and deliberately" in prompt


def test_pace_not_injected_when_no_dialogue_text():
    """Pace phrase must not appear when the shot has no resolved dialogue text."""
    tmpl = _make_template_with_pace("slow")
    # Override shot to have no dialogue text placeholder
    tmpl["shots"][0]["dialogue"] = {
        "placeholder": True, "speaker_slot": "A", "speech_pace": "slow"
    }
    alice = _make_subject("Alice")
    scene = {
        "template_id": "t1", "template_name": "Test", "template": tmpl,
        "slot_assignments": {"A": alice},
        "dialogue": {},  # no text resolved
        "outfit_overrides": {},
    }
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "speaking slowly" not in prompt


# ── trim_to in _build_h3_refplan ──────────────────────────────────────────────

_build_h3_refplan = pa._build_h3_refplan


def _scene_for_refplan(audio_file="alice.wav"):
    alice = _make_subject("Alice", subject_id="alice", audio=audio_file)
    alice["voice"]["audio_start_time"] = 0.0
    alice["voice"]["audio_duration"] = 10.0
    alice["voice"]["audio_retention"] = "timbre"
    alice["voice"]["audio_role"] = ""
    return {"slot_assignments": {"A": alice}, "outfit_overrides": {}}


def test_trim_to_injected_into_standalone_audio():
    scene = _scene_for_refplan()
    refplan = _build_h3_refplan(scene, slot_trim_to={"A": 3.5})
    audio_refs = [r for r in refplan["references"] if r["modality"] == "audio"]
    assert len(audio_refs) == 1
    assert audio_refs[0]["trim_to"] == pytest.approx(3.5)


def test_trim_to_none_when_no_slot_trim_to():
    scene = _scene_for_refplan()
    refplan = _build_h3_refplan(scene)
    audio_refs = [r for r in refplan["references"] if r["modality"] == "audio"]
    assert audio_refs[0]["trim_to"] is None


def test_trim_to_none_for_unmatched_slot():
    scene = _scene_for_refplan()
    # slot_trim_to has B but the audio is in slot A
    refplan = _build_h3_refplan(scene, slot_trim_to={"B": 2.0})
    audio_refs = [r for r in refplan["references"] if r["modality"] == "audio"]
    assert audio_refs[0]["trim_to"] is None


# ── H3 Ref2VA validation helpers ───────────────────────────────────────────────

validate_h3_refs_pre    = pa.validate_h3_refs_pre
validate_h3_audio_clip  = pa.validate_h3_audio_clip
validate_h3_audio_total = pa.validate_h3_audio_total


def _ref(modality, *, aord=1, trim_to=None):
    """Minimal reference dict for validation tests."""
    r = {"modality": modality, "audio_ordinal": aord, "path": f"file.{modality}"}
    if trim_to is not None:
        r["trim_to"] = trim_to
    return r


# ── validate_h3_refs_pre ────────────────────────────────────────────────────────

class TestValidateH3RefsPre:
    def test_empty_refs_ok(self):
        assert validate_h3_refs_pre([]) == []

    def test_image_only_ok(self):
        assert validate_h3_refs_pre([_ref("image")]) == []

    def test_video_only_ok(self):
        assert validate_h3_refs_pre([_ref("video")]) == []

    def test_audio_with_image_ok(self):
        assert validate_h3_refs_pre([_ref("image"), _ref("audio")]) == []

    def test_audio_with_video_ok(self):
        assert validate_h3_refs_pre([_ref("video"), _ref("audio")]) == []

    def test_audio_without_visual_fails(self):
        errors = validate_h3_refs_pre([_ref("audio")])
        assert len(errors) == 1
        assert "paired" in errors[0].lower() or "visual" in errors[0].lower()

    def test_soundtrack_audio_alone_does_not_trigger_pairing_error(self):
        # soundtrack_audio is attached to a video — it is not a standalone audio ref
        errors = validate_h3_refs_pre([_ref("soundtrack_audio")])
        assert errors == []

    def test_four_audio_refs_fails(self):
        refs = [_ref("image")] + [_ref("audio", aord=i) for i in range(1, 5)]
        errors = validate_h3_refs_pre(refs)
        assert any("3" in e for e in errors)

    def test_exactly_three_audio_refs_ok(self):
        refs = [_ref("image")] + [_ref("audio", aord=i) for i in range(1, 4)]
        assert validate_h3_refs_pre(refs) == []

    def test_thirteen_counted_refs_fails(self):
        refs = [_ref("image")] * 13
        errors = validate_h3_refs_pre(refs)
        assert any("12" in e for e in errors)

    def test_twelve_counted_refs_ok(self):
        refs = [_ref("image")] * 12
        assert validate_h3_refs_pre(refs) == []

    def test_trim_to_below_2s_fails(self):
        refs = [_ref("image"), _ref("audio", aord=1, trim_to=1.5)]
        errors = validate_h3_refs_pre(refs)
        assert any("1.50" in e for e in errors)

    def test_trim_to_exactly_2s_ok(self):
        refs = [_ref("image"), _ref("audio", aord=1, trim_to=2.0)]
        assert validate_h3_refs_pre(refs) == []

    def test_trim_to_none_skipped(self):
        refs = [_ref("image"), _ref("audio", aord=1, trim_to=None)]
        assert validate_h3_refs_pre(refs) == []

    def test_multiple_errors_all_returned(self):
        # No visual + 4 audio refs → 2 errors
        refs = [_ref("audio", aord=i) for i in range(1, 5)]
        errors = validate_h3_refs_pre(refs)
        assert len(errors) >= 2

    def test_soundtrack_audio_counts_toward_12_limit(self):
        # 11 images + 1 soundtrack_audio = 12 counted → OK
        refs = [_ref("image")] * 11 + [_ref("soundtrack_audio")]
        assert validate_h3_refs_pre(refs) == []
        # add one more → 13 → fail
        refs.append(_ref("image"))
        errors = validate_h3_refs_pre(refs)
        assert any("12" in e for e in errors)


# ── validate_h3_audio_clip ──────────────────────────────────────────────────────

class TestValidateH3AudioClip:
    def test_in_range_ok(self):
        assert validate_h3_audio_clip(5.0, 1, "clip.wav") is None

    def test_exactly_2s_ok(self):
        assert validate_h3_audio_clip(2.0, 1, "clip.wav") is None

    def test_exactly_15s_ok(self):
        assert validate_h3_audio_clip(15.0, 1, "clip.wav") is None

    def test_below_2s_fails(self):
        err = validate_h3_audio_clip(1.8, 1, "clip.wav")
        assert err is not None
        assert "1.80" in err
        assert "minimum" in err.lower()

    def test_above_15s_fails(self):
        err = validate_h3_audio_clip(16.0, 2, "long.wav")
        assert err is not None
        assert "16.00" in err
        assert "maximum" in err.lower()

    def test_basename_in_error(self):
        err = validate_h3_audio_clip(0.5, 1, "myfile.wav")
        assert "myfile.wav" in err

    def test_ordinal_in_error(self):
        err = validate_h3_audio_clip(20.0, 3, "f.wav")
        assert "<Audio 3>" in err


# ── validate_h3_audio_total ─────────────────────────────────────────────────────

class TestValidateH3AudioTotal:
    def test_empty_ok(self):
        assert validate_h3_audio_total([]) is None

    def test_single_ok(self):
        assert validate_h3_audio_total([10.0]) is None

    def test_sum_exactly_15s_ok(self):
        assert validate_h3_audio_total([5.0, 5.0, 5.0]) is None

    def test_sum_above_15s_fails(self):
        err = validate_h3_audio_total([8.0, 8.0])
        assert err is not None
        assert "16.00" in err

    def test_count_in_error(self):
        err = validate_h3_audio_total([8.0, 8.0])
        assert "2" in err

    def test_just_over_15s_fails(self):
        err = validate_h3_audio_total([15.001])
        assert err is not None
