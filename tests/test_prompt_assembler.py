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


def test_picture_cited_inside_subject_line_default_role():
    alice = _make_subject("Alice", sheets=["alice.png"])
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    # Per H3 spec: character reference images are cited inside <Subject N>, not as
    # standalone <Picture N> entries. Check the "whose appearance comes from" phrasing.
    assert "whose appearance comes from <Picture 1>" in result["prompt"]
    # No standalone picture role line should appear
    assert "<Picture 1> is a character reference sheet" not in result["prompt"]


def test_picture_cited_inside_subject_line_portrait():
    alice = _make_subject("Alice", sheets=[{"file": "alice_portrait.png", "role": "portrait"}])
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "whose appearance comes from <Picture 1>" in result["prompt"]
    assert "<Picture 1> is a frontal portrait" not in result["prompt"]


def test_picture_multiple_sheets_cited_inline():
    alice = _make_subject("Alice", sheets=[
        {"file": "alice_portrait.png", "role": "portrait"},
        {"file": "alice_body.png", "role": "full body"},
    ])
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    # Both picture numbers appear in the Subject line citation
    assert "whose appearance comes from <Picture 1> and <Picture 2>" in result["prompt"]
    assert "<Picture 1> is a frontal portrait" not in result["prompt"]
    assert "<Picture 2> is a full-body reference" not in result["prompt"]


def test_picture_mixed_legacy_and_dict_cited_inline():
    alice = _make_subject("Alice", sheets=[
        "alice_sheet.png",
        {"file": "alice_portrait.png", "role": "portrait"},
    ])
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "whose appearance comes from <Picture 1> and <Picture 2>" in result["prompt"]
    assert "<Picture 1> is a character reference sheet" not in result["prompt"]


def test_picture_custom_role_cited_inline():
    alice = _make_subject("Alice", sheets=[{"file": "alice.png", "role": "3/4 view"}])
    scene = _make_scene(slot_A=alice)
    result = assemble_prompt(scene, "h3_ref2va")
    assert "whose appearance comes from <Picture 1>" in result["prompt"]
    assert "<Picture 1> is 3/4 view" not in result["prompt"]


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
    # "a" → "the" still applies; picture is cited inline with "whose appearance comes from"
    assert "is the young woman with red hair whose appearance comes from <Picture 1>" in prompt
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
    assert "is the lead character whose appearance comes from <Picture 1>" in prompt


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


def test_h3_ref2va_retention_analysis_uses_structured_traits():
    # Structured trait fields are used verbatim; appearance_summary is NOT used in retain clause.
    alice = _make_subject("Alice", summary="the young woman with red hair", hair="long red hair")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "retain their long red hair" in prompt
    # The retain line itself must not restate the full appearance_summary noun phrase.
    ra_lines = [ln for ln in prompt.split("\n") if ln.startswith("<Subject") and "fully_preserved" in ln]
    assert all("the young woman with red hair" not in ln for ln in ra_lines)


def test_h3_ref2va_retention_analysis_includes_detail_fields():
    alice = _make_subject("Alice", summary="the young woman", hair="long red hair", body="tall slender frame")
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "retain their long red hair and tall slender frame" in prompt


def test_h3_ref2va_retention_analysis_generic_fallback_no_traits():
    # No structured traits and no short_name → generic "retain their appearance"
    alice = _make_subject("Alice", summary="a young female", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "retain their appearance" in prompt


def test_h3_ref2va_retention_analysis_short_name_fallback():
    # No structured traits but short_name set → "retain {short_name}'s appearance"
    alice = _make_subject("Alice", summary="a young female", subject_id="char_alice")
    alice["_short_name"] = "the young woman"
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "retain the young woman's appearance" in prompt


def test_h3_ref2va_retention_analysis_fallback_when_no_summary():
    alice = _make_subject("Alice")
    alice["appearance"] = {}
    scene = _make_scene(slot_A=alice)
    prompt = assemble_prompt(scene, "h3_ref2va")["prompt"]
    assert "retain their appearance" in prompt


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


def test_video_identity_reference_excludes_background_by_default():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = _video_entries(("char_alice", "alice.mp4"))
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "ignore the background/setting of <Video 1>" in prompt


def test_video_identity_reference_include_video_background_opts_out():
    alice = _make_subject("Alice", subject_id="char_alice")
    scene = _make_scene(slot_A=alice)
    ve = [{"subject_id": "char_alice", "video_file": "alice.mp4", "include_video_background": True}]
    prompt = assemble_prompt(scene, "h3_ref2va", ve)["prompt"]
    assert "ignore the background/setting" not in prompt


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


# ── validate_h3_load_failures ───────────────────────────────────────────────────

validate_h3_load_failures = pa.validate_h3_load_failures


class TestValidateH3LoadFailures:
    def test_no_failures_ok(self):
        assert validate_h3_load_failures([]) is None

    def test_names_label_and_path(self):
        err = validate_h3_load_failures([("<Picture 2>", "chars/a.png")])
        assert "<Picture 2>" in err
        assert "chars/a.png" in err

    def test_lists_every_failure_in_order(self):
        err = validate_h3_load_failures([
            ("<Picture 1>", "a.png"), ("<Video 1>", "b.mp4"), ("<Audio 1>", "c.wav"),
        ])
        assert err.index("a.png") < err.index("b.mp4") < err.index("c.wav")

    def test_count_in_message(self):
        err = validate_h3_load_failures([("<Picture 1>", "a.png"), ("<Audio 1>", "c.wav")])
        assert "2 reference(s)" in err

    def test_missing_path_is_labelled(self):
        err = validate_h3_load_failures([("<Audio 1>", "")])
        assert "(no path)" in err


# ── assemble_composition: slot_descriptors + appearance_overrides ───────────────

assemble_composition = pa.assemble_composition


def _simple_composition(subjects_keys=("A",), slot_descriptors=None, appearance_overrides=None):
    comp = {
        "id": "test_comp",
        "name": "Test",
        "model_type": "h3_ref2va",
        "style": "",
        "subjects": {k: k.lower() for k in subjects_keys},
        "shots": [
            {
                "id": "shot_1", "timestamp": None,
                "camera": "Close-up", "action": "{A} speaks.",
                "dialogue": {"speaker": "A", "text": "Hello."}, "sound_events": None,
            }
        ],
        "overall_soundscape": "", "non_diegetic_music": "N/A",
    }
    if slot_descriptors:
        comp["slot_descriptors"] = slot_descriptors
    if appearance_overrides:
        comp["appearance_overrides"] = appearance_overrides
    return comp


def _make_resolved_subjects(keys=("A",), summary="original summary", face="original face"):
    return {
        k: {
            "subject_id": k.lower(),
            "name": k,
            "appearance": {"summary": summary, "face": face, "hair": "", "body": "", "default_outfit": ""},
            "voice": {"description": "", "audio_reference_file": "", "language": "en-us"},
            "character_sheet_images": [],
            "concept_id": "",
        }
        for k in keys
    }


class TestSlotDescriptors:
    def test_slot_descriptor_overrides_summary(self):
        comp = _simple_composition(slot_descriptors={"A": "custom appearance description"})
        resolved = _make_resolved_subjects()
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "custom appearance description" in result["prompt"]

    def test_slot_descriptor_does_not_mutate_original_subject(self):
        comp = _simple_composition(slot_descriptors={"A": "override text"})
        resolved = _make_resolved_subjects()
        orig_summary = resolved["A"]["appearance"]["summary"]
        assemble_composition(comp, resolved, None, "h3_ref2va")
        assert resolved["A"]["appearance"]["summary"] == orig_summary

    def test_empty_slot_descriptor_uses_profile_summary(self):
        comp = _simple_composition(slot_descriptors={"A": ""})
        resolved = _make_resolved_subjects(summary="profile summary")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "profile summary" in result["prompt"]

    def test_whitespace_slot_descriptor_treated_as_empty(self):
        comp = _simple_composition(slot_descriptors={"A": "   "})
        resolved = _make_resolved_subjects(summary="profile summary")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "profile summary" in result["prompt"]

    def test_unknown_slot_key_ignored(self):
        comp = _simple_composition(slot_descriptors={"Z99": "should be ignored"})
        resolved = _make_resolved_subjects(summary="profile summary")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "profile summary" in result["prompt"]


class TestAppearanceOverrides:
    def test_field_override_updates_face(self):
        comp = _simple_composition(appearance_overrides={"A": {"face": "new face description"}})
        resolved = _make_resolved_subjects(face="original face")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        # face appears in h3 subject_definitions section
        assert "new face description" in result["prompt"]

    def test_field_override_does_not_mutate_original(self):
        comp = _simple_composition(appearance_overrides={"A": {"face": "overridden face"}})
        resolved = _make_resolved_subjects(face="original face")
        assemble_composition(comp, resolved, None, "h3_ref2va")
        assert resolved["A"]["appearance"]["face"] == "original face"

    def test_empty_string_override_not_applied(self):
        comp = _simple_composition(appearance_overrides={"A": {"face": ""}})
        resolved = _make_resolved_subjects(face="original face")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "original face" in result["prompt"]

    def test_descriptor_and_field_override_both_applied(self):
        comp = _simple_composition(
            slot_descriptors={"A": "custom summary"},
            appearance_overrides={"A": {"face": "custom face"}},
        )
        resolved = _make_resolved_subjects(summary="original summary", face="original face")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "custom summary" in result["prompt"]
        assert "custom face" in result["prompt"]

    def test_no_overrides_passes_through_cleanly(self):
        comp = _simple_composition()
        resolved = _make_resolved_subjects(summary="untouched summary")
        result = assemble_composition(comp, resolved, None, "h3_ref2va")
        assert "untouched summary" in result["prompt"]


# ── _build_background_slot (shared by assemble_composition and
#    SourceProfileClipPrompt, Plan: component-reference-system Phase 1) ─────────

_build_background_slot = pa._build_background_slot


class TestBuildBackgroundSlot:
    def test_returns_none_without_reference_images(self):
        assert _build_background_slot({"name": "Beach", "description": "a beach"}) is None

    def test_returns_none_for_empty_reference_images_list(self):
        assert _build_background_slot({"name": "Beach", "reference_images": []}) is None

    def test_returns_slot_shape_with_reference_images(self):
        slot = _build_background_slot({
            "name": "Beach House", "description": "a beach house at sunset",
            "reference_images": ["beach1.png"],
        })
        assert slot is not None
        assert slot["entity_type"] == "location"
        assert slot["pronoun_style"] == "location"
        assert slot["short_name"] == "the beach house"
        assert slot["appearance"]["summary"] == "a beach house at sunset"
        assert slot["character_sheet_images"] == [{"file": "beach1.png", "role": "scene reference"}]

    def test_normalizes_dict_reference_images(self):
        slot = _build_background_slot({
            "name": "Office", "reference_images": [{"file": "office1.png", "role": "wide shot"}],
        })
        assert slot["character_sheet_images"] == [{"file": "office1.png", "role": "wide shot"}]

    def test_appends_lighting_to_description(self):
        slot = _build_background_slot({
            "name": "Office", "description": "a modern office", "lighting": "warm afternoon light",
            "reference_images": ["office1.png"],
        })
        assert slot["appearance"]["summary"] == "a modern office, warm afternoon light"

    def test_lighting_join_ignores_trailing_periods_on_either_part(self):
        slot = _build_background_slot({
            "name": "Office", "description": "a modern office.", "lighting": "warm afternoon light.",
            "reference_images": ["office1.png"],
        })
        assert slot["appearance"]["summary"] == "a modern office, warm afternoon light"

    def test_lighting_without_a_description_is_used_alone(self):
        slot = _build_background_slot({
            "name": "Office", "description": "", "lighting": "warm afternoon light",
            "reference_images": ["office1.png"],
        })
        assert slot["appearance"]["summary"] == "warm afternoon light"

    def test_description_without_lighting_is_unchanged(self):
        slot = _build_background_slot({
            "name": "Office", "description": "a modern office.", "lighting": "",
            "reference_images": ["office1.png"],
        })
        assert slot["appearance"]["summary"] == "a modern office."

    def test_short_name_falls_back_to_setting_when_unnamed(self):
        slot = _build_background_slot({"reference_images": ["x.png"]})
        assert slot["short_name"] == "the setting"


# ── assemble_composition: background_as_reference (regression coverage for the
#    _build_background_slot extraction — same behavior before and after) ───────

class TestAssembleCompositionBackground:
    def test_background_as_reference_mints_subject_and_picture(self):
        comp = _simple_composition()
        comp["background_as_reference"] = True
        resolved = _make_resolved_subjects()
        background = {
            "name": "Rooftop", "description": "a rooftop at night",
            "reference_images": ["rooftop1.png"],
        }
        result = assemble_composition(comp, resolved, background, "h3_ref2va")
        assert "<Picture" in result["prompt"]
        assert "retain the rooftop" in result["prompt"].lower()

    def test_background_as_reference_false_does_not_mint_subject(self):
        comp = _simple_composition()
        comp["background_as_reference"] = False
        resolved = _make_resolved_subjects()
        background = {"name": "Rooftop", "reference_images": ["rooftop1.png"]}
        result = assemble_composition(comp, resolved, background, "h3_ref2va")
        assert "rooftop" not in result["prompt"].lower()

    def test_background_without_reference_images_does_not_mint_subject(self):
        # The background's own description still feeds a plain detailed_description line
        # regardless of whether it becomes a Subject (separate, pre-existing behavior) --
        # what this test actually checks is that no *second* Subject/Picture gets minted
        # for it when there's no image to back one.
        comp = _simple_composition()
        comp["background_as_reference"] = True
        resolved = _make_resolved_subjects()
        background = {"name": "Rooftop", "description": "a rooftop at night"}
        result = assemble_composition(comp, resolved, background, "h3_ref2va")
        assert "<Subject 2>" not in result["prompt"]
        assert "<Picture" not in result["prompt"]


# ── assemble_composition: hybrid cast retention markers (bundle-slot fix) ───────
#
# apply_cast_to_subjects (utils/prompt_compositions.py) mints an extra
# "<slot>_bundle" key for a hybrid cast entry (bundle replacing a subject that
# also has a source video), pairing the donor slot ("_cast_retention":
# "replaced") with the replacement slot ("_cast_retention": "attribute_transfer").
# These tests confirm assemble_composition surfaces that pairing end-to-end
# into the assembled prompt text, exercising both the slot-letter-minting fix
# for the new "_bundle" key and the "_transfer_to_slot" remap fix
# (apply_cast_to_subjects stamps composition-level slot keys like "A_bundle",
# but ref_map/slot_assignments are keyed by template letters — since base
# subject keys are already letters this is usually identity, but the bundle
# key itself still needs a freshly-minted letter of its own, and
# "_transfer_to_slot" must be translated to point at it).

pc = import_test_module("utils/prompt_compositions.py")
apply_cast_to_subjects = pc.apply_cast_to_subjects


class _HybridBundleRegistry:
    def __init__(self, bundles):
        self._b = bundles

    def get(self, bundle_id):
        return self._b.get(bundle_id)


def _hybrid_composition():
    return {
        "id": "test_hybrid_comp",
        "name": "Hybrid Test",
        "model_type": "h3_ref2va",
        "style": "",
        "subjects": {"A": "alice"},
        "shots": [
            {
                "id": "shot_1", "timestamp": None,
                "camera": "Close-up of {A}", "action": "{A} walks forward.",
                "dialogue": None, "sound_events": None,
            }
        ],
        "overall_soundscape": "", "non_diegetic_music": "N/A",
    }


class TestHybridCastRetentionInAssembledPrompt:
    def test_motion_transfer_sentence_appears_for_bundle_replacement(self):
        comp = _hybrid_composition()
        alice = _make_resolved_subjects(summary="a woman with long red hair")["A"]
        alice["subject_id"] = "alice"
        cast = {"entries": [{
            "subject_id": "alice",
            "bundle_id": "b1",
            "source_profile_id": "sp1",
            "source_subject_id": "src1",
            "role_description": "a woman with long red hair",
            "retention": "partially_preserved",
            "visual_mode": "images",
            "use_audio": False,
        }]}
        breg = _HybridBundleRegistry({"b1": {
            "visual": {"type": "images", "files": ["bundle_ref.png"], "file": ""},
            "audio": {"source": "none", "file": ""},
            "appearance_override": "a woman with short dark hair",
        }})
        enriched = apply_cast_to_subjects({"A": alice}, comp, cast, breg)

        # Sanity: apply_cast_to_subjects produced the paired donor/replacement
        # slots with the composition-level "_transfer_to_slot" values that
        # assemble_composition must remap through slot_map.
        assert enriched["A"]["_cast_retention"] == "replaced"
        assert enriched["A"]["_transfer_to_slot"] == "A_bundle"
        assert enriched["A_bundle"]["_cast_retention"] == "attribute_transfer"
        assert enriched["A_bundle"]["_transfer_to_slot"] == "A"

        result = assemble_composition(comp, enriched, None, "h3_ref2va")
        prompt = result["prompt"]

        # The bundle-replacement subject gets a motion-transfer sentence citing
        # the donor subject's appearance — this only fires when
        # retention_marker == "attribute_transfer" AND the remapped
        # "_transfer_to_slot" resolves to a real ref_map entry (the donor).
        assert "Their pose, movement, and screen position in the scene match those of" in prompt
        assert "a woman with long red hair" in prompt

    def test_replacement_bundle_appearance_and_images_survive_into_prompt(self):
        # Companion to test_hybrid_entry_keeps_bundle_appearance_and_images in
        # test_cast_enrichment.py — confirms the bundle's own data (not the
        # donor's) is what actually reaches the assembled prompt text for the
        # replacement <Subject N>.
        comp = _hybrid_composition()
        alice = _make_resolved_subjects(summary="a woman with long red hair")["A"]
        alice["subject_id"] = "alice"
        cast = {"entries": [{
            "subject_id": "alice",
            "bundle_id": "b1",
            "source_profile_id": "sp1",
            "source_subject_id": "src1",
            "role_description": "a woman with long red hair",
            "retention": "partially_preserved",
            "visual_mode": "images",
            "use_audio": False,
        }]}
        breg = _HybridBundleRegistry({"b1": {
            "visual": {"type": "images", "files": ["bundle_ref.png"], "file": ""},
            "audio": {"source": "none", "file": ""},
            "appearance_override": "a woman with short dark hair",
        }})
        enriched = apply_cast_to_subjects({"A": alice}, comp, cast, breg)
        result = assemble_composition(comp, enriched, None, "h3_ref2va")
        # Definite-article substitution turns "a woman…" into "the woman…" when a
        # picture/video reference is cited, so match on the descriptive tail only.
        assert "woman with short dark hair" in result["prompt"]


# ── Scene still as the swap source (Source Profile with image media) ───────────
#
# A replaced subject's picture in the "scene layout" role is the still the replacement is placed
# into: it fixes pose, orientation, position and setting, never identity. The scene image is
# attached to the first source slot, so it sorts ahead of the bundle's images (K, L, ...) and is
# always <Picture 1>.

_SCENE_STILL = "/out/scenes/s1/base.png"


def _scene_still_instance(include_tags=True, layout_role=None):
    return {
        "template_id": "sp", "template_name": "sp",
        "task_flags": ["video editing", "reference generation"],
        "scene_synopsis": "",
        "include_original_subject_tags": include_tags,
        "slot_assignments": {
            "A": {
                "subject_id": "girl", "name": "girl", "concept_id": None,
                "character_sheet_images": [
                    {"file": _SCENE_STILL, "role": layout_role or pa.SCENE_LAYOUT_ROLE}
                ],
                "appearance": {"summary": "the girl", "hair": "", "face": "", "body": "",
                               "default_outfit": ""},
                "voice": {}, "_cast_retention": "replaced", "_transfer_to_slot": "K",
                "_pronoun_style": "feminine", "entity_type": "person",
            },
            "K": {
                "subject_id": "bun", "name": "Bun", "concept_id": None,
                "character_sheet_images": [{"file": "bun.png", "role": "character sheet"}],
                "appearance": {"summary": "a young woman with platinum braided hair", "hair": "",
                               "face": "", "body": "", "default_outfit": ""},
                "voice": {}, "_cast_retention": "attribute_transfer", "_transfer_to_slot": "A",
                "_pronoun_style": "feminine",
            },
        },
        "dialogue": {}, "outfit_overrides": {},
        "template": {
            "shots": [{"id": "c_shot_1", "action": "{A} sits on a chair facing right.",
                       "camera": "", "dialogue": None}],
            "environment": {}, "style": "", "overall_soundscape": "Silence.",
            "non_diegetic_music": "N/A",
        },
    }


class TestSceneStillAsSwapSource:
    def test_scene_still_is_picture_1_and_bundle_images_follow(self):
        refs = pa._build_h3_refplan(_scene_still_instance(), [])["references"]
        assert [(r["picture_ordinal"], r["path"]) for r in refs] == [
            (1, _SCENE_STILL), (2, "bun.png"),
        ]

    def test_replaced_subject_is_placed_in_the_picture_not_sourced_from_it(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        # The original person is "in" the still; no subject takes its appearance from it.
        assert "<Subject 2> is the girl in <Picture 1>." in prompt
        assert "appearance comes from <Picture 1>" not in prompt
        # The bundle subject's identity comes from its own picture and nothing else.
        assert "<Subject 1> is the young woman with platinum braided hair whose appearance comes from <Picture 2>" in prompt

    def test_bundle_subject_takes_pose_orientation_and_position_from_the_picture(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        assert "match those of the girl in <Picture 1>" in prompt
        assert (
            "<Subject 2> (originally in <Picture 1>): attribute_transfer - transfer her pose, "
            "body orientation and position to <Subject 1>; discard her visual identity "
            "including head, face, body, hair and wardrobe."
        ) in prompt

    def test_scene_still_has_its_own_retention_line_and_is_not_an_identity_reference(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        ra = prompt.split("retention_analysis:")[1].split("detailed_description:")[0]
        assert "<Picture 1> (scene layout reference): partially_preserved" in ra
        assert "discard the original appearance of the girl" in ra
        # Only the bundle's picture carries the "facial features, hair, and clothing" line.
        assert "<Picture 1>: fully_preserved" not in ra
        assert "<Picture 2>: fully_preserved - <Subject 1>'s facial features" in ra

    def test_summary_names_the_picture_the_scene_is_recreated_from(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        assert "The target video recreates the scene shown in <Picture 1>." in prompt

    def test_bundle_video_is_identity_only_not_the_edit_source(self):
        ve = [{"subject_id": "bun", "video_file": "bun.mp4"}]
        inst = _scene_still_instance()
        inst["scene_synopsis"] = "Replace {A} in <Picture 1> with {K}"
        prompt = pa.assemble_prompt(inst, "h3_ref2va", ve)["prompt"]
        assert "<Video 1>" in prompt
        assert "edited version" not in prompt
        assert "motion comes from" not in prompt
        assert "[video editing + reference generation" in prompt
        assert "The target video recreates the scene shown in <Picture 1>. Replace" in prompt

    def test_cast_background_replaces_the_setting_of_the_scene_still(self):
        inst = _scene_still_instance()
        inst["slot_assignments"]["O"] = pa._build_background_slot({
            "name": "Beach", "description": "a sunlit beach", "lighting": "golden hour",
            "reference_images": ["beach.png"],
        })
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        layout = next(l for l in prompt.splitlines() if "(scene layout reference)" in l)
        assert "setting details" not in layout and "lighting and framing" not in layout
        assert "original setting and lighting of <Picture 1> are replaced by <Subject" in layout
        assert "pose, body orientation and position" in layout
        assert "recreates the composition shown in <Picture 1> in the setting of <Subject" in prompt
        assert "<Picture 3>: fully_preserved" not in prompt

    def _video_edit_instance(self, with_bg):
        inst = _scene_still_instance()
        inst["task_flags"] = ["video editing", "reference generation"]
        inst["slot_assignments"]["A"]["character_sheet_images"] = []
        if with_bg:
            inst["slot_assignments"]["O"] = pa._build_background_slot({
                "name": "Beach", "description": "a sunlit beach", "lighting": "golden hour",
                "reference_images": ["beach.png"],
            })
        return inst

    def test_cast_background_replaces_the_setting_of_a_source_video(self):
        ve = [{"subject_id": "girl", "video_file": "src.mp4"}]
        prompt = pa.assemble_prompt(self._video_edit_instance(True), "h3_ref2va", ve)["prompt"]
        assert "edited version of <Video 1>, set in <Subject" in prompt
        assert "the original setting and lighting of <Video 1> are replaced by <Subject" in prompt
        assert "preserve setting details" not in prompt

    def test_source_video_setting_is_kept_without_a_cast_background(self):
        ve = [{"subject_id": "girl", "video_file": "src.mp4"}]
        prompt = pa.assemble_prompt(self._video_edit_instance(False), "h3_ref2va", ve)["prompt"]
        assert "preserve setting details, scene composition, lighting" in prompt
        assert "instead of its original setting" not in prompt

    def test_text_only_background_replaces_the_setting_as_a_description(self):
        inst = _scene_still_instance()
        inst["setting_text"] = "a sunlit beach"
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        layout = next(l for l in prompt.splitlines() if "(scene layout reference)" in l)
        assert "original setting and lighting of <Picture 1> are replaced by a sunlit beach" in layout
        assert "recreates the composition shown in <Picture 1> in the setting of a sunlit beach." in prompt
        assert prompt.count("<Subject ") == pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"].count("<Subject ")

    def test_text_only_background_replaces_the_setting_of_a_source_video(self):
        inst = self._video_edit_instance(False)
        inst["setting_text"] = "a sunlit beach"
        ve = [{"subject_id": "girl", "video_file": "src.mp4"}]
        prompt = pa.assemble_prompt(inst, "h3_ref2va", ve)["prompt"]
        assert "the original setting and lighting of <Video 1> are replaced by a sunlit beach" in prompt
        assert "preserve setting details" not in prompt

    def test_detailed_description_starts_at_shot_1_without_a_style(self):
        for setting in ("", "a sunlit beach"):
            inst = _scene_still_instance()
            inst["setting_text"] = setting
            dd = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"].split("detailed_description:")[1]
            assert dd.lstrip().startswith("[Shot 1]")

    def test_style_opens_detailed_description_and_converts_placeholders(self):
        inst = _scene_still_instance()
        inst["template"]["style"] = "a photorealistic recreation of <Picture 1> featuring {K}"
        dd = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"].split("detailed_description:")[1]
        assert dd.lstrip().startswith("A photorealistic recreation of <Picture 1> featuring <Subject 1>, ")
        assert dd.index("featuring") < dd.index("[Shot 1]")

    def test_preamble_describes_each_subject_after_its_first_tag_only(self):
        inst = _scene_still_instance()
        inst["template"]["style"] = "{A} waits. Later {A} smiles"
        dd = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"].split("detailed_description:")[1]
        pre = dd.split("[Shot 1]")[0]
        phrase = "the young woman with platinum braided hair"
        assert f"<Subject 1>, {phrase} waits. Later <Subject 1> smiles." in pre
        assert pre.count(phrase) == 1
        assert "<Subject 2>" not in pre

    def test_shot_one_still_describes_the_subject_after_a_described_preamble(self):
        inst = _scene_still_instance()
        inst["template"]["style"] = "{A} waits"
        shots = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"].split("detailed_description:")[1].split("[Shot 1]")[1]
        assert "<Subject 1>, the young woman with platinum braided hair sits on a chair" in shots

    def test_soundscape_and_music_stay_bare_labels(self):
        inst = _scene_still_instance()
        inst["template"]["overall_soundscape"] = "Room tone around {A}."
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        assert "Room tone around <Subject 1>." in prompt

    def test_still_source_has_no_video_edit_preamble_with_the_editing_tag(self):
        ve = [{"subject_id": "bun", "video_file": "bun.mp4"}]
        dd = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", ve)["prompt"].split("detailed_description:")[1]
        assert "identity-replacement" not in dd and "recreation of" not in dd

    def test_video_preamble_names_the_cast_setting_without_the_original(self):
        ve = [{"subject_id": "girl", "video_file": "src.mp4"}]
        dd = pa.assemble_prompt(self._video_edit_instance(True), "h3_ref2va", ve)["prompt"].split("detailed_description:")[1].split("[Shot 1]")[0]
        assert "retaining its original motion and camera work, set in <Subject" in dd
        assert "lighting" not in dd

    def test_soundscape_and_music_convert_subject_placeholders(self):
        inst = _scene_still_instance()
        inst["template"]["overall_soundscape"] = "{K} hums softly while {A} listens."
        inst["template"]["non_diegetic_music"] = "A theme for {K}."
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        sc = prompt.split("overall_soundscape:")[1].split("non_diegetic_music:")[0]
        assert "<Subject 1> hums softly while <Subject 1> listens." in sc
        assert "A theme for <Subject 1>." in prompt
        assert "{" not in sc

    def test_preamble_soundscape_and_music_name_the_replacement_not_the_replaced_subject(self):
        # {A} is the replaced source subject (<Subject 2>); like the shot text, every other
        # section must name the bundle subject that replaces it (<Subject 1>).
        inst = _scene_still_instance()
        inst["template"]["style"] = "a close portrait of {A}"
        inst["template"]["overall_soundscape"] = "Quiet room tone around {A}."
        inst["template"]["non_diegetic_music"] = "A theme for {A}."
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        dd = prompt.split("detailed_description:")[1]
        assert dd.lstrip().startswith("A close portrait of <Subject 1>, the young woman with platinum braided hair.")
        assert "Quiet room tone around <Subject 1>." in prompt
        assert "A theme for <Subject 1>." in prompt
        for section in (dd.split("[Shot 1]")[0], prompt.split("overall_soundscape:")[1]):
            assert "<Subject 2>" not in section

    def test_without_a_cast_background_the_scene_setting_is_kept(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        assert "setting details, scene composition, lighting and framing" in prompt
        assert "in the setting of" not in prompt

    def test_works_without_tagging_the_original_subject(self):
        prompt = pa.assemble_prompt(_scene_still_instance(include_tags=False), "h3_ref2va", [])["prompt"]
        assert "<Subject 2>" not in prompt
        assert "<Picture 1> (scene layout reference)" in prompt
        assert "match those of the girl in <Picture 1>" in prompt

    def test_action_placeholder_resolves_to_the_bundle_subject(self):
        prompt = pa.assemble_prompt(_scene_still_instance(), "h3_ref2va", [])["prompt"]
        assert "<Subject 1>, the young woman with platinum braided hair sits on a chair" in prompt

    def test_other_picture_roles_keep_the_appearance_wording(self):
        # A scene image that is not tagged as layout, or whose subject is retained, is still
        # an ordinary appearance reference.
        inst = _scene_still_instance(layout_role="character sheet")
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        assert "scene layout reference" not in prompt
        assert "<Picture 1>: fully_preserved" in prompt

        inst = _scene_still_instance()
        inst["slot_assignments"]["A"]["_cast_retention"] = "fully_preserved"
        inst["slot_assignments"]["A"].pop("_transfer_to_slot")
        prompt = pa.assemble_prompt(inst, "h3_ref2va", [])["prompt"]
        assert "scene layout reference" not in prompt
        assert "recreates the scene shown" not in prompt


class TestAttachSceneLayout:
    def _slots(self, a="fully_preserved", b="fully_preserved"):
        return {
            "A": {"_cast_retention": a, "character_sheet_images": []},
            "B": {"_cast_retention": b, "character_sheet_images": []},
            "K": {"_cast_retention": "attribute_transfer", "character_sheet_images": ["bun.png"]},
        }

    def test_goes_to_the_first_replaced_source_slot(self):
        slots = self._slots(b="replaced")
        assert pa.attach_scene_layout(slots, ["A", "B"], "s.png") == "B"
        assert slots["B"]["character_sheet_images"] == [{"file": "s.png", "role": pa.SCENE_LAYOUT_ROLE}]
        assert slots["A"]["character_sheet_images"] == []

    def test_falls_back_to_the_first_source_slot(self):
        slots = self._slots()
        assert pa.attach_scene_layout(slots, ["A", "B"], "s.png") == "A"

    def test_is_picture_1_ahead_of_the_bundle_images(self):
        slots = self._slots(b="replaced")
        pa.attach_scene_layout(slots, ["A", "B"], "s.png")
        inst = {"slot_assignments": slots}
        paths = [r["path"] for r in pa._build_h3_refplan(inst, [])["references"]]
        assert paths == ["s.png", "bun.png"]

    def test_nothing_to_attach(self):
        slots = self._slots()
        assert pa.attach_scene_layout(slots, [], "s.png") == ""
        assert pa.attach_scene_layout(slots, ["A"], "") == ""
        assert pa.attach_scene_layout(slots, ["Z"], "s.png") == ""
        assert slots["A"]["character_sheet_images"] == []
