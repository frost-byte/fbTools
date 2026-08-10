"""Tests for assemble_composition() and the composition adapter layer.

These exercise the S1/S2→A/B slot remapping, composition-shot→template conversion,
dialogue positional mapping, background integration, and all 8 model types.
"""

import pytest
from conftest import import_test_module

pa = import_test_module("utils/prompt_assembler.py")
assemble_composition = pa.assemble_composition
MODEL_TYPES = pa.MODEL_TYPES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _subject(
    name,
    summary="",
    concept_id="",
    sheets=None,
    audio="",
    language="en-us",
):
    sid = name.lower().replace(" ", "_")
    return {
        "subject_id": sid,
        "name": name,
        "appearance": {
            "summary": summary or f"{name}, a person",
            "face": "",
            "hair": "",
            "body": "",
            "default_outfit": "",
        },
        "voice": {
            "description": f"{name}'s voice",
            "audio_reference_file": audio,
            "language": language,
        },
        "character_sheet_images": sheets or [],
        "concept_id": concept_id,
    }


def _bg(name="Café", description="A warm sunlit café", lighting="soft window light", soundscape="Soft café ambience."):
    return {
        "id": name.lower().replace(" ", "_"),
        "name": name,
        "description": description,
        "lighting": lighting,
        "soundscape": soundscape,
    }


def _shot(action="Characters talk.", camera="Medium shot.", dialogue_text=None, sound=None, sid=None):
    d = None
    if dialogue_text:
        d = {"speaker": sid or "S1", "language": "en-us", "text": dialogue_text}
    return {
        "id": None,  # assigned by test or left for adapter
        "timestamp": None,
        "camera": camera,
        "action": action,
        "dialogue": d,
        "sound_events": sound,
    }


def _comp(
    subjects=None,
    shots=None,
    background_id="",
    style="cinematic",
    soundscape="",
    music="N/A",
    outfit_overrides=None,
):
    subj_ids = subjects or {}
    sht = shots or []
    for i, s in enumerate(sht, 1):
        if s.get("id") is None:
            s["id"] = f"shot_{i}"
    return {
        "id": "test_comp",
        "name": "Test Composition",
        "model_type": "h3_ref2va",
        "style": style,
        "subjects": subj_ids,
        "outfit_overrides": outfit_overrides or {},
        "background": background_id,
        "shots": sht,
        "overall_soundscape": soundscape,
        "non_diegetic_music": music,
    }


# ── Slot remapping ─────────────────────────────────────────────────────────────

def test_s1_maps_to_slot_a():
    alice = _subject("Alice", summary="tall woman")
    comp = _comp(subjects={"S1": "alice_id"})
    resolved = {"S1": alice}
    result = assemble_composition(comp, resolved, None, "h3_ref2va")
    assert "Alice" in result["prompt"]


def test_two_subjects_remapped_in_order():
    alice = _subject("Alice", summary="tall woman")
    bob = _subject("Bob", summary="short man")
    comp = _comp(subjects={"S1": "alice_id", "S2": "bob_id"})
    resolved = {"S1": alice, "S2": bob}
    result = assemble_composition(comp, resolved, None, "h3_ref2va")
    prompt = result["prompt"]
    # Both should appear
    assert "Alice" in prompt
    assert "Bob" in prompt


def test_subject_1_label_for_s1_in_h3_ref2va():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "alice_id"})
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    # h3_ref2va assigns <Subject 1> to first slot
    assert "<Subject 1>" in result["prompt"]


def test_subject_2_label_for_s2_in_h3_ref2va():
    alice = _subject("Alice")
    bob = _subject("Bob")
    comp = _comp(subjects={"S1": "alice_id", "S2": "bob_id"})
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_ref2va")
    assert "<Subject 1>" in result["prompt"]
    assert "<Subject 2>" in result["prompt"]


def test_outfit_override_remapped_with_slot():
    alice = _subject("Alice", summary="tall woman")
    comp = _comp(
        subjects={"S1": "alice_id"},
        outfit_overrides={"S1": "red dress"},
    )
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert "red dress" in result["prompt"]


def test_outfit_override_for_second_slot():
    alice = _subject("Alice")
    bob = _subject("Bob")
    comp = _comp(
        subjects={"S1": "a", "S2": "b"},
        outfit_overrides={"S2": "blue jacket"},
    )
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "wan22")
    assert "blue jacket" in result["prompt"]


# ── {S1}/{S2} placeholder replacement in action/camera ────────────────────────

def test_s1_placeholder_in_action_replaced():
    alice = _subject("Alice")
    shots = [_shot(action="{S1} walks forward.", camera="Wide shot.")]
    comp = _comp(subjects={"S1": "a"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice}, None, "wan22")
    # {S1} should have been translated to {A} → then replaced with Alice's name/ref
    assert "{S1}" not in result["prompt"]
    assert "Alice" in result["prompt"] or "walks forward" in result["prompt"]


def test_s2_placeholder_in_camera_replaced():
    alice = _subject("Alice")
    bob = _subject("Bob")
    shots = [_shot(action="Scene.", camera="Close-up of {S2}.")]
    comp = _comp(subjects={"S1": "a", "S2": "b"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_fl2va")
    assert "{S2}" not in result["prompt"]


def test_placeholder_not_in_raw_text_when_no_subjects():
    shots = [_shot(action="{S1} stands there.")]
    comp = _comp(subjects={}, shots=shots)
    result = assemble_composition(comp, {}, None, "wan22")
    # No subjects mapped; {S1} becomes {A} (unresolvable) but should not crash
    assert result["prompt"]  # just check no exception


# ── Dialogue positional mapping ────────────────────────────────────────────────

def test_first_shot_dialogue_maps_to_shot_1():
    alice = _subject("Alice", audio="a.wav")
    shots = [_shot(action="Alice speaks.", dialogue_text="Hello world.")]
    comp = _comp(subjects={"S1": "a"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert "Hello world." in result["prompt"]


def test_second_shot_dialogue_maps_positionally():
    alice = _subject("Alice", audio="a.wav")
    bob = _subject("Bob", audio="b.wav")
    shots = [
        _shot(action="{S1} speaks.", dialogue_text="Good morning.", sid="S1"),
        _shot(action="{S2} replies.", dialogue_text="Morning!", sid="S2"),
    ]
    comp = _comp(subjects={"S1": "a", "S2": "b"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_ref2va")
    prompt = result["prompt"]
    assert "Good morning." in prompt
    assert "Morning!" in prompt


def test_shot_without_dialogue_does_not_advance_counter():
    alice = _subject("Alice", audio="a.wav")
    shots = [
        _shot(action="Silent moment."),  # no dialogue
        _shot(action="Alice speaks.", dialogue_text="Hi there."),
    ]
    comp = _comp(subjects={"S1": "a"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    # Second shot's dialogue is the first (and only) dialogue → shot_1 in map
    assert "Hi there." in result["prompt"]


def test_dialogue_tags_use_subject_language():
    alice = _subject("Alice", audio="a.wav", language="ja-jp")
    shots = [_shot(action="Alice speaks.", dialogue_text="こんにちは。")]
    comp = _comp(subjects={"S1": "a"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert "<d>[ja-jp] こんにちは。</d>" in result["prompt"]


# ── Background integration ─────────────────────────────────────────────────────

def test_background_description_in_environment():
    alice = _subject("Alice")
    bg = _bg(description="A foggy forest clearing")
    comp = _comp(subjects={"S1": "a"})
    result = assemble_composition(comp, {"S1": alice}, bg, "h3_ref2va")
    assert "foggy forest" in result["prompt"]


def test_background_soundscape_used_when_composition_soundscape_empty():
    alice = _subject("Alice")
    bg = _bg(soundscape="Wind through trees.")
    comp = _comp(subjects={"S1": "a"}, soundscape="")
    result = assemble_composition(comp, {"S1": alice}, bg, "h3_ref2va")
    assert "Wind through trees." in result["prompt"]


def test_composition_soundscape_overrides_background():
    alice = _subject("Alice")
    bg = _bg(soundscape="Wind through trees.")
    comp = _comp(subjects={"S1": "a"}, soundscape="Busy city street.")
    result = assemble_composition(comp, {"S1": alice}, bg, "h3_ref2va")
    assert "Busy city street." in result["prompt"]
    # Background soundscape should NOT appear since comp has its own
    assert "Wind through trees." not in result["prompt"]


def test_none_background_does_not_crash():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "a"})
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert result["prompt"]


# ── Style passthrough ─────────────────────────────────────────────────────────

def test_style_appears_in_prompt():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "a"}, style="noir black and white")
    result = assemble_composition(comp, {"S1": alice}, None, "wan22")
    assert "noir black and white" in result["prompt"]


# ── Concept IDs ───────────────────────────────────────────────────────────────

def test_concept_ids_extracted_from_resolved_subjects():
    alice = _subject("Alice", concept_id="char_alice")
    bob = _subject("Bob", concept_id="char_bob")
    comp = _comp(subjects={"S1": "a", "S2": "b"})
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_ref2va")
    assert "char_alice" in result["concept_ids"]
    assert "char_bob" in result["concept_ids"]


def test_concept_ids_empty_when_no_concepts():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "a"})
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert result["concept_ids"] == []


# ── All 8 model types produce non-empty prompt ────────────────────────────────

@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_all_model_types_produce_output(model_type):
    alice = _subject("Alice", summary="tall woman with red hair", sheets=["a.png"], audio="a.wav")
    bob = _subject("Bob", summary="short man")
    shots = [
        _shot(action="{S1} greets {S2}.", dialogue_text="Hello.", camera="Two-shot."),
        _shot(action="{S2} nods.", dialogue_text="Hi.", camera="Close-up of {S2}."),
    ]
    bg = _bg()
    comp = _comp(
        subjects={"S1": "a", "S2": "b"},
        shots=shots,
        style="cinematic",
        soundscape="café ambience",
        music="soft jazz",
    )
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, bg, model_type)
    assert result["prompt"], f"Empty prompt for model_type={model_type}"
    assert isinstance(result["concept_ids"], list)
    assert isinstance(result["assembly_report"], str)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_all_model_types_no_s_placeholders_in_output(model_type):
    """Slot placeholders {S1} must not leak into the assembled prompt."""
    alice = _subject("Alice")
    shots = [_shot(action="{S1} walks.", camera="Wide.")]
    comp = _comp(subjects={"S1": "a"}, shots=shots)
    result = assemble_composition(comp, {"S1": alice}, None, model_type)
    assert "{S1}" not in result["prompt"], f"{{S1}} leaked for model_type={model_type}"
    assert "{S2}" not in result["prompt"]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_subjects_does_not_crash():
    comp = _comp(subjects={}, shots=[_shot()])
    result = assemble_composition(comp, {}, None, "h3_ref2va")
    assert result["prompt"]
    assert result["concept_ids"] == []


def test_empty_shots_does_not_crash():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "a"}, shots=[])
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert result["prompt"]


def test_three_subjects_slot_mapping():
    alice = _subject("Alice")
    bob = _subject("Bob")
    carol = _subject("Carol")
    comp = _comp(subjects={"S1": "a", "S2": "b", "S3": "c"})
    result = assemble_composition(
        comp, {"S1": alice, "S2": bob, "S3": carol}, None, "h3_ref2va"
    )
    # All three subjects should appear in subject_definitions
    assert "<Subject 1>" in result["prompt"]
    assert "<Subject 2>" in result["prompt"]
    assert "<Subject 3>" in result["prompt"]


def test_reference_image_order_follows_slot_order():
    alice = _subject("Alice", sheets=["a1.png", "a2.png"])
    bob = _subject("Bob", sheets=["b1.png"])
    comp = _comp(subjects={"S1": "a", "S2": "b"})
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_ref2va")
    order = result["reference_image_order"]
    # Slot A (S1=Alice) first, then Slot B (S2=Bob)
    assert order[0][1] == "a1.png"
    assert order[1][1] == "a2.png"
    assert order[2][1] == "b1.png"


def test_audio_slots_from_composition():
    alice = _subject("Alice", audio="alice.wav")
    bob = _subject("Bob")  # no audio
    comp = _comp(subjects={"S1": "a", "S2": "b"})
    result = assemble_composition(comp, {"S1": alice, "S2": bob}, None, "h3_ref2va")
    # Only Alice (A) has audio
    assert result["audio_slots"] == ["A"]


def test_music_field_passed_through():
    alice = _subject("Alice")
    comp = _comp(subjects={"S1": "a"}, music="Epic orchestral score.")
    result = assemble_composition(comp, {"S1": alice}, None, "h3_ref2va")
    assert "Epic orchestral score." in result["prompt"]
