"""Tests for the Scene Composition system."""

import pytest
from conftest import import_test_module

sc = import_test_module("utils/scene_compose.py")

compose_scene = sc.compose_scene
validate_scene = sc.validate_scene
format_scene_summary = sc.format_scene_summary


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_template(slot_ids=("A", "B"), n_dialogue_placeholders=2, needs_voice=True, needs_sheet=True):
    slots = {
        sid: {"role": f"speaker {sid}", "needs_voice": needs_voice, "needs_character_sheet": needs_sheet}
        for sid in slot_ids
    }
    shots = []
    for i, sid in enumerate(slot_ids[:n_dialogue_placeholders]):
        shots.append({
            "id": f"shot_{i+1}",
            "timestamp": None if i == 0 else f"00:0{i*4}.000",
            "camera": f"Medium shot of {{{sid}}}",
            "action": f"{{{sid}}} speaks.",
            "dialogue": {"speaker_slot": sid, "placeholder": True, "default_text": None},
            "sound_events": None,
        })
    # Add a non-dialogue shot
    shots.append({
        "id": f"shot_{len(shots)+1}",
        "timestamp": "00:08.000",
        "camera": "Wide shot",
        "action": "Scene continues.",
        "dialogue": None,
        "sound_events": "ambient noise",
    })
    return {
        "version": 1,
        "id": "test_template",
        "name": "Test Template",
        "description": "A test template",
        "slots": slots,
        "environment": {"summary": "a test environment", "lighting": "natural"},
        "style": "cinematic",
        "shots": shots,
        "overall_soundscape": "ambient",
        "non_diegetic_music": "N/A",
    }


def _make_subject(name, concept_id="", sheets=None, audio=""):
    return {
        "subject_id": name.lower().replace(" ", "_"),
        "name": name,
        "appearance": {
            "summary": f"a person named {name}",
            "face": "distinctive features",
            "hair": "natural hair",
            "body": "average build",
            "default_outfit": "casual clothes",
        },
        "voice": {
            "description": f"{name}'s voice",
            "audio_reference_file": audio,
            "language": "en-us",
        },
        "character_sheet_images": sheets or [],
        "concept_id": concept_id,
    }


# ── compose_scene ──────────────────────────────────────────────────────────────

def test_compose_basic():
    template = _make_template()
    alice = _make_subject("Alice", concept_id="alice_c")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template,
        {"A": alice, "B": bob},
        ["Hello there.", "Hi back."],
        {},
    )
    assert instance["template_id"] == "test_template"
    assert instance["template_name"] == "Test Template"
    assert "A" in instance["slot_assignments"]
    assert "B" in instance["slot_assignments"]


def test_compose_maps_dialogue_positionally():
    template = _make_template(slot_ids=("A", "B"), n_dialogue_placeholders=2)
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template,
        {"A": alice, "B": bob},
        ["First line.", "Second line."],
        {},
    )
    # shot_1 is slot A's placeholder, shot_2 is slot B's
    assert instance["dialogue"]["shot_1"] == "First line."
    assert instance["dialogue"]["shot_2"] == "Second line."


def test_compose_empty_dialogue_strings_skipped():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template,
        {"A": alice, "B": bob},
        ["  ", ""],
        {},
    )
    assert instance["dialogue"] == {}


def test_compose_partial_dialogue():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template,
        {"A": alice, "B": bob},
        ["Only A speaks."],
        {},
    )
    assert "shot_1" in instance["dialogue"]
    assert "shot_2" not in instance["dialogue"]


def test_compose_filters_none_slots():
    template = _make_template()
    alice = _make_subject("Alice")
    instance = compose_scene(
        template,
        {"A": alice, "B": None},
        [],
        {},
    )
    assert "A" in instance["slot_assignments"]
    assert "B" not in instance["slot_assignments"]


def test_compose_applies_outfit_overrides():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template,
        {"A": alice, "B": bob},
        [],
        {"A": "red dress", "B": ""},
    )
    assert instance["outfit_overrides"] == {"A": "red dress"}


def test_compose_blank_outfit_overrides_stripped():
    template = _make_template()
    alice = _make_subject("Alice")
    instance = compose_scene(template, {"A": alice}, [], {"A": "   "})
    assert instance["outfit_overrides"] == {}


def test_compose_does_not_mutate_inputs():
    template = _make_template()
    alice = _make_subject("Alice")
    original_name = alice["name"]
    compose_scene(template, {"A": alice}, [], {})
    assert alice["name"] == original_name


def test_compose_embeds_full_template():
    template = _make_template()
    alice = _make_subject("Alice")
    instance = compose_scene(template, {"A": alice}, [], {})
    assert instance["template"]["id"] == "test_template"
    assert "shots" in instance["template"]


def test_compose_deep_copies_subjects():
    template = _make_template()
    alice = _make_subject("Alice")
    instance = compose_scene(template, {"A": alice}, [], {})
    instance["slot_assignments"]["A"]["name"] = "MUTATED"
    assert alice["name"] == "Alice"


# ── validate_scene ─────────────────────────────────────────────────────────────

def test_validate_all_slots_filled_returns_empty():
    template = _make_template(needs_voice=False, needs_sheet=False)
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    warnings = validate_scene(template, {"A": alice, "B": bob})
    assert warnings == []


def test_validate_missing_slot_warns():
    template = _make_template()
    alice = _make_subject("Alice")
    warnings = validate_scene(template, {"A": alice, "B": None})
    assert any("Slot B" in w for w in warnings)
    assert any("not assigned" in w for w in warnings)


def test_validate_all_missing_warns_each():
    template = _make_template()
    warnings = validate_scene(template, {})
    assert any("Slot A" in w for w in warnings)
    assert any("Slot B" in w for w in warnings)


def test_validate_missing_voice_reference_warns():
    template = _make_template(needs_voice=True)
    alice = _make_subject("Alice", audio="")  # no audio file
    bob = _make_subject("Bob")
    warnings = validate_scene(template, {"A": alice, "B": bob})
    assert any("Slot A" in w and "voice" in w for w in warnings)


def test_validate_missing_character_sheet_warns():
    template = _make_template(needs_sheet=True)
    alice = _make_subject("Alice", sheets=[])
    bob = _make_subject("Bob")
    warnings = validate_scene(template, {"A": alice, "B": bob})
    assert any("Slot A" in w and "character sheet" in w for w in warnings)


def test_validate_with_voice_and_sheets_passes():
    template = _make_template(needs_voice=True, needs_sheet=True)
    alice = _make_subject("Alice", sheets=["alice.png"], audio="alice.wav")
    bob = _make_subject("Bob", sheets=["bob.png"], audio="bob.wav")
    warnings = validate_scene(template, {"A": alice, "B": bob})
    assert warnings == []


# ── format_scene_summary ───────────────────────────────────────────────────────

def test_summary_includes_template_name():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(template, {"A": alice, "B": bob}, [], {})
    summary = format_scene_summary(instance)
    assert "Test Template" in summary


def test_summary_includes_slot_names():
    template = _make_template()
    alice = _make_subject("Alice", concept_id="alice_c")
    bob = _make_subject("Bob")
    instance = compose_scene(template, {"A": alice, "B": bob}, [], {})
    summary = format_scene_summary(instance)
    assert "Alice" in summary
    assert "Bob" in summary
    assert "alice_c" in summary


def test_summary_shows_media_counts():
    template = _make_template()
    alice = _make_subject("Alice", sheets=["a1.png", "a2.png"], audio="a.wav")
    bob = _make_subject("Bob")
    instance = compose_scene(template, {"A": alice, "B": bob}, [], {})
    summary = format_scene_summary(instance)
    assert "2 sheets" in summary
    assert "voice" in summary


def test_summary_includes_dialogue():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template, {"A": alice, "B": bob},
        ["Hello there.", "Hi!"], {}
    )
    summary = format_scene_summary(instance)
    assert "Hello there." in summary
    assert "Hi!" in summary


def test_summary_includes_outfit_override():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(
        template, {"A": alice, "B": bob}, [],
        {"A": "red dress"}
    )
    summary = format_scene_summary(instance)
    assert "red dress" in summary
    assert "casual clothes" in summary  # "overrides: casual clothes"


def test_summary_shows_validation_ok():
    template = _make_template(needs_voice=False, needs_sheet=False)
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    instance = compose_scene(template, {"A": alice, "B": bob}, [], {})
    summary = format_scene_summary(instance, validation_warnings=[])
    assert "Validation: OK" in summary


def test_summary_shows_validation_warnings():
    template = _make_template()
    alice = _make_subject("Alice")
    instance = compose_scene(template, {"A": alice, "B": None}, [], {})
    warnings = validate_scene(template, {"A": alice, "B": None})
    summary = format_scene_summary(instance, validation_warnings=warnings)
    assert "⚠" in summary
    assert "Slot B" in summary


def test_summary_truncates_long_dialogue():
    template = _make_template()
    alice = _make_subject("Alice")
    bob = _make_subject("Bob")
    long_line = "A" * 100
    instance = compose_scene(template, {"A": alice, "B": bob}, [long_line, ""], {})
    summary = format_scene_summary(instance)
    assert "…" in summary


def test_summary_no_slots_assigned():
    template = _make_template()
    instance = compose_scene(template, {}, [], {})
    summary = format_scene_summary(instance)
    assert "none assigned" in summary
