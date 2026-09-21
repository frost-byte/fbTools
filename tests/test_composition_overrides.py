"""Tests for apply_composition_overrides (SceneCastBuild background override) and its
effect on assemble_composition."""
import copy

from conftest import import_test_module

pc = import_test_module("utils/prompt_compositions.py")
pa = import_test_module("utils/prompt_assembler.py")

apply_overrides = pc.apply_composition_overrides


def _bgs():
    return {
        "beach": {"id": "beach", "name": "Beach", "description": "a sunlit beach", "lighting": "golden hour",
                  "soundscape": "waves", "reference_images": [{"file": "beach1.png", "role": "scene reference"}]},
        "cafe": {"id": "cafe", "name": "Cafe", "description": "a warm cafe", "lighting": "window light",
                 "soundscape": "murmur", "reference_images": []},
    }


def _comp(**kw):
    c = {"id": "c1", "name": "C1", "model_type": "h3_ref2va", "style": "cinematic",
         "subjects": {"A": "alice"}, "background": "cafe", "background_as_reference": False,
         "shots": [{"id": "s1", "camera": "wide", "action": "{A} walks.", "dialogue": None, "sound_events": None}],
         "overall_soundscape": "", "non_diegetic_music": "N/A"}
    c.update(kw)
    return c


def _alice():
    return {"subject_id": "alice", "name": "Alice",
            "appearance": {"summary": "tall woman", "face": "", "hair": "", "body": "", "default_outfit": ""},
            "voice": {"description": "v", "audio_reference_file": "", "language": "en-us"},
            "character_sheet_images": [], "concept_id": ""}


def test_no_overrides_returns_equal_copy():
    comp = _comp()
    out, warns = apply_overrides(comp, None, _bgs())
    assert out == comp and out is not comp and warns == []
    out, warns = apply_overrides(comp, {}, _bgs())
    assert out == comp and warns == []


def test_background_swap_sets_id_and_snapshot():
    comp = _comp(_background_snapshot={"id": "cafe"})
    out, warns = apply_overrides(comp, {"background": "beach"}, _bgs())
    assert out["background"] == "beach"
    assert out["_background_snapshot"]["id"] == "beach"
    assert warns == []


def test_background_none_clears_everything():
    comp = _comp(_background_snapshot={"id": "cafe"})
    out, _ = apply_overrides(comp, {"background": "none"}, _bgs())
    assert out["background"] == "" and "_background_snapshot" not in out


def test_unknown_background_warns_and_keeps_original():
    comp = _comp()
    out, warns = apply_overrides(comp, {"background": "ghost"}, _bgs())
    assert out["background"] == "cafe"
    assert len(warns) == 1 and "ghost" in warns[0]


def test_as_reference_flag_is_applied_both_ways():
    out, _ = apply_overrides(_comp(), {"background_as_reference": True}, _bgs())
    assert out["background_as_reference"] is True
    out, _ = apply_overrides(_comp(background_as_reference=True), {"background_as_reference": False}, _bgs())
    assert out["background_as_reference"] is False


def test_input_not_mutated():
    comp = _comp()
    before = copy.deepcopy(comp)
    apply_overrides(comp, {"background": "beach", "background_as_reference": True}, _bgs())
    assert comp == before


def test_assembly_follows_override_with_reference_images():
    bgs = _bgs()
    comp, _ = apply_overrides(_comp(), {"background": "beach", "background_as_reference": True}, bgs)
    bg = pc.resolve_background(comp, bgs)
    with_ref = pa.assemble_composition(comp, {"A": _alice()}, bg, "h3_ref2va")
    assert "sunlit beach" in with_ref["prompt"]

    comp_off, _ = apply_overrides(_comp(), {"background": "beach", "background_as_reference": False}, bgs)
    bg_off = pc.resolve_background(comp_off, bgs)
    without_ref = pa.assemble_composition(comp_off, {"A": _alice()}, bg_off, "h3_ref2va")
    assert "sunlit beach" in without_ref["prompt"]
    # The reference variant mints an extra subject slot for the background images.
    assert with_ref["prompt"] != without_ref["prompt"]


def test_ref_plan_from_assembled_scene_includes_background_images():
    """The H3 ref plan must be built from assemble_composition's own scene_instance so the
    background-reference slot (minted during assembly) is present and ordinals match."""
    bgs = _bgs()
    comp, _ = apply_overrides(_comp(), {"background": "beach", "background_as_reference": True}, bgs)
    bg = pc.resolve_background(comp, bgs)
    alice = _alice()
    alice["character_sheet_images"] = [{"file": "alice1.png", "role": "portrait"}]
    result = pa.assemble_composition(comp, {"A": alice}, bg, "h3_ref2va")

    plan = pa._build_h3_refplan(result["scene_instance"])
    paths = [r["path"] for r in plan["references"] if r["modality"] == "image"]
    assert paths == ["alice1.png", "beach1.png"]
    ordinals = [r["picture_ordinal"] for r in plan["references"] if r["modality"] == "image"]
    assert ordinals == [1, 2]
    # The prompt references both pictures too (subject + background).
    assert result["reference_image_order"] == [("A", "alice1.png"), ("B", "beach1.png")]

    # Flag off -> the background image is not in the plan.
    comp_off, _ = apply_overrides(_comp(), {"background": "beach", "background_as_reference": False}, bgs)
    result_off = pa.assemble_composition(comp_off, {"A": alice}, pc.resolve_background(comp_off, bgs), "h3_ref2va")
    plan_off = pa._build_h3_refplan(result_off["scene_instance"])
    assert [r["path"] for r in plan_off["references"] if r["modality"] == "image"] == ["alice1.png"]
