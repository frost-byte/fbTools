"""Tests for utils/subject_merge.py (pure logic behind scripts/merge_subjects.py)."""
import json

import pytest

from conftest import import_test_module

sm = import_test_module("utils/subject_merge.py")


def _subjects():
    return {
        "a": {"name": "a", "appearance": {"summary": "tall", "hair": ""}, "concept_id": "",
              "pronoun_style": "feminine", "short_name": "woman", "voice": {},
              "character_sheet_images": [{"file": "a1.png", "role": "portrait"}]},
        "b": {"name": "b", "appearance": {"summary": "short", "hair": "red"}, "concept_id": "cb",
              "pronoun_style": "feminine", "short_name": "", "voice": {"language": "en"},
              "character_sheet_images": [{"file": "a1.png", "role": "portrait"}, "b2.png"]},
        "c": {"name": "c", "appearance": {"summary": "x"}, "concept_id": "cc", "pronoun_style": "feminine"},
    }


def test_parse_source_spec():
    assert sm.parse_source_spec("alex") == ("alex", None)
    assert sm.parse_source_spec(" alex : bun ") == ("alex", "bun")
    for bad in ("", ":bun", "alex:"):
        with pytest.raises(ValueError):
            sm.parse_source_spec(bad)


def test_merged_profile_base_is_primary_and_fills_empty_fields():
    m = sm.build_merged_profile(_subjects(), "merged", "a", ["a", "b", "c"])
    assert m["name"] == "merged"
    assert m["appearance"]["summary"] == "tall"       # primary wins
    assert m["appearance"]["hair"] == "red"            # empty sub-key filled from b
    assert m["concept_id"] == "cb"                     # first non-empty donor
    assert m["short_name"] == "woman"
    assert m["voice"] == {"language": "en"}


def test_merged_profile_unions_sheet_images_without_duplicates():
    m = sm.build_merged_profile(_subjects(), "merged", "a", ["a", "b"])
    files = [e["file"] if isinstance(e, dict) else e for e in m["character_sheet_images"]]
    assert files == ["a1.png", "b2.png"]


def test_existing_target_is_the_base_and_is_not_overwritten():
    subs = _subjects()
    subs["merged"] = {"name": "merged", "appearance": {"summary": "keep me"}, "concept_id": "", "pronoun_style": "feminine"}
    m = sm.build_merged_profile(subs, "merged", "a", ["a", "b"])
    assert m["appearance"]["summary"] == "keep me"
    assert m["concept_id"] == "cb"


def test_inputs_are_not_mutated():
    subs = _subjects()
    before = json.dumps(subs, sort_keys=True)
    sm.build_merged_profile(subs, "merged", "a", ["a", "b"])
    assert json.dumps(subs, sort_keys=True) == before


def _bundles():
    return {"b1": {"subject_id": "a"}, "b2": {"subject_id": "b"}, "b3": {"subject_id": "b"},
            "b4": {"subject_id": "other"}}


def test_retarget_whole_subjects_and_single_bundles():
    bundles = _bundles()
    moved, warns = sm.retarget_bundles(bundles, "m", {"a"}, [("b", "b3")])
    assert sorted(moved) == ["b1", "b3"]
    assert bundles["b1"]["subject_id"] == "m" and bundles["b3"]["subject_id"] == "m"
    assert bundles["b2"]["subject_id"] == "b" and warns == []


def test_retarget_warns_on_unknown_or_foreign_bundle():
    bundles = _bundles()
    moved, warns = sm.retarget_bundles(bundles, "m", set(), [("a", "nope"), ("a", "b2")])
    assert moved == [] and len(warns) == 2
    assert bundles["b2"]["subject_id"] == "b"


def test_rewrite_composition_repoints_slots_and_snapshots():
    comp = {"subjects": {"A": "a", "B": "z"}, "_subject_snapshots": {"A": {"name": "a"}, "B": {"name": "z"}}}
    merged = {"name": "m"}
    assert sm.rewrite_composition(comp, {"a": "m"}, merged) is True
    assert comp["subjects"] == {"A": "m", "B": "z"}
    assert comp["_subject_snapshots"]["A"] == {"name": "m"} and comp["_subject_snapshots"]["B"] == {"name": "z"}
    assert sm.rewrite_composition(comp, {"a": "m"}, merged) is False   # idempotent


def test_rewrite_scene_casts():
    casts = {"c1": {"entries": [{"subject_id": "a"}, {"subject_id": "z"}]}, "c2": {"entries": []}}
    assert sm.rewrite_scene_casts(casts, {"a": "m"}) == 1
    assert casts["c1"]["entries"][0]["subject_id"] == "m"


def test_rewrite_workflow_fixes_cast_entry_strings_in_both_forms():
    entries = json.dumps([{"subject_id": "a", "bundle_id": "b1"}, {"subject_id": "z"}])
    wf = {"nodes": [{"widgets_values": [entries, 3], "widgets_values_named": {"cast_entries_json": entries}}],
          "note": "[not json subject_id"}
    assert sm.rewrite_workflow(wf, {"a": "m"}) == 2
    fixed = json.loads(wf["nodes"][0]["widgets_values"][0])
    assert fixed[0]["subject_id"] == "m" and fixed[1]["subject_id"] == "z"
    assert json.loads(wf["nodes"][0]["widgets_values_named"]["cast_entries_json"])[0]["subject_id"] == "m"
    assert wf["note"] == "[not json subject_id"
    assert sm.rewrite_workflow(wf, {"a": "m"}) == 0
