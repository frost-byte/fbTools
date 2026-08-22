"""Tests for the Source Profile system."""

import json
import os
import pytest
from conftest import import_test_module

sp = import_test_module("utils/source_profiles.py")

SourceProfileRegistry = sp.SourceProfileRegistry
load_registry         = sp.load_registry
save_registry         = sp.save_registry
ENTITY_TYPES          = sp.ENTITY_TYPES
MEDIA_TYPES           = sp.MEDIA_TYPES
MEDIA_DIRS            = sp.MEDIA_DIRS


# ── define_profile ─────────────────────────────────────────────────────────────

def test_define_profile_creates_entry():
    reg = SourceProfileRegistry()
    reg2 = reg.define_profile("prof_1", name="Theater Scene", media_filename="video_1.mp4",
                               media_dir="input", media_type="video")
    assert "prof_1" in reg2.profiles
    p = reg2.profiles["prof_1"]
    assert p["name"] == "Theater Scene"
    assert p["media_filename"] == "video_1.mp4"
    assert p["media_type"] == "video"
    assert p["media_dir"] == "input"
    assert p["subjects"] == []


def test_define_profile_does_not_mutate_original():
    reg = SourceProfileRegistry()
    reg.define_profile("prof_a", name="A")
    assert "prof_a" not in reg.profiles


def test_define_profile_preserves_subjects_on_update():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_b", name="B", media_filename="b.mp4")
    reg = reg.define_subject("prof_b", "subj_1", label="person A")
    reg = reg.define_profile("prof_b", name="B Updated", media_filename="b_new.mp4")
    assert reg.profiles["prof_b"]["name"] == "B Updated"
    assert len(reg.profiles["prof_b"]["subjects"]) == 1
    assert reg.profiles["prof_b"]["subjects"][0]["label"] == "person A"


def test_define_profile_defaults_name_to_id():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_unnamed")
    assert reg.profiles["prof_unnamed"]["name"] == "prof_unnamed"


def test_define_profile_rejects_invalid_media_dir():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_c", media_dir="invalid_dir")
    assert reg.profiles["prof_c"]["media_dir"] == "input"


def test_define_profile_rejects_invalid_media_type():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_d", media_type="audio")
    assert reg.profiles["prof_d"]["media_type"] == "video"


def test_define_profile_accepts_image_type():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_e", media_type="image", media_filename="ref.png")
    assert reg.profiles["prof_e"]["media_type"] == "image"


def test_define_profile_output_dir():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_f", media_dir="output", media_filename="out.mp4")
    assert reg.profiles["prof_f"]["media_dir"] == "output"


# ── define_subject ─────────────────────────────────────────────────────────────

def test_define_subject_appends_to_profile():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", name="P1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_girl", label="girl (original)",
                              role_description="the girl originally in the video",
                              entity_type="person")
    subj = reg.get_subject("prof_1", "subj_girl")
    assert subj is not None
    assert subj["label"] == "girl (original)"
    assert subj["role_description"] == "the girl originally in the video"
    assert subj["entity_type"] == "person"


def test_define_subject_does_not_mutate_original():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    original = reg
    reg.define_subject("prof_1", "subj_x", label="X")
    assert reg.get_subject("prof_1", "subj_x") is None


def test_define_subject_updates_existing_by_id():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_a", label="first label")
    reg = reg.define_subject("prof_1", "subj_a", label="updated label")
    assert len(reg.profiles["prof_1"]["subjects"]) == 1
    assert reg.get_subject("prof_1", "subj_a")["label"] == "updated label"


def test_define_subject_preserves_order():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_1", label="first")
    reg = reg.define_subject("prof_1", "subj_2", label="second")
    reg = reg.define_subject("prof_1", "subj_3", label="third")
    ids = reg.subject_ids("prof_1")
    assert ids == ["subj_1", "subj_2", "subj_3"]


def test_define_subject_creates_profile_shell_if_absent():
    reg = SourceProfileRegistry()
    reg = reg.define_subject("new_prof", "subj_x", label="X")
    assert "new_prof" in reg.profiles
    assert reg.get_subject("new_prof", "subj_x") is not None


def test_define_subject_rejects_invalid_entity_type():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_bad", entity_type="robot")
    assert reg.get_subject("prof_1", "subj_bad")["entity_type"] == "person"


def test_define_subject_accepts_all_entity_types():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    for etype in ENTITY_TYPES:
        reg = reg.define_subject("prof_1", f"subj_{etype}", entity_type=etype)
        assert reg.get_subject("prof_1", f"subj_{etype}")["entity_type"] == etype


def test_define_subject_defaults_label_to_id():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_unlabelled")
    assert reg.get_subject("prof_1", "subj_unlabelled")["label"] == "subj_unlabelled"


def test_multiple_profiles_accumulate():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_a", media_filename="a.mp4")
    reg = reg.define_profile("prof_b", media_filename="b.mp4")
    assert set(reg.profile_ids()) == {"prof_a", "prof_b"}


# ── remove_subject / remove_profile ────────────────────────────────────────────

def test_remove_subject_removes_correctly():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg = reg.define_subject("prof_1", "subj_a", label="A")
    reg = reg.define_subject("prof_1", "subj_b", label="B")
    reg = reg.remove_subject("prof_1", "subj_a")
    assert reg.get_subject("prof_1", "subj_a") is None
    assert reg.get_subject("prof_1", "subj_b") is not None


def test_remove_subject_noop_for_missing():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg2 = reg.remove_subject("prof_1", "nonexistent")
    assert reg2.subject_ids("prof_1") == []


def test_remove_subject_noop_for_missing_profile():
    reg = SourceProfileRegistry()
    reg2 = reg.remove_subject("nonexistent_prof", "subj_x")
    assert reg2.profile_ids() == []


def test_remove_profile_removes_correctly():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_a", media_filename="a.mp4")
    reg = reg.define_profile("prof_b", media_filename="b.mp4")
    reg = reg.remove_profile("prof_a")
    assert "prof_a" not in reg.profiles
    assert "prof_b" in reg.profiles


def test_remove_profile_noop_for_missing():
    reg = SourceProfileRegistry()
    reg2 = reg.remove_profile("nonexistent")
    assert reg2.profile_ids() == []


# ── queries ────────────────────────────────────────────────────────────────────

def test_get_profile_returns_none_for_missing():
    reg = SourceProfileRegistry()
    assert reg.get_profile("nonexistent") is None


def test_get_subject_returns_none_for_missing_profile():
    reg = SourceProfileRegistry()
    assert reg.get_subject("nonexistent_prof", "subj_x") is None


def test_get_subject_returns_none_for_missing_subject():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    assert reg.get_subject("prof_1", "nonexistent_subj") is None


def test_profile_ids_empty():
    reg = SourceProfileRegistry()
    assert reg.profile_ids() == []


def test_subject_ids_empty_profile():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    assert reg.subject_ids("prof_1") == []


def test_subject_ids_missing_profile():
    reg = SourceProfileRegistry()
    assert reg.subject_ids("nonexistent") == []


# ── build_subject_wire_dict ────────────────────────────────────────────────────

def test_build_subject_wire_dict_correct_structure():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", name="Theater", media_filename="video_1.mp4",
                              media_dir="input", media_type="video")
    reg = reg.define_subject("prof_1", "subj_man", label="man (left)",
                              role_description="the man on the left",
                              entity_type="person", notes="sits front row")
    wire = reg.build_subject_wire_dict("prof_1", "subj_man")
    assert wire is not None
    assert wire["subject_id"]        == "subj_man"
    assert wire["name"]              == "man (left)"
    assert wire["concept_id"]        is None
    assert wire["source_profile_id"] == "prof_1"
    assert wire["source_media_type"] == "video"
    assert wire["source_media_file"] == "video_1.mp4"
    assert wire["source_media_dir"]  == "input"
    assert wire["role_description"]  == "the man on the left"
    assert wire["entity_type"]       == "person"


def test_build_subject_wire_dict_returns_none_missing_profile():
    reg = SourceProfileRegistry()
    assert reg.build_subject_wire_dict("nonexistent", "subj_x") is None


def test_build_subject_wire_dict_returns_none_missing_subject():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    assert reg.build_subject_wire_dict("prof_1", "nonexistent") is None


def test_build_subject_wire_dict_object_entity():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4", media_type="video")
    reg = reg.define_subject("prof_1", "subj_table", label="table",
                              role_description="the small wooden table",
                              entity_type="object")
    wire = reg.build_subject_wire_dict("prof_1", "subj_table")
    assert wire["entity_type"] == "object"


def test_build_subject_wire_dict_image_source():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_img", media_filename="ref.png",
                              media_dir="input", media_type="image")
    reg = reg.define_subject("prof_img", "subj_person", label="person",
                              role_description="the person in the reference image")
    wire = reg.build_subject_wire_dict("prof_img", "subj_person")
    assert wire["source_media_type"] == "image"
    assert wire["source_media_file"] == "ref.png"


# ── list_profiles ──────────────────────────────────────────────────────────────

def test_list_profiles_empty():
    reg = SourceProfileRegistry()
    assert "No source profiles" in reg.list_profiles()


def test_list_profiles_shows_entries():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", name="Theater Scene", media_filename="v.mp4",
                              media_type="video")
    reg = reg.define_subject("prof_1", "subj_girl", label="girl (original)",
                              entity_type="person",
                              role_description="the girl in the video")
    reg = reg.define_subject("prof_1", "subj_table", label="table", entity_type="object",
                              role_description="the small wooden table")
    listing = reg.list_profiles()
    assert "Theater Scene" in listing
    assert "girl (original)" in listing
    assert "table" in listing
    assert "[person]" in listing
    assert "[object]" in listing


def test_list_profiles_filter_by_type_video():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_v", name="Video", media_filename="v.mp4", media_type="video")
    reg = reg.define_profile("prof_i", name="Image", media_filename="i.png", media_type="image")
    listing = reg.list_profiles(filter_type="video")
    assert "Video" in listing
    assert "Image" not in listing


def test_list_profiles_filter_by_type_image():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_v", name="Video", media_filename="v.mp4", media_type="video")
    reg = reg.define_profile("prof_i", name="Image", media_filename="i.png", media_type="image")
    listing = reg.list_profiles(filter_type="image")
    assert "Image" in listing
    assert "Video" not in listing


def test_list_profiles_filter_no_match():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_v", media_filename="v.mp4", media_type="video")
    listing = reg.list_profiles(filter_type="image")
    assert "No source profiles match" in listing


# ── serialisation ──────────────────────────────────────────────────────────────

def test_to_dict_round_trips():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", name="Theater", media_filename="video_1.mp4",
                              media_type="video")
    reg = reg.define_subject("prof_1", "subj_man", label="man (left)",
                              role_description="the man on the left", entity_type="person")
    data = reg.to_dict()
    assert data["version"] == 1
    assert "prof_1" in data["profiles"]

    reg2 = SourceProfileRegistry.from_dict(data)
    assert reg2.get_profile("prof_1")["name"] == "Theater"
    subj = reg2.get_subject("prof_1", "subj_man")
    assert subj is not None
    assert subj["label"] == "man (left)"
    assert subj["entity_type"] == "person"


def test_from_dict_handles_missing_version():
    data = {
        "profiles": {
            "prof_1": {
                "name": "P1",
                "media_filename": "v.mp4",
                "media_type": "video",
                "media_dir": "input",
                "subjects": [],
            }
        }
    }
    reg = SourceProfileRegistry.from_dict(data)
    assert reg.version == 1
    assert "prof_1" in reg.profiles


def test_from_dict_normalises_subjects():
    data = {
        "version": 1,
        "profiles": {
            "prof_1": {
                "name": "P1",
                "media_filename": "v.mp4",
                "media_type": "video",
                "media_dir": "input",
                "subjects": [
                    {"id": "s1", "label": "L1", "role_description": "R1",
                     "entity_type": "person", "notes": ""},
                ],
            }
        },
    }
    reg = SourceProfileRegistry.from_dict(data)
    subj = reg.get_subject("prof_1", "s1")
    assert subj["label"] == "L1"
    assert subj["entity_type"] == "person"


# ── persistence ────────────────────────────────────────────────────────────────

def test_load_registry_missing_file_returns_empty(tmp_path):
    reg = load_registry(str(tmp_path / "nonexistent.json"))
    assert reg.profiles == {}
    assert reg.version == 1


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "source_profiles.json")
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", name="Theater", media_filename="video_1.mp4",
                              media_type="video")
    reg = reg.define_subject("prof_1", "subj_girl", label="girl (original)",
                              role_description="the girl in the video", entity_type="person")
    save_registry(reg, path, backup=False)

    assert os.path.exists(path)
    loaded = load_registry(path)
    assert "prof_1" in loaded.profiles
    assert loaded.profiles["prof_1"]["name"] == "Theater"
    subj = loaded.get_subject("prof_1", "subj_girl")
    assert subj is not None
    assert subj["label"] == "girl (original)"


def test_save_creates_backup(tmp_path):
    path = str(tmp_path / "source_profiles.json")
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    save_registry(reg, path, backup=False)
    save_registry(reg, path, backup=True)
    assert os.path.exists(path + ".bak")


def test_save_creates_parent_directories(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "source_profiles.json")
    reg = SourceProfileRegistry()
    save_registry(reg, path, backup=False)
    assert os.path.exists(path)


def test_load_registry_sets_file_path(tmp_path):
    path = str(tmp_path / "source_profiles.json")
    reg = SourceProfileRegistry()
    save_registry(reg, path, backup=False)
    loaded = load_registry(path)
    assert loaded.file_path == path


def test_save_updates_file_path_on_registry(tmp_path):
    path = str(tmp_path / "source_profiles.json")
    reg = SourceProfileRegistry()
    save_registry(reg, path, backup=False)
    assert reg.file_path == path


# ── save convenience method ────────────────────────────────────────────────────

def test_registry_save_convenience(tmp_path):
    path = str(tmp_path / "source_profiles.json")
    reg = SourceProfileRegistry(file_path=path)
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    reg.file_path = path
    saved_path = reg.save(backup=False)
    assert saved_path == path
    assert os.path.exists(path)


def test_registry_save_raises_without_path():
    reg = SourceProfileRegistry()
    with pytest.raises(ValueError):
        reg.save()


def test_registry_save_explicit_path_overrides(tmp_path):
    path = str(tmp_path / "explicit.json")
    reg = SourceProfileRegistry()
    reg = reg.define_profile("prof_1", media_filename="v.mp4")
    saved = reg.save(path=path, backup=False)
    assert saved == path
    assert os.path.exists(path)
