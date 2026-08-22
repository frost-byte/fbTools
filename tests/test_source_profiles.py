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


# ── Constants ──────────────────────────────────────────────────────────────────

def test_default_segment_duration_constant():
    assert sp.DEFAULT_SEGMENT_DURATION == 10.0

def test_default_select_every_nth_constant():
    assert sp.DEFAULT_SELECT_EVERY_NTH == 2

def test_default_frame_load_cap_constant():
    assert sp.DEFAULT_FRAME_LOAD_CAP == 120


# ── _normalize_clip ────────────────────────────────────────────────────────────

def test_normalize_clip_fills_defaults():
    c = sp._normalize_clip({"id": "clip_1"})
    assert c["id"] == "clip_1"
    assert c["label"] == ""
    assert c["start_time"] == 0.0
    assert c["end_time"] == 0.0
    assert c["select_every_nth"] == sp.DEFAULT_SELECT_EVERY_NTH
    assert c["frame_load_cap"] == sp.DEFAULT_FRAME_LOAD_CAP
    assert c["subjects"] == []
    assert c["action"] == ""

def test_normalize_clip_preserves_values():
    c = sp._normalize_clip({
        "id": "c2", "label": "Intro", "start_time": 5.5, "end_time": 15.0,
        "select_every_nth": 3, "frame_load_cap": 60,
        "subjects": ["s1", "s2"], "action": "They talk",
    })
    assert c["start_time"] == 5.5
    assert c["end_time"] == 15.0
    assert c["select_every_nth"] == 3
    assert c["frame_load_cap"] == 60
    assert c["subjects"] == ["s1", "s2"]
    assert c["action"] == "They talk"


# ── set_clips / upsert_clip / remove_clip ─────────────────────────────────────

def _reg_with_profile() -> SourceProfileRegistry:
    r = SourceProfileRegistry()
    return r.define_profile("p1", media_filename="vid.mp4", media_type="video")

def test_set_clips_replaces_all():
    reg = _reg_with_profile()
    clips = [
        {"id": "c1", "start_time": 0.0, "end_time": 10.0},
        {"id": "c2", "start_time": 10.0, "end_time": 20.0},
    ]
    reg2 = reg.set_clips("p1", clips)
    assert len(reg2.profiles["p1"]["clips"]) == 2
    assert reg2.profiles["p1"]["clips"][0]["id"] == "c1"

def test_set_clips_immutable():
    reg = _reg_with_profile()
    reg2 = reg.set_clips("p1", [{"id": "c1", "start_time": 0.0, "end_time": 5.0}])
    assert reg.profiles["p1"]["clips"] == []

def test_upsert_clip_appends_new():
    reg = _reg_with_profile()
    reg2 = reg.upsert_clip("p1", {"id": "c1", "start_time": 0.0, "end_time": 10.0})
    assert len(reg2.profiles["p1"]["clips"]) == 1

def test_upsert_clip_updates_existing():
    reg = _reg_with_profile()
    reg2 = reg.upsert_clip("p1", {"id": "c1", "start_time": 0.0, "end_time": 10.0, "action": "v1"})
    reg3 = reg2.upsert_clip("p1", {"id": "c1", "start_time": 0.0, "end_time": 10.0, "action": "v2"})
    assert len(reg3.profiles["p1"]["clips"]) == 1
    assert reg3.profiles["p1"]["clips"][0]["action"] == "v2"

def test_remove_clip_removes_correctly():
    reg = _reg_with_profile()
    reg2 = reg.set_clips("p1", [
        {"id": "c1", "start_time": 0.0, "end_time": 10.0},
        {"id": "c2", "start_time": 10.0, "end_time": 20.0},
    ])
    reg3 = reg2.remove_clip("p1", "c1")
    ids = [c["id"] for c in reg3.profiles["p1"]["clips"]]
    assert ids == ["c2"]

def test_remove_clip_noop_for_missing():
    reg = _reg_with_profile()
    reg = reg.set_clips("p1", [{"id": "c1", "start_time": 0.0, "end_time": 10.0}])
    reg2 = reg.remove_clip("p1", "nonexistent")
    # Profile still has its original clip
    assert len(reg2.profiles["p1"]["clips"]) == 1
    assert reg2.profiles["p1"]["clips"][0]["id"] == "c1"


# ── auto_partition ─────────────────────────────────────────────────────────────

def test_auto_partition_creates_equal_clips():
    reg = _reg_with_profile()
    reg2 = reg.auto_partition("p1", video_duration=30.0, segment_duration=10.0)
    clips = reg2.profiles["p1"]["clips"]
    assert len(clips) == 3
    assert clips[0]["start_time"] == 0.0
    assert clips[0]["end_time"] == 10.0
    assert clips[1]["start_time"] == 10.0
    assert clips[1]["end_time"] == 20.0
    assert clips[2]["start_time"] == 20.0
    assert clips[2]["end_time"] == 30.0

def test_auto_partition_last_clip_clamps_to_duration():
    reg = _reg_with_profile()
    reg2 = reg.auto_partition("p1", video_duration=25.0, segment_duration=10.0)
    clips = reg2.profiles["p1"]["clips"]
    assert len(clips) == 3
    assert clips[2]["end_time"] == 25.0

def test_auto_partition_uses_profile_default_segment_duration():
    reg = SourceProfileRegistry()
    reg = reg.define_profile("p2", media_filename="v.mp4", default_segment_duration=15.0)
    reg2 = reg.auto_partition("p2", video_duration=30.0)
    clips = reg2.profiles["p2"]["clips"]
    assert len(clips) == 2
    assert clips[0]["end_time"] == 15.0

def test_auto_partition_falls_back_to_global_default():
    reg = _reg_with_profile()
    reg2 = reg.auto_partition("p1", video_duration=30.0)  # no segment_duration
    clips = reg2.profiles["p1"]["clips"]
    assert len(clips) == 3  # 30 / 10 = 3

def test_auto_partition_noop_for_zero_duration():
    reg = _reg_with_profile()
    reg2 = reg.auto_partition("p1", video_duration=0.0)
    assert reg2 is reg

def test_auto_partition_clip_ids_sequential():
    reg = _reg_with_profile()
    reg2 = reg.auto_partition("p1", video_duration=20.0, segment_duration=10.0)
    clips = reg2.profiles["p1"]["clips"]
    assert clips[0]["id"] == "clip_1"
    assert clips[1]["id"] == "clip_2"


# ── get_clip / clip_load_params ────────────────────────────────────────────────

def test_get_clip_returns_correct_clip():
    reg = _reg_with_profile()
    reg2 = reg.set_clips("p1", [
        {"id": "c1", "start_time": 0.0, "end_time": 10.0},
        {"id": "c2", "start_time": 10.0, "end_time": 20.0},
    ])
    c = reg2.get_clip("p1", "c2")
    assert c is not None
    assert c["start_time"] == 10.0

def test_get_clip_returns_none_missing_profile():
    reg = _reg_with_profile()
    assert reg.get_clip("nonexistent", "c1") is None

def test_get_clip_returns_none_missing_clip():
    reg = _reg_with_profile()
    reg2 = reg.set_clips("p1", [{"id": "c1", "start_time": 0.0, "end_time": 10.0}])
    assert reg2.get_clip("p1", "c_missing") is None

def test_clip_load_params_returns_correct_values():
    reg = _reg_with_profile()
    reg2 = reg.set_clips("p1", [{
        "id": "c1", "start_time": 5.0, "end_time": 15.0,
        "select_every_nth": 3, "frame_load_cap": 60,
    }])
    lp = reg2.clip_load_params("p1", "c1")
    assert lp is not None
    assert lp["start_time"] == 5.0
    assert lp["duration"] == 10.0
    assert lp["select_every_nth"] == 3
    assert lp["frame_load_cap"] == 60
    assert lp["force_rate"] == 0
    assert lp["skip_first_frames"] == 0

def test_clip_load_params_returns_none_for_empty_clip_id():
    reg = _reg_with_profile()
    assert reg.clip_load_params("p1", "") is None

def test_clip_load_params_returns_none_for_missing_clip():
    reg = _reg_with_profile()
    assert reg.clip_load_params("p1", "nonexistent") is None

def test_clip_load_params_duration_clamped_to_zero():
    reg = _reg_with_profile()
    # end_time < start_time → duration = 0
    reg2 = reg.set_clips("p1", [{"id": "c1", "start_time": 10.0, "end_time": 5.0}])
    lp = reg2.clip_load_params("p1", "c1")
    assert lp["duration"] == 0.0
