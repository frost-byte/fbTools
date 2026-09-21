"""Tests for utils/composition_track_summary.py — readable Run History rows."""
from conftest import import_test_module

cts = import_test_module("utils/composition_track_summary.py")
summarize_scene_cast = cts.summarize_scene_cast
summarize_loras = cts.summarize_loras


def _lookup(bundles):
    return bundles.get


def _cast(*entries):
    return {"id": "_inline", "name": "_inline", "entries": list(entries)}


def _bundle(files=(), video="", vstart=0, vdur=0, audio=None):
    return {"visual": {"files": list(files), "file": video, "start_time": vstart, "duration": vdur},
            "audio": audio or {"source": "none"}}


def test_head_row_names_subject_bundle_mode_and_flags():
    rows = summarize_scene_cast(
        _cast({"subject_id": "alex", "bundle_id": "b1", "visual_mode": "both", "use_audio": True,
               "retention": "fully_preserved"}),
        _lookup({"b1": _bundle()}))
    assert rows["Cast 1: alex"] == "b1 · both · +audio · fully_preserved"


def test_images_all_by_default_and_selection_honoured():
    b = {"b1": _bundle(files=["a.png", "b.png", "c.png"])}
    all_rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1"}), _lookup(b))
    assert all_rows["Cast 1 images"] == "a.png, b.png, c.png"
    sel = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "image_selection": [0, 2]}), _lookup(b))
    assert sel["Cast 1 images"] == "a.png, c.png"
    legacy = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "image_selection": 1}), _lookup(b))
    assert legacy["Cast 1 images"] == "b.png"


def test_dict_style_image_entries_and_out_of_range_selection():
    b = {"b1": _bundle(files=[{"file": "a.png", "role": "face"}])}
    rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "image_selection": [0, 9]}), _lookup(b))
    assert rows["Cast 1 images"] == "a.png"


def test_video_row_only_in_video_modes_with_timing():
    b = {"b1": _bundle(files=["a.png"], video="clip.mp4", vstart=1.5, vdur=4)}
    both = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "visual_mode": "both"}), _lookup(b))
    assert both["Cast 1 video"] == "clip.mp4 (start 1.5s, 4s)"
    images = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "visual_mode": "images"}), _lookup(b))
    assert "Cast 1 video" not in images
    video = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "visual_mode": "video"}), _lookup(b))
    assert "Cast 1 images" not in video and "Cast 1 video" in video


def test_video_without_timing_has_no_parentheses():
    b = {"b1": _bundle(video="clip.mp4")}
    rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "visual_mode": "video"}), _lookup(b))
    assert rows["Cast 1 video"] == "clip.mp4"


def test_audio_variants():
    entry = {"subject_id": "s", "bundle_id": "b1", "visual_mode": "video", "use_audio": True}
    file_b = {"b1": _bundle(video="v.mp4", audio={"source": "file", "file": "voice.wav", "start_time": 2.5, "duration": 8})}
    assert summarize_scene_cast(_cast(entry), _lookup(file_b))["Cast 1 audio"] == "voice.wav (start 2.5s, 8s)"
    vid_b = {"b1": _bundle(video="v.mp4", audio={"source": "extract_from_video", "video_file": "other.mp4", "start_time": 1})}
    assert summarize_scene_cast(_cast(entry), _lookup(vid_b))["Cast 1 audio"] == "from other.mp4 (start 1s)"
    vis_b = {"b1": _bundle(video="v.mp4", audio={"source": "extract_from_visual"})}
    assert summarize_scene_cast(_cast(entry), _lookup(vis_b))["Cast 1 audio"] == "from reference video v.mp4"
    none_b = {"b1": _bundle(video="v.mp4")}
    assert summarize_scene_cast(_cast(entry), _lookup(none_b))["Cast 1 audio"] == "(none configured)"


def test_audio_row_omitted_when_use_audio_false():
    b = {"b1": _bundle(audio={"source": "file", "file": "voice.wav"})}
    rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1", "use_audio": False}), _lookup(b))
    assert "Cast 1 audio" not in rows


def test_missing_bundle_is_reported():
    rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "ghost"}), _lookup({}))
    assert rows["Cast 1 bundle"] == "(bundle 'ghost' not found)"


def test_source_and_dialogue_rows_and_truncation():
    entry = {"subject_id": "s", "bundle_id": "b1", "source_profile_id": "sp", "source_subject_id": "sub_2",
             "dialogue": "x" * 300}
    rows = summarize_scene_cast(_cast(entry), _lookup({"b1": _bundle()}))
    assert rows["Cast 1 source"] == "sp: sub_2"
    assert len(rows["Cast 1 dialogue"]) == 121 and rows["Cast 1 dialogue"].endswith("…")


def test_row_length_is_capped():
    b = {"b1": _bundle(files=[f"file_{i:04d}.png" for i in range(500)])}
    rows = summarize_scene_cast(_cast({"subject_id": "s", "bundle_id": "b1"}), _lookup(b))
    assert len(rows["Cast 1 images"]) <= cts.MAX_ROW_LEN + 1


def test_multiple_entries_are_numbered_and_bad_input_is_safe():
    rows = summarize_scene_cast(
        _cast({"subject_id": "a", "bundle_id": ""}, "junk", {"subject_id": "b", "bundle_id": ""}), _lookup({}))
    assert "Cast 1: a" in rows and "Cast 3: b" in rows and "Cast 2: ?" not in rows
    assert summarize_scene_cast(None, _lookup({})) == {}
    assert summarize_scene_cast({"entries": []}, _lookup({})) == {}


def test_loras_style_matches_enabled_summary():
    text = summarize_loras([
        {"name": "loras/style_lora_v2.safetensors", "weight": 0.75, "target": "MiniMaxH3"},
        {"name": "off.safetensors", "weight": 1, "enabled": False},
        {"name": "b.safetensors", "weight": 1.0},
        {"name": "x" * 80 + ".safetensors", "weight": 0.5},
    ])
    lines = text.splitlines()
    assert lines[0] == "style_lora_v2 0.75 (MiniMaxH3)"
    assert lines[1] == "b 1"
    assert lines[2] == "x" * 48 + " 0.5"
    assert len(lines) == 3


def test_loras_empty():
    assert summarize_loras([]) == "" and summarize_loras(None) == ""


# ── summarize_composition_meta (values the loader no longer outputs) ────────────

summarize_composition_meta = cts.summarize_composition_meta


def test_meta_composition_default_resolves_to_the_compositions_model_type():
    comp = {"model_type": "h3_ref2va", "subjects": {}}
    assert summarize_composition_meta(comp, "composition default", lambda s: None)["Model Type Used"] == "h3_ref2va"
    assert summarize_composition_meta(comp, "", lambda s: None)["Model Type Used"] == "h3_ref2va"
    assert summarize_composition_meta(comp, "wan22", lambda s: None)["Model Type Used"] == "wan22"


def test_meta_concept_ids_in_slot_order_deduped_with_composition_last():
    comp = {"subjects": {"A": "amy", "B": "", "C": "bob", "D": "cara"}, "concept_id": "scene_cid"}
    lookup = {"amy": {"concept_id": "c_amy"}, "bob": {"concept_id": "c_amy"}, "cara": {"concept_id": "c_cara"}}.get
    assert summarize_composition_meta(comp, "x", lookup)["Concept IDs"] == "c_amy, c_cara, scene_cid"


def test_meta_no_concept_ids_row_when_none():
    rows = summarize_composition_meta({"subjects": {"A": "amy"}}, "x", lambda s: {})
    assert "Concept IDs" not in rows
