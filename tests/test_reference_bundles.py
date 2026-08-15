"""Tests for the Reference Bundle system."""

import json
import os
import pytest
from conftest import import_test_module

rb = import_test_module("utils/reference_bundles.py")

BundleRegistry = rb.BundleRegistry
load_registry = rb.load_registry
save_registry = rb.save_registry
validate_bundle = rb.validate_bundle
VISUAL_TYPES = rb.VISUAL_TYPES
AUDIO_SOURCES = rb.AUDIO_SOURCES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _video_bundle(bundle_id="b_vid", subject_id="char_alice"):
    return {
        "id": bundle_id,
        "name": "Alice Video",
        "subject_id": subject_id,
        "visual": {"type": "video", "file": "alice.mp4", "files": []},
        "audio": {"source": "extract_from_visual", "file": ""},
        "appearance_override": "",
        "tags": ["autumn"],
    }


def _images_bundle(bundle_id="b_img", subject_id="char_bob"):
    return {
        "id": bundle_id,
        "name": "Bob Images",
        "subject_id": subject_id,
        "visual": {"type": "images", "file": "", "files": ["bob_a.jpg", "bob_b.jpg"]},
        "audio": {"source": "none", "file": ""},
        "appearance_override": "tall man",
        "tags": [],
    }


# ── BundleRegistry basics ──────────────────────────────────────────────────────

def test_empty_registry():
    reg = BundleRegistry()
    assert reg.bundles == {}
    assert reg.bundle_ids() == []
    assert reg.list_bundles() == []


def test_upsert_creates_entry():
    reg = BundleRegistry()
    reg2 = reg.upsert(_video_bundle())
    assert "b_vid" in reg2.bundles
    assert reg2.get("b_vid")["name"] == "Alice Video"


def test_upsert_does_not_mutate_original():
    reg = BundleRegistry()
    reg.upsert(_video_bundle())
    assert "b_vid" not in reg.bundles


def test_upsert_preserves_created_on_update():
    reg = BundleRegistry()
    r1 = reg.upsert(_video_bundle())
    created_ts = r1.get("b_vid")["created"]

    import time; time.sleep(0.01)
    updated = _video_bundle()
    updated["name"] = "Alice Video v2"
    r2 = r1.upsert(updated)

    assert r2.get("b_vid")["created"] == created_ts
    assert r2.get("b_vid")["name"] == "Alice Video v2"


def test_upsert_refreshes_modified():
    reg = BundleRegistry()
    r1 = reg.upsert(_video_bundle())
    mod1 = r1.get("b_vid")["modified"]

    import time; time.sleep(0.01)
    r2 = r1.upsert(_video_bundle())
    mod2 = r2.get("b_vid")["modified"]

    assert mod2 >= mod1


def test_upsert_defaults_missing_fields():
    reg = BundleRegistry()
    r = reg.upsert({"id": "minimal", "subject_id": "char_x"})
    b = r.get("minimal")
    assert b["visual"]["type"] == "images"
    assert b["visual"]["file"] == ""
    assert b["visual"]["files"] == []
    assert b["visual"]["force_rate"] == 0
    assert b["visual"]["frame_load_cap"] == 96
    assert b["visual"]["skip_first_frames"] == 0
    assert b["visual"]["select_every_nth"] == 1
    assert b["audio"]["source"] == "none"
    assert b["audio"]["file"] == ""
    assert b["audio"]["force_rate"] == 0
    assert b["audio"]["frame_load_cap"] == 0
    assert b["audio"]["skip_first_frames"] == 0
    assert b["audio"]["select_every_nth"] == 1
    assert b["audio"]["start_time"] == 0.0
    assert b["audio"]["duration"] == 0.0
    assert b["appearance_override"] == ""
    assert b["tags"] == []


def test_delete_removes_entry():
    reg = BundleRegistry()
    r1 = reg.upsert(_video_bundle()).upsert(_images_bundle())
    r2 = r1.delete("b_vid")
    assert "b_vid" not in r2.bundles
    assert "b_img" in r2.bundles


def test_delete_noop_on_missing():
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle())
    r2 = r.delete("nonexistent")
    assert r2.bundle_ids() == ["b_vid"]


def test_get_returns_copy():
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle())
    b = r.get("b_vid")
    b["name"] = "MODIFIED"
    assert r.get("b_vid")["name"] == "Alice Video"


def test_get_missing_returns_none():
    assert BundleRegistry().get("nope") is None


# ── list_bundles filtering ─────────────────────────────────────────────────────

def test_list_bundles_no_filter():
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle()).upsert(_images_bundle())
    assert len(r.list_bundles()) == 2


def test_list_bundles_filters_by_subject():
    reg = BundleRegistry()
    r = (reg
         .upsert(_video_bundle(bundle_id="a1", subject_id="char_alice"))
         .upsert(_video_bundle(bundle_id="a2", subject_id="char_alice"))
         .upsert(_images_bundle(bundle_id="b1", subject_id="char_bob")))

    alice_bundles = r.list_bundles(subject_id="char_alice")
    assert len(alice_bundles) == 2
    assert all(b["subject_id"] == "char_alice" for b in alice_bundles)

    bob_bundles = r.list_bundles(subject_id="char_bob")
    assert len(bob_bundles) == 1


def test_list_bundles_unknown_subject_returns_empty():
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle())
    assert r.list_bundles(subject_id="nobody") == []


# ── serialisation roundtrip ────────────────────────────────────────────────────

def test_to_dict_from_dict_roundtrip():
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle()).upsert(_images_bundle())
    data = r.to_dict()
    r2 = BundleRegistry.from_dict(data)
    assert r2.bundle_ids() == r.bundle_ids()
    assert r2.get("b_vid")["visual"]["file"] == "alice.mp4"


def test_from_dict_missing_bundles_key():
    r = BundleRegistry.from_dict({"version": 1})
    assert r.bundles == {}


# ── persistence ────────────────────────────────────────────────────────────────

def test_save_and_load(tmp_path):
    path = str(tmp_path / "bundles.json")
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle()).upsert(_images_bundle())
    save_registry(r, path, backup=False)

    loaded = load_registry(path)
    assert loaded.bundle_ids() == r.bundle_ids()
    assert loaded.get("b_vid")["visual"]["file"] == "alice.mp4"
    assert loaded.file_path == path


def test_load_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "no_such_file.json")
    reg = load_registry(path)
    assert reg.bundles == {}
    assert reg.file_path == path


def test_save_creates_backup(tmp_path):
    path = str(tmp_path / "bundles.json")
    reg = BundleRegistry()
    r = reg.upsert(_video_bundle())
    save_registry(r, path, backup=False)
    save_registry(r, path, backup=True)
    assert os.path.exists(path + ".bak")


def test_registry_save_method(tmp_path):
    path = str(tmp_path / "bundles.json")
    reg = BundleRegistry(file_path=path)
    r = reg.upsert(_video_bundle())
    r.file_path = path
    r.save(backup=False)
    assert os.path.exists(path)


def test_save_no_path_raises():
    reg = BundleRegistry()
    with pytest.raises(ValueError, match="No file path"):
        reg.save()


# ── validate_bundle ────────────────────────────────────────────────────────────

def test_valid_video_bundle_no_warnings():
    assert validate_bundle(_video_bundle()) == []


def test_valid_images_bundle_no_warnings():
    assert validate_bundle(_images_bundle()) == []


def test_video_without_file_warns():
    b = _video_bundle()
    b["visual"]["file"] = ""
    warnings = validate_bundle(b)
    assert any("visual.file is empty" in w for w in warnings)


def test_images_without_files_warns():
    b = _images_bundle()
    b["visual"]["files"] = []
    warnings = validate_bundle(b)
    assert any("visual.files is empty" in w for w in warnings)


def test_extract_from_visual_on_images_warns():
    b = _images_bundle()
    b["audio"]["source"] = "extract_from_visual"
    warnings = validate_bundle(b)
    assert any("extract_from_visual" in w for w in warnings)


def test_audio_file_source_without_file_warns():
    b = _video_bundle()
    b["audio"] = {"source": "file", "file": ""}
    warnings = validate_bundle(b)
    assert any("audio.file is empty" in w for w in warnings)


def test_invalid_visual_type_warns():
    b = _video_bundle()
    b["visual"]["type"] = "gif"
    warnings = validate_bundle(b)
    assert any("visual.type" in w for w in warnings)


def test_invalid_audio_source_warns():
    b = _video_bundle()
    b["audio"]["source"] = "magic"
    warnings = validate_bundle(b)
    assert any("audio.source" in w for w in warnings)


def test_valid_audio_file_source():
    b = _video_bundle()
    b["audio"] = {"source": "file", "file": "alice_voice.wav"}
    assert validate_bundle(b) == []


# ── Frame-sampling params ──────────────────────────────────────────────────────

def test_upsert_preserves_custom_visual_frame_params():
    reg = BundleRegistry()
    bundle = _video_bundle()
    bundle["visual"]["force_rate"] = 24
    bundle["visual"]["frame_load_cap"] = 32
    bundle["visual"]["skip_first_frames"] = 10
    bundle["visual"]["select_every_nth"] = 2
    r = reg.upsert(bundle)
    b = r.get("b_vid")
    assert b["visual"]["force_rate"] == 24
    assert b["visual"]["frame_load_cap"] == 32
    assert b["visual"]["skip_first_frames"] == 10
    assert b["visual"]["select_every_nth"] == 2


def test_upsert_preserves_custom_audio_frame_params():
    reg = BundleRegistry()
    bundle = _video_bundle()
    bundle["audio"]["force_rate"] = 8
    bundle["audio"]["frame_load_cap"] = 16
    bundle["audio"]["skip_first_frames"] = 60
    bundle["audio"]["select_every_nth"] = 3
    r = reg.upsert(bundle)
    b = r.get("b_vid")
    assert b["audio"]["force_rate"] == 8
    assert b["audio"]["frame_load_cap"] == 16
    assert b["audio"]["skip_first_frames"] == 60
    assert b["audio"]["select_every_nth"] == 3


def test_upsert_preserves_audio_time_params():
    reg = BundleRegistry()
    bundle = _video_bundle()
    bundle["audio"]["source"] = "file"
    bundle["audio"]["file"] = "voice.wav"
    bundle["audio"]["start_time"] = 3.5
    bundle["audio"]["duration"] = 10.0
    r = reg.upsert(bundle)
    b = r.get("b_vid")
    assert b["audio"]["start_time"] == 3.5
    assert b["audio"]["duration"] == 10.0


def test_upsert_defaults_frame_params_when_absent():
    reg = BundleRegistry()
    # Bundle with no frame params at all — should get defaults
    r = reg.upsert({"id": "x", "subject_id": "s", "visual": {"type": "video", "file": "v.mp4", "files": []}})
    b = r.get("x")
    assert b["visual"]["force_rate"] == 0
    assert b["visual"]["frame_load_cap"] == 96
    assert b["visual"]["skip_first_frames"] == 0
    assert b["visual"]["select_every_nth"] == 1
    assert b["audio"]["force_rate"] == 0
    assert b["audio"]["frame_load_cap"] == 0
    assert b["audio"]["skip_first_frames"] == 0
    assert b["audio"]["select_every_nth"] == 1
    assert b["audio"]["start_time"] == 0.0
    assert b["audio"]["duration"] == 0.0


def test_upsert_does_not_override_existing_frame_params():
    """setdefault must not overwrite values already present on the incoming bundle."""
    reg = BundleRegistry()
    bundle = _video_bundle()
    bundle["visual"]["frame_load_cap"] = 48
    r = reg.upsert(bundle)
    assert r.get("b_vid")["visual"]["frame_load_cap"] == 48


def test_roundtrip_preserves_frame_params(tmp_path):
    path = str(tmp_path / "bundles.json")
    reg = BundleRegistry()
    bundle = _video_bundle()
    bundle["visual"]["force_rate"] = 12
    bundle["audio"]["skip_first_frames"] = 30
    bundle["audio"]["start_time"] = 1.5
    r = reg.upsert(bundle)
    save_registry(r, path, backup=False)
    loaded = load_registry(path)
    b = loaded.get("b_vid")
    assert b["visual"]["force_rate"] == 12
    assert b["audio"]["skip_first_frames"] == 30
    assert b["audio"]["start_time"] == 1.5
