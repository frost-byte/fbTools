"""Tests for the Scene Cast system."""

import json
import os
import pytest
from conftest import import_test_module

sc = import_test_module("utils/scene_casts.py")

CastRegistry = sc.CastRegistry
load_registry = sc.load_registry
save_registry = sc.save_registry
validate_cast = sc.validate_cast
resolve_cast_for_subject = sc.resolve_cast_for_subject
make_entry = sc.make_entry
VISUAL_MODES = sc.VISUAL_MODES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _cast(cast_id="cast_01", name="Test Cast"):
    return {
        "id": cast_id,
        "name": name,
        "entries": [
            make_entry("char_alice", "alice_autumn", "video", True),
            make_entry("char_bob", "bob_casual", "images", False),
        ],
    }


# ── make_entry ─────────────────────────────────────────────────────────────────

def test_make_entry_fields():
    e = make_entry("char_alice", "b_alice", "video", True)
    assert e["subject_id"] == "char_alice"
    assert e["bundle_id"] == "b_alice"
    assert e["visual_mode"] == "video"
    assert e["use_audio"] is True


def test_make_entry_defaults():
    e = make_entry("char_x", "b_x")
    assert e["visual_mode"] == "images"
    assert e["use_audio"] is False


# ── CastRegistry basics ────────────────────────────────────────────────────────

def test_empty_registry():
    reg = CastRegistry()
    assert reg.casts == {}
    assert reg.cast_ids() == []
    assert reg.list_casts() == []


def test_upsert_creates_entry():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    assert "cast_01" in r.casts
    assert r.get("cast_01")["name"] == "Test Cast"


def test_upsert_does_not_mutate_original():
    reg = CastRegistry()
    reg.upsert(_cast())
    assert "cast_01" not in reg.casts


def test_upsert_preserves_created_on_update():
    reg = CastRegistry()
    r1 = reg.upsert(_cast())
    created_ts = r1.get("cast_01")["created"]

    import time; time.sleep(0.01)
    updated = _cast()
    updated["name"] = "Renamed Cast"
    r2 = r1.upsert(updated)

    assert r2.get("cast_01")["created"] == created_ts
    assert r2.get("cast_01")["name"] == "Renamed Cast"


def test_upsert_defaults_missing_fields():
    reg = CastRegistry()
    r = reg.upsert({"id": "minimal_cast"})
    c = r.get("minimal_cast")
    assert c["name"] == "minimal_cast"
    assert c["entries"] == []


def test_delete_removes_cast():
    reg = CastRegistry()
    r = reg.upsert(_cast("c1")).upsert(_cast("c2"))
    r2 = r.delete("c1")
    assert "c1" not in r2.casts
    assert "c2" in r2.casts


def test_delete_noop_on_missing():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.delete("nonexistent")
    assert r2.cast_ids() == ["cast_01"]


def test_get_returns_copy():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    c = r.get("cast_01")
    c["name"] = "TAMPERED"
    assert r.get("cast_01")["name"] == "Test Cast"


def test_get_missing_returns_none():
    assert CastRegistry().get("nope") is None


# ── find_entry ─────────────────────────────────────────────────────────────────

def test_find_entry_returns_matching():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    entry = r.find_entry("cast_01", "char_alice")
    assert entry is not None
    assert entry["bundle_id"] == "alice_autumn"


def test_find_entry_missing_subject_returns_none():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    assert r.find_entry("cast_01", "char_missing") is None


def test_find_entry_missing_cast_returns_none():
    reg = CastRegistry()
    assert reg.find_entry("no_such_cast", "char_alice") is None


def test_find_entry_returns_copy():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    entry = r.find_entry("cast_01", "char_alice")
    entry["bundle_id"] = "TAMPERED"
    assert r.find_entry("cast_01", "char_alice")["bundle_id"] == "alice_autumn"


# ── update_entry ───────────────────────────────────────────────────────────────

def test_update_entry_changes_bundle():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.update_entry("cast_01", "char_alice", bundle_id="alice_new_bundle")
    assert r2.find_entry("cast_01", "char_alice")["bundle_id"] == "alice_new_bundle"


def test_update_entry_changes_visual_mode():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.update_entry("cast_01", "char_alice", visual_mode="images")
    assert r2.find_entry("cast_01", "char_alice")["visual_mode"] == "images"


def test_update_entry_changes_use_audio():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.update_entry("cast_01", "char_bob", use_audio=True)
    assert r2.find_entry("cast_01", "char_bob")["use_audio"] is True


def test_update_entry_none_fields_unchanged():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.update_entry("cast_01", "char_alice", bundle_id=None, visual_mode=None, use_audio=None)
    entry = r2.find_entry("cast_01", "char_alice")
    assert entry["bundle_id"] == "alice_autumn"
    assert entry["visual_mode"] == "video"
    assert entry["use_audio"] is True


def test_update_entry_appends_new_subject():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.update_entry("cast_01", "char_carol", bundle_id="carol_b", visual_mode="images")
    entry = r2.find_entry("cast_01", "char_carol")
    assert entry is not None
    assert entry["bundle_id"] == "carol_b"


def test_update_entry_does_not_mutate_original():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r.update_entry("cast_01", "char_alice", bundle_id="changed")
    assert r.find_entry("cast_01", "char_alice")["bundle_id"] == "alice_autumn"


def test_update_entry_missing_cast_raises():
    reg = CastRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.update_entry("no_such_cast", "char_alice", bundle_id="b")


# ── remove_entry ───────────────────────────────────────────────────────────────

def test_remove_entry_removes_subject():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.remove_entry("cast_01", "char_alice")
    assert r2.find_entry("cast_01", "char_alice") is None
    assert r2.find_entry("cast_01", "char_bob") is not None


def test_remove_entry_noop_if_subject_absent():
    reg = CastRegistry()
    r = reg.upsert(_cast())
    r2 = r.remove_entry("cast_01", "nobody")
    assert len(r2.get("cast_01")["entries"]) == 2


def test_remove_entry_missing_cast_raises():
    reg = CastRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.remove_entry("no_such_cast", "char_alice")


# ── serialisation roundtrip ────────────────────────────────────────────────────

def test_to_dict_from_dict_roundtrip():
    reg = CastRegistry()
    r = reg.upsert(_cast("c1")).upsert(_cast("c2", "Cast Two"))
    data = r.to_dict()
    r2 = CastRegistry.from_dict(data)
    assert r2.cast_ids() == r.cast_ids()
    assert r2.find_entry("c1", "char_alice")["bundle_id"] == "alice_autumn"


def test_from_dict_missing_casts_key():
    r = CastRegistry.from_dict({"version": 1})
    assert r.casts == {}


# ── persistence ────────────────────────────────────────────────────────────────

def test_save_and_load(tmp_path):
    path = str(tmp_path / "casts.json")
    reg = CastRegistry()
    r = reg.upsert(_cast())
    save_registry(r, path, backup=False)

    loaded = load_registry(path)
    assert loaded.cast_ids() == r.cast_ids()
    assert loaded.find_entry("cast_01", "char_alice")["visual_mode"] == "video"
    assert loaded.file_path == path


def test_load_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "no_such_file.json")
    reg = load_registry(path)
    assert reg.casts == {}
    assert reg.file_path == path


def test_save_creates_backup(tmp_path):
    path = str(tmp_path / "casts.json")
    reg = CastRegistry()
    r = reg.upsert(_cast())
    save_registry(r, path, backup=False)
    save_registry(r, path, backup=True)
    assert os.path.exists(path + ".bak")


def test_save_no_path_raises():
    reg = CastRegistry()
    with pytest.raises(ValueError, match="No file path"):
        reg.save()


# ── validate_cast ──────────────────────────────────────────────────────────────

def test_valid_cast_no_warnings():
    assert validate_cast(_cast()) == []


def test_duplicate_subject_warns():
    c = _cast()
    c["entries"].append(make_entry("char_alice", "alice_extra"))
    warnings = validate_cast(c)
    assert any("more than once" in w for w in warnings)


def test_empty_bundle_id_warns():
    c = _cast()
    c["entries"][0]["bundle_id"] = ""
    warnings = validate_cast(c)
    assert any("bundle_id is empty" in w for w in warnings)


def test_empty_subject_id_warns():
    c = _cast()
    c["entries"][0]["subject_id"] = ""
    warnings = validate_cast(c)
    assert any("subject_id is empty" in w for w in warnings)


def test_invalid_visual_mode_warns():
    c = _cast()
    c["entries"][0]["visual_mode"] = "gif"
    warnings = validate_cast(c)
    assert any("visual_mode" in w for w in warnings)


def test_empty_entries_no_warnings():
    c = {"id": "empty_cast", "name": "Empty", "entries": []}
    assert validate_cast(c) == []


# ── resolve_cast_for_subject ───────────────────────────────────────────────────

def test_resolve_returns_entry():
    cast = _cast()
    entry = resolve_cast_for_subject(cast, "char_bob")
    assert entry is not None
    assert entry["bundle_id"] == "bob_casual"


def test_resolve_returns_none_if_not_in_cast():
    cast = _cast()
    assert resolve_cast_for_subject(cast, "char_carol") is None


def test_resolve_returns_copy():
    cast = _cast()
    entry = resolve_cast_for_subject(cast, "char_alice")
    entry["bundle_id"] = "TAMPERED"
    assert resolve_cast_for_subject(cast, "char_alice")["bundle_id"] == "alice_autumn"
