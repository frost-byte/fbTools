"""Tests for the Subject Profile system."""

import json
import os
import pytest
from conftest import import_test_module

sp = import_test_module("utils/subject_profiles.py")

SubjectRegistry = sp.SubjectRegistry
load_registry = sp.load_registry
save_registry = sp.save_registry
SUPPORTED_LANGUAGES = sp.SUPPORTED_LANGUAGES


# ── SubjectRegistry.define ─────────────────────────────────────────────────────

def test_define_new_subject_creates_entry():
    reg = SubjectRegistry()
    new_reg = reg.define(
        "char_alice",
        name="Alice",
        appearance_summary="a woman with auburn hair",
        face="high cheekbones",
        concept_id="char_alice",
    )
    assert "char_alice" in new_reg.subjects
    s = new_reg.subjects["char_alice"]
    assert s["name"] == "Alice"
    assert s["appearance"]["summary"] == "a woman with auburn hair"
    assert s["appearance"]["face"] == "high cheekbones"
    assert s["concept_id"] == "char_alice"


def test_define_does_not_mutate_original():
    reg = SubjectRegistry()
    new_reg = reg.define("char_a", name="A")
    assert "char_a" not in reg.subjects
    assert "char_a" in new_reg.subjects


def test_define_overwrites_text_fields():
    reg = SubjectRegistry()
    reg1 = reg.define("char_b", name="Bob", appearance_summary="tall man")
    reg2 = reg1.define("char_b", name="Bobby", appearance_summary="short man")
    assert reg2.subjects["char_b"]["name"] == "Bobby"
    assert reg2.subjects["char_b"]["appearance"]["summary"] == "short man"


def test_define_preserves_character_sheet_images():
    reg = SubjectRegistry(subjects={
        "char_c": {
            "name": "Carol",
            "appearance": {"summary": "", "face": "", "hair": "", "body": "", "default_outfit": ""},
            "voice": {"description": "", "audio_reference_file": "", "language": "en-us"},
            "character_sheet_images": ["carol1.png", "carol2.png"],
            "concept_id": "",
        }
    })
    new_reg = reg.define("char_c", name="Carol Updated", appearance_summary="updated summary")
    assert new_reg.subjects["char_c"]["character_sheet_images"] == ["carol1.png", "carol2.png"]


def test_define_defaults_name_to_subject_id():
    reg = SubjectRegistry()
    new_reg = reg.define("unnamed_char")
    assert new_reg.subjects["unnamed_char"]["name"] == "unnamed_char"


def test_define_multiple_subjects_accumulate():
    reg = SubjectRegistry()
    reg = reg.define("char_a", name="Alice")
    reg = reg.define("char_b", name="Bob")
    assert len(reg.subjects) == 2
    assert "char_a" in reg.subjects
    assert "char_b" in reg.subjects


# ── SubjectRegistry.get_subject ───────────────────────────────────────────────

def test_get_subject_returns_none_for_missing():
    reg = SubjectRegistry()
    assert reg.get_subject("nonexistent") is None


def test_get_subject_returns_entry():
    reg = SubjectRegistry()
    reg = reg.define("char_d", name="Diana", appearance_summary="elegant woman")
    result = reg.get_subject("char_d")
    assert result is not None
    assert result["name"] == "Diana"


# ── SubjectRegistry.subject_ids ───────────────────────────────────────────────

def test_subject_ids_empty():
    reg = SubjectRegistry()
    assert reg.subject_ids() == []


def test_subject_ids_lists_all():
    reg = SubjectRegistry()
    reg = reg.define("z_char", name="Z")
    reg = reg.define("a_char", name="A")
    ids = reg.subject_ids()
    assert set(ids) == {"z_char", "a_char"}


# ── SubjectRegistry.list_subjects ─────────────────────────────────────────────

def test_list_subjects_empty():
    reg = SubjectRegistry()
    assert "No subjects defined" in reg.list_subjects()


def test_list_subjects_shows_entries():
    reg = SubjectRegistry()
    reg = reg.define("char_e", name="Eve", appearance_summary="mysterious woman")
    listing = reg.list_subjects()
    assert "[char_e]" in listing
    assert "Eve" in listing
    assert "mysterious woman" in listing


def test_list_subjects_truncates_long_summary():
    reg = SubjectRegistry()
    long_summary = "a" * 100
    reg = reg.define("char_f", name="Frank", appearance_summary=long_summary)
    listing = reg.list_subjects()
    assert "..." in listing


def test_list_subjects_shows_concept_id():
    reg = SubjectRegistry()
    reg = reg.define("char_g", name="Grace", concept_id="grace_concept")
    listing = reg.list_subjects()
    assert "[grace_concept]" in listing


# ── SubjectRegistry serialisation ─────────────────────────────────────────────

def test_to_dict_round_trips():
    reg = SubjectRegistry()
    reg = reg.define("char_h", name="Heidi", appearance_summary="blonde woman", language="de")
    data = reg.to_dict()
    assert data["version"] == 1
    assert "char_h" in data["subjects"]
    reg2 = SubjectRegistry.from_dict(data)
    assert reg2.subjects["char_h"]["name"] == "Heidi"
    assert reg2.subjects["char_h"]["voice"]["language"] == "de"


def test_from_dict_handles_missing_version():
    data = {"subjects": {"char_i": {"name": "Iris", "appearance": {}, "voice": {},
                                     "character_sheet_images": [], "concept_id": ""}}}
    reg = SubjectRegistry.from_dict(data)
    assert reg.version == 1
    assert "char_i" in reg.subjects


# ── Persistence ────────────────────────────────────────────────────────────────

def test_load_registry_missing_file_returns_empty(tmp_path):
    reg = load_registry(str(tmp_path / "nonexistent.json"))
    assert reg.subjects == {}
    assert reg.version == 1


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "subjects.json")
    reg = SubjectRegistry()
    reg = reg.define("char_j", name="Julia", appearance_summary="red-haired woman",
                     concept_id="julia_c", language="en-us")
    save_registry(reg, path, backup=False)

    assert os.path.exists(path)
    loaded = load_registry(path)
    assert "char_j" in loaded.subjects
    assert loaded.subjects["char_j"]["name"] == "Julia"
    assert loaded.subjects["char_j"]["concept_id"] == "julia_c"


def test_save_creates_backup(tmp_path):
    path = str(tmp_path / "subjects.json")
    reg = SubjectRegistry()
    reg = reg.define("char_k", name="Kate")
    save_registry(reg, path, backup=False)
    save_registry(reg, path, backup=True)
    assert os.path.exists(path + ".bak")


def test_save_creates_parent_directories(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "subjects.json")
    reg = SubjectRegistry()
    save_registry(reg, path, backup=False)
    assert os.path.exists(path)


def test_load_registry_sets_file_path(tmp_path):
    path = str(tmp_path / "subjects.json")
    reg = SubjectRegistry()
    save_registry(reg, path, backup=False)
    loaded = load_registry(path)
    assert loaded.file_path == path


# ── SUPPORTED_LANGUAGES ────────────────────────────────────────────────────────

def test_supported_languages_includes_common():
    assert "en-us" in SUPPORTED_LANGUAGES
    assert "ja" in SUPPORTED_LANGUAGES
    assert "ko" in SUPPORTED_LANGUAGES
    assert "zh" in SUPPORTED_LANGUAGES


# ── SubjectRegistry.save convenience method ───────────────────────────────────

def test_registry_save_convenience(tmp_path):
    path = str(tmp_path / "subjects.json")
    reg = SubjectRegistry(file_path=path)
    reg = reg.define("char_l", name="Laura")
    reg.file_path = path
    saved_path = reg.save(backup=False)
    assert saved_path == path
    assert os.path.exists(path)


def test_registry_save_raises_without_path():
    reg = SubjectRegistry()
    with pytest.raises(ValueError):
        reg.save()
