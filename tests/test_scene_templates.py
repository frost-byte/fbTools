"""Tests for the Scene Template system."""

import json
import os
import pytest
from conftest import import_test_module

st = import_test_module("utils/scene_templates.py")

SceneTemplate = st.SceneTemplate
load_template = st.load_template
scan_templates = st.scan_templates
template_ids = st.template_ids
format_template_list = st.format_template_list
dir_fingerprint = st.dir_fingerprint


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_template(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


_CAFE_2P = {
    "version": 1,
    "id": "cafe_2p",
    "name": "Café Conversation",
    "description": "Two people in a café",
    "slots": {
        "A": {"role": "primary speaker", "needs_voice": True, "needs_character_sheet": True},
        "B": {"role": "secondary speaker", "needs_voice": False, "needs_character_sheet": True},
    },
    "environment": {"summary": "warm café", "lighting": "window light"},
    "style": "cinematic",
    "shots": [
        {
            "id": "shot_1",
            "timestamp": None,
            "camera": "Medium two-shot",
            "action": "{A} speaks to {B}.",
            "dialogue": {"speaker_slot": "A", "placeholder": True, "default_text": None},
            "sound_events": None,
        },
        {
            "id": "shot_2",
            "timestamp": "00:04.000",
            "camera": "Close-up of {B}",
            "action": "{B} responds.",
            "dialogue": {"speaker_slot": "B", "placeholder": True, "default_text": None},
            "sound_events": None,
        },
        {
            "id": "shot_3",
            "timestamp": "00:08.000",
            "camera": "Wide shot",
            "action": "{A} laughs.",
            "dialogue": None,
            "sound_events": "laughter",
        },
    ],
    "overall_soundscape": "Café ambience",
    "non_diegetic_music": "N/A",
}

_MONO = {
    "version": 1,
    "id": "monologue",
    "name": "Monologue",
    "description": "Single speaker",
    "slots": {
        "A": {"role": "speaker", "needs_voice": True, "needs_character_sheet": True},
    },
    "shots": [
        {
            "id": "shot_1",
            "timestamp": None,
            "camera": "Medium close-up",
            "action": "{A} speaks.",
            "dialogue": {"speaker_slot": "A", "placeholder": True, "default_text": None},
            "sound_events": None,
        },
    ],
    "overall_soundscape": "Room tone",
    "non_diegetic_music": "N/A",
}


# ── SceneTemplate properties ───────────────────────────────────────────────────

def test_template_id_from_data():
    t = SceneTemplate(_CAFE_2P)
    assert t.template_id == "cafe_2p"


def test_template_id_falls_back_to_filename(tmp_path):
    path = str(tmp_path / "my_template.json")
    data = dict(_CAFE_2P)
    del data["id"]
    t = SceneTemplate(data, file_path=path)
    assert t.template_id == "my_template"


def test_template_name():
    t = SceneTemplate(_CAFE_2P)
    assert t.name == "Café Conversation"


def test_template_description():
    t = SceneTemplate(_CAFE_2P)
    assert t.description == "Two people in a café"


def test_template_slots():
    t = SceneTemplate(_CAFE_2P)
    assert set(t.slot_ids) == {"A", "B"}
    assert t.slot_count == 2


def test_template_slot_count_monologue():
    t = SceneTemplate(_MONO)
    assert t.slot_count == 1


def test_template_shots():
    t = SceneTemplate(_CAFE_2P)
    assert len(t.shots) == 3


def test_template_does_not_mutate_input():
    data = dict(_CAFE_2P)
    t = SceneTemplate(data)
    data["name"] = "MUTATED"
    assert t.name == "Café Conversation"


def test_to_dict_round_trips():
    t = SceneTemplate(_CAFE_2P)
    d = t.to_dict()
    assert d["id"] == "cafe_2p"
    assert d["slots"]["A"]["role"] == "primary speaker"


# ── format_slot_info ───────────────────────────────────────────────────────────

def test_format_slot_info_two_slots():
    t = SceneTemplate(_CAFE_2P)
    info = t.format_slot_info()
    assert "Slot A:" in info
    assert "Slot B:" in info
    assert "primary speaker" in info
    assert "voice: required" in info
    assert "character sheet: required" in info


def test_format_slot_info_one_slot():
    t = SceneTemplate(_MONO)
    info = t.format_slot_info()
    assert "Slot A:" in info
    assert "Slot B:" not in info


def test_format_slot_info_no_slots():
    t = SceneTemplate({"id": "empty"})
    assert "No slots defined" in t.format_slot_info()


def test_format_slot_info_optional_needs():
    data = {
        "id": "t1",
        "slots": {
            "A": {"role": "observer", "needs_voice": False, "needs_character_sheet": False},
        },
    }
    t = SceneTemplate(data)
    info = t.format_slot_info()
    assert "observer" in info
    assert "voice" not in info
    assert "character sheet" not in info


# ── dialogue_placeholders ──────────────────────────────────────────────────────

def test_dialogue_placeholders_count():
    t = SceneTemplate(_CAFE_2P)
    placeholders = t.dialogue_placeholders()
    assert len(placeholders) == 2


def test_dialogue_placeholders_excludes_null_dialogue():
    t = SceneTemplate(_CAFE_2P)
    placeholders = t.dialogue_placeholders()
    for p in placeholders:
        assert p["dialogue"] is not None


def test_dialogue_placeholders_excludes_fixed():
    data = {
        "id": "t2",
        "slots": {"A": {}},
        "shots": [
            {"id": "s1", "dialogue": {"speaker_slot": "A", "placeholder": False, "default_text": "Hi"}, "action": "", "camera": "", "timestamp": None, "sound_events": None},
            {"id": "s2", "dialogue": {"speaker_slot": "A", "placeholder": True, "default_text": None}, "action": "", "camera": "", "timestamp": None, "sound_events": None},
        ],
    }
    t = SceneTemplate(data)
    assert len(t.dialogue_placeholders()) == 1


# ── scan_templates ─────────────────────────────────────────────────────────────

def test_scan_templates_empty_dir(tmp_path):
    assert scan_templates(str(tmp_path)) == []


def test_scan_templates_missing_dir(tmp_path):
    assert scan_templates(str(tmp_path / "nonexistent")) == []


def test_scan_templates_finds_json_files(tmp_path):
    _write_template(tmp_path / "alpha.json", _CAFE_2P)
    _write_template(tmp_path / "beta.json", _MONO)
    (tmp_path / "readme.txt").write_text("not a template")

    results = scan_templates(str(tmp_path))
    ids = [r["template_id"] for r in results]
    assert "alpha" in ids
    assert "beta" in ids
    assert len(results) == 2


def test_scan_templates_sorted(tmp_path):
    _write_template(tmp_path / "zzz.json", _MONO)
    _write_template(tmp_path / "aaa.json", _MONO)
    results = scan_templates(str(tmp_path))
    assert results[0]["template_id"] == "aaa"
    assert results[1]["template_id"] == "zzz"


def test_scan_templates_extracts_metadata(tmp_path):
    _write_template(tmp_path / "cafe.json", _CAFE_2P)
    results = scan_templates(str(tmp_path))
    r = results[0]
    assert r["name"] == "Café Conversation"
    assert r["description"] == "Two people in a café"
    assert r["slot_count"] == 2


def test_scan_templates_tolerates_bad_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json")
    results = scan_templates(str(tmp_path))
    assert results[0]["template_id"] == "broken"
    assert "error" in results[0]["description"]


# ── template_ids ───────────────────────────────────────────────────────────────

def test_template_ids_returns_sorted_list(tmp_path):
    _write_template(tmp_path / "zzz.json", _MONO)
    _write_template(tmp_path / "aaa.json", _MONO)
    ids = template_ids(str(tmp_path))
    assert ids == ["aaa", "zzz"]


def test_template_ids_empty_dir(tmp_path):
    assert template_ids(str(tmp_path)) == []


# ── format_template_list ───────────────────────────────────────────────────────

def test_format_template_list_empty(tmp_path):
    assert "No templates found" in format_template_list(str(tmp_path))


def test_format_template_list_shows_all(tmp_path):
    _write_template(tmp_path / "cafe.json", _CAFE_2P)
    _write_template(tmp_path / "mono.json", _MONO)
    listing = format_template_list(str(tmp_path))
    assert "[cafe]" in listing
    assert "[mono]" in listing
    assert "Café Conversation" in listing
    assert "2 slots" in listing
    assert "1 slot" in listing


def test_format_template_list_includes_description(tmp_path):
    _write_template(tmp_path / "cafe.json", _CAFE_2P)
    listing = format_template_list(str(tmp_path))
    assert "Two people in a café" in listing


# ── dir_fingerprint ────────────────────────────────────────────────────────────

def test_dir_fingerprint_empty_dir(tmp_path):
    assert dir_fingerprint(str(tmp_path)) == ()


def test_dir_fingerprint_missing_dir(tmp_path):
    assert dir_fingerprint(str(tmp_path / "missing")) == ()


def test_dir_fingerprint_changes_on_add(tmp_path):
    fp1 = dir_fingerprint(str(tmp_path))
    _write_template(tmp_path / "t.json", _MONO)
    fp2 = dir_fingerprint(str(tmp_path))
    assert fp1 != fp2


def test_dir_fingerprint_includes_filename(tmp_path):
    _write_template(tmp_path / "alpha.json", _MONO)
    fp = dir_fingerprint(str(tmp_path))
    assert any("alpha.json" in item[0] for item in fp)


# ── load_template ──────────────────────────────────────────────────────────────

def test_load_template(tmp_path):
    path = str(tmp_path / "cafe.json")
    _write_template(path, _CAFE_2P)
    t = load_template(path)
    assert t.name == "Café Conversation"
    assert t.slot_count == 2
    assert t.file_path == path


def test_load_template_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_template(str(tmp_path / "missing.json"))


# ── Bundled example templates ──────────────────────────────────────────────────

BUNDLED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scene_templates",
)


def test_bundled_templates_exist():
    assert os.path.isdir(BUNDLED_DIR), "scene_templates/ directory must exist in repo root"
    jsons = [f for f in os.listdir(BUNDLED_DIR) if f.endswith(".json")]
    assert len(jsons) >= 3, "At least 3 bundled example templates expected"


@pytest.mark.parametrize("fname", [
    "monologue_indoor.json",
    "cafe_conversation_2p.json",
    "meeting_room_3p.json",
])
def test_bundled_template_loads(fname):
    path = os.path.join(BUNDLED_DIR, fname)
    t = load_template(path)
    assert t.template_id, "template must have an id"
    assert t.name, "template must have a name"
    assert t.slot_count >= 1, "template must have at least one slot"
    assert len(t.shots) >= 1, "template must have at least one shot"


def test_bundled_monologue_has_one_slot():
    t = load_template(os.path.join(BUNDLED_DIR, "monologue_indoor.json"))
    assert t.slot_count == 1


def test_bundled_cafe_has_two_slots():
    t = load_template(os.path.join(BUNDLED_DIR, "cafe_conversation_2p.json"))
    assert t.slot_count == 2


def test_bundled_meeting_has_three_slots():
    t = load_template(os.path.join(BUNDLED_DIR, "meeting_room_3p.json"))
    assert t.slot_count == 3
