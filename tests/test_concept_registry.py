"""Tests for the Concept Registry system."""

import json
import os
import pytest
from conftest import import_test_module

cr = import_test_module("utils/concept_registry.py")

ConceptRegistry = cr.ConceptRegistry
load_registry = cr.load_registry
save_registry = cr.save_registry
resolve_concepts = cr.resolve_concepts
assemble_prompt = cr.assemble_prompt
format_resolved_info = cr.format_resolved_info
parse_concept_ids = cr.parse_concept_ids
build_model_entry = cr.build_model_entry
MODEL_PROFILES = cr.MODEL_PROFILES
MODEL_TYPE_IDS = cr.MODEL_TYPE_IDS


# ── MODEL_PROFILES ─────────────────────────────────────────────────────────────

def test_model_profiles_split_flags():
    assert MODEL_PROFILES["wan22"]["split"] is True
    assert MODEL_PROFILES["bernini"]["split"] is True
    assert MODEL_PROFILES["ltx23"]["split"] is False
    assert MODEL_PROFILES["flux2"]["split"] is False
    assert MODEL_PROFILES["krea2"]["split"] is False
    assert MODEL_PROFILES["qwen"]["split"] is False
    assert MODEL_PROFILES["minimax_h3"]["split"] is False


def test_model_type_ids_covers_all_profiles():
    assert set(MODEL_TYPE_IDS) == set(MODEL_PROFILES.keys())


# ── ConceptRegistry.define ─────────────────────────────────────────────────────

def _make_reg(**kwargs) -> ConceptRegistry:
    return ConceptRegistry(**kwargs)


def test_define_new_concept_creates_entry():
    reg = _make_reg()
    entry = {"lora": "char.safetensors", "weight": 0.8, "trigger": "XY"}
    new_reg = reg.define("char_alice", "Alice", "main character", "flux2", entry)

    assert "char_alice" in new_reg.concepts
    c = new_reg.concepts["char_alice"]
    assert c["name"] == "Alice"
    assert c["description"] == "main character"
    assert c["models"]["flux2"] == entry


def test_define_does_not_mutate_original():
    reg = _make_reg()
    entry = {"lora": "a.safetensors", "weight": 1.0, "trigger": ""}
    new_reg = reg.define("c1", "C1", "", "flux2", entry)

    assert "c1" not in reg.concepts
    assert "c1" in new_reg.concepts


def test_define_different_model_types_accumulate():
    reg = _make_reg()
    entry_flux = {"lora": "a_flux.safetensors", "weight": 1.0, "trigger": "abc"}
    entry_wan = {"lora_high": "a_wan_h.safetensors", "lora_low": "a_wan_l.safetensors",
                 "weight_high": 0.8, "weight_low": 0.7, "trigger": "abc"}

    r1 = reg.define("char_a", "Char A", "", "flux2", entry_flux)
    r2 = r1.define("char_a", "Char A", "", "wan22", entry_wan)

    assert "flux2" in r2.concepts["char_a"]["models"]
    assert "wan22" in r2.concepts["char_a"]["models"]
    assert len(r2.concepts["char_a"]["models"]) == 2


def test_define_same_model_type_overwrites():
    reg = _make_reg()
    entry_v1 = {"lora": "v1.safetensors", "weight": 1.0, "trigger": "old"}
    entry_v2 = {"lora": "v2.safetensors", "weight": 0.9, "trigger": "new"}

    r1 = reg.define("char_a", "Char A", "", "flux2", entry_v1)
    r2 = r1.define("char_a", "Char A", "", "flux2", entry_v2)

    assert r2.concepts["char_a"]["models"]["flux2"]["lora"] == "v2.safetensors"
    assert r2.concepts["char_a"]["models"]["flux2"]["trigger"] == "new"


def test_define_chain_three_concepts():
    reg = _make_reg()
    for i in range(3):
        reg = reg.define(f"c{i}", f"C{i}", "", "flux2",
                         {"lora": f"lora{i}.safetensors", "weight": 1.0, "trigger": ""})
    assert len(reg.concepts) == 3


# ── Persistence ────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "registry.json")
    reg = _make_reg()
    reg = reg.define("c1", "Concept One", "desc", "flux2",
                     {"lora": "c1.safetensors", "weight": 1.0, "trigger": "TRIGGER"})
    save_registry(reg, path, backup=False)

    loaded = load_registry(path)
    assert "c1" in loaded.concepts
    assert loaded.concepts["c1"]["models"]["flux2"]["trigger"] == "TRIGGER"


def test_save_creates_backup(tmp_path):
    path = str(tmp_path / "registry.json")
    reg = _make_reg()
    reg = reg.define("c1", "C1", "", "flux2", {"lora": "l.safetensors", "weight": 1.0, "trigger": ""})
    save_registry(reg, path, backup=False)
    save_registry(reg, path, backup=True)

    assert os.path.exists(path + ".bak")


def test_load_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "nonexistent.json")
    reg = load_registry(path)
    assert reg.concepts == {}
    assert reg.file_path == path


# ── resolve_concepts ───────────────────────────────────────────────────────────

def _registry_with_concepts() -> ConceptRegistry:
    reg = _make_reg()
    # Single-model type (flux2)
    reg = reg.define("char_alice", "Alice", "", "flux2",
                     {"lora": "alice_flux.safetensors", "weight": 0.9, "trigger": "ALICE_TRIGGER"})
    # Split-model type (wan22)
    reg = reg.define("style_cinematic", "Cinematic", "", "wan22", {
        "lora_high": "cinematic_h.safetensors",
        "lora_low":  "cinematic_l.safetensors",
        "weight_high": 0.8,
        "weight_low":  0.6,
        "trigger": "cinematic",
    })
    # Concept defined for both types
    reg = reg.define("effect_glow", "Glow Effect", "", "flux2",
                     {"lora": "glow_flux.safetensors", "weight": 0.7, "trigger": "glow"})
    reg = reg.define("effect_glow", "Glow Effect", "", "wan22", {
        "lora_high": "glow_wan_h.safetensors",
        "lora_low":  "glow_wan_l.safetensors",
        "weight_high": 0.7,
        "weight_low":  0.5,
        "trigger": "glow",
    })
    return reg


def test_resolve_single_model_type():
    reg = _registry_with_concepts()
    result = resolve_concepts(reg, ["char_alice"], "flux2")

    assert len(result) == 1
    r = result[0]
    assert r.concept_id == "char_alice"
    assert r.lora_high == "alice_flux.safetensors"
    assert r.lora_low is None
    assert r.weight_high == pytest.approx(0.9)
    assert r.trigger == "ALICE_TRIGGER"
    assert r.error is None


def test_resolve_split_model_type():
    reg = _registry_with_concepts()
    result = resolve_concepts(reg, ["style_cinematic"], "wan22")

    r = result[0]
    assert r.lora_high == "cinematic_h.safetensors"
    assert r.lora_low == "cinematic_l.safetensors"
    assert r.weight_high == pytest.approx(0.8)
    assert r.weight_low == pytest.approx(0.6)


def test_resolve_missing_concept_returns_error():
    reg = _registry_with_concepts()
    result = resolve_concepts(reg, ["does_not_exist"], "flux2")

    assert result[0].error == "NOT FOUND"


def test_resolve_concept_missing_model_type():
    reg = _registry_with_concepts()
    # char_alice is only defined for flux2, not for wan22
    result = resolve_concepts(reg, ["char_alice"], "wan22")

    assert result[0].error is not None
    assert "wan22" in result[0].error


def test_resolve_multiple_concepts():
    reg = _registry_with_concepts()
    result = resolve_concepts(reg, ["char_alice", "effect_glow"], "flux2")

    assert len(result) == 2
    assert result[0].concept_id == "char_alice"
    assert result[1].concept_id == "effect_glow"


# ── assemble_prompt ────────────────────────────────────────────────────────────

def test_assemble_prepend():
    result = assemble_prompt(["trigger1", "trigger2"], "base text", "prepend")
    assert result == "trigger1, trigger2, base text"


def test_assemble_append():
    result = assemble_prompt(["glow"], "beautiful portrait", "append")
    assert result == "beautiful portrait, glow"


def test_assemble_empty_triggers_returns_base():
    result = assemble_prompt([], "base text", "prepend")
    assert result == "base text"


def test_assemble_empty_base_returns_triggers():
    result = assemble_prompt(["trigger"], "", "append")
    assert result == "trigger"


def test_assemble_skips_blank_triggers():
    result = assemble_prompt(["  ", "glow", ""], "portrait", "prepend")
    assert result == "glow, portrait"


# ── parse_concept_ids ──────────────────────────────────────────────────────────

def test_parse_comma_separated():
    result = parse_concept_ids("char_alice, effect_glow, style_cinematic")
    assert result == ["char_alice", "effect_glow", "style_cinematic"]


def test_parse_newline_separated():
    result = parse_concept_ids("char_alice\neffect_glow\nstyle_cinematic")
    assert result == ["char_alice", "effect_glow", "style_cinematic"]


def test_parse_mixed_separators():
    result = parse_concept_ids("a, b\nc")
    assert result == ["a", "b", "c"]


def test_parse_trims_whitespace():
    result = parse_concept_ids("  a  ,  b  ")
    assert result == ["a", "b"]


def test_parse_skips_empty_parts():
    result = parse_concept_ids("a,,b,,c")
    assert result == ["a", "b", "c"]


# ── build_model_entry ──────────────────────────────────────────────────────────

def test_build_entry_single_model():
    entry = build_model_entry("flux2", "lora.safetensors", "None", 0.9, 0.5, "TRIGGER")
    assert entry == {"lora": "lora.safetensors", "weight": 0.9, "trigger": "TRIGGER"}
    assert "lora_high" not in entry
    assert "lora_low" not in entry


def test_build_entry_split_model():
    entry = build_model_entry("wan22", "h.safetensors", "l.safetensors", 0.8, 0.6, "style")
    assert entry == {
        "lora_high": "h.safetensors",
        "lora_low":  "l.safetensors",
        "weight_high": 0.8,
        "weight_low":  0.6,
        "trigger": "style",
    }


def test_build_entry_none_lora_stored_as_empty():
    entry = build_model_entry("flux2", "None", "None", 1.0, 1.0, "")
    assert entry["lora"] == ""


# ── list_concepts ──────────────────────────────────────────────────────────────

def test_list_all_concepts():
    reg = _registry_with_concepts()
    listing = reg.list_concepts()
    assert "char_alice" in listing
    assert "effect_glow" in listing
    assert "style_cinematic" in listing


def test_list_filtered_by_model_type():
    reg = _registry_with_concepts()
    listing = reg.list_concepts("flux2")
    assert "char_alice" in listing
    assert "effect_glow" in listing
    assert "style_cinematic" not in listing


def test_list_empty_registry():
    reg = _make_reg()
    assert reg.list_concepts() == "No concepts defined."


# ── format_resolved_info ───────────────────────────────────────────────────────

def test_format_resolved_info_includes_model_type():
    reg = _registry_with_concepts()
    resolved = resolve_concepts(reg, ["char_alice"], "flux2")
    info = format_resolved_info("flux2", resolved, "assembled text")
    assert "flux2" in info
    assert "Flux 2" in info
    assert "assembled text" in info
