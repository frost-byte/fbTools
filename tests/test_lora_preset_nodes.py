"""Tests for LoraPresetDefine and LoraPresetSelect logic."""

from conftest import import_test_module
import pytest

lp = import_test_module("utils/lora_presets.py")
preset_define = lp.preset_define
preset_select = lp.preset_select

# Minimal LORA_STACK_DATA values: list of dicts
STACK_A = [{"lora": "lora_a.safetensors", "strength_model": 1.0, "strength_clip": 1.0, "model_target": "all", "enabled": True}]
STACK_B = [{"lora": "lora_b.safetensors", "strength_model": 0.8, "strength_clip": 0.8, "model_target": "all", "enabled": True}]


# ── preset_define ──────────────────────────────────────────────────────────────

def test_define_single_node_no_input():
    result = preset_define("Portrait", STACK_A, "a woman in a park")
    assert len(result) == 1
    assert result[0]["name"] == "Portrait"
    assert result[0]["lora_stack"] == STACK_A
    assert result[0]["prompt"] == "a woman in a park"


def test_define_accepts_none_stack():
    result = preset_define("Empty", None, "prompt")
    assert result[0]["lora_stack"] is None


def test_define_chain_three_nodes():
    r1 = preset_define("Style A", STACK_A, "prompt a")
    r2 = preset_define("Style B", STACK_B, "prompt b", r1)
    r3 = preset_define("Style C", None,   "prompt c", r2)

    assert len(r3) == 3
    assert r3[0]["name"] == "Style A"
    assert r3[1]["name"] == "Style B"
    assert r3[2]["name"] == "Style C"


def test_define_does_not_mutate_incoming_list():
    r1 = preset_define("Style A", STACK_A, "prompt a")
    original_len = len(r1)
    preset_define("Style B", STACK_B, "prompt b", r1)
    assert len(r1) == original_len


def test_define_prompt_multiline_and_unicode():
    prompt = "line one\nline two, \"quoted\", 日本語"
    result = preset_define("Unicode", None, prompt)
    assert result[0]["prompt"] == prompt


# ── preset_select ──────────────────────────────────────────────────────────────

def _build_list(n: int) -> list:
    presets = None
    for i in range(n):
        stack = [{"lora": f"lora_{i}.safetensors", "strength_model": 1.0, "strength_clip": 1.0, "model_target": "all", "enabled": True}]
        presets = preset_define(f"Style {i}", stack, f"prompt {i}", presets)
    return presets


def test_select_by_name_first():
    name, lora_stack, lora_stack_native, prompt, _ = preset_select(_build_list(3), "Style 0")
    assert name == "Style 0"
    assert lora_stack == [{"lora": "lora_0.safetensors", "strength_model": 1.0, "strength_clip": 1.0, "model_target": "all", "enabled": True}]
    assert prompt == "prompt 0"
    # Native stack should be auto-generated from LORA_STACK_DATA
    assert lora_stack_native is not None
    assert lora_stack_native[0][0] == "lora_0.safetensors"


def test_select_by_name_last():
    name, lora_stack, _, _, _ = preset_select(_build_list(3), "Style 2")
    assert name == "Style 2"
    assert lora_stack[0]["lora"] == "lora_2.safetensors"


def test_select_unknown_name_falls_back_to_first():
    name, _, _, _, _ = preset_select(_build_list(3), "Nonexistent")
    assert name == "Style 0"


def test_select_none_default_falls_back_to_first():
    # "none" is the default combo value before the user runs the node
    name, _, _, _, _ = preset_select(_build_list(3), "none")
    assert name == "Style 0"


def test_select_available_presets_format():
    _, _, _, _, available = preset_select(_build_list(3), "Style 0")
    assert "[0] Style 0" in available
    assert "[1] Style 1" in available
    assert "[2] Style 2" in available


def test_select_empty_list():
    name, lora_stack, lora_stack_native, prompt, available = preset_select([], "anything")
    assert name == ""
    assert lora_stack is None
    assert lora_stack_native is None
    assert prompt == ""
    assert "No presets available" in available


# ── native LORA_STACK auto-generation ─────────────────────────────────────────

def test_native_auto_generated_from_stack_data():
    result = preset_define("Portrait", STACK_A, "prompt")
    native = result[0]["lora_stack_native"]
    assert native is not None
    assert len(native) == 1
    assert native[0][0] == "lora_a.safetensors"
    assert native[0][1] == pytest.approx(1.0)
    assert native[0][2] == pytest.approx(1.0)


def test_native_not_generated_when_stack_is_none():
    result = preset_define("Empty", None, "prompt")
    assert result[0]["lora_stack_native"] is None


def test_explicit_native_not_overwritten_by_auto_gen():
    custom_native = [("custom.safetensors", 0.5, 0.7)]
    result = preset_define("Custom", STACK_A, "prompt", lora_stack_native=custom_native)
    assert result[0]["lora_stack_native"] == custom_native


def test_native_only_no_stack_data():
    native_input = [("lora_n.safetensors", 0.8, 0.9)]
    result = preset_define("NativeOnly", None, "prompt", lora_stack_native=native_input)
    assert result[0]["lora_stack"] is None
    assert result[0]["lora_stack_native"] == native_input


def test_native_excludes_disabled_entries():
    stack_with_disabled = [
        {"lora": "a.safetensors", "strength_model": 1.0, "strength_clip": 1.0, "model_target": "all", "enabled": True},
        {"lora": "b.safetensors", "strength_model": 0.5, "strength_clip": 0.5, "model_target": "all", "enabled": False},
    ]
    result = preset_define("Mixed", stack_with_disabled, "prompt")
    native = result[0]["lora_stack_native"]
    assert len(native) == 1
    assert native[0][0] == "a.safetensors"
