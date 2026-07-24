"""Tests for WanPresetDefine and WanPresetSelect logic."""

import pytest
from conftest import import_test_module

wp = import_test_module("utils/wan_presets.py")
preset_define = wp.preset_define
preset_select = wp.preset_select

# Minimal LORA_STACK values: list of (lora_name, model_strength, clip_strength)
STACK_H = [("lora_a_high.safetensors", 1.0, 1.0)]
STACK_L = [("lora_a_low.safetensors", 0.8, 0.8)]
STACK_H2 = [("lora_b_high.safetensors", 1.0, 1.0)]
STACK_L2 = [("lora_b_low.safetensors", 0.8, 0.8)]


# ── preset_define ─────────────────────────────────────────────────────────────

def test_define_single_node_no_input():
    result = preset_define("Style A", STACK_H, STACK_L, "a woman walking on the beach")
    assert len(result) == 1
    assert result[0]["name"] == "Style A"
    assert result[0]["lora_h"] == STACK_H
    assert result[0]["lora_l"] == STACK_L
    assert result[0]["prompt"] == "a woman walking on the beach"


def test_define_accepts_none_stacks():
    result = preset_define("Empty", None, None, "prompt")
    assert result[0]["lora_h"] is None
    assert result[0]["lora_l"] is None


def test_define_chain_three_nodes():
    r1 = preset_define("Style A", STACK_H, STACK_L, "prompt a")
    r2 = preset_define("Style B", STACK_H2, STACK_L2, "prompt b", r1)
    r3 = preset_define("Style C", None, None, "prompt c", r2)

    assert len(r3) == 3
    assert r3[0]["name"] == "Style A"
    assert r3[1]["name"] == "Style B"
    assert r3[2]["name"] == "Style C"


def test_define_does_not_mutate_incoming_list():
    r1 = preset_define("Style A", STACK_H, STACK_L, "prompt a")
    original_len = len(r1)
    preset_define("Style B", STACK_H2, STACK_L2, "prompt b", r1)
    assert len(r1) == original_len


def test_define_prompt_multiline_and_unicode():
    prompt = "line one\nline two, \"quoted\", 日本語"
    result = preset_define("Unicode", None, None, prompt)
    assert result[0]["prompt"] == prompt


# ── preset_select ─────────────────────────────────────────────────────────────

def _build_list(n: int) -> list:
    presets = None
    for i in range(n):
        stack = [(f"lora_{i}.safetensors", 1.0, 1.0)]
        presets = preset_define(f"Style {i}", stack, stack, f"prompt {i}", presets)
    return presets


def test_select_index_zero():
    name, lora_h, lora_l, prompt, _ = preset_select(_build_list(3), 0)
    assert name == "Style 0"
    assert lora_h == [("lora_0.safetensors", 1.0, 1.0)]
    assert lora_l == [("lora_0.safetensors", 1.0, 1.0)]
    assert prompt == "prompt 0"


def test_select_index_two():
    name, lora_h, _, _, _ = preset_select(_build_list(3), 2)
    assert name == "Style 2"
    assert lora_h == [("lora_2.safetensors", 1.0, 1.0)]


def test_select_index_out_of_range_clamps():
    name, _, _, _, _ = preset_select(_build_list(3), 99)
    assert name == "Style 2"


def test_select_available_presets_format():
    _, _, _, _, available = preset_select(_build_list(3), 0)
    assert "[0] Style 0" in available
    assert "[1] Style 1" in available
    assert "[2] Style 2" in available


def test_select_empty_list():
    name, lora_h, lora_l, prompt, available = preset_select([], 0)
    assert name == ""
    assert lora_h is None
    assert lora_l is None
    assert prompt == ""
    assert "No presets available" in available
