"""Tests for utils/llm_scanner.py — pure filesystem scanner, no ComfyUI deps."""

import json
import os
import pytest
from conftest import import_test_module

scanner = import_test_module("utils/llm_scanner.py")
scan_llm_dirs    = scanner.scan_llm_dirs
_scan_gguf_dir   = scanner._scan_gguf_dir
_scan_hf_dir     = scanner._scan_hf_dir
_scan_directory  = scanner._scan_directory
capability_tags  = scanner.capability_tags
capability_note  = scanner.capability_note
DEFAULT_MODEL    = scanner.DEFAULT_MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_gguf_dir(tmp_path, name="MyGGUF", main_size=100, mmproj=False):
    d = tmp_path / name
    d.mkdir()
    (d / f"{name}-Q4_K_M.gguf").write_bytes(b"G" * main_size)
    if mmproj:
        (d / "mmproj-vision-f16.gguf").write_bytes(b"P" * 10)
    return d


def _make_hf_dir(tmp_path, name="MyHF", architectures=None, model_type="", has_vision_config=False,
                 has_preprocessor=False, extra_safetensors=None):
    d = tmp_path / name
    d.mkdir()
    config = {}
    if architectures:
        config["architectures"] = architectures
    if model_type:
        config["model_type"] = model_type
    if has_vision_config:
        config["vision_config"] = {"hidden_size": 1152}
    (d / "config.json").write_text(json.dumps(config))
    if has_preprocessor:
        (d / "preprocessor_config.json").write_text("{}")
    if extra_safetensors:
        for fname, size in extra_safetensors.items():
            (d / fname).write_bytes(b"W" * size)
    return d


# ── GGUF scanner ──────────────────────────────────────────────────────────────

def test_gguf_dir_detected(tmp_path):
    d = _make_gguf_dir(tmp_path, "Qwen3B")
    result = _scan_gguf_dir(str(d), "Qwen3B")
    assert result is not None
    assert result["format"] == "gguf"
    assert result["name"] == "Qwen3B"
    assert result["supports_vision"] is False


def test_gguf_dir_with_mmproj_is_vision(tmp_path):
    d = _make_gguf_dir(tmp_path, "LLaVA", mmproj=True)
    result = _scan_gguf_dir(str(d), "LLaVA")
    assert result is not None
    assert result["supports_vision"] is True
    assert result["mmproj_file"] is not None


def test_gguf_dir_no_gguf_returns_none(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "readme.txt").write_text("hi")
    assert _scan_gguf_dir(str(d), "empty") is None


def test_gguf_picks_largest_as_main(tmp_path):
    d = tmp_path / "Multi"
    d.mkdir()
    (d / "Multi-Q8.gguf").write_bytes(b"X" * 200)
    (d / "Multi-Q4.gguf").write_bytes(b"X" * 100)
    result = _scan_gguf_dir(str(d), "Multi")
    assert result["main_file"] == "Multi-Q8.gguf"


def test_gguf_native_video_always_false(tmp_path):
    d = _make_gguf_dir(tmp_path, "G", mmproj=True)
    result = _scan_gguf_dir(str(d), "G")
    assert result["native_video"] is False


def test_gguf_supports_video_matches_vision(tmp_path):
    d = _make_gguf_dir(tmp_path, "V", mmproj=True)
    result = _scan_gguf_dir(str(d), "V")
    assert result["supports_video"] is True

    d2 = _make_gguf_dir(tmp_path, "T")
    result2 = _scan_gguf_dir(str(d2), "T")
    assert result2["supports_video"] is False


# ── HF scanner ────────────────────────────────────────────────────────────────

def test_hf_dir_text_only(tmp_path):
    d = _make_hf_dir(tmp_path, "LlamaText", architectures=["LlamaForCausalLM"], model_type="llama")
    result = _scan_hf_dir(str(d), "LlamaText")
    assert result is not None
    assert result["format"] == "hf"
    assert result["supports_vision"] is False
    assert result["native_video"] is False


def test_hf_dir_vision_via_architecture(tmp_path):
    d = _make_hf_dir(tmp_path, "LLaVA", architectures=["LlavaForConditionalGeneration"])
    result = _scan_hf_dir(str(d), "LLaVA")
    assert result["supports_vision"] is True


def test_hf_dir_vision_via_vision_config(tmp_path):
    d = _make_hf_dir(tmp_path, "VModel", has_vision_config=True)
    result = _scan_hf_dir(str(d), "VModel")
    assert result["supports_vision"] is True


def test_hf_dir_vision_via_preprocessor(tmp_path):
    d = _make_hf_dir(tmp_path, "PP", has_preprocessor=True)
    result = _scan_hf_dir(str(d), "PP")
    assert result["supports_vision"] is True


def test_hf_dir_vision_via_model_type(tmp_path):
    d = _make_hf_dir(tmp_path, "Qwen", model_type="qwen2_vl")
    result = _scan_hf_dir(str(d), "Qwen")
    assert result["supports_vision"] is True


def test_hf_dir_native_video_qwen_vl(tmp_path):
    d = _make_hf_dir(tmp_path, "QwenVL", architectures=["Qwen2_5_VLForConditionalGeneration"])
    result = _scan_hf_dir(str(d), "QwenVL")
    assert result["native_video"] is True


def test_hf_dir_no_config_returns_none(tmp_path):
    d = tmp_path / "NoConfig"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"X" * 100)
    assert _scan_hf_dir(str(d), "NoConfig") is None


def test_hf_size_estimate(tmp_path):
    d = _make_hf_dir(tmp_path, "Sized", extra_safetensors={"model.safetensors": 1024 * 1024})
    result = _scan_hf_dir(str(d), "Sized")
    assert result["size_mb"] == 1


# ── Directory walker ──────────────────────────────────────────────────────────

def test_walk_finds_nested_gguf(tmp_path):
    # tmp_path/LLMs/MyModel/*.gguf
    model_dir = tmp_path / "LLMs" / "MyModel"
    model_dir.mkdir(parents=True)
    (model_dir / "MyModel-Q4.gguf").write_bytes(b"G" * 50)

    results = _scan_directory(str(tmp_path / "LLMs"))
    assert any(r["name"] == "MyModel" for r in results)


def test_walk_finds_hf_nested(tmp_path):
    model_dir = tmp_path / "LLMs" / "MyHF"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

    results = _scan_directory(str(tmp_path / "LLMs"))
    assert any(r["name"] == "MyHF" for r in results)


def test_walk_empty_dir_returns_empty(tmp_path):
    assert _scan_directory(str(tmp_path)) == []


def test_walk_nonexistent_returns_empty(tmp_path):
    assert _scan_directory(str(tmp_path / "nonexistent")) == []


def test_walk_does_not_recurse_into_gguf_model(tmp_path):
    """A GGUF dir with a sub-subdir should not be descended into."""
    model_dir = tmp_path / "M"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_bytes(b"G" * 10)
    nested = model_dir / "sub"
    nested.mkdir()
    (nested / "n.gguf").write_bytes(b"G" * 10)

    results = _scan_directory(str(tmp_path))
    # Only the top-level model dir should be found
    assert len(results) == 1
    assert results[0]["name"] == "M"


# ── scan_llm_dirs (mocked folder_paths) ──────────────────────────────────────

def test_scan_llm_dirs_with_extra_dirs(tmp_path):
    model_dir = tmp_path / "MyVisionModel"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"G" * 10)
    (model_dir / "mmproj-f16.gguf").write_bytes(b"P" * 5)

    results = scan_llm_dirs(extra_dirs=[str(tmp_path)])
    found = [r for r in results if r["name"] == "MyVisionModel"]
    assert len(found) == 1
    assert found[0]["supports_vision"] is True
    assert "capability_tags" in found[0]
    assert "capability_note" in found[0]


def test_scan_llm_dirs_deduplicates_same_path(tmp_path):
    model_dir = tmp_path / "Dup"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_bytes(b"G" * 10)

    results = scan_llm_dirs(extra_dirs=[str(tmp_path), str(tmp_path)])
    names = [r["name"] for r in results]
    assert names.count("Dup") == 1


def test_scan_llm_dirs_sorted_by_name(tmp_path):
    for name in ["Zoo", "Alpha", "Middle"]:
        d = tmp_path / name
        d.mkdir()
        (d / "m.gguf").write_bytes(b"G" * 10)

    results = scan_llm_dirs(extra_dirs=[str(tmp_path)])
    names = [r["name"] for r in results]
    assert names == sorted(names, key=str.lower)


# ── Capability tags ───────────────────────────────────────────────────────────

def test_text_only_tag():
    m = {"supports_vision": False, "native_video": False}
    assert capability_tags(m) == ["🔤 Text only"]


def test_vision_frame_sampling_tag():
    m = {"supports_vision": True, "native_video": False}
    tags = capability_tags(m)
    assert "📷 Vision" in tags
    assert "🎬 Video (frames)" in tags
    assert "🔤 Text only" not in tags


def test_native_video_tag():
    m = {"supports_vision": True, "native_video": True}
    tags = capability_tags(m)
    assert "🎬 Video (native)" in tags
    assert "🎬 Video (frames)" not in tags


def test_capability_note_text_only():
    m = {"supports_vision": False, "native_video": False}
    note = capability_note(m)
    assert "Text only" in note
    assert "image" in note.lower()


def test_capability_note_vision():
    m = {"supports_vision": True, "native_video": False}
    note = capability_note(m)
    assert "image" in note.lower()
    assert "frame" in note.lower()


def test_capability_note_native_video():
    m = {"supports_vision": True, "native_video": True}
    note = capability_note(m)
    assert "temporal" in note.lower()


# ── Default model descriptor ──────────────────────────────────────────────────

def test_default_model_has_required_fields():
    for key in ["repo_id", "filename", "mmproj", "name", "size_hint", "description",
                "supports_vision", "supports_video", "capability_note"]:
        assert key in DEFAULT_MODEL, f"Missing key: {key}"


def test_default_model_is_vision_capable():
    assert DEFAULT_MODEL["supports_vision"] is True
    assert DEFAULT_MODEL["supports_video"] is True
