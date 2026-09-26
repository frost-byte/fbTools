"""Tests for utils/h3_template_runner.py — pure logic, no ComfyUI dependencies."""
import pytest

from conftest import import_test_module

h3_template_runner = import_test_module("utils/h3_template_runner.py")
find_node_by_title = h3_template_runner.find_node_by_title
patch_prompt = h3_template_runner.patch_prompt


def _template():
    """A minimal but representative API-format template with the 4 contract titles."""
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "placeholder.png", "upload": "image"},
            "_meta": {"title": "IN:image"},
        },
        "2": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {"prompt": "placeholder prompt", "width": 1344, "height": 768},
            "_meta": {"title": "IN:prompt"},
        },
        "3": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 0, "control_after_generate": "fixed"},
            "_meta": {"title": "IN:seed"},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI"},
            "_meta": {"title": "OUT:save"},
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "some_clip.safetensors"},
            "_meta": {"title": ""},
        },
        "6": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "default_model.safetensors", "weight_dtype": "default"},
            "_meta": {"title": "IN:model"},
        },
    }


def test_find_node_by_title_happy_path():
    assert find_node_by_title(_template(), "IN:image") == "1"
    assert find_node_by_title(_template(), "OUT:save") == "4"


def test_find_node_by_title_missing_raises_with_title_named():
    with pytest.raises(ValueError, match="IN:missing"):
        find_node_by_title(_template(), "IN:missing")


def test_find_node_by_title_duplicate_raises():
    tpl = _template()
    tpl["6"] = {"class_type": "LoadImage", "inputs": {}, "_meta": {"title": "IN:image"}}
    with pytest.raises(ValueError, match="Multiple nodes titled"):
        find_node_by_title(tpl, "IN:image")


def test_patch_prompt_sets_all_four_contract_fields():
    tpl = _template()
    patched = patch_prompt(
        tpl,
        image="fbtools_tmp/frame.jpg",
        prompt_text="remove the people",
        seed=12345,
        filename_prefix="fbtools/h3_background_plates/abc123",
    )
    assert patched["1"]["inputs"]["image"] == "fbtools_tmp/frame.jpg"
    assert patched["2"]["inputs"]["prompt"] == "remove the people"
    assert patched["3"]["inputs"]["noise_seed"] == 12345
    assert patched["4"]["inputs"]["filename_prefix"] == "fbtools/h3_background_plates/abc123"


def test_patch_prompt_does_not_mutate_original_template():
    tpl = _template()
    original_image = tpl["1"]["inputs"]["image"]
    patch_prompt(
        tpl,
        image="different.jpg",
        prompt_text="different prompt",
        seed=999,
        filename_prefix="different/prefix",
    )
    assert tpl["1"]["inputs"]["image"] == original_image


def test_patch_prompt_missing_title_raises():
    tpl = _template()
    del tpl["4"]  # drop OUT:save
    with pytest.raises(ValueError, match="OUT:save"):
        patch_prompt(tpl, image="x.png", prompt_text="p", seed=1, filename_prefix="pfx")


def test_find_node_by_title_optional_missing_returns_none():
    assert find_node_by_title(_template(), "IN:missing", required=False) is None


def test_find_node_by_title_optional_present_still_returns_id():
    assert find_node_by_title(_template(), "IN:model", required=False) == "6"


def test_find_node_by_title_optional_duplicate_still_raises():
    tpl = _template()
    tpl["7"] = {"class_type": "UNETLoader", "inputs": {}, "_meta": {"title": "IN:model"}}
    with pytest.raises(ValueError, match="Multiple nodes titled"):
        find_node_by_title(tpl, "IN:model", required=False)


def test_patch_prompt_applies_override_for_title_present_in_template():
    tpl = _template()
    patched = patch_prompt(
        tpl, image="x.png", prompt_text="p", seed=1, filename_prefix="pfx",
        overrides={"IN:model": {"unet_name": "turbo_hybrid.safetensors"}},
    )
    assert patched["6"]["inputs"]["unet_name"] == "turbo_hybrid.safetensors"


def test_patch_prompt_silently_skips_override_for_title_absent_from_template():
    tpl = _template()
    del tpl["6"]  # no IN:model in this template
    patched = patch_prompt(
        tpl, image="x.png", prompt_text="p", seed=1, filename_prefix="pfx",
        overrides={"IN:model": {"unet_name": "turbo_hybrid.safetensors"}},
    )
    # No exception, and the 4 required fields still got patched normally.
    assert patched["1"]["inputs"]["image"] == "x.png"
    assert "6" not in patched


def test_patch_prompt_overrides_does_not_mutate_original_template():
    tpl = _template()
    original = tpl["6"]["inputs"]["unet_name"]
    patch_prompt(
        tpl, image="x.png", prompt_text="p", seed=1, filename_prefix="pfx",
        overrides={"IN:model": {"unet_name": "turbo_hybrid.safetensors"}},
    )
    assert tpl["6"]["inputs"]["unet_name"] == original
