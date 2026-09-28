"""Tests for utils/h3_template_runner.py — pure logic, no ComfyUI dependencies."""
import pytest

from conftest import import_test_module

h3_template_runner = import_test_module("utils/h3_template_runner.py")
find_node_by_title = h3_template_runner.find_node_by_title
patch_prompt = h3_template_runner.patch_prompt
patch_character_sheet_prompt = h3_template_runner.patch_character_sheet_prompt


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


# ── patch_character_sheet_prompt ────────────────────────────────────────────────

def _char_sheet_template():
    """A minimal but representative API-format template with the character-sheet contract titles."""
    return {
        "1": {
            "class_type": "DenoAdvancedImageSourceLoader",
            "inputs": {"image_paths": "", "mode": "Keep Input Ratio"},
            "_meta": {"title": "IN:refs"},
        },
        "2": {
            "class_type": "LazySwitchKJ",
            "inputs": {"switch": False},
            "_meta": {"title": "IN:mode"},
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
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "default_model.safetensors", "weight_dtype": "default"},
            "_meta": {"title": "IN:model"},
        },
    }


def test_patch_character_sheet_prompt_sets_required_contract_fields():
    tpl = _char_sheet_template()
    patched = patch_character_sheet_prompt(
        tpl,
        ref_images=["a.png", "b.png", "c.png"],
        mode_select=True,
        seed=42,
        filename_prefix="fbtools/h3_character_sheets/abc123",
    )
    assert patched["1"]["inputs"]["image_paths"] == "a.png\nb.png\nc.png\na.png\nb.png\nc.png\na.png\nb.png\nc.png"
    assert patched["2"]["inputs"]["switch"] is True
    assert patched["3"]["inputs"]["noise_seed"] == 42
    assert patched["4"]["inputs"]["filename_prefix"] == "fbtools/h3_character_sheets/abc123"


def test_patch_character_sheet_prompt_pads_single_image_to_nine_slots():
    # The 9 ref_image_N sockets each read a fixed index (0-8) from the same shared batch this
    # loader produces — a batch smaller than 9 crashes ComfyUI execution with a raw IndexError
    # (confirmed live), so fewer than 9 supplied images must be padded, not sent as-is.
    tpl = _char_sheet_template()
    patched = patch_character_sheet_prompt(
        tpl, ref_images=["a.png"], mode_select=False, seed=1, filename_prefix="pfx",
    )
    assert patched["1"]["inputs"]["image_paths"] == "\n".join(["a.png"] * 9)


def test_patch_character_sheet_prompt_pads_and_truncates_non_divisor_count():
    tpl = _char_sheet_template()
    patched = patch_character_sheet_prompt(
        tpl, ref_images=["a.png", "b.png", "c.png", "d.png"], mode_select=False, seed=1, filename_prefix="pfx",
    )
    assert patched["1"]["inputs"]["image_paths"] == "\n".join(
        ["a.png", "b.png", "c.png", "d.png", "a.png", "b.png", "c.png", "d.png", "a.png"]
    )


def test_patch_character_sheet_prompt_nine_images_unchanged():
    tpl = _char_sheet_template()
    nine = [f"{i}.png" for i in range(9)]
    patched = patch_character_sheet_prompt(
        tpl, ref_images=nine, mode_select=False, seed=1, filename_prefix="pfx",
    )
    assert patched["1"]["inputs"]["image_paths"] == "\n".join(nine)


def test_patch_character_sheet_prompt_does_not_mutate_original_template():
    tpl = _char_sheet_template()
    patch_character_sheet_prompt(
        tpl, ref_images=["a.png"], mode_select=True, seed=1, filename_prefix="pfx",
    )
    assert tpl["1"]["inputs"]["image_paths"] == ""
    assert tpl["2"]["inputs"]["switch"] is False


def test_patch_character_sheet_prompt_rejects_empty_ref_images():
    with pytest.raises(ValueError, match="at least one image"):
        patch_character_sheet_prompt(
            _char_sheet_template(), ref_images=[], mode_select=False, seed=1, filename_prefix="pfx",
        )


def test_patch_character_sheet_prompt_rejects_more_than_nine_ref_images():
    with pytest.raises(ValueError, match="at most 9 images"):
        patch_character_sheet_prompt(
            _char_sheet_template(), ref_images=[f"{i}.png" for i in range(10)],
            mode_select=False, seed=1, filename_prefix="pfx",
        )


def test_patch_character_sheet_prompt_applies_override_for_title_present_in_template():
    tpl = _char_sheet_template()
    patched = patch_character_sheet_prompt(
        tpl, ref_images=["a.png"], mode_select=False, seed=1, filename_prefix="pfx",
        overrides={"IN:model": {"unet_name": "turbo_hybrid.safetensors"}},
    )
    assert patched["5"]["inputs"]["unet_name"] == "turbo_hybrid.safetensors"


def test_patch_character_sheet_prompt_silently_skips_override_for_title_absent_from_template():
    tpl = _char_sheet_template()
    del tpl["5"]  # no IN:model in this template
    patched = patch_character_sheet_prompt(
        tpl, ref_images=["a.png"], mode_select=False, seed=1, filename_prefix="pfx",
        overrides={"IN:model": {"unet_name": "turbo_hybrid.safetensors"}},
    )
    assert patched["1"]["inputs"]["image_paths"] == "\n".join(["a.png"] * 9)
    assert "5" not in patched


def test_patch_character_sheet_prompt_missing_required_title_raises():
    tpl = _char_sheet_template()
    del tpl["2"]  # drop IN:mode
    with pytest.raises(ValueError, match="IN:mode"):
        patch_character_sheet_prompt(
            tpl, ref_images=["a.png"], mode_select=False, seed=1, filename_prefix="pfx",
        )
