"""Tests for utils/image_text_overlay.py — pure logic, no ComfyUI dependencies."""
import numpy as np
import pytest

from conftest import import_test_module

ito = import_test_module("utils/image_text_overlay.py")
overlay_text = ito.overlay_text
hex_to_rgb255 = ito.hex_to_rgb255


def _solid_image(h=64, w=64, color=(20, 20, 20)):
    return np.tile(np.array(color, dtype=np.uint8), (h, w, 1))


# ── overlay_text ─────────────────────────────────────────────────────────────────

def test_output_shape_and_dtype_unchanged():
    image = _solid_image()
    result = overlay_text(image, "hello", position="bottom-left")
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_input_not_modified_in_place():
    image = _solid_image()
    original = image.copy()
    overlay_text(image, "hello")
    assert np.array_equal(image, original)


def test_text_actually_changes_pixels():
    image = _solid_image(color=(20, 20, 20))
    result = overlay_text(image, "A", position="bottom-left", text_color=(255, 255, 255))
    assert not np.array_equal(image, result)


def test_empty_text_still_draws_background_box_but_no_error():
    image = _solid_image()
    result = overlay_text(image, "", position="top-left")
    assert result.shape == image.shape


def test_multiline_text_covers_more_area_than_single_line():
    image = _solid_image()
    single = overlay_text(image, "A", position="top-left", bg_opacity=1.0)
    multi = overlay_text(image, "A\nB\nC", position="top-left", bg_opacity=1.0)
    single_changed = np.count_nonzero(np.any(single != image, axis=-1))
    multi_changed = np.count_nonzero(np.any(multi != image, axis=-1))
    assert multi_changed > single_changed


def test_bg_opacity_zero_leaves_background_color_unblended():
    # With bg_opacity 0, only the text glyphs themselves should change pixels, not the box.
    image = _solid_image(color=(20, 20, 20))
    result = overlay_text(image, "A", position="top-left", bg_opacity=0.0, text_color=(255, 255, 255))
    corner_far_from_text = result[-1, -1]
    assert tuple(corner_far_from_text) == (20, 20, 20)


@pytest.mark.parametrize("position", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_all_positions_accepted(position):
    image = _solid_image()
    result = overlay_text(image, "test", position=position)
    assert result.shape == image.shape


def test_invalid_position_raises():
    with pytest.raises(ValueError, match="position must be one of"):
        overlay_text(_solid_image(), "x", position="middle")


def test_invalid_image_ndim_raises():
    with pytest.raises(ValueError, match="must be"):
        overlay_text(np.zeros((64, 64)), "x")


def test_top_left_and_bottom_left_change_different_rows():
    image = _solid_image()
    top = overlay_text(image, "A", position="top-left", bg_opacity=1.0)
    bottom = overlay_text(image, "A", position="bottom-left", bg_opacity=1.0)
    top_row_changed = np.any(top[0] != image[0])
    bottom_row_changed = np.any(bottom[0] != image[0])
    assert top_row_changed
    assert not bottom_row_changed


# ── hex_to_rgb255 ────────────────────────────────────────────────────────────────

def test_hex_to_rgb255_with_hash():
    assert hex_to_rgb255("#FF00FF") == (255, 0, 255)


def test_hex_to_rgb255_without_hash():
    assert hex_to_rgb255("00FF00") == (0, 255, 0)


def test_hex_to_rgb255_wrong_length_raises():
    with pytest.raises(ValueError, match="6-digit hex"):
        hex_to_rgb255("#FFF")


def test_hex_to_rgb255_non_hex_raises():
    with pytest.raises(ValueError, match="6-digit hex"):
        hex_to_rgb255("#GGGGGG")
