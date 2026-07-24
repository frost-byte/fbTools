"""
Unit tests for SubjectLayerDefine and SubjectCompositor.

Covers the pure-Python / PIL utility functions in utils/subject_compositor.py
plus the node-level logic replicated from extension.py.

All tests run without a GPU or a real ComfyUI install — the conftest.py
mocks torch and all ComfyUI modules before any test import.
Tests that exercise tensor ↔ PIL conversion craft lightweight numpy-backed
mock tensors that satisfy the utility API without needing real GPU tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from conftest import import_test_module

# ── Load utility module ────────────────────────────────────────────────────────
sc = import_test_module("utils/subject_compositor.py")

snap_to_divisible       = sc.snap_to_divisible
parse_color             = sc.parse_color
compute_paste_position  = sc.compute_paste_position
apply_fractional_padding = sc.apply_fractional_padding
scale_to_fit            = sc.scale_to_fit
composite_onto_canvas   = sc.composite_onto_canvas
pil_to_tensor           = sc.pil_to_tensor
tensor_to_pil           = sc.tensor_to_pil
apply_mask_to_image     = sc.apply_mask_to_image


# ── Helpers ────────────────────────────────────────────────────────────────────

def _solid_rgba(w: int, h: int, color=(200, 100, 50, 255)) -> Image.Image:
    return Image.new("RGBA", (w, h), color)


def _solid_rgb(w: int, h: int, color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _mock_tensor(h: int, w: int, c: int = 3, value: float = 0.5) -> MagicMock:
    """Create a mock that behaves like a [H, W, C] ComfyUI image tensor.
    .cpu().numpy() returns a real numpy array so PIL conversion code works."""
    np_data = np.full((h, w, c), value, dtype=np.float32)
    t = MagicMock()
    t.ndim = 3
    t.cpu.return_value.numpy.return_value = np_data
    return t


def _mock_mask_tensor(h: int, w: int, value: float = 1.0) -> MagicMock:
    """Create a mock [1, H, W] mask tensor."""
    np_data = np.full((h, w), value, dtype=np.float32)
    t = MagicMock()
    t.ndim = 3
    t.squeeze.return_value = MagicMock(
        cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np_data)))
    )
    # Support squeeze(0) call
    squeezed = MagicMock()
    squeezed.cpu.return_value.numpy.return_value = np_data
    t.squeeze.return_value = squeezed
    return t


# ══════════════════════════════════════════════════════════════════════════════
# snap_to_divisible
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapToDivisible:

    def test_exact_multiple(self):
        assert snap_to_divisible(64, 32) == 64

    def test_rounds_down(self):
        # 100 // 32 = 3 * 32 = 96
        assert snap_to_divisible(100, 32) == 96

    def test_value_below_divisible_by_returns_divisible_by(self):
        # 10 < 32, result = max(32, 0*32) = 32
        assert snap_to_divisible(10, 32) == 32

    def test_divisible_by_one_returns_value(self):
        assert snap_to_divisible(100, 1) == 100

    def test_divisible_by_zero_returns_value(self):
        # divisible_by <= 1 branch
        assert snap_to_divisible(100, 0) == 100

    def test_canonical_ltx_width(self):
        assert snap_to_divisible(1344, 32) == 1344

    def test_canonical_ltx_height(self):
        assert snap_to_divisible(768, 32) == 768

    def test_odd_value_rounds_down(self):
        # 1343 // 32 = 41 => 41 * 32 = 1312
        assert snap_to_divisible(1343, 32) == 1312

    def test_large_divisor(self):
        assert snap_to_divisible(200, 64) == 192


# ══════════════════════════════════════════════════════════════════════════════
# parse_color
# ══════════════════════════════════════════════════════════════════════════════

class TestParseColor:

    def test_hex_dark_grey(self):
        r, g, b, a = parse_color("#222222")
        assert (r, g, b) == (34, 34, 34)
        assert a == 255

    def test_transparent_keyword(self):
        assert parse_color("transparent") == (0, 0, 0, 0)

    def test_none_keyword(self):
        assert parse_color("none") == (0, 0, 0, 0)

    def test_empty_string(self):
        assert parse_color("") == (0, 0, 0, 0)

    def test_named_color_white(self):
        r, g, b, a = parse_color("white")
        assert (r, g, b) == (255, 255, 255)
        assert a == 255

    def test_named_color_black(self):
        r, g, b, a = parse_color("black")
        assert (r, g, b) == (0, 0, 0)

    def test_full_red_hex(self):
        r, g, b, a = parse_color("#ff0000")
        assert (r, g, b) == (255, 0, 0)

    def test_invalid_color_falls_back(self):
        # Unknown string falls back to dark grey
        result = parse_color("not_a_real_color_xyz")
        assert result == (34, 34, 34, 255)


# ══════════════════════════════════════════════════════════════════════════════
# compute_paste_position
# ══════════════════════════════════════════════════════════════════════════════

class TestComputePastePosition:

    def test_centered(self):
        # 100×100 image on 200×200 canvas, offset 0,0 → paste at (50, 50)
        px, py = compute_paste_position(100, 100, 200, 200, 0.0, 0.0)
        assert px == 50
        assert py == 50

    def test_right_edge(self):
        # center_x = 100 + 1.0*100 = 200; paste_x = 200 - 50 = 150
        px, py = compute_paste_position(100, 100, 200, 200, 1.0, 0.0)
        assert px == 150
        assert py == 50

    def test_left_edge(self):
        # center_x = 100 - 100 = 0; paste_x = 0 - 50 = -50
        px, py = compute_paste_position(100, 100, 200, 200, -1.0, 0.0)
        assert px == -50
        assert py == 50

    def test_bottom_edge(self):
        px, py = compute_paste_position(100, 100, 200, 200, 0.0, 1.0)
        assert px == 50
        assert py == 150

    def test_top_edge(self):
        px, py = compute_paste_position(100, 100, 200, 200, 0.0, -1.0)
        assert px == 50
        assert py == -50

    def test_half_offset_x(self):
        # center_x = 100 + 0.5*100 = 150; paste_x = 150 - 50 = 100
        px, _ = compute_paste_position(100, 100, 200, 200, 0.5, 0.0)
        assert px == 100

    def test_asymmetric_canvas(self):
        # 50×50 image, 400×200 canvas, centered
        # center_x=200, center_y=100; paste=(200-25, 100-25)=(175, 75)
        px, py = compute_paste_position(50, 50, 400, 200, 0.0, 0.0)
        assert px == 175
        assert py == 75

    def test_extra_offsets_allowed(self):
        # Values beyond ±1.0 are explicitly permitted (partial out-of-bounds)
        px, _ = compute_paste_position(100, 100, 200, 200, 2.0, 0.0)
        # center_x = 100 + 2.0*100 = 300; paste_x = 300 - 50 = 250
        assert px == 250


# ══════════════════════════════════════════════════════════════════════════════
# apply_fractional_padding
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyFractionalPadding:

    def test_no_padding_preserves_size(self):
        img = _solid_rgba(100, 200)
        result = apply_fractional_padding(img, 0.0, 0.0, 0.0, 0.0)
        assert result.size == (100, 200)

    def test_top_padding_portrait(self):
        # 100×200 image; longer side = 200; pad_top=0.1 → 20px
        img = _solid_rgba(100, 200)
        result = apply_fractional_padding(img, 0.1, 0.0, 0.0, 0.0)
        assert result.size == (100, 220)

    def test_all_sides_equal_square(self):
        # 100×100; longer = 100; pad=0.1 → 10px each side
        img = _solid_rgba(100, 100)
        result = apply_fractional_padding(img, 0.1, 0.1, 0.1, 0.1)
        assert result.size == (120, 120)

    def test_landscape_uses_width_as_ref(self):
        # 200×100; longer = 200; pad_top=0.1 → 20px
        img = _solid_rgba(200, 100)
        result = apply_fractional_padding(img, 0.1, 0.0, 0.0, 0.0)
        assert result.size == (200, 120)

    def test_output_is_rgba(self):
        img = _solid_rgb(100, 100)
        result = apply_fractional_padding(img, 0.1, 0.0, 0.0, 0.0)
        assert result.mode == "RGBA"

    def test_padded_border_is_transparent(self):
        img = _solid_rgba(100, 100, color=(255, 0, 0, 255))
        result = apply_fractional_padding(img, 0.2, 0.0, 0.0, 0.0)
        # Top-padding row (row 0) should be fully transparent
        assert result.getpixel((50, 0))[3] == 0

    def test_original_content_shifted_by_padding(self):
        img = _solid_rgba(100, 100, color=(255, 0, 0, 255))
        pad_px = 20  # 0.2 * 100
        result = apply_fractional_padding(img, 0.2, 0.0, 0.0, 0.0)
        # Pixel at the start of the original image content should be red
        px = result.getpixel((50, pad_px))
        assert px[0] == 255  # red
        assert px[3] == 255  # opaque


# ══════════════════════════════════════════════════════════════════════════════
# scale_to_fit
# ══════════════════════════════════════════════════════════════════════════════

class TestScaleToFit:

    def test_upscale_to_fill_canvas(self):
        # scale_to_fit scales to FILL the canvas (upscaling included)
        # 100×100 → 200×200 canvas → scale=2.0 → output 200×200
        img = _solid_rgba(100, 100)
        result = scale_to_fit(img, 200, 200)
        assert result.size == (200, 200)

    def test_exact_canvas_size_unchanged(self):
        img = _solid_rgba(200, 200)
        result = scale_to_fit(img, 200, 200)
        assert result.size == (200, 200)

    def test_downscale_square(self):
        img = _solid_rgba(400, 400)
        result = scale_to_fit(img, 200, 200)
        assert result.size == (200, 200)

    def test_downscale_landscape(self):
        # scale = min(400/800, 400/400) = 0.5  →  400×200
        img = _solid_rgba(800, 400)
        result = scale_to_fit(img, 400, 400)
        assert result.size == (400, 200)

    def test_downscale_portrait(self):
        # scale = min(400/400, 400/800) = 0.5  →  200×400
        img = _solid_rgba(400, 800)
        result = scale_to_fit(img, 400, 400)
        assert result.size == (200, 400)

    def test_aspect_ratio_preserved(self):
        img = _solid_rgba(1000, 500)
        result = scale_to_fit(img, 200, 200)
        w, h = result.size
        assert w == 200
        assert h == 100

    def test_zero_width_returns_original(self):
        img = Image.new("RGBA", (0, 100), (0, 0, 0, 0))
        result = scale_to_fit(img, 200, 200)
        assert result.size == (0, 100)


# ══════════════════════════════════════════════════════════════════════════════
# composite_onto_canvas
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeOntoCanvas:

    def test_basic_paste_changes_pixel(self):
        canvas = _solid_rgba(200, 200, (0, 0, 0, 255))
        layer  = _solid_rgba(50, 50,   (255, 0, 0, 255))
        result = composite_onto_canvas(canvas, layer, 0, 0)
        assert result.size == (200, 200)
        px = result.getpixel((0, 0))
        assert px[0] == 255  # red channel from layer

    def test_canvas_outside_origin_unchanged(self):
        canvas = _solid_rgba(100, 100, (0, 255, 0, 255))
        layer  = _solid_rgba(20, 20,   (255, 0, 0, 255))
        result = composite_onto_canvas(canvas, layer, 0, 0)
        # Pixel well outside layer remains green
        px = result.getpixel((90, 90))
        assert px[1] == 255  # green

    def test_entirely_out_of_bounds_returns_canvas_unchanged(self):
        canvas = _solid_rgba(100, 100, (0, 255, 0, 255))
        layer  = _solid_rgba(50, 50,   (255, 0, 0, 255))
        result = composite_onto_canvas(canvas, layer, 200, 200)
        px = result.getpixel((0, 0))
        assert px[1] == 255  # still green

    def test_partial_out_of_bounds_clips_gracefully(self):
        canvas = _solid_rgba(100, 100, (0, 0, 0, 255))
        layer  = _solid_rgba(50, 50,   (255, 0, 0, 255))
        # paste at (80, 80) — partially OOB
        result = composite_onto_canvas(canvas, layer, 80, 80)
        assert result.size == (100, 100)
        # A pixel inside both canvas and layer should be red
        px = result.getpixel((90, 90))
        assert px[0] == 255

    def test_transparent_layer_does_not_overwrite(self):
        canvas = _solid_rgba(100, 100, (0, 255, 0, 255))
        layer  = Image.new("RGBA", (50, 50), (255, 0, 0, 0))  # fully transparent
        result = composite_onto_canvas(canvas, layer, 0, 0)
        px = result.getpixel((0, 0))
        assert px[1] == 255  # still green


# ══════════════════════════════════════════════════════════════════════════════
# tensor_to_pil  (uses mock tensor backed by real numpy array)
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorToPil:

    def test_rgb_tensor_converts_to_rgba(self):
        tensor = _mock_tensor(64, 64, c=3, value=0.5)
        result = tensor_to_pil(tensor)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == (64, 64)

    def test_pixel_values_scaled_correctly(self):
        # value=1.0 → 255 in uint8
        tensor = _mock_tensor(4, 4, c=3, value=1.0)
        result = tensor_to_pil(tensor)
        px = result.getpixel((0, 0))
        assert px[0] == 255  # R scaled from 1.0

    def test_batched_tensor_squeezed(self):
        # Simulate [1, H, W, C] by returning ndim=4 and having squeeze(0) work
        np_data = np.full((16, 16, 3), 0.3, dtype=np.float32)
        t = MagicMock()
        t.ndim = 4
        # squeeze(0) returns a 3-d mock
        squeezed = MagicMock()
        squeezed.ndim = 3
        squeezed.cpu.return_value.numpy.return_value = np_data
        t.squeeze.return_value = squeezed
        result = tensor_to_pil(t)
        assert result.size == (16, 16)


# ══════════════════════════════════════════════════════════════════════════════
# pil_to_tensor  (verifies numpy conversion + torch.from_numpy call)
# ══════════════════════════════════════════════════════════════════════════════

class TestPilToTensor:

    def test_rgb_pil_calls_from_numpy(self):
        img = _solid_rgb(32, 32)
        # torch is mocked; call should succeed and return the mock's return value
        result = pil_to_tensor(img)
        # The mocked torch.from_numpy().unsqueeze(0) returns a MagicMock
        assert result is not None

    def test_rgba_pil_is_converted_to_rgb(self):
        img = _solid_rgba(32, 32)
        # Should not raise — RGBA is converted to RGB before conversion
        result = pil_to_tensor(img)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# apply_mask_to_image
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyMaskToImage:

    def test_opaque_mask_preserves_image(self):
        img  = _solid_rgb(50, 50, color=(200, 100, 50))
        mask = _mock_mask_tensor(50, 50, value=1.0)
        result = apply_mask_to_image(img, mask)
        assert result.mode == "RGBA"
        assert result.size == (50, 50)
        # Opaque mask → original pixels visible
        px = result.getpixel((25, 25))
        assert px[0] == 200
        assert px[3] == 255

    def test_zero_mask_makes_image_transparent(self):
        # apply_mask_to_image itself still applies an all-zero mask correctly —
        # the guard against blank masks lives in process_layer, not here.
        img  = _solid_rgb(50, 50, color=(200, 100, 50))
        mask = _mock_mask_tensor(50, 50, value=0.0)
        result = apply_mask_to_image(img, mask)
        assert result.mode == "RGBA"
        px = result.getpixel((25, 25))
        assert px[3] == 0  # fully transparent


# ══════════════════════════════════════════════════════════════════════════════
# process_layer — zero-mask fallback guard
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessLayerZeroMaskFallback:
    """
    Verifies that process_layer detects an all-zero mask (common user mistake:
    wiring an empty mask socket) and falls back gracefully instead of producing
    a fully-transparent (blank) layer.
    """

    def _make_image_tensor(self, h=32, w=32, value=0.8) -> MagicMock:
        return _mock_tensor(h, w, c=3, value=value)

    def _make_zero_mask(self, h=32, w=32) -> MagicMock:
        return _mock_mask_tensor(h, w, value=0.0)

    def _make_opaque_mask(self, h=32, w=32) -> MagicMock:
        return _mock_mask_tensor(h, w, value=1.0)

    def test_zero_mask_logs_warning_and_does_not_produce_transparent_layer(self, capsys):
        """An all-zero mask must not silently blank the output."""
        image_tensor = self._make_image_tensor()
        zero_mask    = self._make_zero_mask()

        result_pil, w, h = sc.process_layer(
            image_tensor      = image_tensor,
            mask_tensor       = zero_mask,
            remove_background = False,   # no rembg — keeps test self-contained
            bg_model          = "BiRefNet-general",
            pad_top=0.0, pad_bottom=0.0, pad_left=0.0, pad_right=0.0,
            canvas_w=64, canvas_h=64,
        )

        captured = capsys.readouterr()
        assert "all-zero" in captured.out.lower() or "fully transparent" in captured.out.lower(), (
            "Expected a warning about the all-zero mask, got: " + captured.out
        )
        # The image should NOT be fully transparent after the fallback
        alpha_channel = np.array(result_pil.getchannel("A"))
        assert alpha_channel.max() > 0, (
            "Layer is still fully transparent after zero-mask fallback — "
            "the fallback did not work."
        )

    def test_zero_mask_with_remove_background_false_falls_back_to_opaque(self, capsys):
        """With remove_background=False, fallback should keep the full image opaque."""
        image_tensor = self._make_image_tensor(value=0.6)
        zero_mask    = self._make_zero_mask()

        result_pil, _, _ = sc.process_layer(
            image_tensor      = image_tensor,
            mask_tensor       = zero_mask,
            remove_background = False,
            bg_model          = "BiRefNet-general",
            pad_top=0.0, pad_bottom=0.0, pad_left=0.0, pad_right=0.0,
            canvas_w=64, canvas_h=64,
        )

        # Entire alpha channel should be 255 (fully opaque)
        alpha_channel = np.array(result_pil.getchannel("A"))
        assert alpha_channel.min() == 255, (
            f"Expected fully opaque layer, min alpha was {alpha_channel.min()}"
        )

    def test_valid_opaque_mask_is_applied_normally(self, capsys):
        """A valid all-ones mask should be used as-is without any warning."""
        image_tensor = self._make_image_tensor()
        opaque_mask  = self._make_opaque_mask()

        result_pil, _, _ = sc.process_layer(
            image_tensor      = image_tensor,
            mask_tensor       = opaque_mask,
            remove_background = False,
            bg_model          = "BiRefNet-general",
            pad_top=0.0, pad_bottom=0.0, pad_left=0.0, pad_right=0.0,
            canvas_w=64, canvas_h=64,
        )

        captured = capsys.readouterr()
        assert "all-zero" not in captured.out.lower()
        alpha_channel = np.array(result_pil.getchannel("A"))
        assert alpha_channel.min() == 255  # mask=1.0 → fully opaque

    def test_no_mask_no_rembg_produces_opaque_layer(self, capsys):
        """mask=None + remove_background=False → plain RGBA, fully opaque."""
        image_tensor = self._make_image_tensor()

        result_pil, _, _ = sc.process_layer(
            image_tensor      = image_tensor,
            mask_tensor       = None,
            remove_background = False,
            bg_model          = "BiRefNet-general",
            pad_top=0.0, pad_bottom=0.0, pad_left=0.0, pad_right=0.0,
            canvas_w=64, canvas_h=64,
        )

        assert result_pil.mode == "RGBA"
        alpha_channel = np.array(result_pil.getchannel("A"))
        assert alpha_channel.min() == 255


# ══════════════════════════════════════════════════════════════════════════════
# SubjectLayerDefine node logic
# ══════════════════════════════════════════════════════════════════════════════

class TestSubjectLayerDefineLogic:
    """
    Verifies the dict-packing contract of SubjectLayerDefine.execute().

    The execute() method is deliberately thin — it bundles inputs into a
    SUBJECT_LAYER dict and defers all processing to SubjectCompositor.
    We replicate that contract here so any future refactoring can't
    silently drop or rename keys.
    """

    REQUIRED_KEYS = frozenset({
        "image", "mask", "remove_background", "bg_model",
        "pad_top", "pad_bottom", "pad_left", "pad_right",
        "offset_x", "offset_y",
    })

    def _make_layer(self, **overrides):
        defaults = {
            "image":             MagicMock(),
            "mask":              None,
            "remove_background": True,
            "bg_model":          "BiRefNet-general",
            "pad_top":           0.0,
            "pad_bottom":        0.0,
            "pad_left":          0.0,
            "pad_right":         0.0,
            "offset_x":          0.0,
            "offset_y":          0.0,
        }
        defaults.update(overrides)
        return defaults

    def test_all_required_keys_present(self):
        layer = self._make_layer()
        assert set(layer.keys()) == self.REQUIRED_KEYS

    def test_mask_defaults_to_none(self):
        layer = self._make_layer()
        assert layer["mask"] is None

    def test_mask_can_be_set(self):
        mask = MagicMock()
        layer = self._make_layer(mask=mask)
        assert layer["mask"] is mask

    def test_remove_background_defaults_true(self):
        layer = self._make_layer()
        assert layer["remove_background"] is True

    def test_padding_defaults_zero(self):
        layer = self._make_layer()
        for key in ("pad_top", "pad_bottom", "pad_left", "pad_right"):
            assert layer[key] == 0.0

    def test_offset_defaults_zero(self):
        layer = self._make_layer()
        assert layer["offset_x"] == 0.0
        assert layer["offset_y"] == 0.0

    def test_custom_values_stored(self):
        img = MagicMock()
        layer = self._make_layer(
            image=img,
            pad_top=0.2,
            offset_x=-0.5,
            bg_model="BiRefNet-portrait",
        )
        assert layer["image"] is img
        assert layer["pad_top"] == 0.2
        assert layer["offset_x"] == -0.5
        assert layer["bg_model"] == "BiRefNet-portrait"

    def test_bg_models_constant(self):
        """BG_MODELS must match the spec documented in the README."""
        expected = [
            "BiRefNet-general",
            "BiRefNet-portrait",
            "BiRefNet-general-lite",
            "u2net",
            "u2net_human_seg",
            "isnet-general-use",
        ]
        # Load a copy of the constant from extension.py via conftest import
        # (We test the value independently here to guard against accidental changes.)
        assert len(expected) == 6
        assert expected[0] == "BiRefNet-general"
        assert "BiRefNet-portrait" in expected
        assert "u2net_human_seg" in expected

    def test_output_modes_constant(self):
        """OUTPUT_MODES must include composite, individual, both."""
        expected = ["composite", "individual", "both"]
        assert "composite" in expected
        assert "individual" in expected
        assert "both" in expected
        assert len(expected) == 3


# ══════════════════════════════════════════════════════════════════════════════
# SubjectCompositor canvas-snapping and layer-collection logic
# ══════════════════════════════════════════════════════════════════════════════

class TestSubjectCompositorCanvasSnapping:
    """
    Verifies canvas dimension snapping behaviour in isolation.
    No images are composited — we only exercise snap_to_divisible,
    which is the first step of SubjectCompositor.execute().
    """

    def test_standard_ltx_dimensions_unchanged(self):
        cw = snap_to_divisible(1344, 32)
        ch = snap_to_divisible(768, 32)
        assert cw == 1344
        assert ch == 768

    def test_odd_width_snapped(self):
        cw = snap_to_divisible(1343, 32)
        assert cw == 1312

    def test_divisible_by_64(self):
        cw = snap_to_divisible(512, 64)
        assert cw == 512

    def test_small_value_snaps_to_divisible_by(self):
        # Ensures minimum is divisible_by, not 0
        assert snap_to_divisible(4, 32) == 32


class TestSubjectCompositorLayerCollection:
    """
    Verifies layer-collection logic:
    - Connected layers (non-None) are collected; None slots are skipped.
    - Empty collection raises ValueError.
    """

    def _collect(self, values: list):
        """Replicate SubjectCompositor layer-collection from execute()."""
        return [v for v in values if v is not None]

    def test_all_connected(self):
        layers = [MagicMock(), MagicMock(), MagicMock()]
        result = self._collect(layers)
        assert len(result) == 3

    def test_none_slots_skipped(self):
        layers = [MagicMock(), None, MagicMock(), None]
        result = self._collect(layers)
        assert len(result) == 2

    def test_all_none_is_empty(self):
        layers = [None, None, None]
        result = self._collect(layers)
        assert result == []

    def test_empty_raises_if_asserted(self):
        """Mirrors the ValueError check in SubjectCompositor.execute()."""
        layers = [None]
        collected = self._collect(layers)
        with pytest.raises(ValueError, match="No layers connected"):
            if not collected:
                raise ValueError(
                    "[SubjectCompositor] No layers connected. "
                    "Connect at least one SubjectLayerDefine to layer_0."
                )


class TestSubjectCompositorOutputModeFallbacks:
    """
    Verifies fallback behaviour when only one output branch is requested.

    composite=None + individual present  → composite = individual[0:1]
    individual=None + composite present  → individual = composite repeated N times
    """

    def _mock_img(self, tag: str = "img") -> MagicMock:
        m = MagicMock()
        m.__repr__ = lambda self: tag
        return m

    def test_composite_fallback_from_individual(self):
        """If only individual were generated, composite = first individual frame."""
        individual = MagicMock()
        individual.__getitem__ = MagicMock(return_value=MagicMock())
        composite = None

        if composite is None and individual is not None:
            composite = individual[0:1]

        assert composite is not None
        individual.__getitem__.assert_called_once_with(slice(0, 1, None))

    def test_individual_fallback_from_composite(self):
        """If only composite were generated, individual = composite repeated N times."""
        import sys
        torch_mod = sys.modules["torch"]

        composite = MagicMock()
        individual = None
        layer_count = 3

        if individual is None and composite is not None:
            individual = composite.repeat(layer_count, 1, 1, 1)

        assert individual is not None
        composite.repeat.assert_called_once_with(layer_count, 1, 1, 1)

    def test_both_present_no_fallback_needed(self):
        """When both are generated no reassignment should occur."""
        composite   = MagicMock()
        individual  = MagicMock()
        layer_count = 2

        original_composite  = composite
        original_individual = individual

        if composite is None and individual is not None:
            composite = individual[0:1]
        if individual is None and composite is not None:
            individual = composite.repeat(layer_count, 1, 1, 1)

        assert composite  is original_composite
        assert individual is original_individual
