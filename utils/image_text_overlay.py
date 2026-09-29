"""Pure logic for drawing a readable text block onto an image, behind a semi-opaque background box
so it stays legible regardless of what's underneath (dark footage, bright footage, a solid marker
color, etc).

No ComfyUI dependencies (see docs/GOTCHAS.md convention) — operates on a plain numpy uint8 [H,W,3]
array, not ComfyUI's torch IMAGE tensor; the node wrapper converts.

Built so a test/demo workflow can always produce one visible image output, even when the node under
test's meaningful results are scalars or strings (counts, indices, flags) with no image of their
own — feed those values in as `text` and this becomes the workflow's visible proof, instead of
pulling in a separate value-display node (and its dependency) just to show a run worked.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def hex_to_rgb255(hex_str: str) -> tuple[int, int, int]:
    """'#FF00FF' -> (255, 0, 255). Raises ValueError on anything not a 6-digit hex string."""
    stripped = (hex_str or "").strip().lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"color must be a 6-digit hex string like '#FFFFFF', got {hex_str!r}")
    try:
        return tuple(int(stripped[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"color must be a 6-digit hex string like '#FFFFFF', got {hex_str!r}")


def overlay_text(
    image: np.ndarray,
    text: str,
    position: str = "bottom-left",
    font_size: int = 16,
    padding: int = 6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_opacity: float = 0.6,
) -> np.ndarray:
    """Draw `text` (newline-separated lines) onto `image` [H,W,3] uint8. Returns a new array;
    `image` is not modified in place.
    """
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"image must be [H,W,3] or [H,W,4], got shape {image.shape}")

    lines = text.split("\n") if text else [""]
    base = Image.fromarray(image[..., :3].astype(np.uint8)).convert("RGB")

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    measure = ImageDraw.Draw(base)
    line_boxes = [measure.textbbox((0, 0), line, font=font) for line in lines]
    line_w = max(box[2] - box[0] for box in line_boxes)
    line_h = max(box[3] - box[1] for box in line_boxes)
    line_spacing = 4
    block_w = line_w + padding * 2
    block_h = (line_h + line_spacing) * len(lines) - line_spacing + padding * 2

    h, w = image.shape[:2]
    if position == "top-left":
        x0, y0 = 0, 0
    elif position == "top-right":
        x0, y0 = max(0, w - block_w), 0
    elif position == "bottom-left":
        x0, y0 = 0, max(0, h - block_h)
    else:  # bottom-right
        x0, y0 = max(0, w - block_w), max(0, h - block_h)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = int(max(0.0, min(1.0, bg_opacity)) * 255)
    ImageDraw.Draw(overlay).rectangle(
        [x0, y0, x0 + block_w, y0 + block_h], fill=(*bg_color, alpha),
    )
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(base)
    for i, line in enumerate(lines):
        ty = y0 + padding + i * (line_h + line_spacing)
        draw.text((x0 + padding, ty), line, fill=text_color, font=font)

    return np.array(base)
