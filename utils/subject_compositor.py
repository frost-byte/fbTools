"""
subject_compositor_utils.py
Core image processing utilities for SubjectLayerDefine and SubjectCompositor nodes.
Handles background removal, fractional padding, alpha compositing, and canvas placement.
"""

from __future__ import annotations

import math
from typing import Optional, Any

import numpy as np
import torch
from PIL import Image, ImageFilter

# Session cache — keyed by model name so each model loads once per process lifetime.
_rembg_sessions: dict[str, Any] = {}


# ── Tensor ↔ PIL helpers ──────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI IMAGE tensor [H, W, C] float32 0-1 to a PIL RGBA image.
    Handles both RGB (3-channel) and RGBA (4-channel) tensors.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # remove batch dim if present

    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    if np_img.shape[2] == 3:
        pil = Image.fromarray(np_img, mode="RGB").convert("RGBA")
    elif np_img.shape[2] == 4:
        pil = Image.fromarray(np_img, mode="RGBA")
    else:
        raise ValueError(f"Unexpected channel count: {np_img.shape[2]}")

    return pil


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image (RGB or RGBA) to a ComfyUI IMAGE tensor [1, H, W, 3] float32 0-1.
    Alpha channel is discarded — caller should composite onto background first.
    """
    rgb = pil.convert("RGB")
    np_img = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W, 3]


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert a 2D uint8 alpha mask [H, W] to a ComfyUI MASK tensor [1, H, W] float32 0-1."""
    return torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)


# ── Canvas color helper ───────────────────────────────────────────────────────

def parse_color(color_str: str) -> tuple[int, int, int, int]:
    """
    Parse a color string to RGBA tuple.
    Accepts: "#RRGGBB", "#RRGGBBAA", "rgb(r,g,b)", named colors, or "transparent".
    Returns (R, G, B, A) with A=255 for opaque, A=0 for transparent.
    """
    s = color_str.strip().lower()

    if s in ("transparent", "none", ""):
        return (0, 0, 0, 0)

    try:
        # Let PIL handle named colors and hex
        dummy = Image.new("RGBA", (1, 1), s)
        return dummy.getpixel((0, 0))
    except (ValueError, AttributeError):
        pass

    # Fallback to dark grey matching typical ComfyUI canvas background
    return (34, 34, 34, 255)


# ── Background removal ────────────────────────────────────────────────────────

def remove_background_birefnet(
    pil_img: Image.Image,
    model_name: str = "BiRefNet-general",
) -> Image.Image:
    """
    Remove background using BiRefNet via the rembg library.
    Returns a PIL RGBA image with background made transparent.

    Falls back gracefully if rembg is not installed.
    Session is cached globally so the model is only loaded once per process.
    """
    global _rembg_sessions

    # Map display model names to rembg session names
    model_map = {
        "BiRefNet-general":      "birefnet-general",
        "BiRefNet-portrait":     "birefnet-portrait",
        "BiRefNet-general-lite": "birefnet-general-lite",
        "u2net":                 "u2net",
        "u2net_human_seg":       "u2net_human_seg",
        "isnet-general-use":     "isnet-general-use",
    }
    session_name = model_map.get(model_name, "birefnet-general")

    try:
        from rembg import remove, new_session

        if session_name not in _rembg_sessions:
            print(f"[SubjectCompositor] Loading rembg session (CPU): {session_name}", flush=True)
            # Force CPU-only ONNX execution — rembg's GPU (CUDA) provider conflicts with
            # PyTorch's live CUDA context and causes a C-level segfault inside ort.InferenceSession.
            _rembg_sessions[session_name] = new_session(
                session_name,
                providers=["CPUExecutionProvider"],
            )

        session = _rembg_sessions[session_name]
        print(f"[SubjectCompositor] Running background removal ({session_name})", flush=True)
        result = remove(pil_img.convert("RGBA"), session=session)
        return result

    except ImportError:
        print(
            "[SubjectCompositor] Warning: rembg not installed. "
            "Background removal skipped. Install with: pip install rembg",
            flush=True,
        )
        return pil_img.convert("RGBA")

    except Exception as e:
        print(f"[SubjectCompositor] Background removal failed: {e}. Skipping.", flush=True)
        return pil_img.convert("RGBA")


def apply_mask_to_image(
    pil_img: Image.Image,
    mask_tensor: torch.Tensor,
) -> Image.Image:
    """
    Apply an external ComfyUI MASK tensor as the alpha channel of a PIL image.
    mask_tensor: [1, H, W] or [H, W] float32 0-1 (1=keep, 0=transparent)
    Returns RGBA PIL image.
    """
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)

    mask_np = (mask_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")

    # Resize mask to match image if needed
    if mask_pil.size != pil_img.size:
        mask_pil = mask_pil.resize(pil_img.size, Image.LANCZOS)

    rgba = pil_img.convert("RGBA")
    r, g, b, _ = rgba.split()
    return Image.merge("RGBA", (r, g, b, mask_pil))


# ── Fractional padding ────────────────────────────────────────────────────────

def apply_fractional_padding(
    pil_img: Image.Image,
    pad_top: float,
    pad_bottom: float,
    pad_left: float,
    pad_right: float,
) -> Image.Image:
    """
    Add transparent padding as a fraction of the image's longer dimension.
    pad_top=0.1 adds 10% of max(width, height) as transparent space at the top.
    Input and output are RGBA PIL images.
    """
    img = pil_img.convert("RGBA")
    w, h = img.size
    ref = max(w, h)

    top_px    = int(round(pad_top    * ref))
    bottom_px = int(round(pad_bottom * ref))
    left_px   = int(round(pad_left   * ref))
    right_px  = int(round(pad_right  * ref))

    new_w = w + left_px + right_px
    new_h = h + top_px  + bottom_px

    padded = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    padded.paste(img, (left_px, top_px))
    return padded


# ── Resize to fit canvas ──────────────────────────────────────────────────────

def scale_to_fit(
    pil_img: Image.Image,
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """
    Scale a PIL image to fit within canvas_w x canvas_h while preserving aspect ratio.
    Never crops. Uses LANCZOS for downscaling, LANCZOS for upscaling too (quality).
    """
    img_w, img_h = pil_img.size
    if img_w == 0 or img_h == 0:
        return pil_img

    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    return pil_img.resize((new_w, new_h), Image.LANCZOS)


# ── Snap to divisible_by ──────────────────────────────────────────────────────

def snap_to_divisible(value: int, divisible_by: int) -> int:
    """Round value down to nearest multiple of divisible_by. Minimum 1."""
    if divisible_by <= 1:
        return max(1, value)
    return max(divisible_by, (value // divisible_by) * divisible_by)


# ── Canvas placement ──────────────────────────────────────────────────────────

def compute_paste_position(
    img_w: int,
    img_h: int,
    canvas_w: int,
    canvas_h: int,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int]:
    """
    Compute top-left paste position for an image on a canvas.

    offset_x / offset_y are fractions of half the canvas:
      0.0, 0.0  → image centered on canvas
      1.0, 0.0  → image centered on right edge
     -1.0, 0.0  → image centered on left edge
      0.0, 1.0  → image centered on bottom edge
      0.0,-1.0  → image centered on top edge

    Returns (paste_x, paste_y) — top-left corner of where to paste.
    Allows partial out-of-bounds (PIL handles clipping automatically).
    """
    center_x = canvas_w / 2.0 + offset_x * (canvas_w / 2.0)
    center_y = canvas_h / 2.0 + offset_y * (canvas_h / 2.0)

    paste_x = int(round(center_x - img_w / 2.0))
    paste_y = int(round(center_y - img_h / 2.0))

    return paste_x, paste_y


# ── Alpha composite onto canvas ───────────────────────────────────────────────

def composite_onto_canvas(
    canvas: Image.Image,
    layer_img: Image.Image,
    paste_x: int,
    paste_y: int,
) -> Image.Image:
    """
    Alpha-composite an RGBA layer onto an RGBA canvas at the given position.
    Handles partial out-of-bounds gracefully by clipping to canvas bounds.
    Returns the updated RGBA canvas.
    """
    canvas = canvas.convert("RGBA")
    layer  = layer_img.convert("RGBA")

    lw, lh = layer.size
    cw, ch = canvas.size

    # Clip region to canvas bounds
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(lw, cw - paste_x)
    src_y2 = min(lh, ch - paste_y)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return canvas  # layer is entirely out of bounds

    dst_x = paste_x + src_x1
    dst_y = paste_y + src_y1

    cropped = layer.crop((src_x1, src_y1, src_x2, src_y2))

    # Create a temp canvas of same size for alpha_composite
    temp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp.paste(cropped, (dst_x, dst_y))

    return Image.alpha_composite(canvas, temp)


# ── Full layer processing pipeline ───────────────────────────────────────────

def process_layer(
    image_tensor: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    remove_background: bool,
    bg_model: str,
    pad_top: float,
    pad_bottom: float,
    pad_left: float,
    pad_right: float,
    canvas_w: int,
    canvas_h: int,
) -> tuple[Image.Image, int, int]:
    """
    Full processing pipeline for a single layer.
    Returns: (processed_rgba_pil, scaled_width, scaled_height)

    Steps:
      1. Tensor → PIL RGBA
      2. Background removal (if requested, or mask supplied)
      3. Apply fractional padding (transparent)
      4. Scale to fit canvas
    """
    # 1. Convert to PIL RGBA
    print("[SubjectCompositor] Step 1: tensor → PIL", flush=True)
    pil = tensor_to_pil(image_tensor)
    print(f"[SubjectCompositor]   image size: {pil.size}", flush=True)

    # 2. Background handling
    print(
        f"[SubjectCompositor] Step 2: background (mask={'connected' if mask_tensor is not None else 'none'}, "
        f"remove_bg={remove_background}, model={bg_model})",
        flush=True,
    )
    if mask_tensor is not None:
        # Validate the mask before using it — warn if all-zero (fully transparent).
        # An all-black mask is almost always a mistake (e.g. a disconnected or empty
        # mask socket was wired in). Fall back to remove_background so the layer
        # still renders rather than silently producing a blank image.
        mask_data = mask_tensor
        if mask_data.ndim == 3:
            mask_data = mask_data.squeeze(0)
        mask_np = mask_data.cpu().numpy()
        mask_max = float(mask_np.max())

        if mask_max < 1e-4:
            print(
                "[SubjectCompositor] Warning: connected mask is all-zero (fully transparent). "
                "The mask will be ignored. "
                "Either disconnect the mask input or supply a valid mask. "
                f"Falling back to remove_background={remove_background}.",
                flush=True,
            )
            if remove_background:
                pil = remove_background_birefnet(pil, bg_model)
            else:
                pil = pil.convert("RGBA")
        else:
            pil = apply_mask_to_image(pil, mask_tensor)
    elif remove_background:
        pil = remove_background_birefnet(pil, bg_model)
    else:
        pil = pil.convert("RGBA")

    print("[SubjectCompositor] Step 3: fractional padding", flush=True)
    pil = apply_fractional_padding(pil, pad_top, pad_bottom, pad_left, pad_right)

    print(f"[SubjectCompositor] Step 4: scale to fit {canvas_w}×{canvas_h}", flush=True)
    pil = scale_to_fit(pil, canvas_w, canvas_h)

    print(f"[SubjectCompositor] process_layer done → {pil.size}", flush=True)
    return pil, pil.width, pil.height
