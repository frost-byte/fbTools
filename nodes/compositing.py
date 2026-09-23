"""Subject-layer compositing nodes, extracted from extension.py (Plan 24).

SubjectLayerDefine + SubjectCompositor + their shared SUBJECT_LAYER custom wire type. Unrelated
to the composition-engine's own Subject Profile system (different meaning of "Subject") or to
nodes/narrative/ — pure image compositing, no cross-domain coupling.
"""
from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from comfy_api.latest import io

from .shared import prefixed_node_id
from ..utils.subject_compositor import (
    tensor_to_pil,
    pil_to_tensor,
    parse_color,
    process_layer,
    compute_paste_position,
    composite_onto_canvas,
    snap_to_divisible,
)


# ── Custom type: SUBJECT_LAYER ────────────────────────────────────────────────

SUBJECT_LAYER_TYPE = "SUBJECT_LAYER"


@io.comfytype(io_type=SUBJECT_LAYER_TYPE)
class SubjectLayer:
    """
    Custom type passed between SubjectLayerDefine and SubjectCompositor.
    Carries the raw image tensor plus all per-layer parameters.
    Processing (bg removal, padding, scaling) is deferred to the compositor
    so it has access to the final canvas dimensions.
    """
    Type = dict  # { image, mask, remove_background, bg_model,
                 #   pad_top, pad_bottom, pad_left, pad_right,
                 #   offset_x, offset_y }

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "layer", **kwargs):
            super().__init__(name, **kwargs)


# ── Node 1: SubjectLayerDefine ────────────────────────────────────────────────

BG_MODELS = [
    "BiRefNet-general",
    "BiRefNet-portrait",
    "BiRefNet-general-lite",
    "u2net",
    "u2net_human_seg",
    "isnet-general-use",
]

OUTPUT_MODES = ["composite", "individual", "both"]
class SubjectLayerDefine(io.ComfyNode):
    """
    Define a single subject layer for use with SubjectCompositor.

    Padding is specified as a fraction of the image's longer dimension.
    pad_top=0.2 adds 20% of max(width, height) as transparent space at the top,
    effectively making the subject appear smaller relative to other layers.

    Offset positions the subject's center relative to the canvas center.
    offset_x=0.0, offset_y=0.0 places the subject at the canvas center.
    offset_x=0.5 shifts the subject halfway toward the right edge.
    offset_x=-0.5 shifts the subject halfway toward the left edge.
    offset_y=-0.5 shifts the subject halfway toward the top edge.

    An optional mask input (ComfyUI MASK) can be supplied instead of using
    automatic background removal — useful when an upstream RMBG node is
    already in the workflow.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("SubjectLayerDefine"),
            display_name="Subject Layer Define",
            category="🧊 frost-byte/compositing",
            description=(
                "Define a subject layer for SubjectCompositor. "
                "Specify padding (as fraction of longer dimension), "
                "canvas offset (fraction of half-canvas from center), "
                "and optional background removal."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="Image",
                    tooltip="Input image containing the subject.",
                ),
                io.Mask.Input(
                    "mask",
                    display_name="Mask",
                    optional=True,
                    tooltip=(
                        "Optional pre-computed alpha mask (ComfyUI MASK, 1=keep). "
                        "If provided, overrides background removal."
                    ),
                ),
                io.Float.Input(
                    "pad_top",
                    display_name="Pad Top",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added above the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_bottom",
                    display_name="Pad Bottom",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added below the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_left",
                    display_name="Pad Left",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added to the left of the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_right",
                    display_name="Pad Right",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added to the right of the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "offset_x",
                    display_name="Offset X",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Horizontal offset of subject center relative to canvas center. "
                        "0.0=center, 1.0=right edge, -1.0=left edge."
                    ),
                ),
                io.Float.Input(
                    "offset_y",
                    display_name="Offset Y",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Vertical offset of subject center relative to canvas center. "
                        "0.0=center, 1.0=bottom edge, -1.0=top edge."
                    ),
                ),
                io.Boolean.Input(
                    "remove_background",
                    display_name="Remove Background",
                    default=True,
                    tooltip="Automatically remove the background. Ignored if a mask is connected.",
                ),
                io.Combo.Input(
                    "bg_model",
                    display_name="BG Removal Model",
                    options=BG_MODELS,
                    default="BiRefNet-general",
                    optional=True,
                    tooltip="Background removal model to use. BiRefNet-portrait works best for people.",
                ),
            ],
            outputs=[
                SubjectLayer.Output(
                    "layer",
                    display_name="Layer",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        pad_top: float = 0.0,
        pad_bottom: float = 0.0,
        pad_left: float = 0.0,
        pad_right: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        remove_background: bool = True,
        bg_model: str = "BiRefNet-general",
        mask: Optional[torch.Tensor] = None,
    ) -> io.NodeOutput:

        layer = {
            "image":             image,
            "mask":              mask,
            "remove_background": remove_background,
            "bg_model":          bg_model,
            "pad_top":           pad_top,
            "pad_bottom":        pad_bottom,
            "pad_left":          pad_left,
            "pad_right":         pad_right,
            "offset_x":          offset_x,
            "offset_y":          offset_y,
        }

        return io.NodeOutput(layer)

# ── Node 2: SubjectCompositor ─────────────────────────────────────────────────

class SubjectCompositor(io.ComfyNode):
    """
    Composite multiple subject layers onto a canvas.

    Accepts 1–20 SUBJECT_LAYER inputs (from SubjectLayerDefine).
    Layers are processed in order — layer_0 is placed first (furthest back),
    later layers are composited on top.

    Output modes:
      composite   — one image with all layers composited together
      individual  — one image per layer, each placed at its offset on its own canvas
      both        — returns both composite and individual images

    The individual images output is a batch tensor [N, H, W, 3] where N is
    the number of connected layers. Each image in the batch corresponds to
    the layer at the same index, and can be fed directly into ReferenceLatent
    or other conditioning nodes.

    The canvas dimensions are snapped to the nearest lower multiple of
    divisible_by (default 32, required by most video/image models).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:

        autogrow_template = io.Autogrow.TemplatePrefix(
            input=SubjectLayer.Input("layer", optional=True),
            prefix="layer",
            min=1,
            max=20,
        )

        return io.Schema(
            node_id=prefixed_node_id("SubjectCompositor"),
            display_name="Subject Compositor",
            category="🧊 frost-byte/compositing",
            description=(
                "Composite multiple subject layers onto a canvas. "
                "Outputs a composite image and/or individual per-subject images "
                "at the target resolution, suitable for ReferenceLatent or "
                "other multi-image conditioning nodes."
            ),
            inputs=[
                io.Int.Input(
                    "canvas_width",
                    display_name="Canvas Width",
                    default=1344,
                    min=64,
                    max=8192,
                    step=1,
                    tooltip="Output canvas width in pixels. Snapped to divisible_by.",
                ),
                io.Int.Input(
                    "canvas_height",
                    display_name="Canvas Height",
                    default=768,
                    min=64,
                    max=8192,
                    step=1,
                    tooltip="Output canvas height in pixels. Snapped to divisible_by.",
                ),
                io.String.Input(
                    "canvas_color",
                    display_name="Canvas Color",
                    default="#222222",
                    multiline=False,
                    tooltip=(
                        "Background color of the canvas. "
                        "Accepts hex (#RRGGBB), named colors, or 'transparent'."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    display_name="Output Mode",
                    options=OUTPUT_MODES,
                    default="both",
                    tooltip=(
                        "composite: one merged image. "
                        "individual: one image per layer. "
                        "both: composite and individual batch."
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    display_name="Divisible By",
                    default=32,
                    min=1,
                    max=256,
                    step=1,
                    tooltip=(
                        "Snap canvas dimensions to nearest lower multiple of this value. "
                        "Use 32 for LTX/most video models, 64 for some diffusion models, "
                        "1 to disable snapping."
                    ),
                ),
                io.Autogrow.Input("layers", template=autogrow_template),
            ],
            outputs=[
                io.Image.Output(
                    "composite",
                    display_name="Composite Image",
                ),
                io.Image.Output(
                    "individual_images",
                    display_name="Individual Images",
                ),
                io.Int.Output(
                    "layer_count",
                    display_name="Layer Count",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        canvas_width: int,
        canvas_height: int,
        canvas_color: str,
        output_mode: str,
        divisible_by: int,
        layers: io.Autogrow.Type,
    ) -> io.NodeOutput:

        # ── 1. Snap canvas dimensions ─────────────────────────────────────────
        cw = snap_to_divisible(canvas_width,  divisible_by)
        ch = snap_to_divisible(canvas_height, divisible_by)

        if cw != canvas_width or ch != canvas_height:
            print(
                f"[SubjectCompositor] Canvas snapped from "
                f"{canvas_width}×{canvas_height} to {cw}×{ch} "
                f"(divisible_by={divisible_by})",
                flush=True,
            )

        # ── 2. Collect connected layers ───────────────────────────────────────
        # Autogrow gives us a dict mapping input names to values.
        # Filter out None entries (unconnected optional slots).
        layer_list = [v for v in layers.values() if v is not None]

        if not layer_list:
            raise ValueError(
                "[SubjectCompositor] No layers connected. "
                "Connect at least one SubjectLayerDefine to layer_0."
            )

        layer_count = len(layer_list)
        bg_color = parse_color(canvas_color)

        # ── 3. Process each layer ─────────────────────────────────────────────
        processed: list[tuple[Image.Image, int, int, float, float]] = []

        for i, layer_def in enumerate(layer_list):
            try:
                img_rgba, scaled_w, scaled_h = process_layer(
                    image_tensor      = layer_def["image"],
                    mask_tensor       = layer_def.get("mask"),
                    remove_background = layer_def.get("remove_background", True),
                    bg_model          = layer_def.get("bg_model", "BiRefNet-general"),
                    pad_top           = layer_def.get("pad_top",    0.0),
                    pad_bottom        = layer_def.get("pad_bottom", 0.0),
                    pad_left          = layer_def.get("pad_left",   0.0),
                    pad_right         = layer_def.get("pad_right",  0.0),
                    canvas_w          = cw,
                    canvas_h          = ch,
                )
                processed.append((
                    img_rgba,
                    scaled_w,
                    scaled_h,
                    layer_def.get("offset_x", 0.0),
                    layer_def.get("offset_y", 0.0),
                ))
            except Exception as e:
                print(f"[SubjectCompositor] Error processing layer {i}: {e}")
                raise

        # ── 4. Build composite image ──────────────────────────────────────────
        composite_tensor = None

        if output_mode in ("composite", "both"):
            canvas = Image.new("RGBA", (cw, ch), bg_color)

            for img_rgba, sw, sh, ox, oy in processed:
                px, py = compute_paste_position(sw, sh, cw, ch, ox, oy)
                canvas = composite_onto_canvas(canvas, img_rgba, px, py)

            # Flatten RGBA to RGB over the background color
            bg = Image.new("RGB", (cw, ch), bg_color[:3])
            bg.paste(canvas.convert("RGB"), mask=canvas.split()[3])
            composite_tensor = pil_to_tensor(bg)

        # ── 5. Build individual images ────────────────────────────────────────
        individual_tensor = None

        if output_mode in ("individual", "both"):
            individual_tensors = []

            for img_rgba, sw, sh, ox, oy in processed:
                ind_canvas = Image.new("RGBA", (cw, ch), bg_color)
                px, py = compute_paste_position(sw, sh, cw, ch, ox, oy)
                ind_canvas = composite_onto_canvas(ind_canvas, img_rgba, px, py)

                # Flatten to RGB
                bg = Image.new("RGB", (cw, ch), bg_color[:3])
                bg.paste(ind_canvas.convert("RGB"), mask=ind_canvas.split()[3])
                individual_tensors.append(pil_to_tensor(bg))  # [1, H, W, 3]

            # Stack into batch [N, H, W, 3]
            individual_tensor = torch.cat(individual_tensors, dim=0)

        # ── 6. Handle output_mode fallbacks ──────────────────────────────────
        # If composite was not generated, create a placeholder (first individual)
        if composite_tensor is None and individual_tensor is not None:
            composite_tensor = individual_tensor[0:1]

        # If individual was not generated, return composite repeated N times
        if individual_tensor is None and composite_tensor is not None:
            individual_tensor = composite_tensor.repeat(layer_count, 1, 1, 1)

        return io.NodeOutput(
            composite_tensor,
            individual_tensor,
            layer_count,
        )
