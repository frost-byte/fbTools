"""ImageTextOverlay node — burns a text block onto every frame of an image batch.

Built for test/demo workflows: a node whose meaningful outputs are scalars or strings (counts,
indices, flags) can still produce a single visible image as proof it ran correctly, by feeding
those values in as `text` here — instead of pulling in a separate value-display node (and its
dependency) just to show a number.
"""
from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ..utils.image_text_overlay import hex_to_rgb255, overlay_text
from .shared import prefixed_node_id


class ImageTextOverlay(io.ComfyNode):
    """Draws a readable text block, on a semi-opaque background box, onto every frame of an image
    batch. The background box keeps the text legible regardless of what's underneath (dark
    footage, bright footage, a solid marker color, etc).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("ImageTextOverlay"),
            display_name="Image Text Overlay",
            category="🧊 frost-byte/Image Processing",
            description=(
                "Burns a text block (newline-separated lines) onto every frame of an image batch, "
                "behind a semi-opaque background box for legibility. Handy for turning a node's "
                "scalar/text outputs into one visible image, e.g. for a test workflow's thumbnail."
            ),
            inputs=[
                io.Image.Input("images"),
                io.String.Input("text", multiline=True, default=""),
                io.Combo.Input(
                    "position", options=["bottom-left", "bottom-right", "top-left", "top-right"],
                    default="bottom-left",
                ),
                io.Int.Input("font_size", default=16, min=6, max=128, step=1),
                io.String.Input("text_color", default="#FFFFFF"),
                io.String.Input("bg_color", default="#000000"),
                io.Float.Input("bg_opacity", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, images, text, position, font_size, text_color, bg_color, bg_opacity) -> io.NodeOutput:
        text_rgb = hex_to_rgb255(text_color)
        bg_rgb = hex_to_rgb255(bg_color)

        frames_np = (images.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        out_frames = np.stack(
            [
                overlay_text(frame, text, position, font_size, 6, text_rgb, bg_rgb, bg_opacity)
                for frame in frames_np
            ],
            axis=0,
        )
        out_tensor = torch.from_numpy(out_frames.astype(np.float32) / 255.0)
        return io.NodeOutput(out_tensor)
