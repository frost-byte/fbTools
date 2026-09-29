"""MarkerFrameSplit node — locate a deliberately-inserted marker-color frame run inside a video
frame batch and split it into the "before" and "after" clip segments.

Built for a clip-bridging pipeline: concatenate two clips with a solid marker-color buffer between
them (e.g. in an editor, before export), load the result as one video, then use this node to find
exactly where clip A ends and clip B begins before feeding the tail of A and the head of B into
MiniMaxH3AddGuide as the two anchor points for a generated bridge. See utils/marker_frame_split.py
for why a deliberate marker beats relying on ML scene-detection for this specific job.
"""
from __future__ import annotations

from comfy_api.latest import io

from ..utils.marker_frame_split import find_marker_run, hex_to_rgb01
from .shared import prefixed_node_id


class MarkerFrameSplit(io.ComfyNode):
    """Split a video frame batch at a deliberately-inserted marker-color segment.

    Raises if no qualifying marker run is found — silently returning the whole input as "clip_a"
    would be a worse failure mode than a loud error here, since a missing marker means the rest of
    a bridging graph would run on the wrong data.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("MarkerFrameSplit"),
            display_name="Marker Frame Split",
            category="🧊 frost-byte/Video",
            description=(
                "Splits a video frame batch at a deliberately-inserted marker-color segment "
                "(e.g. solid magenta) into the frames before and after it — used to locate the "
                "real clip-A/clip-B boundary in a pre-concatenated clip-bridging input."
            ),
            inputs=[
                io.Image.Input("images"),
                io.String.Input(
                    "marker_color", default="#FF00FF",
                    tooltip="6-digit hex color the marker frames were filled with — pick something "
                            "that never occurs in real footage (pure magenta/green, not black).",
                ),
                io.Float.Input(
                    "tolerance", default=0.08, min=0.0, max=1.0, step=0.01,
                    tooltip="Max per-frame mean-color distance (normalized RGB) to still count as a "
                            "marker frame. Higher tolerates compression noise around the marker.",
                ),
                io.Int.Input(
                    "min_marker_frames", default=1, min=1, max=9999,
                    tooltip="Shortest run of consecutive marker-colored frames that counts as the "
                            "real marker segment, not an incidental partial match elsewhere.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="clip_a_frames"),
                io.Image.Output(display_name="clip_b_frames"),
                io.Int.Output(display_name="clip_a_end_idx"),
                io.Int.Output(display_name="clip_b_start_idx"),
                io.Int.Output(display_name="marker_frame_count"),
            ],
        )

    @classmethod
    def execute(cls, images, marker_color, tolerance, min_marker_frames) -> io.NodeOutput:
        marker_rgb = hex_to_rgb01(marker_color)
        frames_np = images.detach().cpu().numpy()
        result = find_marker_run(frames_np, marker_rgb, tolerance, min_marker_frames)
        if result is None:
            raise ValueError(
                f"No marker segment found (color={marker_color}, tolerance={tolerance}, "
                f"min_marker_frames={min_marker_frames}) — check the marker color/tolerance match "
                f"what was actually inserted, and that the input video wasn't re-encoded in a way "
                f"that shifted the marker frames' color."
            )

        clip_a_end_idx = result["clip_a_end_idx"]
        clip_b_start_idx = result["clip_b_start_idx"]
        total_frames = images.shape[0]

        clip_a_frames = images[: clip_a_end_idx + 1] if clip_a_end_idx >= 0 else images[:0]
        clip_b_frames = images[clip_b_start_idx:] if clip_b_start_idx < total_frames else images[:0]

        return io.NodeOutput(
            clip_a_frames, clip_b_frames, clip_a_end_idx, clip_b_start_idx, result["marker_frame_count"],
        )
