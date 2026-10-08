"""H3 Source Guides -- anchor frames of a Source Profile clip at their own times in the output.

A reference video is placed ahead of the generated video on the model's time axis and is not
aligned with it. MiniMaxH3AddGuide puts a frame (or a short clip) at a chosen frame of the
generated video instead. This node reads the clip a Source Profile prompt references, picks one
frame every N, and adds each as a guide, so a few frames can carry the clip's progression at its
real timing.

Wire it after CompositionToH3Conditioning, before the guider/sampler:
    Composition -> H3 Conditioning (positive, latent) -> H3 Source Guides -> guider / sampler
Guides are sized to the connected latent, so keep any latent upscale at 1x (as the bridge
template does) or add this node once per pass.
"""
from __future__ import annotations

import hashlib
import json
import os

from comfy_api.latest import io

from .composition_shared import _h3_resolve_path
from .composition_types import H3RefplanType
from .shared import prefixed_node_id, send_status_update
from ..utils.h3_source_guides import describe_plan, plan_guides
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

FPS = 24
_GUIDE_FRAME_OPTIONS = ["1", "5", "22"]
_TIME_MAPPINGS = ["real time", "stretch to output length"]


def _find_video_ref(refplan: dict, video_ordinal: int) -> dict | None:
    for ref in (refplan or {}).get("references", []):
        if ref.get("modality") == "video" and ref.get("video_ordinal") == video_ordinal:
            return ref
    return None


def _source_load_params(ref: dict) -> tuple[dict, int]:
    """Load params for reading every frame of the clip, and how many real frames it holds."""
    lp = dict(ref.get("load_params") or {})
    duration = float(lp.get("duration", 0.0))
    lp.update(force_rate=FPS, select_every_nth=1, frame_load_cap=0)
    real_frames = max(1, round(duration * FPS)) if duration > 0 else 0
    return lp, real_frames


class H3SourceGuides(io.ComfyNode):
    node_id = prefixed_node_id("H3SourceGuides")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name="H3 Source Guides",
            category="🧊 frost-byte/conditioning",
            description=(
                "Adds frames from the Source Profile clip as MiniMaxH3AddGuide guides, one every N "
                "frames, each at its own time in the output. Wire after Composition → H3 "
                "Conditioning, before the guider. Guides are sized to the connected latent."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                H3RefplanType.Input("h3_refplan"),
                io.Int.Input(
                    "video_ordinal", default=1, min=1, max=3,
                    tooltip="Which <Video N> reference to take the frames from.",
                ),
                io.Int.Input(
                    "interval_frames", default=12, min=1, max=1000,
                    tooltip="Output frames between guides (24 frames = 1 second).",
                ),
                io.Combo.Input(
                    "guide_frames", options=_GUIDE_FRAME_OPTIONS, default="1",
                    tooltip="Frames per guide. 1 = a single image; 5 or 22 = a short run of the "
                            "clip at its real motion (a run of 5 is 2 latent frames).",
                ),
                io.Int.Input(
                    "max_guides", default=24, min=1, max=200,
                    tooltip="Upper limit. If the interval would give more, guides are thinned "
                            "evenly (first and last are kept).",
                ),
                io.Boolean.Input(
                    "include_last", default=True,
                    tooltip="Also anchor the end of the clip.",
                ),
                io.Combo.Input(
                    "time_mapping", options=_TIME_MAPPINGS, default=_TIME_MAPPINGS[0],
                    tooltip="'real time': output frame N uses source frame N, and guides past the "
                            "end of the source are dropped. 'stretch': the source is spread over "
                            "the whole output length.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.String.Output(display_name="Summary",
                                 tooltip="Which source frame was anchored at which output frame."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, h3_refplan, video_ordinal=1, interval_frames=12, guide_frames="1",
                           max_guides=24, include_last=True, time_mapping=_TIME_MAPPINGS[0], **_):
        ref = _find_video_ref(h3_refplan, video_ordinal) or {}
        path = _h3_resolve_path(ref.get("path", ""))
        try:
            mtime = f"{os.path.getmtime(path):.3f}"
        except OSError:
            mtime = "missing"
        plan_hash = hashlib.md5(json.dumps(ref, sort_keys=True, default=str).encode()).hexdigest()
        return (plan_hash, path, mtime, video_ordinal, interval_frames, guide_frames,
                max_guides, include_last, time_mapping)

    @classmethod
    def execute(cls, positive, latent, vae, h3_refplan, video_ordinal=1, interval_frames=12,
                guide_frames="1", max_guides=24, include_last=True,
                time_mapping=_TIME_MAPPINGS[0]) -> io.NodeOutput:
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
            from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        except ImportError as exc:
            raise RuntimeError(
                f"MiniMaxH3AddGuide not found in comfy_extras. Ensure ComfyUI includes "
                f"nodes_minimax_h3.py. ({exc})"
            ) from exc
        from .compositions import _h3_load_video_frames

        ref = _find_video_ref(h3_refplan, video_ordinal)
        if ref is None:
            raise ValueError(
                f"H3 Source Guides: the refplan has no <Video {video_ordinal}> reference. "
                f"Use a Source Profile clip prompt (video edit), or pick another video_ordinal."
            )

        load_params, real_frames = _source_load_params(ref)
        frames = _h3_load_video_frames(ref.get("path", ""), load_params)
        if frames is None:
            raise ValueError(
                f"H3 Source Guides: could not load <Video {video_ordinal}> "
                f"({ref.get('path', '(no path)')}). See the log above for the cause."
            )
        # The loader pads to a valid reference length by ping-ponging; only real frames count.
        source_frames = min(frames.shape[0], real_frames) if real_frames else frames.shape[0]

        video_latent = latent["samples"].tensors[0]
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video_latent.shape[2]))
        n_frames = int(guide_frames)

        plan = plan_guides(
            frame_count, source_frames, interval_frames, n_frames, max_guides,
            include_last=include_last, stretch=(time_mapping == _TIME_MAPPINGS[1]),
        )
        summary = describe_plan(plan, frame_count, n_frames, FPS)
        if not plan:
            logger.warning("H3 Source Guides: %d source frame(s) and a %d-frame video give no guides.",
                           source_frames, frame_count)
        logger.info(summary)
        send_status_update(cls.node_id, summary.splitlines()[0])

        for guide in plan:
            start = guide["src_start"]
            result = MiniMaxH3AddGuide.execute(
                positive, latent, guide["frame_idx"],
                vae=vae, image=frames[start:start + n_frames],
            )
            positive = result.args[0]

        return io.NodeOutput(positive, summary)
