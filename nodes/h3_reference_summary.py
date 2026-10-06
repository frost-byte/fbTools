"""H3 Reference Summary -- report the references handed to MiniMax H3's ReferenceToVideo node.

Wire the same sources into this node's ref_* groups that feed MiniMaxH3ReferenceToVideo (the
input group names and numbering match, so the API-format keys are identical). It logs a summary
of what the model is given -- tag numbering (<Picture i> / <Video k> / <Audio j>), frames
actually used after H3's 17k+5 length rule, durations, soundtrack pairing, and warnings -- and
returns the same text as a STRING for an optional ShowText. It is an output node, so it runs
(and logs) even when its string output is left unconnected.
"""
from __future__ import annotations

from comfy_api.latest import io

from .shared import prefixed_node_id
from ..utils.h3_reference_summary import describe_references
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _audio_info(audio) -> dict | None:
    """Reduce an AUDIO value to {"seconds", "sample_rate", "channels"}.

    Indexes by key instead of isinstance(audio, dict): VideoHelperSuite returns a lazy Mapping
    (LazyAudioMap), not a dict -- see docs/GOTCHAS.md.
    """
    if audio is None:
        return None
    try:
        waveform = audio["waveform"]
        rate = int(audio["sample_rate"])
    except (KeyError, TypeError, ValueError):
        return None
    shape = tuple(getattr(waveform, "shape", ()))
    length = int(shape[-1]) if shape else 0
    channels = int(shape[-2]) if len(shape) >= 2 else 1
    return {"seconds": (length / rate) if rate else 0.0, "sample_rate": rate, "channels": channels}


def _frames_info(name: str, frames) -> dict | None:
    """Reduce an IMAGE batch [N, H, W, C] to its frame count and size."""
    shape = tuple(getattr(frames, "shape", ()))
    if len(shape) < 3:
        return None
    return {"name": name, "frames": int(shape[0]), "height": int(shape[1]), "width": int(shape[2])}


class H3ReferenceSummary(io.ComfyNode):
    node_id = prefixed_node_id("H3ReferenceSummary")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name="H3 Reference Summary",
            category="🧊 frost-byte/Video",
            description=(
                "Logs which references MiniMax H3 is given (tag numbering, frames used, durations, "
                "soundtrack pairing, warnings). Wire the same sources as MiniMaxH3ReferenceToVideo. "
                "Output node: runs even when the string output is unconnected."
            ),
            is_output_node=True,
            inputs=[
                io.Int.Input(
                    "length", default=0, min=0, max=3600, optional=True,
                    tooltip="The generation length in frames (the same value fed to the reference node). "
                            "0 = unknown; when set, longer reference videos are reported as truncated.",
                ),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[
                io.String.Output(
                    "summary", display_name="Summary",
                    tooltip="The same text that is logged. Optional: feed a ShowText node.",
                ),
            ],
        )

    @classmethod
    def execute(cls, length: int = 0, ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        images = []
        for name, img in (ref_images or {}).items():
            info = _frames_info(name, img)
            if info:
                images.append({"name": name, "height": info["height"], "width": info["width"]})

        videos = []
        for name, frames in (ref_videos or {}).items():
            info = _frames_info(name, frames)
            if info:
                videos.append(info)

        video_audios = {}
        for name, audio in (ref_video_audios or {}).items():
            info = _audio_info(audio)
            if info:
                video_audios[name.rsplit("_", 1)[-1]] = info

        audios = []
        for name, audio in (ref_audios or {}).items():
            info = _audio_info(audio)
            if info:
                audios.append({"name": name, **info})

        summary, warnings = describe_references(
            images=images, videos=videos, video_audios=video_audios, audios=audios,
            length=int(length) or None,
        )
        if warnings:
            logger.warning("%s", summary)
        else:
            logger.info("%s", summary)
        return io.NodeOutput(summary, ui={"text": [summary]})
