"""Audio utility node, extracted from extension.py (Plan 24)."""
from __future__ import annotations

from comfy_api.latest import io

from .shared import prefixed_node_id


class AudioFixShape(io.ComfyNode):
    """Fixes audio waveform tensor shape by ensuring the batch dimension exists.

    Useful after nodes that strip the batch dimension, leaving a 1-D or 2-D
    tensor instead of the expected (batch, channels, samples) layout.
    """
    node_id = prefixed_node_id("AudioFixShape")
    display_name = "Audio Fix Shape"
    category = "🧊 frost-byte/Audio"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            description=cls.__doc__.split("\n")[0].strip(),
            category=cls.category,
            inputs=[
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(),
            ],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        if audio is None:
            return io.NodeOutput(None)

        waveform = audio["waveform"]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        return io.NodeOutput({"waveform": waveform, "sample_rate": audio["sample_rate"]})
