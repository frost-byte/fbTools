"""Prompt Shorthand Expander: expands S1/V1/A1 shorthand into MiniMax H3's
native <Subject 1>/<Video 1>/<Audio 1> reference-label format.

See docs/openshot-bridge-prompt-shorthand.md for the user-facing convention.
"""
from comfy_api.latest import io

from .shared import prefixed_node_id
from ..utils.prompt_shorthand import expand_prompt_shorthand


class PromptShorthandExpander(io.ComfyNode):
    """
    Expands S1/V1/A1-style shorthand into MiniMax H3 Ref2VA's native
    <Subject 1>/<Video 1>/<Audio 1> reference-label format. Case-insensitive
    and word-bounded, so it only matches whole S<digits>/V<digits>/A<digits>
    tokens -- text already using the full <Subject N> form passes through
    unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("PromptShorthandExpander"),
            display_name="Prompt Shorthand Expander",
            category="🧊 frost-byte/Nodes",
            description=(
                "Expand S1/V1/A1 shorthand into <Subject 1>/<Video 1>/<Audio 1> "
                "for MiniMax H3 Ref2VA prompts."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt text using S1/V1/A1 shorthand for subject/video/audio references.",
                ),
            ],
            outputs=[
                io.String.Output("expanded_prompt", display_name="Expanded Prompt"),
            ],
        )

    @classmethod
    def execute(cls, prompt: str = ""):
        return io.NodeOutput(expand_prompt_shorthand(prompt))
