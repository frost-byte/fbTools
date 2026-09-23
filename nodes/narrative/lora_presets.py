"""LoRA presets domain (distinct from LoRA stacks, in ../lora_stacks.py), extracted from
extension.py (Plan 23).

Sibling to nodes/narrative/scene.py and nodes/narrative/story.py in the same nodes/narrative/
subpackage — imports SceneInfo/default_pose_options from .scene directly, and LoraStackData from
..lora_stacks (only ever used as a wire-type annotation, not for any stack-building logic).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from comfy_api.latest import io, ui

from ..lora_stacks import LoraStackData
from ..shared import prefixed_node_id, default_scenes_dir
from .scene import SceneInfo, default_pose_options
from ...utils.images import make_empty_image
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Preset scene-image loader ─────────────────────────────────────────────────

def _load_preset_scene_images(
    scene_name: str,
    pose_image_type: str,
) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
    """Return (base_image, pose_image) tensors for a preset's linked scene.

    Returns (None, None) when scene_name is "none"/empty or the directory
    doesn't exist.  Callers should substitute a placeholder before wiring
    these to io.Image outputs.
    """
    if not scene_name or scene_name == "none":
        return None, None
    scene_dir = os.path.join(default_scenes_dir(), scene_name)
    if not os.path.isdir(scene_dir):
        logger.warning("_load_preset_scene_images: scene_dir '%s' not found", scene_dir)
        return None, None
    pose_attr = default_pose_options.get(pose_image_type, "pose_open_image")
    try:
        assets = SceneInfo.load_preview_assets(
            scene_dir,
            depth_attr="depth_image",
            pose_attr=pose_attr,
            mask_name="",
        )
        return assets.get("base_image"), assets.get("pose_image")
    except Exception:
        logger.exception("_load_preset_scene_images: error loading assets for '%s'", scene_name)
        return None, None


def _preset_scene_ui_and_images(
    preset: dict,
    names: list[str],
) -> "tuple[torch.Tensor, torch.Tensor, dict]":
    """Build (base_image, pose_image, ui_data) for a *PresetSelect execute().

    Always returns tensors (placeholder when no scene is linked).
    ui_data includes preset_names and any preview images for is_output_node.
    """
    base_image, pose_image = _load_preset_scene_images(
        preset.get("scene_name", "none"),
        preset.get("pose_image_type", "open"),
    )

    placeholder = make_empty_image(1, 64, 64)
    base_out = base_image if base_image is not None else placeholder
    pose_out = pose_image if pose_image is not None else placeholder

    preview_batch = [t for t in [base_image, pose_image] if t is not None]
    preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None

    ui_data: dict = {"preset_names": names}
    if preview_image:
        pd = preview_image.as_dict()
        ui_data["images"] = pd.get("images", [])
        ui_data["animated"] = pd.get("animated", False)

    return base_out, pose_out, ui_data


# ── Custom type: LORA_PRESET_LIST ────────────────────────────────────────────

LORA_PRESET_LIST_TYPE = "LORA_PRESET_LIST"


@io.comfytype(io_type=LORA_PRESET_LIST_TYPE)
class LoraPresetList:
    """
    Carries an ordered list of LoRA presets between nodes.
    Each entry is a dict: { name, lora_stack, prompt, scene_name, pose_image_type }.
    lora_stack holds a LORA_STACK_DATA value (list of dicts from LoraStackCollect).
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "preset_list", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: LoraPresetDefine ────────────────────────────────────────────────────

class LoraPresetDefine(io.ComfyNode):
    """
    Define one LoRA preset and append it to an optional incoming preset list.
    Chain multiple LoraPresetDefine nodes sequentially to build a collection;
    leave preset_list unconnected on the first node.

    Each preset holds a name, a single LORA_STACK_DATA, optional prompt, and
    an optional linked scene (for base/pose image output from LoraPresetSelect).
    Use this instead of WanPresetDefine for models with a single sampler stage.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraPresetDefine"),
            display_name="LoRA Preset Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one LoRA preset (name, LoRA stack, prompt, optional scene) "
                "and append it to an optional incoming preset list. "
                "Chain multiple nodes to build a preset collection."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    display_name="Preset Name",
                    default="Preset",
                    tooltip="Human-readable name for this preset.",
                ),
                LoraStackData.Input(
                    "lora_stack",
                    display_name="LoRA Stack (LORA_STACK_DATA)",
                    optional=True,
                    tooltip="Rich per-target stack from LoraStackCollect's 'Stack Data' output. Auto-generates the native LORA_STACK output.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_stack_native",
                    display_name="LoRA Stack (Native)",
                    optional=True,
                    tooltip="Native (name, model_str, clip_str) stack from any easy-use compatible source. Use instead of or alongside the LORA_STACK_DATA input.",
                ),
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    default="",
                    multiline=True,
                    tooltip="Positive prompt text for this preset.",
                ),
                io.Combo.Input(
                    "scene_name",
                    display_name="Scene",
                    options=["none"],
                    default="none",
                    tooltip=(
                        "Optional scene to associate with this preset. "
                        "When selected, LoraPresetSelect outputs the scene's base and pose images."
                    ),
                ),
                io.Combo.Input(
                    "pose_image_type",
                    display_name="Pose Image Type",
                    options=list(default_pose_options.keys()),
                    default="open",
                    tooltip="Which pose image variant to load from the scene.",
                ),
                LoraPresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    optional=True,
                    tooltip="Incoming list from a previous LoraPresetDefine node. Leave unconnected on the first node in the chain.",
                ),
            ],
            outputs=[
                LoraPresetList.Output("preset_list", display_name="Preset List"),
            ],
        )

    @classmethod
    def validate_inputs(cls, scene_name: str = "none", **kwargs) -> bool | str:
        # scene_name is populated dynamically by the frontend; bypass static validation.
        return True

    @classmethod
    def execute(
        cls,
        name: str,
        lora_stack: Optional[list] = None,
        lora_stack_native: Optional[list] = None,
        prompt: str = "",
        scene_name: str = "none",
        pose_image_type: str = "open",
        preset_list: Optional[list] = None,
    ) -> io.NodeOutput:
        from ...utils.lora_presets import preset_define
        return io.NodeOutput(
            preset_define(name, lora_stack, prompt, preset_list, scene_name, pose_image_type, lora_stack_native)
        )


# ── Node: LoraPresetSelect ────────────────────────────────────────────────────

class LoraPresetSelect(io.ComfyNode):
    """
    Select one preset from a LoraPresetDefine chain by name.
    Outputs the preset's LoRA stack, prompt, scene images, and a summary of
    all available presets (wire to a Show Text node).

    Falls back to the first preset if the selected name is not found.
    If the preset has a linked scene, base_image and pose_image are loaded
    from that scene; otherwise placeholder 64×64 black images are returned.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraPresetSelect"),
            display_name="LoRA Preset Select",
            category="🧊 frost-byte/lora",
            description=(
                "Select one preset from a completed LoraPresetDefine chain. "
                "Outputs the LoRA stack, prompt, and optional scene images."
            ),
            inputs=[
                LoraPresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    tooltip="The complete preset list from the end of a LoraPresetDefine chain.",
                ),
                io.Combo.Input(
                    "selected_preset",
                    display_name="Preset",
                    options=["none"],
                    default="none",
                    tooltip="Select a preset by name. Connect a Preset List and run this node to populate the dropdown.",
                ),
            ],
            outputs=[
                io.String.Output("name",              display_name="Name"),
                LoraStackData.Output("lora_stack",    display_name="LoRA Stack (LORA_STACK_DATA)",
                    tooltip="Rich per-target stack. Connect to LoraStackApply."),
                io.String.Output("prompt",            display_name="Prompt"),
                io.String.Output("available_presets", display_name="Available Presets"),
                io.Image.Output("base_image",         display_name="Base Image",
                    tooltip="Base image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Image.Output("pose_image",         display_name="Pose Image",
                    tooltip="Pose image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Custom("LORA_STACK").Output("lora_stack_native", display_name="LoRA Stack (Native)",
                    tooltip="Native (name, model_str, clip_str) stack. Connect to EasyLoraStack, PowerLoraLoader, or any easy-use compatible node."),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, selected_preset: str, **kwargs) -> bool | str:
        # Accept any string — options are populated dynamically by the frontend
        # after execution, so the static schema list ["none"] is just a placeholder.
        return True

    @classmethod
    def execute(
        cls,
        preset_list: list,
        selected_preset: str,
    ) -> io.NodeOutput:
        from ...utils.lora_presets import preset_select
        name, lora_stack, lora_stack_native, prompt, available = preset_select(preset_list, selected_preset)
        names = [p.get("name", "") for p in preset_list] if preset_list else []
        selected = next((p for p in preset_list if p.get("name") == name), {}) if preset_list else {}
        base_image, pose_image, ui_data = _preset_scene_ui_and_images(selected, names)
        return io.NodeOutput(name, lora_stack, prompt, available, base_image, pose_image, lora_stack_native, ui=ui_data)


# ── Custom type: PRESET_LIST ──────────────────────────────────────────────────

PRESET_LIST_TYPE = "PRESET_LIST"


@io.comfytype(io_type=PRESET_LIST_TYPE)
class PresetList:
    """
    Carries an ordered list of Wan video generation presets between nodes.
    Each entry is a dict: { name, lora_h, lora_l, prompt, scene_name, pose_image_type }.
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "preset_list", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: WanPresetDefine ─────────────────────────────────────────────────────

class WanPresetDefine(io.ComfyNode):
    """
    Define one Wan video generation preset and append it to an optional
    incoming preset list.  Chain multiple WanPresetDefine nodes sequentially
    to build a collection; leave preset_list unconnected on the first node.

    lora_h / lora_l carry LoRA stacks for the high-noise and low-noise model
    stages respectively.  An optional linked scene provides base/pose images
    that WanPresetSelect outputs when this preset is selected.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("WanPresetDefine"),
            display_name="Wan Preset Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one Wan video preset (name, high/low LoRA, prompt, optional scene) "
                "and append it to an optional incoming preset list. "
                "Chain multiple nodes to build a preset collection."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    display_name="Preset Name",
                    default="Preset",
                    tooltip="Human-readable name for this preset.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_h",
                    display_name="LoRA Stack (High Noise)",
                    optional=True,
                    tooltip="LoRA stack for the high-noise model stage. Connect from LoraStackCollect, EasyLoraStack, PowerLoraLoader, or any LORA_STACK source.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_l",
                    display_name="LoRA Stack (Low Noise)",
                    optional=True,
                    tooltip="LoRA stack for the low-noise model stage. Connect from LoraStackCollect, EasyLoraStack, PowerLoraLoader, or any LORA_STACK source.",
                ),
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    default="",
                    multiline=True,
                    tooltip="Positive prompt text for this preset.",
                ),
                io.Combo.Input(
                    "scene_name",
                    display_name="Scene",
                    options=["none"],
                    default="none",
                    tooltip=(
                        "Optional scene to associate with this preset. "
                        "When selected, WanPresetSelect outputs the scene's base and pose images."
                    ),
                ),
                io.Combo.Input(
                    "pose_image_type",
                    display_name="Pose Image Type",
                    options=list(default_pose_options.keys()),
                    default="open",
                    tooltip="Which pose image variant to load from the scene.",
                ),
                PresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    optional=True,
                    tooltip="Incoming list from a previous WanPresetDefine node. Leave unconnected on the first node in the chain.",
                ),
            ],
            outputs=[
                PresetList.Output("preset_list", display_name="Preset List"),
            ],
        )

    @classmethod
    def validate_inputs(cls, scene_name: str = "none", **kwargs) -> bool | str:
        # scene_name is populated dynamically by the frontend; bypass static validation.
        return True

    @classmethod
    def execute(
        cls,
        name: str,
        lora_h: Optional[list] = None,
        lora_l: Optional[list] = None,
        prompt: str = "",
        scene_name: str = "none",
        pose_image_type: str = "open",
        preset_list: Optional[list] = None,
    ) -> io.NodeOutput:
        from ...utils.wan_presets import preset_define
        return io.NodeOutput(preset_define(name, lora_h, lora_l, prompt, preset_list, scene_name, pose_image_type))


# ── Node: WanPresetSelect ─────────────────────────────────────────────────────

class WanPresetSelect(io.ComfyNode):
    """
    Select one preset from a WanPresetDefine chain by name.
    Outputs individual fields for downstream consumption, a formatted summary
    of all available presets, and scene images if the preset has a linked scene.

    Falls back to the first preset if the selected name is not found.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("WanPresetSelect"),
            display_name="Wan Preset Select",
            category="🧊 frost-byte/lora",
            description=(
                "Select one preset from a completed WanPresetDefine chain. "
                "Outputs individual fields, scene images, and an available-presets summary."
            ),
            inputs=[
                PresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    tooltip="The complete preset list from the end of a WanPresetDefine chain.",
                ),
                io.Combo.Input(
                    "selected_preset",
                    display_name="Preset",
                    options=["none"],
                    default="none",
                    tooltip="Select a preset by name. Connect a Preset List and run this node to populate the dropdown.",
                ),
            ],
            outputs=[
                io.String.Output("name",              display_name="Name"),
                io.Custom("LORA_STACK").Output("lora_h", display_name="LoRA Stack (High Noise)"),
                io.Custom("LORA_STACK").Output("lora_l", display_name="LoRA Stack (Low Noise)"),
                io.String.Output("prompt",            display_name="Prompt"),
                io.String.Output("available_presets", display_name="Available Presets"),
                io.Image.Output("base_image",         display_name="Base Image",
                    tooltip="Base image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Image.Output("pose_image",         display_name="Pose Image",
                    tooltip="Pose image from the preset's linked scene, or a placeholder if no scene is set."),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, selected_preset: str, **kwargs) -> bool | str:
        # Accept any string — options are populated dynamically by the frontend
        # after execution, so the static schema list ["none"] is just a placeholder.
        return True

    @classmethod
    def execute(
        cls,
        preset_list: list,
        selected_preset: str,
    ) -> io.NodeOutput:
        from ...utils.wan_presets import preset_select
        name, lora_h, lora_l, prompt, available = preset_select(preset_list, selected_preset)
        names = [p.get("name", "") for p in preset_list] if preset_list else []
        selected = next((p for p in preset_list if p.get("name") == name), {}) if preset_list else {}
        base_image, pose_image, ui_data = _preset_scene_ui_and_images(selected, names)
        return io.NodeOutput(name, lora_h, lora_l, prompt, available, base_image, pose_image, ui=ui_data)
