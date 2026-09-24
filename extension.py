from __future__ import annotations

from PIL import Image
import folder_paths
from folder_paths import get_input_directory, get_output_directory
import comfy.model_management as model_management

from typing_extensions import override
from .nodes.shared import (
    prefixed_node_id,
    routes,
    send_status_update,
    user_data_dir,
    default_subject_profiles_path,
    default_source_profiles_path,
    default_bundle_registry_path,
    default_cast_registry_path,
    default_outfit_registry_path,
    reload_counter,
    bump_reload,
)
from .nodes.dataset_caption import DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader
from .nodes.libber import Libber, LibberManager, LibberApply, LibberStateManager
from .nodes.lora_stacks import (
    LoraStackData, LoraStackBuilder, LoraStackApply, LoraEntryDefine, LoraStackCollect, WanVidLoraStack,
)
from .nodes.narrative.scene import (
    SceneSelect, SceneLoraStackSave, SceneCreate, SceneUpdate,
    SceneView, SceneMaskDefinition, SceneOutput, SceneSave, SceneInput,
)
from .nodes.narrative.story import (
    StoryCreate, StoryEdit, StoryView, StorySceneBatch, StoryScenePick,
    StorySave, StoryLoad, StorySceneImageSave, StoryVideoBatch,
)
from .nodes.narrative.lora_presets import LoraPresetDefine, LoraPresetSelect, WanPresetDefine, WanPresetSelect
from .nodes.narrative.scene_prompts import ScenePromptManager, PromptComposer
from .nodes.compositing import SubjectLayerDefine, SubjectCompositor
from .nodes.image_processing import SAMPreprocessNHWC, TailEnhancePro, TailSplit, OpaqueAlpha, MaskProcessor
from .nodes.qwen_conditioning import FBTextEncodeQwenImageEditPlus, QwenAspectRatio
from .nodes.audio import AudioFixShape
from .nodes.utility import SubdirLister
from .nodes.run_tracking import RunMetaCapture, JobCompleteNotifier, register_track_formatter
from .nodes.concepts import ConceptRegistryIOType, ConceptRegistryLoad, ConceptDefine, ConceptResolve, ConceptList
from .nodes.subjects import (
    SubjectProfileIOType, SubjectProfileLoad, SubjectProfileDefine, SubjectProfileList,
    _load_subject_images, _load_subject_audio,
)
from .nodes.scene_templates import SceneTemplateIOType, SceneTemplateLoad, SceneTemplateList
from .nodes.outfits import OutfitRegistryIOType, OutfitRegistryLoad, OutfitDefine, OutfitList
from .nodes.composition_types import (
    SourceProfileIOType, SceneInstanceIOType, CastIOType, H3RefplanType, CompositionIOType,
)
from .nodes.source_profiles import SourceProfileLoad, SourceProfileDefine, SourceProfileList, SourceProfileClipPrompt
from .utils.composition_track_summary import summarize_scene_cast, summarize_loras, summarize_composition_meta
from .nodes import kdenlive_archive as _kdenlive_archive_routes  # noqa: F401  (registers /fbtools/kdenlive/* routes on import)
# Route-only modules: importing them registers their /fbtools/* handlers on the PromptServer routes.
from .nodes import backgrounds_presets as _backgrounds_presets_routes  # noqa: F401
from .nodes import registry_api as _registry_api_routes  # noqa: F401
from .nodes import lora_info as _lora_info_routes  # noqa: F401
from .nodes import media as _media_routes  # noqa: F401
from .nodes import prompt_collections as _prompt_collections_routes  # noqa: F401
from .nodes import llm_assistant as _llm_assistant_routes  # noqa: F401
from .utils.util import (
    draw_pose_json,
    draw_pose,
    extend_scalelist,
    pose_normalized,
    find_node_by_id,
    get_node_inputs
)
from .utils.node_output_tracker import extract_tracked_nodes, stringify_capture_values
from .utils.h3_vram_estimator import tokens_for as h3_tokens_for, max_safe_scale as h3_max_safe_scale

from comfy_api.latest import ComfyExtension, io
import torch
import numpy as np
from typing import List
import os
import json
from dataclasses import asdict
import re
import copy
import hashlib
import random
from collections import deque
from .utils.logging_utils import get_logger
from .captioner import caption_image, get_model
from .utils.subject_profiles import load_registry as _load_subject_registry
from .utils.source_profiles import (
    load_registry as _load_source_registry,
    resolved_pronoun_style as _sp_resolved_pronoun_style,
    resolve_ordinal_subject as _sp_resolve_ordinal_subject,
    resolve_ordinal_from_list as _sp_resolve_ordinal_from_list,
)
from .utils.proxy_cache import ensure_bundle_video_proxy as _ensure_bundle_proxy
from .utils.scene_templates import SceneTemplate
from .utils.scene_compose import (
    compose_scene as _compose_scene,
    validate_scene as _validate_scene,
    format_scene_summary as _format_scene_summary,
)
from .utils.prompt_assembler import (
    assemble_prompt as _assemble_prompt,
    MODEL_TYPES as _PROMPT_MODEL_TYPES,
)
from .utils.reference_bundles import (
    BundleRegistry,
    load_registry as _load_bundle_registry,
    save_registry as _save_bundle_registry,
    validate_bundle as _validate_bundle,
)
from .utils.scene_casts import (
    CastRegistry,
    load_registry as _load_cast_registry,
    save_registry as _save_cast_registry,
    validate_cast as _validate_cast,
)
from .utils.outfit_registry import (
    OutfitRegistry,
    load_outfit_registry as _load_outfit_registry,
)


logger = get_logger(__name__)

try:
    from westNeighbor_comfyui_ultimate_openpose_editor.openpose_editor_nodes import OpenposeEditorNode  # type: ignore
except Exception:
    OpenposeEditorNode = None


OpenposeJSON = dict



# Dataset caption constants + 7 shared helpers moved to nodes/dataset_caption.py (Plan 18)
# _directory_fingerprint moved to nodes/shared.py (Plan 20)

# SUBJECT_LAYER custom type / BG_MODELS / OUTPUT_MODES moved to nodes/compositing.py (Plan 24)


# SubjectLayerDefine moved to nodes/compositing.py (Plan 24)


# SubjectCompositor moved to nodes/compositing.py (Plan 24)

# Dataset caption node classes (DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer,
# DatasetExportSummary, CaptionModelUnloader) moved to nodes/dataset_caption.py (Plan 18)

# Libber class moved to nodes/libber.py (Plan 17)

























def load_pose(
    show_body=True,
    show_face=True,
    show_hands=True,
    resolution_x=-1,
    pose_marker_size=4,
    face_marker_size=3,
    hand_marker_size=2,
    hands_scale=1.0,
    body_scale=1.0,
    head_scale=1.0,
    overall_scale=1.0,
    scalelist_behavior="poses",
    match_scalelist_method="loop extend",
    only_scale_pose_index=99,
    POSE_KEYPOINT=None
):
    if POSE_KEYPOINT is not None:
        POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
        hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
            scalelist_behavior, POSE_JSON, hands_scale, body_scale, head_scale, overall_scale,
            match_scalelist_method, only_scale_pose_index)
        normalized_pose_json = pose_normalized(POSE_JSON)
        pose_imgs, POSE_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
        if pose_imgs:
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return {
                "ui": {"POSE_JSON": [json.dumps(POSE_SCALED, indent=4)]},
                "result": (torch.from_numpy(pose_imgs_np), POSE_SCALED, json.dumps(POSE_SCALED, indent=4))
            }

    # otherwise output blank images
    W=512
    H=768
    pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
    pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
    people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

    W_scaled = resolution_x
    if resolution_x < 64:
        W_scaled = W
    H_scaled = int(H*(W_scaled*1.0/W))
    pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
    pose_img_np = np.array(pose_img).astype(np.float32) / 255

    return {
        "ui": {"POSE_JSON": people},
        "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
    }

# DictType moved to nodes/narrative/scene.py (Plan 21)


# MultiLoraLoader (dead) moved to nodes/utility.py (Plan 24)


# SAMPreprocessNHWC moved to nodes/image_processing.py (Plan 24)

# TailEnhancePro moved to nodes/image_processing.py (Plan 24)

# TailSplit moved to nodes/image_processing.py (Plan 24)

# OpaqueAlpha moved to nodes/image_processing.py (Plan 24)

# MaskProcessor moved to nodes/image_processing.py (Plan 24)

# get_subdirectories moved to nodes/shared.py (Plan 20)

# SubdirLister moved to nodes/utility.py (Plan 24)


# QwenAspectRatio moved to nodes/qwen_conditioning.py (Plan 24)


# ============================================================================
# PROMPT COLLECTION - Flexible Multi-Prompt System
# ============================================================================

# Import the data models from separate module for better testability
# PromptMetadata/PromptCollection import moved with ScenePromptManager/PromptComposer to nodes/narrative/scene_prompts.py (Plan 25)


# RGB/MaskType/MaskDefinition/load_masks_json/save_masks_json/SceneInfo moved to nodes/narrative/scene.py (Plan 20)

# ============================================================================
# STORY MODELS - Imported from story_models.py
# ============================================================================
# SceneInStory and StoryInfo have been extracted to story_models.py for easier
# testing and reusability. See story_models.py for the full definitions.

# _migrate_loras_json_to_stack and load_lora_stack moved to nodes/narrative/scene.py (Plan 20)


# save_lora_stack moved to nodes/narrative/scene.py (Plan 21)


# ── Legacy helpers kept for backward compat with any external callers ─────────

# load_loras/save_loras moved to nodes/narrative/scene.py (Plan 21)

# get_available_stories moved to nodes/narrative/story.py (Plan 22)


# NodeInputSelect (dead) moved to nodes/utility.py (Plan 24)

# SceneSelect moved to nodes/narrative/scene.py (Plan 21)

# default_depth_options/default_pose_options/default_mask_options/resolve_mask_key moved to nodes/narrative/scene.py (Plan 20)

# build_positive_prompt moved to nodes/narrative/story.py (Plan 22)

# SceneWanVideoLoraMultiSave/SceneLoraStackSave/SceneCreate/SceneUpdate/SceneView/SceneMaskDefinition/SceneOutput/SceneSave/SceneInput (+save_lora_stack/load_loras/save_loras) moved to nodes/narrative/scene.py (Plan 21)

# StoryCreate/StoryEdit/StoryView/StorySceneBatch/StoryScenePick/StorySave/StoryLoad/StorySceneImageSave/StoryVideoBatch moved to nodes/narrative/story.py (Plan 22)


# FBTextEncodeQwenImageEditPlus moved to nodes/qwen_conditioning.py (Plan 24)


# LibberManager/LibberApply nodes moved to nodes/libber.py (Plan 17)


# ScenePromptManager/PromptComposer moved to nodes/narrative/scene_prompts.py (Plan 25)


# ============================================================================
# REST API STATE MANAGERS
# ============================================================================

from aiohttp import web





# LibberStateManager moved to nodes/libber.py (Plan 17)










# DATASET CAPTION API ENDPOINTS moved to nodes/dataset_caption.py (Plan 18)

# LIBBER REST API ENDPOINTS moved to nodes/libber.py (Plan 17)


# scene_process_compositions moved to nodes/narrative/scene.py (Plan 21)


# scene_get_prompts moved to nodes/narrative/scene.py (Plan 21)


# scene_save_prompts moved to nodes/narrative/scene.py (Plan 21)


# story_load/story_get_job_ids/story_list/story_regenerate_thumbnails/story_save moved to nodes/narrative/story.py (Plan 22)


# ── LoRA civitai info endpoint ────────────────────────────────────────────────









# LoRA stack classes (LoraEntryDefine, LoraStackCollect, LoraStackView, LoraStackApply,
# WanVidLoraStack, LoraStackBuilder) + shared constants/helpers moved to nodes/lora_stacks.py
# (Plan 19). LoraStackData, LORA_MODEL_TARGETS, _lora_get_list, _lora_entries_for_target,
# _lora_build_wanvid, _lora_json_to_stack are imported back below for the classes here that
# still need them (SceneSelect, SceneLoraStackSave, SceneCreate, SceneUpdate, SceneOutput,
# SceneInput, StoryVideoBatch, SourceProfileClipPrompt, PromptCompositionLoader, ConceptDefine).


# _load_preset_scene_images/_preset_scene_ui_and_images/LoraPresetList/LoraPresetDefine/LoraPresetSelect/PresetList/WanPresetDefine/WanPresetSelect moved to nodes/narrative/lora_presets.py (Plan 23)


# ── Node: AudioFixShape ───────────────────────────────────────────────────────

# AudioFixShape moved to nodes/audio.py (Plan 24)


# ConceptRegistryLoad/ConceptDefine/ConceptResolve/ConceptList (+ ConceptRegistryIOType, _concept_get_ids) moved to nodes/concepts.py (Plan 26)

# SubjectProfileLoad/SubjectProfileDefine/SubjectProfileList (+ SubjectProfileIOType, subject helpers) moved to nodes/subjects.py (Plan 26)


# SourceProfileLoad/SourceProfileDefine/SourceProfileList/SourceProfileClipPrompt (+ source profile helpers) moved to nodes/source_profiles.py (Plan 28)
# ── Subject REST API endpoints ────────────────────────────────────────────────





# Source Profile REST API endpoints moved to nodes/source_profiles.py (Plan 28)


# SceneTemplateLoad/SceneTemplateList (+ SceneTemplateIOType, _template_get_ids) moved to nodes/scene_templates.py (Plan 26)


# ── Scene Template REST API endpoints ─────────────────────────────────────────





# OutfitRegistryLoad/OutfitDefine/OutfitList (+ OutfitRegistryIOType, _outfit_get_ids) moved to nodes/outfits.py (Plan 26)


# SceneInstanceIOType moved to nodes/composition_types.py (Plan 27)
# ── Node: SceneCompose ────────────────────────────────────────────────────────

class SceneCompose(io.ComfyNode):
    """Assign subjects to template slots and fill in dialogue.

    Connects subject profiles to a scene template's placeholder slots, maps
    positional dialogue inputs to shots in order, and optionally overrides
    outfit descriptions per slot.  Outputs a SCENE_INSTANCE ready for
    PromptAssemble in Phase 4.
    """
    node_id = prefixed_node_id("SceneCompose")
    display_name = "Scene Compose"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                SceneTemplateIOType.Input(
                    "template",
                    display_name="Scene Template",
                    tooltip="Template from SceneTemplateLoad.",
                ),
                SubjectProfileIOType.Input(
                    "slot_A",
                    display_name="Slot A",
                    tooltip="Subject assigned to slot A (typically the primary speaker).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_B",
                    display_name="Slot B",
                    tooltip="Subject assigned to slot B (optional).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_C",
                    display_name="Slot C",
                    tooltip="Subject assigned to slot C (optional).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_D",
                    display_name="Slot D",
                    tooltip="Subject assigned to slot D (optional).",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_1",
                    display_name="Dialogue 1",
                    default="",
                    multiline=True,
                    tooltip="Fills the first placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_2",
                    display_name="Dialogue 2",
                    default="",
                    multiline=True,
                    tooltip="Fills the second placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_3",
                    display_name="Dialogue 3",
                    default="",
                    multiline=True,
                    tooltip="Fills the third placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_4",
                    display_name="Dialogue 4",
                    default="",
                    multiline=True,
                    tooltip="Fills the fourth placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "scene_synopsis",
                    display_name="Scene Synopsis",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Concise scene overview used as the summary body in H3 prompts. "
                        "Use {A}/{B}/… slot references (A–J) — expanded to <Subject N> labels at assemble time. "
                        "Overrides the automatic shot-action summary when non-empty."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_A",
                    display_name="Outfit Override A",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot A subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_B",
                    display_name="Outfit Override B",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot B subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_C",
                    display_name="Outfit Override C",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot C subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_D",
                    display_name="Outfit Override D",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot D subject's default outfit for this scene only.",
                    optional=True,
                ),
                OutfitRegistryIOType.Input(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip=(
                        "Optional outfit registry from OutfitRegistryLoad. "
                        "When connected, outfit_*_id inputs resolve to registry descriptions."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_A_id",
                    display_name="Outfit A ID",
                    default="",
                    multiline=False,
                    tooltip=(
                        "ID of an outfit in the registry to use for slot A. "
                        "Ignored if outfit_override_A is non-empty."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_B_id",
                    display_name="Outfit B ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot B.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_C_id",
                    display_name="Outfit C ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot C.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_D_id",
                    display_name="Outfit D ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot D.",
                    optional=True,
                ),
            ],
            outputs=[
                SceneInstanceIOType.Output(
                    "scene_instance",
                    display_name="Scene Instance",
                    tooltip="Composed scene ready for PromptAssemble.",
                ),
                io.String.Output(
                    "scene_summary",
                    display_name="Scene Summary",
                    tooltip="Human-readable summary of the composed scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        template=None,
        slot_A=None,
        slot_B=None,
        slot_C=None,
        slot_D=None,
        scene_synopsis: str = "",
        dialogue_1: str = "",
        dialogue_2: str = "",
        dialogue_3: str = "",
        dialogue_4: str = "",
        outfit_override_A: str = "",
        outfit_override_B: str = "",
        outfit_override_C: str = "",
        outfit_override_D: str = "",
        outfit_registry=None,
        outfit_A_id: str = "",
        outfit_B_id: str = "",
        outfit_C_id: str = "",
        outfit_D_id: str = "",
    ) -> io.NodeOutput:
        if template is None:
            return io.NodeOutput(None, "No template connected.")

        template_dict = template.to_dict() if isinstance(template, SceneTemplate) else template

        slot_assignments = {"A": slot_A, "B": slot_B, "C": slot_C, "D": slot_D}
        dialogue = [dialogue_1, dialogue_2, dialogue_3, dialogue_4]

        # Explicit text overrides win; registry lookups fill in where text is empty
        def _resolve_outfit(text_override: str, outfit_id: str) -> str:
            if text_override.strip():
                return text_override
            if outfit_id.strip() and isinstance(outfit_registry, OutfitRegistry):
                return outfit_registry.get_description(outfit_id.strip())
            return text_override

        outfit_overrides = {
            "A": _resolve_outfit(outfit_override_A, outfit_A_id),
            "B": _resolve_outfit(outfit_override_B, outfit_B_id),
            "C": _resolve_outfit(outfit_override_C, outfit_C_id),
            "D": _resolve_outfit(outfit_override_D, outfit_D_id),
        }

        warnings = _validate_scene(template_dict, slot_assignments)
        instance = _compose_scene(template_dict, slot_assignments, dialogue, outfit_overrides)
        if scene_synopsis.strip():
            instance["scene_synopsis"] = scene_synopsis.strip()
        summary = _format_scene_summary(instance, validation_warnings=warnings)

        if warnings:
            send_status_update(cls.node_id, f"Composed with {len(warnings)} warning(s)", level="warn")
        else:
            send_status_update(cls.node_id, f"Composed: {instance['template_name']}")

        return io.NodeOutput(instance, summary, ui={"scene_summary": summary})


# ── Node: PromptAssemble ──────────────────────────────────────────────────────

class PromptAssemble(io.ComfyNode):
    """Assemble a model-specific prompt from a composed scene instance.

    Takes a SCENE_INSTANCE from SceneCompose and generates the complete
    prompt in the format required by the chosen model type.  Also outputs
    reference images and audio in the order expected by the model's
    conditioning nodes, and the concept IDs for downstream LoRA resolution
    via ConceptResolve.
    """
    node_id = prefixed_node_id("PromptAssemble")
    display_name = "Prompt Assemble"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                SceneInstanceIOType.Input(
                    "scene_instance",
                    display_name="Scene Instance",
                    tooltip="Composed scene from SceneCompose.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_PROMPT_MODEL_TYPES,
                    default="h3_ref2va",
                    display_name="Model Type",
                    tooltip="Target model prompt format.",
                ),
                io.String.Input(
                    "task_flags",
                    default="",
                    display_name="Task Flags (H3)",
                    tooltip=(
                        "Comma-separated H3 task type flags for the summary bracket tag. "
                        "Overrides auto-detection when non-empty. "
                        "Official types: reference generation, keyframe completion, "
                        "video editing, video continuation, audio reference, audio reuse. "
                        "Leave blank to auto-detect (reference generation when refs present, "
                        "audio reference when voice refs present)."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip=(
                        "Optional cast from SceneCastLoad or SceneCastBuild. "
                        "When connected, subjects whose bundle has visual_mode='video' "
                        "receive a <Video N> reference label in H3 prompts."
                    ),
                    optional=True,
                ),
                ConceptRegistryIOType.Input(
                    "concept_registry",
                    display_name="Concept Registry",
                    tooltip="Optional registry for trigger word lookup (future use).",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Assembled model-specific prompt text.",
                ),
                io.Image.Output(
                    "reference_images",
                    display_name="Reference Images",
                    tooltip="Character sheet images stacked in slot order (Picture 1, 2, …).",
                ),
                io.Audio.Output(
                    "reference_audio",
                    display_name="Reference Audio",
                    tooltip="First subject's voice reference audio (slot A), or None.",
                ),
                io.Audio.Output(
                    "additional_audio",
                    display_name="Additional Audio",
                    tooltip="Second subject's voice reference audio (slot B), or None.",
                ),
                io.String.Output(
                    "concept_ids",
                    display_name="Concept IDs",
                    tooltip="Comma-separated concept IDs for wiring into ConceptResolve.",
                ),
                io.String.Output(
                    "assembly_report",
                    display_name="Assembly Report",
                    tooltip="Human-readable summary of what was assembled.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_instance=None,
        model_type: str = "h3_ref2va",
        task_flags: str = "",
        scene_cast=None,
        concept_registry=None,
    ) -> io.NodeOutput:
        if scene_instance is None:
            return io.NodeOutput("", None, None, None, "", "No scene instance connected.")

        # Thread user-specified task flags into scene_instance for h3_ref2va
        if task_flags and task_flags.strip():
            flags = [f.strip() for f in task_flags.split(",") if f.strip()]
            scene_instance = {**scene_instance, "task_flags": flags}

        # Resolve video_entries from cast so subjects flagged as video get <Video N> labels
        video_entries: list[dict] = []
        if scene_cast:
            cast_media = _resolve_cast_media(scene_cast)
            video_entries = cast_media.get("video_entries_full", [])

        result = _assemble_prompt(scene_instance, model_type, video_entries or None)

        # Load reference images in declared order
        image_order = result["reference_image_order"]
        images = _load_subject_images([fname for _, fname in image_order]) if image_order else None

        # Load audio for first two slots with audio
        audio_slots = result["audio_slots"]
        assignments = scene_instance.get("slot_assignments", {})

        def _get_audio(slot_id: str) -> "dict | None":
            subject = assignments.get(slot_id)
            if subject is None:
                return None
            audio_file = subject.get("voice", {}).get("audio_reference_file", "")
            return _load_subject_audio(audio_file)

        reference_audio = _get_audio(audio_slots[0]) if len(audio_slots) > 0 else None
        additional_audio = _get_audio(audio_slots[1]) if len(audio_slots) > 1 else None

        concept_ids_str = ", ".join(result["concept_ids"])
        report = result["assembly_report"]
        prompt = result["prompt"]

        n_images = len(image_order)
        n_audio = len(audio_slots)
        send_status_update(
            cls.node_id,
            f"Assembled {model_type}: {n_images} image{'s' if n_images != 1 else ''}, "
            f"{n_audio} audio",
        )

        return io.NodeOutput(
            prompt, images, reference_audio, additional_audio,
            concept_ids_str, report,
            ui={"assembly_report": report},
        )


# ── Outfit REST API endpoints ─────────────────────────────────────────────────

















# ── Concept REST API endpoints ─────────────────────────────────────────────────





# ── Subject CRUD routes (editor-facing) ───────────────────────────────────────









# ── Reference Bundle routes ───────────────────────────────────────────────────

@routes.get("/fbtools/bundles/list")
async def _bundles_list(request):
    """Return all bundles, optionally filtered by ?subject_id=."""
    try:
        subject_id = request.rel_url.query.get("subject_id") or None
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundles = registry.list_bundles(subject_id=subject_id)
        return web.json_response({"bundles": bundles})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/bundles/get")
async def _bundles_get(request):
    """Return a single bundle by ?id=<bundle_id>."""
    bundle_id = request.rel_url.query.get("id", "")
    if not bundle_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle = registry.get(bundle_id)
        if bundle is None:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)
        return web.json_response(bundle)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Bundle video proxy cache (Plan 16) ────────────────────────────────────────
# Extends the Source Profile clip proxy system (utils/proxy_cache.py) to reference bundles'
# video reference, so generation never has to re-seek/re-decode the original source file. Three
# trigger points share this eligibility check + fire-and-forget builder: Preview (this file,
# preview_sampled below), Save (_bundles_save above... below), and the generation-time fallback
# inside _resolve_cast_media.

# Bundle proxies bake in 24fps unconditionally, same as Source Profile clips' hardcoded
# "force_rate": 24 (SourceProfileClipPrompt, "H3 requires 24fps reference video") — but a
# bundle's force_rate, unlike a clip's, is a genuinely per-bundle configurable field. Swapping in
# a 24fps-baked proxy is only correct when the bundle also targets 24fps; otherwise
# _h3_load_video_frames would resample an already-24fps file against the wrong target, so this
# guard must be checked at every trigger point before ever calling _ensure_bundle_proxy.
_BUNDLE_PROXY_SHORT_EDGE = 768


def _bundle_proxy_eligible(force_rate, duration: float) -> bool:
    try:
        force_rate = int(force_rate or 0)
    except (TypeError, ValueError):
        force_rate = 0
    return duration > 0.0 and force_rate in (0, 24)


def _fire_bundle_proxy_build(bundle_id: str, abs_path: str, start_time: float, duration: float,
                              force_rate) -> None:
    """Fire-and-forget: build (or refresh) a bundle's video proxy in a background thread.

    Never awaited by the caller — Preview/Save must stay fast, and a stale/missing proxy is never
    a hard failure (the generation-time fallback in _resolve_cast_media builds one synchronously,
    on the spot, if this hasn't finished or was never triggered). Broadcasts over the same
    fbtools.status/source="proxy_build" channel prebuild_proxies already uses, so
    js/ui/bundle_editor.js's freshness readout picks it up with the same listener pattern already
    used for Source Profile clips (source_profile_editor.js) and SceneCastBuild's clip preview.
    """
    if not bundle_id or not abs_path or not _bundle_proxy_eligible(force_rate, duration):
        return

    def _build():
        label = os.path.basename(abs_path)
        send_status_update("proxy_build", f"Building bundle proxy: {label}", source="proxy_build")
        try:
            result = _ensure_bundle_proxy(
                source_path=abs_path,
                bundle_id=bundle_id,
                start_time=start_time,
                end_time=start_time + duration,
                short_edge=_BUNDLE_PROXY_SHORT_EDGE,
                base_dir=str(user_data_dir()),
            )
            status = "ready" if result else "failed"
        except Exception as exc:
            status = f"error: {exc}"
            logger.warning("bundle proxy build failed for %r (%s): %s", bundle_id, label, exc)
        send_status_update("proxy_build", f"Bundle proxy build complete: {label} ({status})",
                            source="proxy_build")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build)


@routes.post("/fbtools/bundles/save")
async def _bundles_save(request):
    """Create or update a bundle.  Body: full bundle dict with 'id'."""
    try:
        data = await request.json()
        bundle_id = (data.get("id") or "").strip()
        if not bundle_id:
            return web.json_response({"error": "Bundle 'id' is required"}, status=400)
        path = default_bundle_registry_path()
        registry = _load_bundle_registry(path)
        registry = registry.upsert(data)
        _save_bundle_registry(registry, path)

        visual = data.get("visual") or {}
        vfile = visual.get("file", "")
        if vfile:
            vdir = visual.get("video_dir", "input")
            abs_vfile = os.path.join(
                get_output_directory() if vdir == "output" else get_input_directory(), vfile,
            )
            _fire_bundle_proxy_build(
                bundle_id, abs_vfile,
                float(visual.get("start_time", 0.0)), float(visual.get("duration", 0.0)),
                visual.get("force_rate", 24),
            )

        return web.json_response({"success": True, "id": bundle_id})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/bundles/delete")
async def _bundles_delete(request):
    """Delete a bundle by ?id=<bundle_id>."""
    bundle_id = request.rel_url.query.get("id", "")
    if not bundle_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_bundle_registry_path()
        registry = _load_bundle_registry(path)
        if registry.get(bundle_id) is None:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)
        registry = registry.delete(bundle_id)
        _save_bundle_registry(registry, path)
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Scene Cast routes ─────────────────────────────────────────────────────────









# ── Media file listing ────────────────────────────────────────────────────────

















@routes.post("/fbtools/bundles/preview_sampled")
async def _bundles_preview_sampled(request):
    """Extract sampled video frames and encode as a fragmented MP4 for browser preview.

    Body JSON: {filename, start_time, duration, force_rate, select_every_nth}
    Response: video/mp4 (fragmented) or {"error": "ffmpeg_unavailable"} 503.
    Applies the same time-accumulator resampling as CompositionToH3Conditioning so the
    user sees exactly the frames the model will receive.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    filename = (data.get("filename") or "").strip()
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    src_dir   = (data.get("dir") or "input").strip()
    input_dir = get_output_directory() if src_dir == "output" else get_input_directory()
    path = os.path.realpath(os.path.join(input_dir, filename))
    if not path.startswith(os.path.realpath(input_dir)):
        return web.json_response({"error": "Forbidden"}, status=403)
    if not os.path.isfile(path):
        return web.json_response({"error": f"Not found: {filename}"}, status=404)

    start_time       = float(data.get("start_time",        0.0))
    duration         = float(data.get("duration",          0.0))
    force_rate       = int(data.get("force_rate",          24))  # H3 requires 24fps
    select_every_nth = max(1, int(data.get("select_every_nth", 1)))

    # Plan 16: Preview already pays for equivalent ffmpeg work against these exact settings, so
    # this is the natural moment to also (re)build the bundle's cached generation-time proxy —
    # fire-and-forget, its own independent ffmpeg run, never blocks or affects this response.
    bundle_id = (data.get("bundle_id") or "").strip()
    if bundle_id:
        _fire_bundle_proxy_build(bundle_id, path, start_time, duration, force_rate)

    try:
        import cv2  # noqa: F401
    except ImportError:
        return web.json_response({"error": "ffmpeg_unavailable"}, status=503)

    import shutil as _shutil
    if not _shutil.which("ffmpeg"):
        return web.json_response({"error": "ffmpeg_unavailable"}, status=503)

    def _extract_and_encode():
        import cv2
        import subprocess

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        native_fps        = cap.get(cv2.CAP_PROP_FPS) or 24.0
        target_fps        = float(force_rate) if force_rate > 0 else native_fps
        width             = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_frame_time   = 1.0 / native_fps
        target_frame_time = 1.0 / target_fps

        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

        # Cap at 120 output frames to keep the preview response small
        PREVIEW_CAP = 120
        frame_cap = PREVIEW_CAP
        if duration > 0.0:
            output_approx = max(1, int(duration * target_fps / select_every_nth))
            frame_cap = min(frame_cap, output_approx)

        frames_raw: list = []
        time_offset = target_frame_time   # mirrors VHS cv_frame_generator init
        evaluated   = -1
        sampled     = 0

        ret, current_bgr = cap.read()
        if not ret:
            cap.release()
            return None

        while cap.isOpened():
            if time_offset < target_frame_time:
                ret, bgr = cap.read()
                if not ret:
                    break
                current_bgr = bgr
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            evaluated += 1
            if evaluated % select_every_nth != 0:
                continue
            frames_raw.append(current_bgr.tobytes())
            sampled += 1
            if sampled >= frame_cap:
                break

        cap.release()
        if not frames_raw:
            return None

        effective_fps = target_fps / select_every_nth
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(effective_fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
            "-movflags", "frag_keyframe+empty_moov+faststart",
            "-f", "mp4", "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=b"".join(frames_raw),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.warning(
                "preview_sampled: ffmpeg error: %s",
                proc.stderr.decode(errors="replace")[:500],
            )
            return None
        return proc.stdout

    loop = asyncio.get_event_loop()
    mp4_bytes = await loop.run_in_executor(None, _extract_and_encode)

    if mp4_bytes is None:
        return web.json_response({"error": "extraction failed"}, status=500)

    return web.Response(
        body=mp4_bytes,
        content_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@routes.get("/fbtools/bundles/proxy_status")
async def _bundles_proxy_status(request: web.Request) -> web.Response:
    """Return video-proxy freshness for one bundle (Plan 16).

    Query params:
        bundle_id  str

    Returns: {"fresh": bool, "proxy_path": str|null, "eligible": bool} — "eligible" is false (and
    "fresh" always false) when the bundle has no video reference, no set duration, or a
    force_rate other than 24/0, mirroring _bundle_proxy_eligible()'s own check server-side.
    """
    bundle_id = request.rel_url.query.get("bundle_id", "").strip()
    if not bundle_id:
        return web.json_response({"error": "bundle_id is required"}, status=400)

    try:
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle   = registry.get(bundle_id)
        if not bundle:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)

        visual     = bundle.get("visual", {})
        vfile      = visual.get("file", "")
        start_time = float(visual.get("start_time", 0.0))
        duration   = float(visual.get("duration", 0.0))
        force_rate = visual.get("force_rate", 24)
        eligible   = bool(vfile) and _bundle_proxy_eligible(force_rate, duration)
        if not eligible:
            return web.json_response({"fresh": False, "proxy_path": None, "eligible": False})

        vdir      = visual.get("video_dir", "input")
        video_abs = os.path.join(
            get_output_directory() if vdir == "output" else get_input_directory(), vfile,
        )

        from .utils.proxy_cache import _proxy_dir, _proxy_stem, _is_fresh
        stem       = _proxy_stem(bundle_id, "video", start_time, start_time + duration, _BUNDLE_PROXY_SHORT_EDGE)
        proxy_path = _proxy_dir(str(user_data_dir()), "bundles") / f"{stem}.mp4"
        fresh      = _is_fresh(proxy_path, video_abs) if os.path.exists(video_abs) else False

        return web.json_response({
            "fresh":      fresh,
            "proxy_path": str(proxy_path) if fresh else None,
            "eligible":   True,
        })
    except Exception as exc:
        logger.exception("bundle proxy_status failed for %r", bundle_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/bundles/preprocess_audio")
async def _bundles_preprocess_audio(request):
    """Apply the audio preprocessing pipeline to a bundle's audio source and cache the result.

    Body JSON:
        bundle_id       str      — used for the cache directory name
        filename        str      — audio/video filename
        dir             str      — "input" or "output" (default "input")
        start_time      float    — trim start (seconds); ignored if audio_source == "extract_from_visual"
        duration        float    — trim duration; 0 = to end
        audio_processing dict:
            noise_removal     bool   — spectral denoising (scipy)
            normalize_lufs    bool   — LUFS normalize (pyloudnorm)
            target_lufs       float  — target integrated loudness (default -14)

    Response: {cache_path, duration, lufs_before, lufs_after, fingerprint}
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    bundle_id = (data.get("bundle_id") or "default").strip()
    filename  = (data.get("filename") or "").strip()
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    dir_hint = (data.get("dir") or "input").strip()
    base_dir = get_output_directory() if dir_hint == "output" else get_input_directory()
    src_path = os.path.realpath(os.path.join(base_dir, filename))
    if not src_path.startswith(os.path.realpath(base_dir)):
        return web.json_response({"error": "Forbidden"}, status=403)
    if not os.path.isfile(src_path):
        return web.json_response({"error": f"Not found: {filename}"}, status=404)

    start_time = float(data.get("start_time", 0.0))
    duration   = float(data.get("duration",   0.0))
    proc_cfg   = data.get("audio_processing", {})
    noise_removal  = bool(proc_cfg.get("noise_removal",  False))
    normalize_lufs = bool(proc_cfg.get("normalize_lufs", True))
    target_lufs    = float(proc_cfg.get("target_lufs",  -14.0))

    # Resolve MelBand Roformer model path from saved settings
    melband_path: str | None = None
    melband_reason = ""
    if noise_removal:
        settings     = _read_composition_settings()
        melband_raw  = settings.get("melband_model_path", "").strip()
        if melband_raw:
            if os.path.isabs(melband_raw) and os.path.isfile(melband_raw):
                melband_path = melband_raw
            else:
                resolved = folder_paths.get_full_path("diffusion_models", melband_raw)
                if resolved and os.path.isfile(resolved):
                    melband_path = resolved
            if not melband_path:
                melband_reason = f"configured MelBand path {melband_raw!r} not found under diffusion_models"
        else:
            melband_reason = "no MelBand model path configured in Settings"

    # denoise_method reports what actually ran, independent of from_cache — a cache
    # hit is only served when this same (noise_removal, melband_path) combination
    # produced it (see cache_fingerprint below), so it's always accurate here too.
    denoise_method = "melband" if melband_path else ("spectral_fallback" if noise_removal else "none")
    logger.info(
        "preprocess_audio: %s — denoise=%s%s, normalize_lufs=%s, target_lufs=%.1f",
        os.path.basename(src_path), denoise_method,
        f" ({melband_reason})" if melband_reason else "",
        normalize_lufs, target_lufs,
    )

    from .utils.audio_preprocess import preprocess_audio, cache_fingerprint, measure_lufs

    fp = cache_fingerprint(src_path, start_time, duration, {
        "noise_removal":  noise_removal,
        "normalize_lufs": normalize_lufs,
        "target_lufs":    target_lufs,
        "melband_path":   melband_path or "",
    })

    # Cache dir: user_data_dir/bundles_cache/<bundle_id>/
    safe_bid  = "".join(c if c.isalnum() or c in "-_" else "_" for c in bundle_id)[:64]
    cache_dir = os.path.join(user_data_dir(), "bundles_cache", safe_bid)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"audio_{fp}.wav")

    if os.path.isfile(cache_path):
        # Re-measure from cached file for the response metrics
        def _measure_cached():
            import torchaudio
            wf, sr = torchaudio.load(cache_path)
            import numpy as np
            audio_np = wf.numpy()
            lufs = measure_lufs(audio_np, sr)
            dur = audio_np.shape[-1] / sr if sr > 0 else 0.0
            return {"duration": round(dur, 2), "lufs_before": None,
                    "lufs_after": round(lufs, 1) if np.isfinite(lufs) else None}
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, _measure_cached)
        return web.json_response({
            "cache_path": cache_path,
            "fingerprint": fp,
            "from_cache": True,
            "denoise_method": denoise_method,
            "denoise_reason": melband_reason,
            **metrics,
        })

    def _run_pipeline():
        raw = _h3_load_audio(src_path, start_time, duration)
        if raw is None:
            return None, None
        wf_out, sr_out, metrics = preprocess_audio(
            raw["waveform"], raw["sample_rate"],
            noise_removal=noise_removal,
            normalize_lufs=normalize_lufs,
            target_lufs=target_lufs,
            melband_model_path=melband_path,
        )
        import torchaudio
        torchaudio.save(cache_path, wf_out.squeeze(0), sr_out)
        return metrics, cache_path

    loop = asyncio.get_event_loop()
    try:
        metrics, out_path = await loop.run_in_executor(None, _run_pipeline)
    except Exception as exc:
        logger.error("preprocess_audio: pipeline error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)

    if out_path is None:
        return web.json_response({"error": "audio load failed — check ffmpeg and filename"}, status=500)

    return web.json_response({
        "cache_path":  cache_path,
        "fingerprint": fp,
        "from_cache":  False,
        "denoise_method": denoise_method,
        "denoise_reason": melband_reason,
        **(metrics or {}),
    })


@routes.get("/fbtools/bundles/audio_cache/stream")
async def _bundles_audio_cache_stream(request):
    """Stream a processed audio cache file.

    ?path=<absolute_path>  — must be inside user_data_dir()/bundles_cache/.
    Supports HTTP Range requests so browsers can seek.
    """
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="path required")
    allowed_root = os.path.realpath(os.path.join(user_data_dir(), "bundles_cache"))
    real_path = os.path.realpath(path)
    if not real_path.startswith(allowed_root + os.sep):
        return web.Response(status=403, text="Forbidden")
    if not os.path.isfile(real_path):
        return web.Response(status=404, text="Not found")
    return web.FileResponse(real_path)


# ══════════════════════════════════════════════════════════════════════════════
# Scene Cast nodes  (Reference Bundle & Scene Cast system)
# ══════════════════════════════════════════════════════════════════════════════

# CastIOType/H3RefplanType moved to nodes/composition_types.py (Plan 27)
# ── Scene Cast helpers ────────────────────────────────────────────────────────

def _cast_get_ids() -> list[str]:
    """Return available cast IDs for combo population at schema time."""
    try:
        registry = _load_cast_registry(default_cast_registry_path())
        ids = registry.cast_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: SceneCastLoad ───────────────────────────────────────────────────────

class SceneCastLoad(io.ComfyNode):
    """Load a Scene Cast from disk.

    A Scene Cast assigns a reference bundle to each subject in a composition,
    with per-entry visual mode and audio toggles.  Wire the SCENE_CAST output
    into PromptCompositionLoader to resolve reference media during assembly.
    The combo is populated at extension load time; press R to refresh after
    saving a new cast in the Scene Casts sidebar panel.
    """
    node_id = prefixed_node_id("SceneCastLoad")
    display_name = "Scene Cast Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        cast_ids = _cast_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "cast_id",
                    options=cast_ids,
                    display_name="Cast ID",
                    tooltip="Scene cast to load. Press R to refresh after saving a new cast.",
                ),
            ],
            outputs=[
                CastIOType.Output(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Cast dict for wiring into PromptCompositionLoader.",
                ),
                io.String.Output(
                    "cast_summary",
                    display_name="Cast Summary",
                    tooltip="Human-readable summary of the cast entries.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cast_id: str = "", **_):
        path = default_cast_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, cast_id, mtime, reload_counter("cast"))

    @classmethod
    def execute(cls, cast_id: str = "") -> io.NodeOutput:
        if not cast_id or cast_id == "(none)":
            logger.warning("SceneCastLoad: no cast_id selected")
            return io.NodeOutput(None, "")

        path = default_cast_registry_path()
        registry = _load_cast_registry(path)
        cast = registry.get(cast_id)

        if cast is None:
            logger.warning("SceneCastLoad: cast_id %r not found in %s", cast_id, path)
            return io.NodeOutput(None, f"Cast not found: {cast_id}")

        entries = cast.get("entries", [])
        n = len(entries)
        lines = [f"Cast: {cast.get('name', cast_id)}  ({n} {'subject' if n == 1 else 'subjects'})"]
        for e in entries:
            mode = e.get("visual_mode", "images")
            audio_flag = " + audio" if e.get("use_audio") else ""
            lines.append(
                f"  • {e.get('subject_id', '?')} → {e.get('bundle_id', '?')} [{mode}{audio_flag}]"
            )

        summary = "\n".join(lines)
        send_status_update(
            cls.node_id,
            f"Loaded cast: {cast.get('name', cast_id)} | {n} {'entry' if n == 1 else 'entries'}",
        )
        return io.NodeOutput(cast, summary, ui={"cast_summary": summary})


# ── Node: SceneCastBuild ──────────────────────────────────────────────────────

class SceneCastBuild(io.ComfyNode):
    """Build a Scene Cast inline, with an optional Source Profile input.

    Accepts standalone subject → bundle assignments alongside source-derived
    subjects from a connected SourceProfileLoad node.  Assignments and retention
    modes are encoded in cast_entries_json (managed by the Scene Casts sidebar).

    Each entry may be:
      • Bundle-backed  — subject_id + bundle_id
      • Source-derived — subject_id + source_profile_id + source_subject_id
      • Hybrid         — bundle + source (bundle provides media, source provides video ref)

    Outputs the same SCENE_CAST type as SceneCastLoad, plus source_profile and
    clip_id pass-throughs so SourceProfileClipPrompt can be wired downstream
    without re-connecting the profile.
    """

    node_id = prefixed_node_id("SceneCastBuild")
    display_name = "Scene Cast Build"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "cast_entries_json",
                    display_name="Cast Entries",
                    default="[]",
                    tooltip=(
                        "JSON array of cast entries. "
                        "Managed via the Scene Casts sidebar — do not edit by hand."
                    ),
                ),
                SourceProfileIOType.Input(
                    "source_profile",
                    display_name="Source Profile",
                    optional=True,
                    tooltip="Source profile whose subjects are available as cast pool options.",
                ),
                CompositionIOType.Input(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    optional=True,
                    tooltip=(
                        "Prompt Composition whose subjects are the cast pool. Use this "
                        "instead of a Source Profile; if both are connected the Source "
                        "Profile drives the node and this input is ignored."
                    ),
                ),
                io.String.Input(
                    "clip_id",
                    display_name="Clip ID",
                    default="",
                    tooltip=(
                        "Optional clip ID from the Source Profile. "
                        "When set, only frames within that clip's time window are loaded."
                    ),
                ),
                io.Int.Input(
                    "clip_duration_multiplier",
                    display_name="Clip Duration Multiplier",
                    default=1,
                    min=1,
                    max=4,
                    tooltip=(
                        "Duration multiplier (1x–4x) managed by the timeline UI. "
                        "Wire to SourceProfileClipPrompt to scale the output frame count."
                    ),
                ),
                io.String.Input(
                    "composition_overrides_json",
                    display_name="Composition Overrides",
                    default="{}",
                    optional=True,
                    tooltip=(
                        "JSON object managed by the on-node Composition options block: "
                        "background id and background-as-reference overrides applied on top "
                        "of the connected Prompt Composition. Empty = use the composition's "
                        "own values. Do not edit by hand."
                    ),
                ),
                io.String.Input(
                    "action_preview",
                    display_name="Action Preview",
                    default="",
                    optional=True,
                    multiline=True,
                    tooltip=(
                        "Read-only. The active clip's action text with {A}/{B}/… "
                        "placeholders resolved to bundle names — computed and written "
                        "by the on-node preview widget so it rides along in the "
                        "submitted prompt for Run History tracking. Not read by "
                        "execute(); do not edit by hand."
                    ),
                ),
            ],
            outputs=[
                CastIOType.Output(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Inline cast dict for wiring into PromptCompositionLoader.",
                ),
                io.String.Output(
                    "cast_summary",
                    display_name="Cast Summary",
                    tooltip="Human-readable summary of configured entries.",
                ),
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Pass-through of the connected Source Profile (for SourceProfileClipPrompt).",
                ),
                io.String.Output(
                    "clip_id",
                    display_name="Clip ID",
                    tooltip="Pass-through of the selected Clip ID (for SourceProfileClipPrompt).",
                ),
                io.Int.Output(
                    "clip_duration_multiplier",
                    display_name="Duration Multiplier",
                    tooltip=(
                        "Pass-through of the duration multiplier set in the timeline UI (1–4). "
                        "Wire to SourceProfileClipPrompt clip_duration_multiplier input."
                    ),
                ),
                CompositionIOType.Output(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip="Pass-through of the connected Prompt Composition.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        cast_entries_json: str = "[]",
        source_profile=None,
        clip_id: str = "",
        clip_duration_multiplier: int = 1,
        prompt_composition=None,
        composition_overrides_json: str = "{}",
        **_,
    ):
        bundle_mtime = subject_mtime = source_mtime = 0
        try:
            bundle_mtime = os.path.getmtime(default_bundle_registry_path())
        except OSError:
            pass
        try:
            subject_mtime = os.path.getmtime(
                os.path.join(user_data_dir(), "subject_profiles.json")
            )
        except OSError:
            pass
        try:
            source_mtime = os.path.getmtime(default_source_profiles_path())
        except OSError:
            pass
        sp_id = source_profile.get("id", "") if isinstance(source_profile, dict) else ""
        pc_id = prompt_composition.get("id", "") if isinstance(prompt_composition, dict) else ""
        pc_subjects = json.dumps(prompt_composition.get("subjects", {}), sort_keys=True) if isinstance(prompt_composition, dict) else ""
        return (bundle_mtime, subject_mtime, source_mtime, cast_entries_json, clip_id, sp_id,
                clip_duration_multiplier, pc_id, pc_subjects, composition_overrides_json)

    @classmethod
    def execute(
        cls,
        cast_entries_json: str = "[]",
        source_profile=None,
        clip_id: str = "",
        clip_duration_multiplier: int = 1,
        prompt_composition=None,
        composition_overrides_json: str = "{}",
        **_,
    ) -> io.NodeOutput:
        try:
            raw = json.loads(cast_entries_json or "[]")
            if not isinstance(raw, list):
                raw = []
        except Exception:
            raw = []

        # Build connected source profiles dict: profile_id → profile dict
        connected_profiles: dict = {}
        if isinstance(source_profile, dict):
            pid = source_profile.get("id", "")
            if pid:
                connected_profiles[pid] = source_profile

        # Composition-driven cast: the composition's own subjects are the pool
        # and ordinal entries resolve against them. A Source Profile, when also
        # connected, keeps driving the node (the UI hides the composition then).
        comp_roster = None
        if isinstance(prompt_composition, dict):
            if isinstance(source_profile, dict):
                logger.warning(
                    "SceneCastBuild: both source_profile and prompt_composition are connected; "
                    "using the Source Profile and ignoring the Prompt Composition."
                )
            else:
                comp_roster = _composition_ordinal_roster(
                    prompt_composition,
                    _load_subject_registry(default_subject_profiles_path()).get_subject,
                )

        _RETENTION_BUNDLE = "fully_preserved"
        _RETENTION_SOURCE = "partially_preserved"

        # Bundle/subject registries — only needed to resolve a bundle's own
        # pronoun_style for ordinal-match entries; loaded once up front.
        _ordinal_entries_present = any(str(e.get("match_mode", "")).strip() == "ordinal" for e in raw)
        bundle_reg = subject_reg = None
        if _ordinal_entries_present:
            try:
                bundle_reg = _load_bundle_registry(default_bundle_registry_path())
            except Exception:
                bundle_reg = None
            try:
                subject_reg = _load_subject_registry(default_subject_profiles_path())
            except Exception:
                subject_reg = None

        entries = []
        for e in raw:
            subject_id        = str(e.get("subject_id",        "")).strip()
            source_profile_id = str(e.get("source_profile_id", "")).strip()
            source_subject_id = str(e.get("source_subject_id", "")).strip()
            bundle_id         = str(e.get("bundle_id",         "")).strip()
            retention         = str(e.get("retention",         "")).strip()

            # Ordinal match: resolve source_subject_id fresh against whichever
            # profile/clip is *currently connected* — this node only ever has
            # one source_profile input, so there is nothing to disambiguate
            # and no reason to gate on a previously-stored source_profile_id,
            # which goes stale the moment the upstream Source Profile is
            # swapped for a different one (unlike an explicit subject pick
            # below, which legitimately should invalidate if its specific
            # profile disconnects). No match (or nothing to resolve against)
            # → clear source linkage entirely, falling through to the
            # bundle-only branch below, exactly as if no source had ever been
            # assigned for this subject.
            if (comp_roster is not None
                    and str(e.get("match_mode", "")).strip() == "ordinal" and bundle_id):
                # Composition ordinal: "the Nth subject in this composition sharing
                # my bundle's pronoun_style" becomes the entry's subject_id, and the
                # entry stays a plain bundle-backed one (no source linkage).
                _bundle = bundle_reg.get(bundle_id) if bundle_reg else None
                _resolved = ""
                if _bundle is not None:
                    _bun_subj = (subject_reg.get_subject(_bundle.get("subject_id", ""))
                                 if subject_reg and _bundle.get("subject_id") else None)
                    _pronoun = _sp_resolved_pronoun_style(
                        _bundle.get("entity_type", "person"),
                        (_bun_subj.get("pronoun_style") if _bun_subj else "") or _bundle.get("pronoun_style", ""),
                    )
                    try:
                        _n = int(e.get("ordinal", 0))
                    except (TypeError, ValueError):
                        _n = 0
                    _resolved = _sp_resolve_ordinal_from_list(comp_roster, _pronoun, _n, id_key="subject_id")
                if not _resolved:
                    logger.warning(
                        "SceneCastBuild: ordinal entry for bundle %r matched no subject in the composition, skipping",
                        bundle_id,
                    )
                    continue
                subject_id = _resolved
                source_profile_id = source_subject_id = ""
            elif (str(e.get("match_mode", "")).strip() == "ordinal"
                    and bundle_id and not source_subject_id):
                profile = source_profile if isinstance(source_profile, dict) else None
                bundle  = bundle_reg.get(bundle_id) if bundle_reg else None
                if profile is not None and bundle is not None and clip_id:
                    bun_subj = (subject_reg.get_subject(bundle.get("subject_id", ""))
                                if subject_reg and bundle.get("subject_id") else None)
                    bun_pronoun = _sp_resolved_pronoun_style(
                        bundle.get("entity_type", "person"),
                        (bun_subj.get("pronoun_style") if bun_subj else "") or bundle.get("pronoun_style", ""),
                    )
                    try:
                        ordinal = int(e.get("ordinal", 0))
                    except (TypeError, ValueError):
                        ordinal = 0
                    source_subject_id = _sp_resolve_ordinal_subject(profile, clip_id, bun_pronoun, ordinal)
                    if source_subject_id:
                        source_profile_id = profile.get("id", "")  # always the live connected profile
                if not source_subject_id:
                    source_profile_id = ""  # no match — treat as if never assigned

            if source_profile_id and source_subject_id:
                # Resolve source subject (shared by source-only and hybrid paths)
                profile = connected_profiles.get(source_profile_id)
                if profile is None:
                    logger.warning(
                        "SceneCastBuild: source_profile_id %r not connected, skipping",
                        source_profile_id,
                    )
                    continue
                subject_entry = next(
                    (s for s in profile.get("subjects", [])
                     if s.get("id") == source_subject_id),
                    None,
                )
                if subject_entry is None:
                    logger.warning(
                        "SceneCastBuild: subject %r not found in profile %r, skipping",
                        source_subject_id, source_profile_id,
                    )
                    continue
                src_fields = {
                    "source_profile_id": source_profile_id,
                    "source_subject_id": source_subject_id,
                    "role_description":  subject_entry.get("role_description", ""),
                    "entity_type":       subject_entry.get("entity_type", "person"),
                    "source_media_file": profile.get("media_filename", ""),
                    "source_media_dir":  profile.get("media_dir", "input"),
                    "source_media_type": profile.get("media_type", "video"),
                }

                if bundle_id:
                    # Hybrid: bundle provides appearance/media, source provides video reference
                    visual_mode    = e.get("visual_mode", "images")
                    use_audio      = bool(e.get("use_audio", False))
                    image_selection = e.get("image_selection")
                    entries.append({
                        "subject_id":      subject_id or source_subject_id,
                        "bundle_id":       bundle_id,
                        "visual_mode":     visual_mode if visual_mode in ("images", "video", "both") else "images",
                        "use_audio":       use_audio,
                        "image_selection": image_selection,
                        "retention":       retention or _RETENTION_SOURCE,
                        "dialogue":        str(e.get("dialogue", "") or "").strip(),
                        **src_fields,
                    })
                else:
                    # Source-only: no bundle
                    entries.append({
                        "subject_id": subject_id or source_subject_id,
                        "retention":  retention or _RETENTION_SOURCE,
                        "dialogue":   str(e.get("dialogue", "") or "").strip(),
                        **src_fields,
                    })

            elif subject_id and bundle_id:
                # Bundle-only: no source video reference
                visual_mode     = e.get("visual_mode", "images")
                use_audio       = bool(e.get("use_audio", False))
                image_selection = e.get("image_selection")
                entries.append({
                    "subject_id":      subject_id,
                    "bundle_id":       bundle_id,
                    "visual_mode":     visual_mode if visual_mode in ("images", "video", "both") else "images",
                    "use_audio":       use_audio,
                    "image_selection": image_selection,
                    "retention":       retention or _RETENTION_BUNDLE,
                    "dialogue":        str(e.get("dialogue", "") or "").strip(),
                })

        # Flag (never silently resolve) two or more entries landing on the same
        # source subject — most likely two ordinal entries whose bundles share
        # a pronoun_style classification and whose ordinals don't actually pick
        # out distinct subjects in this clip, or an ordinal entry colliding
        # with an explicit one. Only one bundle can really replace a given
        # subject; whichever entry appears last in cast_entries_json is the one
        # whose bundle_id/visual settings "win" for that subject downstream,
        # but neither claim is intentional here, so surface it loudly.
        _claims: dict[str, list[str]] = {}
        for entry in entries:
            sid = entry.get("source_subject_id")
            if sid:
                _claims.setdefault(sid, []).append(entry.get("subject_id", "?"))
        for sid, claimants in _claims.items():
            if len(claimants) > 1:
                logger.warning(
                    "SceneCastBuild: %d cast entries (%s) all resolved to the same source "
                    "subject %r for clip %r — check for an ordinal/explicit assignment "
                    "conflict (e.g. two bundles sharing a pronoun_style whose ordinals "
                    "don't pick out distinct subjects in this clip).",
                    len(claimants), ", ".join(claimants), sid, clip_id,
                )

        # Map profile_id → clip_id
        clip_ids: dict = {}
        if isinstance(source_profile, dict) and clip_id and clip_id.strip():
            pid = source_profile.get("id", "")
            if pid:
                clip_ids[pid] = clip_id.strip()

        cast = {
            "id":              "_inline",
            "name":            "_inline",
            "entries":         entries,
            "source_profiles": connected_profiles,
            "clip_ids":        clip_ids,
        }
        # Per-run overrides for the connected composition (background etc.).
        # Ignored in Source Profile mode, where the composition is not the driver.
        if isinstance(prompt_composition, dict) and prompt_composition and not connected_profiles:
            try:
                ov = json.loads(composition_overrides_json or "{}")
            except Exception:
                ov = {}
            if isinstance(ov, dict) and ov:
                cast["composition_overrides"] = ov

        n = len(entries)
        has_sp = bool(connected_profiles)
        lines = [f"Inline cast  ({n} {'subject' if n == 1 else 'subjects'}"
                 + (" + source profile" if has_sp else "") + ")"]
        for e in entries:
            ret_str = f" [{e.get('retention', '')}]" if e.get("retention") else ""
            if e.get("source_profile_id") and e.get("bundle_id"):
                # Hybrid
                audio_flag = " + audio" if e.get("use_audio") else ""
                sp_subj = e.get("source_subject_id", "?")
                lines.append(
                    f"  • {e['subject_id']} [{e['bundle_id']},"
                    f" {e['visual_mode']}{audio_flag}] + src:{sp_subj}{ret_str}"
                )
            elif e.get("source_profile_id"):
                # Source-only
                etype = e.get("entity_type", "?")
                role  = e.get("role_description", e.get("subject_id", "?"))
                lines.append(f"  • [{etype}] {role}{ret_str}")
            else:
                # Bundle-only
                audio_flag = " + audio" if e.get("use_audio") else ""
                lines.append(
                    f"  • {e['subject_id']} → {e['bundle_id']} [{e['visual_mode']}{audio_flag}]{ret_str}"
                )
        summary = "\n".join(lines)

        send_status_update(
            cls.hidden.unique_id,
            f"Inline cast: {n} {'entry' if n == 1 else 'entries'}"
            + (" | source profile" if has_sp else ""),
        )
        mult = max(1, int(clip_duration_multiplier or 1))
        return io.NodeOutput(cast, summary, source_profile or {}, clip_id or "", mult, prompt_composition or {})


# ── Scene Cast reload endpoint ────────────────────────────────────────────────




# ── Prompt Composition routes ─────────────────────────────────────────────────

from .utils.prompt_compositions import (
    list_compositions as _list_compositions,
    load_composition as _load_composition,
    composition_ordinal_roster as _composition_ordinal_roster,
    save_composition as _save_composition,
    delete_composition as _delete_composition,
    resolve_subjects as _resolve_composition_subjects,
    resolve_background as _resolve_composition_background,
    validate_composition as _validate_composition,
    apply_cast_to_subjects as _apply_cast_to_subjects,
    apply_composition_overrides as _apply_composition_overrides,
)
from .utils.composition_resources import (
    list_backgrounds as _list_backgrounds,
    get_background as _get_background,
    save_background as _save_background,
    delete_background as _delete_background,
    list_camera_presets as _list_camera_presets,
    save_camera_preset as _save_camera_preset,
    delete_camera_preset as _delete_camera_preset,
    list_sound_presets as _list_sound_presets,
    save_sound_preset as _save_sound_preset,
    delete_sound_preset as _delete_sound_preset,
    load_backgrounds as _load_backgrounds_dict,
)
from .utils.prompt_assembler import assemble_composition as _assemble_composition
from .utils.prompt_assembler import _build_h3_refplan
from .utils.prompt_assembler import estimate_speech_duration as _estimate_speech_duration
from .utils.prompt_assembler import _PACE_CHARS_PER_SEC
from .utils.prompt_assembler import (
    validate_h3_refs_pre     as _validate_h3_refs_pre,
    validate_h3_audio_clip   as _validate_h3_audio_clip,
    validate_h3_audio_total  as _validate_h3_audio_total,
)
from .utils.prompt_assembler import _build_ref_map as _pa_build_ref_map


@routes.get("/fbtools/compositions/list")
async def _compositions_list(request):
    try:
        items = _list_compositions(user_data_dir())
        return web.json_response({"compositions": items})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/compositions/get")
async def _compositions_get(request):
    cid = request.rel_url.query.get("id", "")
    if not cid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        composition = _load_composition(user_data_dir(), cid)
        return web.json_response(composition)
    except FileNotFoundError:
        return web.json_response({"error": f"Composition '{cid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/save")
async def _compositions_save(request):
    try:
        composition = await request.json()
        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        saved = _save_composition(user_data_dir(), composition, registry, backgrounds)
        return web.json_response({"success": True, "id": saved["id"]})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/compositions/delete")
async def _compositions_delete(request):
    cid = request.rel_url.query.get("id", "")
    if not cid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        _delete_composition(user_data_dir(), cid)
        return web.json_response({"success": True})
    except FileNotFoundError:
        return web.json_response({"error": f"Composition '{cid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/assemble")
async def _compositions_assemble(request):
    """Assemble a prompt from a composition.

    Body: {"scene_id": "...", "model_type": "..."} or
          {"composition": {...}, "model_type": "..."}  (inline, unsaved)
    """
    try:
        body = await request.json()
        model_type = body.get("model_type", "h3_ref2va")

        if "scene_id" in body:
            composition = _load_composition(user_data_dir(), body["scene_id"])
        elif "composition" in body:
            composition = body["composition"]
        else:
            return web.json_response({"error": "scene_id or composition required"}, status=400)

        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        outfit_reg = _load_outfit_registry(default_outfit_registry_path())

        resolved_subjects = _resolve_composition_subjects(composition, registry)
        resolved_background = _resolve_composition_background(composition, backgrounds)

        outfit_ids = composition.get("outfit_ids", {})
        resolved_outfits = {
            sk: outfit_reg.get_outfit(oid)
            for sk, oid in outfit_ids.items()
            if oid and outfit_reg.get_outfit(oid)
        }

        warnings = _validate_composition(composition)
        result = _assemble_composition(
            composition, resolved_subjects, resolved_background, model_type,
            resolved_outfits=resolved_outfits,
        )
        result["warnings"] = warnings
        return web.json_response(result)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Composition assembly failed")
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/reload")
async def _compositions_reload(request):
    """Increment reload counter so PromptCompositionLoader nodes re-execute."""
    _composition_reload_counter = bump_reload("composition")
    logger.info("Composition reload requested (counter=%d)", _composition_reload_counter)
    return web.json_response({"success": True, "counter": _composition_reload_counter})


# ── Composition system settings ───────────────────────────────────────────────

def _composition_settings_path() -> str:
    return os.path.join(user_data_dir(), "composition_settings.json")


_COMPOSITION_SETTINGS_DEFAULTS: dict = {
    "libber_delimiter":           "%",
    "libber_max_depth":           10,
    "default_speech_pace":        "normal",
    "default_audio_noise_removal":  False,
    "default_audio_normalize_lufs": True,
    "default_audio_target_lufs":   -14.0,
    "melband_model_path":          "",  # Kijai/MelBandRoFormer_comfy — fp16 or fp32 .safetensors
    # H3 model output limits
    "h3_max_frames":               360,  # 15 s × 24 fps — 0 = unclamped
}


def _read_composition_settings() -> dict:
    defaults = dict(_COMPOSITION_SETTINGS_DEFAULTS)
    path = _composition_settings_path()
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {**defaults, **data}
    except Exception:
        return defaults


def _write_composition_settings(settings: dict) -> None:
    path = _composition_settings_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


@routes.get("/fbtools/compositions/settings")
async def _compositions_settings_get(request):
    """Return global composition system settings."""
    return web.json_response(_read_composition_settings())


@routes.post("/fbtools/compositions/settings")
async def _compositions_settings_post(request):
    """Update global composition system settings."""
    try:
        body = await request.json()
        settings = _read_composition_settings()

        if "libber_delimiter" in body:
            d = str(body["libber_delimiter"])
            if len(d) == 1:
                settings["libber_delimiter"] = d

        if "libber_max_depth" in body:
            v = int(body["libber_max_depth"])
            settings["libber_max_depth"] = max(1, min(50, v))

        if "default_speech_pace" in body:
            pace = str(body["default_speech_pace"])
            if pace in ("slow", "normal", "fast"):
                settings["default_speech_pace"] = pace

        if "default_audio_noise_removal" in body:
            settings["default_audio_noise_removal"] = bool(body["default_audio_noise_removal"])

        if "default_audio_normalize_lufs" in body:
            settings["default_audio_normalize_lufs"] = bool(body["default_audio_normalize_lufs"])

        if "default_audio_target_lufs" in body:
            v = float(body["default_audio_target_lufs"])
            settings["default_audio_target_lufs"] = max(-36.0, min(-6.0, v))

        if "melband_model_path" in body:
            settings["melband_model_path"] = str(body["melband_model_path"]).strip()

        if "h3_max_frames" in body:
            v = int(body["h3_max_frames"])
            settings["h3_max_frames"] = max(0, min(9999, v))

        _write_composition_settings(settings)
        return web.json_response(settings)
    except Exception as exc:
        logger.exception("Error saving composition settings")
        return web.json_response({"error": str(exc)}, status=500)


# ── LLM assistant routes ──────────────────────────────────────────────────────

from .utils import unsloth_client as _unsloth_client
from .utils import modal_deploy as _modal_deploy
import asyncio

# ── Unsloth client startup configuration ──────────────────────────────────────
# Resolve workspace and stored API key at import time so the client is ready
# the moment the user opens the LLM panel, without any extra button clicks.
try:
    _unsloth_workspace = _modal_deploy.get_workspace() or ""
    _unsloth_api_key   = _modal_deploy.load_api_key(user_data_dir()) or ""
    _unsloth_client.configure(workspace=_unsloth_workspace, api_key=_unsloth_api_key)
    if _unsloth_workspace:
        logger.info("Unsloth client configured: workspace=%s key=%s",
                    _unsloth_workspace, "stored" if _unsloth_api_key else "not found")
except Exception as _exc:
    logger.debug("Unsloth startup configure skipped: %s", _exc)














# ── Active-backend routing ─────────────────────────────────────────────────────
# All inference calls go through _route_llm(). The active backend is whichever
# one the user last activated in the LLM panel — exactly one at a time.
# Activate endpoints enforce mutual exclusion (Unsloth ↔ Modal).

















# ── Unified LLM history ────────────────────────────────────────────────────────
# Supersedes the old video_describe_history.json (migrated on first read).
















# ── Modal backend control routes ──────────────────────────────────────────────











# ── Unsloth Studio routes ─────────────────────────────────────────────────────





























# ── VLM activity log routes ────────────────────────────────────────────────────







# ── Background routes ─────────────────────────────────────────────────────────











# ── Preset routes ─────────────────────────────────────────────────────────────













# ── Node: PromptCompositionLoader ─────────────────────────────────────────────

_COMP_MODEL_TYPE_OPTIONS = ["composition default"] + list(_PROMPT_MODEL_TYPES)


def _composition_get_names() -> list[str]:
    """Return saved composition names for the combo widget, sorted by name."""
    try:
        items = _list_compositions(user_data_dir())
        names = [c["name"] for c in items]
        return names if names else ["(none)"]
    except Exception:
        return ["(none)"]


# CompositionIOType moved to nodes/composition_types.py (Plan 27)
class CompositionLoad(io.ComfyNode):
    """Load a saved Prompt Composition and expose it for wiring into SceneCastBuild.

    Mirrors SourceProfileLoad: the composition's subjects become the cast pool
    in SceneCastBuild, and changing the selection here refreshes it downstream.
    Assembling a prompt is still PromptCompositionLoader's job.
    """
    node_id = prefixed_node_id("CompositionLoad")
    display_name = "Composition Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "composition_name",
                    options=_composition_get_names(),
                    display_name="Composition",
                    tooltip="Saved Prompt Composition to load. Press R to refresh the list after adding new compositions.",
                ),
            ],
            outputs=[
                CompositionIOType.Output(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip="Full composition dict for wiring into SceneCastBuild.",
                ),
                io.String.Output(
                    "subject_info",
                    display_name="Subject Info",
                    tooltip="Slot letter, subject and pronoun style for each subject in this composition.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, composition_name: str = "", **_):
        comps_dir = os.path.join(user_data_dir(), "prompt_compositions")
        try:
            dir_mtime = os.path.getmtime(comps_dir)
        except OSError:
            dir_mtime = 0
        file_mtime = 0
        if composition_name:
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == composition_name), None)
            if matched:
                try:
                    file_mtime = os.path.getmtime(os.path.join(comps_dir, f"{matched['id']}.json"))
                except OSError:
                    pass
        return (comps_dir, composition_name, dir_mtime, file_mtime, reload_counter("composition"))

    @classmethod
    def execute(cls, composition_name: str = "") -> io.NodeOutput:
        matched = None
        if composition_name and composition_name != "(none)":
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == composition_name), None)
        if matched is None:
            logger.warning("CompositionLoad: composition %r not found", composition_name)
            return io.NodeOutput(None, "")

        composition = _load_composition(user_data_dir(), matched["id"])
        registry = _load_subject_registry(default_subject_profiles_path())
        lines = []
        for slot, sid in composition.get("subjects", {}).items():
            subj = registry.get_subject(sid) if sid else None
            name = (subj or {}).get("name") or sid or "(empty)"
            pronoun = (subj or {}).get("pronoun_style", "")
            lines.append(f"{slot}: {name}" + (f" [{pronoun}]" if pronoun else ""))
        subject_info = "\n".join(lines) if lines else "(no subjects defined)"

        send_status_update(cls.node_id, f"Loaded: {composition_name} | {len(lines)} subject(s)")
        return io.NodeOutput(composition, subject_info)


def _resolve_cast_media(
    scene_cast: dict,
    bundle_registry,
) -> "dict":
    """Extract reference media and frame-sampling params from a scene cast dict.

    Iterates entries in cast order.  Returns a dict with:
      reference_video    – absolute path of first video-mode entry's file, or "".
      reference_images   – [N,H,W,3] tensor from all image-mode entries, or None.
      video_force_rate   – fps override for the visual Load Video node.
      video_frame_cap    – frame cap for the visual Load Video node.
      video_skip_first   – skip-first-frames for the visual Load Video node.
      video_every_nth    – select-every-nth for the visual Load Video node.
      audio_source       – "extract_from_visual" | "extract_from_video" | "file" | "none".
      audio_file         – absolute path to audio file (source=="file"), or "".
      audio_force_rate   – fps override for the audio Load Video node (extract_from_visual).
      audio_frame_cap    – frame cap for the audio Load Video node.
      audio_skip_first   – skip-first-frames for the audio Load Video node.
      audio_every_nth    – select-every-nth for the audio Load Video node.
      audio_start_time   – start time in seconds for Load Audio (source=="file").
      audio_duration     – duration in seconds for Load Audio; 0 = to end.
      video_entries_full – list of all video-mode entry descriptors (for refplan):
                           [{subject_id, video_file, load_params, audio_source,
                             audio_path, audio_start_time, audio_duration,
                             audio_retention, audio_role}, …]
    """
    entries = scene_cast.get("entries", [])
    reference_video = ""
    image_files: list[str] = []
    video_params: dict = {"force_rate": 24, "frame_load_cap": 96, "skip_first_frames": 0, "select_every_nth": 1}
    audio_source = "none"
    audio_file = ""
    audio_params: dict = {"force_rate": 0, "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}
    audio_time: dict = {"start_time": 0.0, "duration": 0.0}
    video_entries_full: list[dict] = []

    input_dir = get_input_directory()

    for entry in entries:
        bundle_id = entry.get("bundle_id", "")
        visual_mode = entry.get("visual_mode", "images")
        subject_id = entry.get("subject_id", "")
        if not bundle_id:
            continue
        bundle = bundle_registry.get(bundle_id)
        if bundle is None:
            logger.debug("_resolve_cast_media: bundle %r not found", bundle_id)
            continue
        visual = bundle.get("visual", {})
        audio = bundle.get("audio", {})

        want_video  = visual_mode in ("video", "both")
        want_images = visual_mode != "video"

        if want_video:
            vfile = visual.get("file", "")
            if vfile:
                vdir = visual.get("video_dir", "input")
                abs_vfile = os.path.join(
                    get_output_directory() if vdir == "output" else input_dir,
                    vfile,
                )
                entry_load_params = {
                    "start_time":        float(visual.get("start_time", 0.0)),
                    "duration":          float(visual.get("duration",   0.0)),
                    "force_rate":        visual.get("force_rate", 24),  # H3 requires 24fps
                    "frame_load_cap":    visual.get("frame_load_cap", 96),
                    "skip_first_frames": visual.get("skip_first_frames", 0),
                    "select_every_nth":  visual.get("select_every_nth", 1),
                }
                # Legacy flat output: first video only
                if not reference_video:
                    reference_video = abs_vfile
                    video_params = entry_load_params

                # Determine audio config for this video entry
                a_src = audio.get("source", "none")
                entry_audio_source = "none"
                entry_audio_path   = ""
                entry_audio_start  = 0.0
                entry_audio_dur    = 0.0
                if a_src == "extract_from_visual":
                    entry_audio_source = "extract_from_visual"
                    entry_audio_path   = abs_vfile
                    # Convert frame-based params to time for ffmpeg seeking.
                    # start_time/duration in the audio bundle are for "file" source only.
                    a_fps  = audio.get("force_rate", 0) or visual.get("force_rate", 0) or 0
                    a_skip = audio.get("skip_first_frames", 0)
                    a_cap  = audio.get("frame_load_cap", 0)
                    entry_audio_start = (a_skip / a_fps) if (a_fps > 0 and a_skip > 0) else 0.0
                    entry_audio_dur   = (a_cap  / a_fps) if (a_fps > 0 and a_cap  > 0) else 0.0
                elif a_src == "extract_from_video":
                    av_file = audio.get("video_file", "")
                    if av_file:
                        av_dir = audio.get("video_dir", "input")
                        entry_audio_source = "extract_from_visual"
                        entry_audio_path   = os.path.join(
                            get_output_directory() if av_dir == "output" else input_dir,
                            av_file,
                        )
                        entry_audio_start  = audio.get("start_time", 0.0)
                        entry_audio_dur    = audio.get("duration",   0.0)
                elif a_src == "file":
                    af = audio.get("file", "")
                    if af:
                        af_dir = audio.get("audio_dir", "input")
                        entry_audio_source = "file"
                        entry_audio_path   = os.path.join(
                            get_output_directory() if af_dir == "output" else input_dir,
                            af,
                        )
                        entry_audio_start  = audio.get("start_time", 0.0)
                        entry_audio_dur    = audio.get("duration", 0.0)

                # Generation-time proxy fallback (Plan 16): if a fresh proxy already exists this
                # is a near-instant filesystem check; if not, this builds one on the spot. Audio
                # extraction above always keeps using abs_vfile (the real original) — proxies are
                # silent (-an), never a valid audio source. Never blocks generation on failure.
                video_file_for_entry = vfile
                if _bundle_proxy_eligible(entry_load_params["force_rate"], entry_load_params["duration"]):
                    try:
                        proxy_path = _ensure_bundle_proxy(
                            source_path=abs_vfile,
                            bundle_id=bundle_id,
                            start_time=entry_load_params["start_time"],
                            end_time=entry_load_params["start_time"] + entry_load_params["duration"],
                            short_edge=_BUNDLE_PROXY_SHORT_EDGE,
                            base_dir=str(user_data_dir()),
                        )
                    except Exception as _proxy_exc:
                        proxy_path = None
                        logger.warning(
                            "_resolve_cast_media: proxy generation failed for bundle %r (%s): %s",
                            bundle_id, os.path.basename(abs_vfile), _proxy_exc,
                        )
                    if proxy_path:
                        video_file_for_entry = proxy_path
                        entry_load_params = dict(entry_load_params)
                        entry_load_params["start_time"] = 0.0  # proxy is already trimmed

                video_entries_full.append({
                    "subject_id":       subject_id,
                    "video_file":       video_file_for_entry,
                    "load_params":      entry_load_params,
                    "audio_source":     entry_audio_source,
                    "audio_path":       entry_audio_path,
                    "audio_start_time": entry_audio_start,
                    "audio_duration":   entry_audio_dur,
                    "audio_retention":  audio.get("retention", "timbre"),
                    "audio_role":       audio.get("role", ""),
                    "audio_cache":      audio.get("audio_cache", ""),
                })

        if want_images:
            raw_files = visual.get("files", [])
            # image_selection: None=all, list[int]=specific indices, int=legacy single
            img_sel = entry.get("image_selection")
            if img_sel is not None:
                if isinstance(img_sel, list):
                    raw_files = [raw_files[i] for i in img_sel if isinstance(i, int) and 0 <= i < len(raw_files)]
                else:
                    try:
                        idx = int(img_sel)
                        raw_files = [raw_files[idx]] if 0 <= idx < len(raw_files) else []
                    except (TypeError, ValueError):
                        pass
            image_files.extend(raw_files)

        # Legacy flat audio: first non-"none" source across all entries
        if audio_source == "none":
            a_src = audio.get("source", "none")
            if a_src == "extract_from_visual" and want_video and visual.get("file"):
                audio_source = a_src
                audio_params = {
                    "force_rate":        audio.get("force_rate", 0),
                    "frame_load_cap":    audio.get("frame_load_cap", 0),
                    "skip_first_frames": audio.get("skip_first_frames", 0),
                    "select_every_nth":  audio.get("select_every_nth", 1),
                }
                a_fps  = audio.get("force_rate", 0) or visual.get("force_rate", 0) or 0
                a_skip = audio.get("skip_first_frames", 0)
                a_cap  = audio.get("frame_load_cap", 0)
                audio_time = {
                    "start_time": (a_skip / a_fps) if (a_fps > 0 and a_skip > 0) else 0.0,
                    "duration":   (a_cap  / a_fps) if (a_fps > 0 and a_cap  > 0) else 0.0,
                }
            elif a_src == "extract_from_video":
                av_file = audio.get("video_file", "")
                if av_file:
                    audio_source = "extract_from_video"
                    audio_file   = os.path.join(input_dir, av_file)
                    audio_params = {
                        "force_rate":        audio.get("force_rate", 0),
                        "frame_load_cap":    audio.get("frame_load_cap", 0),
                        "skip_first_frames": audio.get("skip_first_frames", 0),
                        "select_every_nth":  audio.get("select_every_nth", 1),
                    }
                    audio_time = {
                        "start_time": audio.get("start_time", 0.0),
                        "duration":   audio.get("duration",   0.0),
                    }
            elif a_src == "file":
                a_file = audio.get("file", "")
                if a_file:
                    audio_source = a_src
                    audio_file = os.path.join(input_dir, a_file)
                    audio_time = {
                        "start_time": audio.get("start_time", 0.0),
                        "duration":   audio.get("duration", 0.0),
                    }

    # ── Source-derived entries ─────────────────────────────────────────────────
    # Source profiles are grouped by profile_id; subjects sharing the same
    # profile reference the same source media, so one video_entries_full entry
    # is emitted per unique profile (not per subject).
    source_profiles_data = scene_cast.get("source_profiles", {})
    seen_profile_ids: set = set()

    for entry in entries:
        sp_id = entry.get("source_profile_id", "")
        if not sp_id or sp_id in seen_profile_ids:
            continue
        seen_profile_ids.add(sp_id)

        profile = source_profiles_data.get(sp_id)
        if not profile:
            continue

        media_type = profile.get("media_type", "video")
        if media_type != "video":
            continue  # image-only sources: no video_entries_full entry

        media_file = profile.get("media_filename", "")
        if not media_file:
            continue

        media_dir = profile.get("media_dir", "input")
        abs_file = os.path.join(
            get_output_directory() if media_dir == "output" else input_dir,
            media_file,
        )

        # First source video fills the legacy flat reference_video output
        if not reference_video:
            reference_video = abs_file

        # Only include subject_ids from pure source-only entries (no bundle).
        # Hybrid entries (bundle + source) use the bundle for visual references;
        # including the source profile video would produce a spurious <Video N>
        # even when the user selected images mode for that slot.
        sp_subject_ids = [
            e.get("subject_id") or e.get("source_subject_id", "")
            for e in entries
            if e.get("source_profile_id") == sp_id and not e.get("bundle_id")
        ]
        if not sp_subject_ids:
            # All cast entries for this profile have bundles — skip source video
            continue

        # Resolve clip load_params if a clip_id was specified for this profile
        clip_ids_map = scene_cast.get("clip_ids", {})
        clip_id = clip_ids_map.get(sp_id, "")
        load_params: dict
        if clip_id:
            _sp_path = default_source_profiles_path()
            _sp_reg = _load_source_registry(_sp_path)
            _clip_lp = _sp_reg.clip_load_params(sp_id, clip_id)
            load_params = _clip_lp if _clip_lp else {
                "start_time": 0.0, "duration": 0.0,
                "force_rate": 24, "frame_load_cap": 96,  # H3 requires 24fps
                "skip_first_frames": 0, "select_every_nth": 1,
            }
        else:
            load_params = {
                "start_time": 0.0, "duration": 0.0,
                "force_rate": 24, "frame_load_cap": 96,  # H3 requires 24fps
                "skip_first_frames": 0, "select_every_nth": 1,
            }

        video_entries_full.append({
            "subject_id":        sp_subject_ids[0] if sp_subject_ids else "",
            "subject_ids":       sp_subject_ids,
            "source_profile_id": sp_id,
            "clip_id":           clip_id,
            "video_file":        media_file,
            "load_params":       load_params,
            "audio_source":     "none",
            "audio_path":       "",
            "audio_start_time": 0.0,
            "audio_duration":   0.0,
            "audio_retention":  "timbre",
            "audio_role":       "",
            "audio_cache":      "",
        })

    reference_images = _load_subject_images(image_files) if image_files else None
    return {
        "reference_video":    reference_video,
        "reference_images":   reference_images,
        "video_force_rate":   video_params["force_rate"],
        "video_frame_cap":    video_params["frame_load_cap"],
        "video_skip_first":   video_params["skip_first_frames"],
        "video_every_nth":    video_params["select_every_nth"],
        "audio_source":       audio_source,
        "audio_file":         audio_file,
        "audio_force_rate":   audio_params["force_rate"],
        "audio_frame_cap":    audio_params["frame_load_cap"],
        "audio_skip_first":   audio_params["skip_first_frames"],
        "audio_every_nth":    audio_params["select_every_nth"],
        "audio_start_time":   audio_time["start_time"],
        "audio_duration":     audio_time["duration"],
        "video_entries_full": video_entries_full,
    }


def _apply_composition_libbers(text: str, libbers: list, delimiter: str, manager) -> str:
    """Apply libbers in order to text, resolving references in three passes.

    Pass 0 — %*%        : random value drawn from the combined pool of all attached
                          libbers (sampling without replacement; wraps when exhausted).
    Pass 1 — %key:N%    : named key from the Nth libber (1-based).
             %*:N%      : random value from the Nth libber (sampling without replacement).
    Pass 2 — %key%      : plain ref chained through all libbers in order (first match wins).

    Within a single call every %*:N% (or %*%) draws a DIFFERENT key from that libber's
    shuffled queue.  When all keys have been drawn the queue refills with a new shuffle
    so no key repeats until every other key has been used at least once.
    """
    if not libbers or not text:
        return text

    # Load all referenced libbers; strip .json extension for the state manager key
    loaded: list = []
    for name in libbers:
        key = name[:-5] if name.endswith(".json") else name
        loaded.append(manager.ensure_libber(key))

    esc = re.escape(delimiter)

    # ── Per-libber shuffled queues (for %*:N%) ────────────────────────────────
    _per_queue: dict[int, deque] = {}

    def _libber_keys(idx: int) -> list[str]:
        lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
        return list(lb.libs.keys()) if lb else []

    def _pop_from_libber(idx: int) -> str | None:
        """Pop the next key from libber idx's shuffled deque; refill when empty."""
        if idx not in _per_queue:
            keys = _libber_keys(idx)
            if not keys:
                return None
            random.shuffle(keys)
            _per_queue[idx] = deque(keys)
        q = _per_queue[idx]
        if not q:
            keys = _libber_keys(idx)
            if not keys:
                return None
            random.shuffle(keys)
            q.extend(keys)
        return q.popleft()

    def _resolve_value(lb, key: str) -> str:
        val = lb.get_lib(key)
        if val is None:
            return ""
        temp = Libber(lib_dict=dict(lb.libs), delimiter=delimiter, max_depth=lb.max_depth)
        return temp.substitute(val)

    # ── Combined queue (for %*%) ───────────────────────────────────────────────
    # Each entry is (libber_index, key) — shuffle once across all attached libbers.
    _combined_queue: deque = deque()

    def _fill_combined() -> None:
        pairs = [
            (i, key)
            for i, lb in enumerate(loaded)
            if lb
            for key in lb.libs
        ]
        random.shuffle(pairs)
        _combined_queue.extend(pairs)

    def _pop_combined() -> str:
        if not _combined_queue:
            _fill_combined()
        if not _combined_queue:
            return ""
        idx, key = _combined_queue.popleft()
        lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
        return _resolve_value(lb, key) if lb else ""

    # ── Pass 0: %*% — combined-pool wildcard ──────────────────────────────────
    _fill_combined()
    text = re.sub(re.escape(delimiter) + r'\*' + re.escape(delimiter), lambda _: _pop_combined(), text)

    # ── Pass 1: %key:N% and %*:N% — indexed refs ─────────────────────────────
    def _replace_indexed(m: re.Match) -> str:
        inner = m.group(1)   # e.g. "key:2", "*:1" or "*.1"
        if inner.startswith("*."):
            inner = "*:" + inner[2:]
        raw_key, idx_str = inner.rsplit(":", 1)
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return m.group(0)

        if raw_key == "*":
            key = _pop_from_libber(idx)
            if key is None:
                return ""
            lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
            return _resolve_value(lb, key) if lb else ""

        # Named key from specific libber slot
        if 0 <= idx < len(loaded) and loaded[idx]:
            return _resolve_value(loaded[idx], raw_key) or m.group(0)
        return m.group(0)

    # Accept "%*.N%" as a spelling of "%*:N%" (a common typo that otherwise never resolves).
    indexed_pat = esc + r'((?:[a-z0-9_]+:|\*[:.])[0-9]+)' + esc
    text = re.sub(indexed_pat, _replace_indexed, text)

    # ── Pass 2: plain unindexed refs — chain through libbers in order ─────────
    for lb in loaded:
        if lb:
            temp = Libber(lib_dict=dict(lb.libs), delimiter=delimiter, max_depth=lb.max_depth)
            text = temp.substitute(text)

    return text


class PromptCompositionLoader(io.ComfyNode):
    """Load a saved prompt composition and assemble it into a model-specific prompt.

    Select a composition by name from the dropdown. The assembled prompt and
    concept IDs are ready to wire into text-conditioning and ConceptResolve nodes.

    Optionally wire a SCENE_CAST from SceneCastLoad or SceneCastBuild to resolve
    reference media (video file path and/or image batch) from the cast's bundles.

    After saving a composition in the Prompt Compositions sidebar panel, any
    PromptCompositionLoader nodes on the canvas automatically re-execute to pick
    up the latest content (via the reload counter). A full page refresh is needed
    for brand-new compositions to appear in the name dropdown.
    """

    node_id = prefixed_node_id("PromptCompositionLoader")
    display_name = "Prompt Composition Loader"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        names = _composition_get_names()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.Combo.Input(
                    "composition_name",
                    options=names,
                    display_name="Composition",
                    tooltip="Select a saved composition. Press R to refresh the list after saving new ones.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_COMP_MODEL_TYPE_OPTIONS,
                    display_name="Model Type",
                    tooltip="'composition default' uses the model type stored inside the composition.",
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    default="",
                    tooltip=(
                        "Optional prefix prepended literally to the composition name for the filename_prefix output. "
                        "Include a trailing '/' to make it a folder (e.g. 'video/' → 'video/bbc_ride'). "
                        "Wire the output into a VHS_VideoCombine filename_prefix input."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Optional cast from SceneCastLoad or SceneCastBuild. Resolves reference media (video path and images) from bundles.",
                    optional=True,
                ),
                CompositionIOType.Input(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip=(
                        "Optional composition from CompositionLoad (or SceneCastBuild's pass-through). "
                        "When connected it drives this node and the Composition dropdown is ignored."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Prompt"),
                io.String.Output("composition_name", display_name="Composition Name"),
                io.String.Output(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="prefix + composition name (e.g. 'video/bbc_ride'). Wire into VHS_VideoCombine filename_prefix.",
                ),
                LoraStackData.Output(
                    "lora_stack_data",
                    display_name="LoRA Stack",
                    tooltip=(
                        "LoRAs attached to this composition as LORA_STACK_DATA. "
                        "Wire into LoraStackApply. Empty if no LoRAs are attached."
                    ),
                ),
                H3RefplanType.Output(
                    "h3_refplan",
                    display_name="H3 Ref Plan",
                    tooltip=(
                        "Ordered reference descriptor bundle (FBTOOLS_H3_REFPLAN). "
                        "Wire into CompositionToH3Conditioning to load media and build conditioning "
                        "without manual VHS/audio-loader wiring."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, composition_name: str = "", model_type: str = "composition default", filename_prefix: str = "", scene_cast=None, prompt_composition=None, **_):
        comps_dir = os.path.join(user_data_dir(), "prompt_compositions")
        try:
            dir_mtime = os.path.getmtime(comps_dir)
        except OSError:
            dir_mtime = 0
        # Also key off the matched composition file's own mtime so that out-of-band
        # JSON edits (editing the file on disk without touching the editor) invalidate
        # the node without needing the reload counter.
        file_mtime = 0
        if composition_name:
            items = _list_compositions(user_data_dir())
            matched = next((c for c in items if c["name"] == composition_name), None)
            if matched:
                cpath = os.path.join(comps_dir, f"{matched['id']}.json")
                try:
                    file_mtime = os.path.getmtime(cpath)
                except OSError:
                    pass
        # A wired composition replaces the dropdown; key off its full content so any edit re-runs.
        pc_digest = ""
        if isinstance(prompt_composition, dict) and prompt_composition:
            pc_digest = hashlib.md5(json.dumps(prompt_composition, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        cast_id = scene_cast.get("id", "") if scene_cast else ""
        cast_modified = scene_cast.get("modified", "") if scene_cast else ""
        try:
            bundle_mtime = os.path.getmtime(default_bundle_registry_path()) if scene_cast else 0
        except OSError:
            bundle_mtime = 0
        try:
            settings_mtime = os.path.getmtime(_composition_settings_path())
        except OSError:
            settings_mtime = 0
        return (comps_dir, composition_name, model_type, dir_mtime, file_mtime,
                reload_counter("composition"), cast_id, cast_modified, bundle_mtime, settings_mtime, pc_digest)

    @classmethod
    def execute(
        cls,
        composition_name: str = "",
        model_type: str = "composition default",
        filename_prefix: str = "",
        scene_cast=None,
        prompt_composition=None,
    ) -> io.NodeOutput:
        if isinstance(prompt_composition, dict) and prompt_composition:
            # Wired composition (CompositionLoad): drives everything; the dropdown is ignored.
            composition = copy.deepcopy(prompt_composition)
            composition_name = composition.get("name") or composition_name
            filename_prefix_out = f"{filename_prefix}{composition_name}"
        else:
            filename_prefix_out = f"{filename_prefix}{composition_name}"

            items = _list_compositions(user_data_dir())
            matched = next((c for c in items if c["name"] == composition_name), None)
            if matched is None:
                logger.warning("PromptCompositionLoader: composition %r not found", composition_name)
                return io.NodeOutput("", composition_name, filename_prefix_out, None, None)

            composition = _load_composition(user_data_dir(), matched["id"])

        model_type_used = (
            composition.get("model_type", "h3_ref2va")
            if model_type == "composition default"
            else model_type
        )

        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        outfit_reg = _load_outfit_registry(default_outfit_registry_path())
        # Per-run overrides from SceneCastBuild (background, background-as-reference)
        _cast_overrides = scene_cast.get("composition_overrides") if isinstance(scene_cast, dict) else None
        if _cast_overrides:
            composition, _ov_warnings = _apply_composition_overrides(composition, _cast_overrides, backgrounds)
            for _w in _ov_warnings:
                logger.warning("PromptCompositionLoader: %s", _w)
        resolved_subjects = _resolve_composition_subjects(composition, registry)
        resolved_background = _resolve_composition_background(composition, backgrounds)

        outfit_ids = composition.get("outfit_ids", {})
        resolved_outfits_node = {
            sk: outfit_reg.get_outfit(oid)
            for sk, oid in outfit_ids.items()
            if oid and outfit_reg.get_outfit(oid)
        }

        # Resolve cast: build video_entries for assembler + reference media tensors
        cast_media: dict = {
            "reference_video": "", "reference_images": None,
            "video_force_rate": 24, "video_frame_cap": 16, "video_skip_first": 0, "video_every_nth": 1,
            "audio_source": "none", "audio_file": "",
            "audio_force_rate": 0, "audio_frame_cap": 0, "audio_skip_first": 0, "audio_every_nth": 1,
            "audio_start_time": 0.0, "audio_duration": 0.0,
            "video_entries_full": [],
        }
        if scene_cast:
            bundle_registry = _load_bundle_registry(default_bundle_registry_path())
            cast_media = _resolve_cast_media(scene_cast, bundle_registry)

            # Replace/enrich subjects by cast position, then apply bundle media
            resolved_subjects = _apply_cast_to_subjects(
                resolved_subjects, composition, scene_cast, bundle_registry, registry
            )

        # video_entries_full carries full descriptors (paths, load params, audio config)
        # for all video-mode cast entries; used by both the prompt assembler (for
        # <Video N> labels) and _build_h3_refplan (for the terminal node).
        video_entries: list[dict] = cast_media["video_entries_full"]

        result = _assemble_composition(
            composition, resolved_subjects, resolved_background, model_type_used,
            video_entries, resolved_outfits=resolved_outfits_node,
        )

        prompt = result.get("prompt", "")

        # Apply attached libbers using the composition delimiter from global settings
        libbers_list = composition.get("libbers", [])
        if libbers_list:
            cs = _read_composition_settings()
            prompt = _apply_composition_libbers(
                prompt, libbers_list, cs.get("libber_delimiter", "%"), LibberStateManager.instance()
            )


        # Build LORA_STACK_DATA from composition's attached LoRAs
        lora_stack_data = [
            {
                "lora":           e["name"],
                "strength_model": float(e.get("weight", 1.0)),
                "strength_clip":  float(e.get("weight", 1.0)),
                "enabled":        True,
                "model_target":   e.get("target", "MiniMaxH3"),
                "audio_enabled":  False,
            }
            for e in composition.get("loras", [])
            if e.get("name")
        ] or None

        # Compute per-slot trim_to from shot dialogue durations.
        # Each shot's dialogue text is resolved through Libbers independently —
        # a different random pick than what landed in the assembled prompt, but
        # durations are close enough for trim purposes since Libber values for
        # the same key are authored in similar length ranges.
        # Composition subject-slot keys are already letters (A, B, ...), so the
        # dialogue speaker key IS the slot_trim_to key directly — no mapping needed.
        slot_trim_to: dict[str, float] = {}
        _comp_subjects = composition.get("subjects", {})
        _cs = _read_composition_settings()
        _libbers_l = composition.get("libbers", [])
        _delim = _cs.get("libber_delimiter", "%")
        for _shot in composition.get("shots", []):
            _dlg = _shot.get("dialogue")
            if not _dlg or not _dlg.get("text"):
                continue
            _raw = _dlg["text"]
            _resolved = (
                _apply_composition_libbers(_raw, _libbers_l, _delim, LibberStateManager.instance())
                if _libbers_l else _raw
            )
            _cps = _PACE_CHARS_PER_SEC.get(_dlg.get("speech_pace") or "normal", 13.0)
            _dur = _estimate_speech_duration(_resolved, _cps)
            _letter = _dlg.get("speaker", "")
            if _letter and _letter in _comp_subjects:
                slot_trim_to[_letter] = slot_trim_to.get(_letter, 0.0) + _dur

        # Build FBTOOLS_H3_REFPLAN from enriched subjects + full video descriptors
        # Use the scene_instance the prompt was assembled from: it carries the minted
        # background / outfit-reference / bundle slots, so ref-plan picture ordinals match
        # the <Subject N> labels in the prompt. (Fallback keeps older assemblers working.)
        scene_instance_for_plan = result.get("scene_instance") or {
            "slot_assignments": resolved_subjects,
            "outfit_overrides": composition.get("outfit_overrides", {}),
        }
        h3_refplan = _build_h3_refplan(scene_instance_for_plan, video_entries, slot_trim_to or None)
        h3_refplan["prompt"]          = prompt
        h3_refplan["model_type"]      = model_type_used
        h3_refplan["ref_image_size"]  = "match"
        h3_refplan["has_turbo_lora"]  = any(
            "turbo" in (e.get("name", "") or "").lower()
            for e in composition.get("loras", [])
            if e.get("name")
        )

        cast_note = f" | cast: {scene_cast.get('name', '?')}" if scene_cast else ""
        send_status_update(
            cls.node_id,
            f"Loaded: {composition_name} | {model_type_used} | {len(prompt)} chars{cast_note}",
        )

        comp_name_out = composition.get("name", composition_name)
        filename_prefix_out = f"{filename_prefix}{comp_name_out}"

        # Reference media, audio timing, model type and concept IDs are no longer separate
        # outputs: the media travels in h3_refplan, and Run History shows the rest.
        return io.NodeOutput(prompt, comp_name_out, filename_prefix_out, lora_stack_data, h3_refplan)


# =============================================================================
# CompositionToH3Conditioning — media-loading terminal node
# =============================================================================

def _h3_resolve_path(path: str) -> str:
    """Return an absolute path, checking input then output directory."""
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(get_input_directory(), path)
    if os.path.exists(candidate):
        return candidate
    candidate_out = os.path.join(get_output_directory(), path)
    if os.path.exists(candidate_out):
        return candidate_out
    return path  # let callers decide what to do with a missing path


def _h3_load_image(path: str):
    """Load an image file and return a [1,H,W,3] float32 tensor in [0,1]."""
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: image not found: %s", path)
        return None
    try:
        pil = Image.open(resolved).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0   # [H,W,3]
        return torch.from_numpy(arr).unsqueeze(0)        # [1,H,W,3]
    except Exception as exc:
        logger.warning("CompositionToH3: failed to load image %s: %s", path, exc)
        return None


def _h3_load_video_frames(path: str, load_params: dict):
    """Load video frames → [B,H,W,3] float32 in [0,1] (RGB) using cv2.

    VHS (VideoHelperSuite) is intentionally not used here.  VHS loads its own
    sub-package via relative imports, which caches modules under its parent
    package key in sys.modules.  An absolute ``from videohelpersuite.load_video_nodes``
    import looks for a different key and fails because ComfyUI adds
    ``custom_nodes/`` — not ``custom_nodes/ComfyUI-VideoHelperSuite/`` — to
    sys.path.  cv2 is always available (it is a direct dependency of both
    ComfyUI and VHS) and produces identical output.
    """
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: video not found: %s", path)
        return None

    start_time        = float(load_params.get("start_time",        0.0))
    duration          = float(load_params.get("duration",          0.0))
    force_rate        = int(load_params.get("force_rate",          0))
    frame_load_cap    = int(load_params.get("frame_load_cap",      0))
    skip_first_frames = int(load_params.get("skip_first_frames",   0))
    select_every_nth  = int(load_params.get("select_every_nth",    1)) or 1

    # start_time (time-based seek) and skip_first_frames (legacy frame-count seek)
    # are two alternative mechanisms for the same purpose.  When start_time is set,
    # suppress skip_first_frames to avoid double-offsetting into old bundles that
    # had skip_first_frames set before the trim slider was introduced.
    if start_time > 0.0 and skip_first_frames > 0:
        logger.warning(
            "CompositionToH3: both start_time=%.2fs and skip_first_frames=%d are set "
            "for %s — skip_first_frames ignored when start_time is used. "
            "Clear skip_first_frames in the bundle to suppress this warning.",
            start_time, skip_first_frames, os.path.basename(path),
        )
        skip_first_frames = 0

    try:
        import cv2
    except ImportError:
        logger.error("CompositionToH3: cv2 not available; cannot load video %s", path)
        return None
    try:
        cap = cv2.VideoCapture(resolved)
        if not cap.isOpened():
            logger.warning("CompositionToH3: cv2 cannot open %s", path)
            return None
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        target_fps = float(force_rate) if force_rate > 0 else native_fps
        base_frame_time   = 1.0 / native_fps
        target_frame_time = 1.0 / target_fps

        # Time-based start: seek directly to start_time using cv2.
        # This is preferred over skip_first_frames for large offsets and is
        # what users naturally specify (seconds, not frame counts).
        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

        # Time-based duration: convert to an output-frame cap so the accumulator
        # loop can stop without tracking wall-clock time on every frame.
        # `duration` wins over frame_load_cap when both are set. `frame_load_cap` is compared
        # against `sampled` below (17441-17443), which counts frames *after* the select_every_nth
        # filter — so the cap must be expressed in post-filter frames too, or it silently reads
        # select_every_nth times too much of the source (e.g. select_every_nth=2 → exactly 2x the
        # requested duration is decoded before the cap is hit).
        if duration > 0.0:
            duration_cap = max(1, int(duration * target_fps / select_every_nth))
            frame_load_cap = duration_cap if frame_load_cap == 0 else min(frame_load_cap, duration_cap)

        # Time-accumulator resampling — mirrors VHS cv_frame_generator logic:
        # read native frames until the virtual clock reaches the next output slot,
        # then emit the most recently decoded frame for that slot.  Starting
        # time_offset at target_frame_time means the first native frame is
        # immediately emitted without needing extra reads.
        frames: list = []
        current_bgr = None
        time_offset = target_frame_time
        total_count = 0   # virtual frames seen (drives skip_first_frames)
        evaluated   = -1  # frames after skip (drives select_every_nth)
        sampled     = 0   # frames actually appended

        # Pre-read the first native frame (VHS does an initial grab() before its loop).
        ret, current_bgr = cap.read()
        if not ret:
            cap.release()
            logger.warning("CompositionToH3: cv2 loaded no frames from %s", path)
            return None

        while cap.isOpened():
            # Advance native frames until the virtual clock reaches the next slot.
            if time_offset < target_frame_time:
                ret, bgr = cap.read()
                if not ret:
                    break
                current_bgr = bgr
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue

            time_offset -= target_frame_time
            total_count += 1
            if total_count <= skip_first_frames:
                continue
            evaluated += 1
            if evaluated % select_every_nth != 0:
                continue

            rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
            sampled += 1
            if frame_load_cap > 0 and sampled >= frame_load_cap:
                break

        cap.release()
        if not frames:
            logger.warning("CompositionToH3: cv2 loaded no frames from %s", path)
            return None

        # Ping-pong to the nearest valid H3 frame count (17k+5: 5, 22, 39, …).
        # MiniMaxH3ReferenceToVideo trims DOWN to 17k+5 and errors below 5, so
        # whenever the loaded count is not already a valid number we ping-pong up
        # to the next valid count — never further.  This avoids both the model
        # error and unnecessary frame trimming while keeping fabricated motion
        # to the minimum needed.
        n = len(frames)
        _k = max(1, (n - 5 + 16) // 17)   # ceiling: smallest k with 17k+5 >= n, floor k=1 (22 frames)
        target = 17 * _k + 5
        if n < target:
            logger.warning(
                "CompositionToH3: %d frame(s) from %s is not a valid H3 count "
                "(17k+5) — ping-pong looping to %d.",
                n, os.path.basename(path), target,
            )
            # Single frame: repeat.  Multi-frame: bounce (omit endpoints from
            # the reversed half so they are not duplicated at each turn).
            cycle = frames if n == 1 else frames + list(reversed(frames[1:-1]))
            result: list = []
            i = 0
            while len(result) < target:
                result.append(cycle[i % len(cycle)])
                i += 1
            frames = result

        logger.debug(
            "CompositionToH3: loaded %d frames from %s (start=%.2fs, dur=%.2fs, cap=%d, skip=%d)",
            len(frames), path, start_time, duration, frame_load_cap, skip_first_frames,
        )
        return torch.stack(frames, dim=0)  # [B,H,W,3]
    except Exception as exc:
        logger.warning("CompositionToH3: cv2 frame load failed for %s: %s", path, exc)
        return None


def _h3_load_audio(path: str, start_time: float = 0.0, duration: float = 0.0):
    """Load audio → {'waveform': [1,C,L] float32, 'sample_rate': int}.

    Uses ffmpeg directly (same approach as VHS get_audio) so video files and all
    audio codecs are handled correctly with accurate start_time/duration seeking.
    VHS is intentionally not imported — see _h3_load_video_frames for the reason.
    """
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: audio file not found: %s", path)
        return None

    # Prefer imageio_ffmpeg (ships with ComfyUI); fall back to system ffmpeg.
    ffmpeg_exe = None
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
    except Exception:
        pass
    if not ffmpeg_exe:
        import shutil as _shutil
        ffmpeg_exe = _shutil.which("ffmpeg")
    if not ffmpeg_exe:
        logger.error(
            "CompositionToH3: ffmpeg not found; cannot extract audio from %s", path
        )
        return None

    import re
    import subprocess
    args = [ffmpeg_exe, "-i", resolved]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    args += ["-f", "f32le", "-"]

    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "CompositionToH3: ffmpeg failed for %s: %s",
            path,
            exc.stderr.decode("utf-8", "backslashreplace")[:300],
        )
        return None

    stderr_text = res.stderr.decode("utf-8", "backslashreplace")
    match = re.search(r", (\d+) Hz, (\w+),", stderr_text)
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
    else:
        ar = 44100
        ac = 2
        logger.warning(
            "CompositionToH3: could not parse audio format from ffmpeg stderr for %s; "
            "assuming 44100 Hz stereo",
            path,
        )

    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}


def _track_format_prompt_composition_loader(kwargs: dict):
    """Run History rows for a tracked Prompt Composition Loader. Covers what the node no
    longer exposes as outputs (model type used, concept IDs, reference media and audio,
    via the cast rows) plus its LoRAs. See utils/composition_track_summary.py."""
    rows: dict = {}
    consumed: set = set()
    composition = None
    wired = kwargs.get("prompt_composition")
    if isinstance(wired, dict) and wired:
        # The wired composition drives the node; the dropdown value is stale/ignored.
        consumed.add("prompt_composition")
        composition = wired
        rows["Composition (wired)"] = str(wired.get("name") or wired.get("id") or "?")
    else:
        name = kwargs.get("composition_name")
        if name:
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == name), None)
            if matched:
                composition = _load_composition(user_data_dir(), matched["id"])

    if composition is not None:
        subject_lookup = _load_subject_registry(default_subject_profiles_path()).get_subject
        rows.update(summarize_composition_meta(composition, kwargs.get("model_type", ""), subject_lookup))

    scene_cast = kwargs.get("scene_cast")
    if composition is not None:
        # Effective background (after any SceneCastBuild override) so a run's setting is recoverable.
        overrides = scene_cast.get("composition_overrides") if isinstance(scene_cast, dict) else None
        bgs = _load_backgrounds_dict(user_data_dir())
        eff, _ = _apply_composition_overrides(composition, overrides, bgs)
        bg = _resolve_composition_background(eff, bgs)
        rows["Background"] = (
            (bg.get("name") or eff.get("background") or "?") if bg else "(none)"
        ) + (" (override)" if overrides and "background" in overrides else "")
        as_ref = bool(eff.get("background_as_reference"))
        has_refs = bool(bg and bg.get("reference_images"))
        rows["Background as reference"] = (
            ("yes" if as_ref else "no")
            + ("" if not as_ref or has_refs else " (no reference images, text only)")
            + (" (override)" if overrides and "background_as_reference" in overrides else "")
        )
        if overrides and overrides.get("background_soundscape"):
            rows["Soundscape"] = "from the background (override)"
    if isinstance(scene_cast, dict):
        bundle_reg = _load_bundle_registry(default_bundle_registry_path())
        rows.update(summarize_scene_cast(scene_cast, bundle_reg.get))
        consumed.add("scene_cast")

    if composition is not None:
        loras = summarize_loras(composition.get("loras", []))
        if loras:
            rows["LoRAs"] = loras
    return consumed, rows


register_track_formatter(prefixed_node_id("PromptCompositionLoader"), _track_format_prompt_composition_loader)


class CompositionToH3Conditioning(io.ComfyNode):
    """Terminal node: consume FBTOOLS_H3_REFPLAN, decode all media internally,
    and delegate to MiniMaxH3ReferenceToVideo to produce conditioning + latent.

    Common-case graph: PromptCompositionLoader → CompositionToH3Conditioning → sampler.
    """

    node_id = prefixed_node_id("CompositionToH3Conditioning")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name="Composition → H3 Conditioning",
            category="🧊 frost-byte/conditioning",
            description=(
                "Decodes all references in an FBTOOLS_H3_REFPLAN bundle "
                "(images, videos, audio) and calls MiniMaxH3ReferenceToVideo "
                "to produce conditioning + AV latent."
            ),
            inputs=[
                H3RefplanType.Input("h3_refplan"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("width",  default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768,  min=32, max=8192, step=32),
                io.Int.Input("length", default=124,  min=5,  max=3600, step=17,
                             tooltip="Frame count at 24 fps (124 = ~5s)"),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip=(
                        "'match' scales refs to the generation canvas area (faster). "
                        "'max' uses full 2048px short-edge fidelity (slower)."
                    ),
                ),
                io.Boolean.Input(
                    "estimate_vram",
                    display_name="Estimate VRAM",
                    default=True,
                    tooltip=(
                        "Estimate second-pass attention memory from this canvas + the "
                        "resolved references, and recommend a safe scale for "
                        "MinimaxH3LatentUpscaler3D's 'scale' input. A heuristic "
                        "calibrated from real OOM incidents on this machine — not a "
                        "guarantee. Disable to skip the computation entirely."
                    ),
                ),
                io.Float.Input(
                    "vram_safety_buffer",
                    display_name="VRAM Safety Buffer",
                    default=0.85, min=0.1, max=1.0, step=0.05,
                    optional=True,
                    tooltip=(
                        "Fraction of the estimated safe headroom to actually recommend "
                        "(0.85 = recommend 85% of the theoretical max scale). Lower this "
                        "if recommended scales still OOM in practice."
                    ),
                ),
                io.Float.Input(
                    "desired_scale",
                    display_name="Desired Scale",
                    default=0.0, min=0.0, max=4.0, step=0.05,
                    optional=True,
                    tooltip=(
                        "Target scale for MinimaxH3LatentUpscaler3D's 'scale' input. "
                        "0.0 = auto (use the calculated safe maximum). Any other value "
                        "is validated against the estimate: passed through unchanged if "
                        "it fits within budget, clamped down to the safe maximum (with "
                        "a warning) if it doesn't."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Float.Output(display_name="Recommended Scale",
                                tooltip="Wire into MinimaxH3LatentUpscaler3D's 'scale' input."),
                io.String.Output(display_name="VRAM Estimate",
                                  tooltip="Human-readable summary of the estimate — also logged."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, h3_refplan, width, height, length, ref_image_size,
                            estimate_vram=True, vram_safety_buffer=0.85, desired_scale=0.0, **_):
        bundle_hash = hashlib.md5(
            json.dumps(h3_refplan or {}, sort_keys=True).encode()
        ).hexdigest()
        file_mtimes = []
        for ref in (h3_refplan or {}).get("references", []):
            path = _h3_resolve_path(ref.get("path", ""))
            try:
                file_mtimes.append(f"{path}:{os.path.getmtime(path):.3f}")
            except OSError:
                file_mtimes.append(f"{path}:missing")
        return (bundle_hash, *file_mtimes, width, height, length, ref_image_size,
                estimate_vram, vram_safety_buffer, desired_scale)

    @classmethod
    def execute(cls, h3_refplan, clip, vae, audio_vae,
                width, height, length, ref_image_size="match",
                estimate_vram=True, vram_safety_buffer=0.85, desired_scale=0.0) -> io.NodeOutput:
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except ImportError as exc:
            raise RuntimeError(
                f"MiniMaxH3ReferenceToVideo not found in comfy_extras. "
                f"Ensure ComfyUI includes nodes_minimax_h3.py. ({exc})"
            ) from exc

        if not h3_refplan:
            raise ValueError("h3_refplan is required — wire PromptCompositionLoader's "
                             "'H3 Ref Plan' output into this node.")

        references = h3_refplan.get("references", [])
        prompt     = h3_refplan.get("prompt", "")

        # ── §1 pre-load validation (MiniMax H3 Ref2VA hard limits) ─────────────
        pre_errors = _validate_h3_refs_pre(references)
        if pre_errors:
            raise ValueError(
                "H3 Ref2VA validation failed:\n"
                + "\n".join(f"  • {e}" for e in pre_errors)
            )

        # ── Turbo LoRA warning ──────────────────────────────────────────────────
        # Any audio reference counts, whether it is a video's soundtrack or a standalone clip.
        audio_refs = [r for r in references if r.get("modality") in ("audio", "soundtrack_audio")]
        if h3_refplan.get("has_turbo_lora") and audio_refs:
            _turbo_msg = (
                "Turbo LoRA detected with audio references. "
                "Turbo LoRA is known to degrade audio quality badly — "
                "disable it for voice/dialogue generation."
            )
            logger.warning("CompositionToH3: %s", _turbo_msg)
            send_status_update(cls.node_id, f"⚠️ {_turbo_msg}")

        # ── Load references ─────────────────────────────────────────────────────
        ref_images       = {}
        ref_videos       = {}
        ref_video_audios = {}
        ref_audios       = {}
        standalone_idx   = 0
        loaded_audio_durations: list[float] = []

        # VRAM estimate: reference token count. With ref_image_size=="match",
        # MiniMaxH3ReferenceToVideo rescales every reference to the generation
        # canvas's own pixel area, so approximate each reference frame's cost
        # as the canvas's own per-frame token cost rather than the reference's
        # original (pre-rescale) resolution. With "max" (full 2048px-short-edge
        # fidelity), use the reference's actual loaded resolution instead.
        reference_tokens = 0.0
        canvas_tokens_per_frame = h3_tokens_for(width, height, 1) if estimate_vram else 0.0

        logger.info("CompositionToH3: loading %d reference item(s) — canvas %dx%d, %d frames",
                    len(references), width, height, length)

        for ref in references:
            modality = ref.get("modality", "")
            path     = ref.get("path", "")
            fname    = os.path.basename(path) if path else "(no path)"

            if modality == "image":
                frames = _h3_load_image(path)
                if frames is not None:
                    n = ref["picture_ordinal"] - 1
                    ref_images[f"ref_image_{n}"] = frames
                    h, w = frames.shape[1], frames.shape[2]
                    logger.info("  <Picture %d>  image  %s  %dx%d",
                                ref["picture_ordinal"], fname, w, h)
                    if estimate_vram:
                        reference_tokens += (
                            canvas_tokens_per_frame if ref_image_size == "match"
                            else h3_tokens_for(w, h, 1)
                        )
                else:
                    logger.warning("  <Picture %d>  image  %s  FAILED TO LOAD",
                                   ref["picture_ordinal"], fname)

            elif modality == "video":
                frames = _h3_load_video_frames(path, ref.get("load_params", {}))
                if frames is not None:
                    n = ref["video_ordinal"] - 1
                    ref_videos[f"ref_video_{n}"] = frames
                    lp = ref.get("load_params", {})
                    n_frames = frames.shape[0]
                    logger.info("  <Video %d>    video  %s  %d frames  start=%.1fs dur=%.1fs",
                                ref["video_ordinal"], fname, n_frames,
                                lp.get("start_time", 0.0), lp.get("duration", 0.0))
                    if estimate_vram:
                        reference_tokens += (
                            canvas_tokens_per_frame * n_frames if ref_image_size == "match"
                            else h3_tokens_for(frames.shape[2], frames.shape[1], n_frames)
                        )
                else:
                    logger.warning("  <Video %d>    video  %s  FAILED TO LOAD",
                                   ref["video_ordinal"], fname)

            elif modality == "soundtrack_audio":
                _cache = ref.get("audio_cache", "")
                if _cache and os.path.isfile(_cache):
                    audio = _h3_load_audio(_cache, 0.0, 0.0)
                    src_note = f"cache:{os.path.basename(_cache)}"
                else:
                    audio = _h3_load_audio(path, ref.get("start_time", 0.0),
                                           ref.get("duration", 0.0))
                    src_note = fname
                if audio is not None:
                    n = ref["video_ordinal"] - 1
                    ref_video_audios[f"ref_video_audio_{n}"] = audio
                    dur = audio["waveform"].shape[-1] / max(audio["sample_rate"], 1)
                    logger.info("  <Audio %d>    soundtrack  %s  %.2fs  retention=%s  (paired with <Video %d>)",
                                ref["audio_ordinal"], src_note, dur,
                                ref.get("retention", "timbre"), ref["video_ordinal"])
                else:
                    logger.warning("  <Audio %d>    soundtrack  %s  FAILED TO LOAD",
                                   ref["audio_ordinal"], src_note)

            elif modality == "audio":
                _cache = ref.get("audio_cache", "")
                if _cache and os.path.isfile(_cache):
                    audio = _h3_load_audio(_cache, 0.0, 0.0)
                    src_note = f"cache:{os.path.basename(_cache)}"
                else:
                    audio = _h3_load_audio(path, ref.get("start_time", 0.0),
                                           ref.get("duration", 0.0))
                    src_note = fname
                if audio is None:
                    logger.warning("  <Audio %d>    standalone  %s  FAILED TO LOAD",
                                   ref["audio_ordinal"], src_note)
                    continue

                # Apply trim_to: shorten waveform to estimated dialogue line duration
                trim_to = ref.get("trim_to")
                if trim_to is not None and trim_to > 0:
                    sr = audio["sample_rate"]
                    target_samples = int(trim_to * sr)
                    audio["waveform"] = audio["waveform"][:, :, :target_samples]

                # Per-clip duration validation (post-trim, actual samples)
                actual_samples = audio["waveform"].shape[-1]
                sr = audio["sample_rate"]
                actual_dur = actual_samples / sr if sr > 0 else 0.0
                clip_err = _validate_h3_audio_clip(
                    actual_dur, ref.get("audio_ordinal", "?"), os.path.basename(path)
                )
                if clip_err:
                    raise ValueError(clip_err)

                trim_note = f"  trimmed→{actual_dur:.2f}s" if trim_to else ""
                logger.info("  <Audio %d>    standalone  %s  %.2fs  retention=%s%s",
                            ref["audio_ordinal"], src_note, actual_dur,
                            ref.get("retention", "timbre"), trim_note)

                loaded_audio_durations.append(actual_dur)
                ref_audios[f"ref_audio_{standalone_idx}"] = audio
                standalone_idx += 1

        # Total audio duration check (uses actual loaded/trimmed durations)
        total_audio = sum(loaded_audio_durations)
        total_err = _validate_h3_audio_total(loaded_audio_durations)
        if total_err:
            raise ValueError(total_err)

        logger.info(
            "CompositionToH3: passing to MiniMaxH3ReferenceToVideo — "
            "%d image(s), %d video(s), %d soundtrack(s), %d standalone audio(s)%s",
            len(ref_images), len(ref_videos), len(ref_video_audios), len(ref_audios),
            f" | {total_audio:.1f}s audio total" if loaded_audio_durations else "",
        )

        audio_note = f" | audio {total_audio:.1f}s total" if loaded_audio_durations else ""
        send_status_update(
            cls.node_id,
            f"H3 conditioning: {len(ref_images)} image(s), "
            f"{len(ref_videos)} video(s), "
            f"{len(ref_video_audios)} soundtrack(s), "
            f"{len(ref_audios)} standalone audio(s){audio_note}",
        )

        mm_result = MiniMaxH3ReferenceToVideo.execute(
            clip, prompt, width, height, length,
            ref_image_size=ref_image_size,
            vae=vae,
            audio_vae=audio_vae,
            ref_images=ref_images or None,
            ref_videos=ref_videos or None,
            ref_video_audios=ref_video_audios or None,
            ref_audios=ref_audios or None,
        )
        positive, latent = mm_result.args[0], mm_result.args[1]

        if estimate_vram:
            recommended_scale, vram_summary = cls._vram_estimate_summary(
                width, height, length, reference_tokens, vram_safety_buffer, desired_scale,
            )
        else:
            recommended_scale = desired_scale if desired_scale > 0.0 else 1.0
            vram_summary = "VRAM estimate disabled (Estimate VRAM input is off)."

        return io.NodeOutput(positive, latent, recommended_scale, vram_summary)

    @classmethod
    def _vram_estimate_summary(cls, width, height, length, reference_tokens,
                                vram_safety_buffer, desired_scale=0.0):
        """Compute (output_scale, summary_text) for the second-pass upscale
        factor to feed MinimaxH3LatentUpscaler3D's 'scale' input. Heuristic —
        see utils/h3_vram_estimator.py for the calibration this is based on.

        desired_scale <= 0.0 means "auto": output the calculated safe maximum.
        Any other value is treated as a target: passed through unchanged if it
        fits within the estimate, clamped down to the safe maximum (with a
        warning) if it doesn't.
        """
        main_tokens = h3_tokens_for(width, height, length)
        try:
            budget_gib = torch.cuda.mem_get_info()[1] / (1024 ** 3) if torch.cuda.is_available() else None
        except Exception:
            budget_gib = None

        if budget_gib is None:
            fallback = desired_scale if desired_scale > 0.0 else 1.0
            summary = (
                f"VRAM estimate unavailable (no CUDA device detected) — "
                f"passing through scale {fallback:.2f}x unvalidated."
            )
            logger.warning("CompositionToH3: %s", summary)
            return fallback, summary

        max_scale, at_risk = h3_max_safe_scale(
            main_tokens, reference_tokens, budget_gib, safety_buffer=vram_safety_buffer,
        )

        auto_mode = desired_scale <= 0.0
        if auto_mode:
            output_scale = max_scale
            clamped = False
        else:
            clamped = desired_scale > max_scale
            output_scale = max_scale if clamped else desired_scale

        align = 32
        second_w = round(width * output_scale / align) * align
        second_h = round(height * output_scale / align) * align
        mode_str = "auto" if auto_mode else ("target, clamped" if clamped else "target, confirmed")
        summary = (
            f"VRAM estimate: pass 1 {width}x{height} ({main_tokens:,.0f} main + "
            f"{reference_tokens:,.0f} ref tokens) -> scale {output_scale:.2f}x [{mode_str}] "
            f"for pass 2 (~{second_w}x{second_h}) | safe max {max_scale:.2f}x | "
            f"budget {budget_gib:.1f} GiB, buffer {vram_safety_buffer:.0%}"
        )
        if clamped:
            summary += f" | WARNING: requested {desired_scale:.2f}x exceeds estimated safe max — clamped"
            logger.warning("CompositionToH3: %s", summary)
        elif at_risk:
            summary += " | WARNING: even scale=1.0 may be tight at this canvas/reference load"
            logger.warning("CompositionToH3: %s", summary)
        else:
            logger.info("CompositionToH3: %s", summary)
        return output_scale, summary


class FBToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SubjectLayerDefine,
            SubjectCompositor,
            CaptionModelUnloader,
            DatasetCaptioner,
            DatasetCaptionEditor,
            DatasetCaptionViewer,
            DatasetExportSummary,
            FBTextEncodeQwenImageEditPlus,
            SAMPreprocessNHWC,
            QwenAspectRatio,
            SubdirLister,
            # NodeInputSelect,
            SceneCreate,
            SceneUpdate,
            SceneMaskDefinition,
            SceneSave,
            SceneInput,
            SceneOutput,
            SceneView,
            SceneSelect,
            SceneLoraStackSave,
            StorySceneBatch,
            StoryScenePick,
            StoryVideoBatch,
            StoryCreate,
            StoryEdit,
            StoryView,
            StorySave,
            StoryLoad,
            StorySceneImageSave,
            OpaqueAlpha,
            MaskProcessor,
            TailSplit,
            TailEnhancePro,
            # Libber nodes
            LibberManager,
            LibberApply,
            # Scene Prompt Management nodes
            ScenePromptManager,
            PromptComposer,
            # LoRA scene nodes
            LoraStackBuilder,
            LoraStackApply,
            LoraEntryDefine,
            LoraStackCollect,
            WanVidLoraStack,
            # Wan preset nodes
            LoraPresetDefine,
            LoraPresetSelect,
            WanPresetDefine,
            WanPresetSelect,
            # Audio nodes
            AudioFixShape,
            # Concept Registry nodes
            ConceptRegistryLoad,
            ConceptDefine,
            ConceptResolve,
            ConceptList,
            # Subject Profile nodes
            SubjectProfileLoad,
            SubjectProfileDefine,
            SubjectProfileList,
            # Source Profile nodes
            SourceProfileLoad,
            SourceProfileDefine,
            SourceProfileList,
            SourceProfileClipPrompt,
            # Scene Template nodes
            SceneTemplateLoad,
            SceneTemplateList,
            # Outfit Registry nodes
            OutfitRegistryLoad,
            OutfitDefine,
            OutfitList,
            # Scene Composition nodes
            SceneCompose,
            PromptAssemble,
            CompositionLoad,
            PromptCompositionLoader,
            CompositionToH3Conditioning,
            # Scene Cast nodes
            SceneCastLoad,
            SceneCastBuild,
            # Run tracking
            RunMetaCapture,
            # Notifications
            JobCompleteNotifier,
        ]