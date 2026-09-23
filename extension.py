from __future__ import annotations

from PIL import Image
import math
import folder_paths
from folder_paths import get_input_directory, get_output_directory
import comfy.model_management as model_management

from typing_extensions import override
from .nodes.shared import (
    prefixed_node_id,
    routes,
    send_status_update,
    user_data_dir,
    default_registry_path,
    default_subject_profiles_path,
    default_source_profiles_path,
    default_bundle_registry_path,
    default_cast_registry_path,
    default_outfit_registry_path,
    default_scene_templates_dir,
    reload_counter,
    bump_reload,
)
from .nodes.dataset_caption import DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader
from .nodes.libber import Libber, LibberManager, LibberApply, LibberStateManager
from .nodes.llm_assistant import _SPA_STATUS_ID, _route_llm, _run_text_inference, _run_vision_inference, _run_vision_inference_clip
from .nodes.lora_stacks import (
    LoraStackData, LoraStackBuilder, LoraStackApply, LoraEntryDefine, LoraStackCollect, WanVidLoraStack,
    _lora_get_list,
)
from .nodes.media import _audio_get_list
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
from .utils.composition_track_summary import summarize_scene_cast, summarize_loras, summarize_composition_meta
from .nodes import kdenlive_archive as _kdenlive_archive_routes  # noqa: F401  (registers /fbtools/kdenlive/* routes on import)
# Route-only modules: importing them registers their /fbtools/* handlers on the PromptServer routes.
from .nodes import backgrounds_presets as _backgrounds_presets_routes  # noqa: F401
from .nodes import registry_api as _registry_api_routes  # noqa: F401
from .nodes import outfits as _outfits_routes  # noqa: F401
from .nodes import lora_info as _lora_info_routes  # noqa: F401
from .nodes import prompt_collections as _prompt_collections_routes  # noqa: F401
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

from .utils.images import load_image_comfyui, make_placeholder_tensor, normalize_image_tensor
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
from .utils.concept_registry import (
    ConceptRegistry,
    load_registry as _load_concept_registry,
    save_registry as _save_concept_registry,
    resolve_concepts as _resolve_concepts,
    assemble_prompt as _assemble_concept_prompt,
    format_resolved_info as _format_resolved_info,
    parse_concept_ids as _parse_concept_ids,
    build_model_entry as _build_model_entry,
    MODEL_PROFILES as _CONCEPT_MODEL_PROFILES,
    MODEL_TYPE_IDS as _CONCEPT_MODEL_TYPE_IDS,
)
from .utils.subject_profiles import (
    SubjectRegistry,
    load_registry as _load_subject_registry,
    save_registry as _save_subject_registry,
    SUPPORTED_LANGUAGES as _SUBJECT_LANGUAGES,
)
from .utils.source_profiles import (
    SourceProfileRegistry,
    load_registry as _load_source_registry,
    save_registry as _save_source_registry,
    ENTITY_TYPES as _SOURCE_ENTITY_TYPES,
    MEDIA_TYPES as _SOURCE_MEDIA_TYPES,
    MEDIA_DIRS as _SOURCE_MEDIA_DIRS,
    resolved_pronoun_style as _sp_resolved_pronoun_style,
    resolve_ordinal_subject as _sp_resolve_ordinal_subject,
    resolve_ordinal_from_list as _sp_resolve_ordinal_from_list,
)
from .utils.source_profile_analysis import (
    build_segment_detection_prompt as _spa_build_segment_prompt,
    build_clip_description_prompt as _spa_build_clip_desc_prompt,
    _parse_segments_response as _spa_parse_segments,
    build_subject_inference_prompt as _spa_build_subject_inference_prompt,
    parse_inferred_subjects_response as _spa_parse_inferred_subjects,
    parse_clip_description_response as _spa_parse_clip_desc,
    build_prompt as _spa_build_prompt,
    build_multi_prompt as _spa_build_multi_prompt,
    _parse_vlm_json_response as _spa_parse_response,
    append_history_entry as _spa_append_history,
    history_for_profile as _spa_history_for_profile,
    extract_video_frame as _spa_extract_frame,
    extract_frame_at_time as _spa_extract_frame_at_time,
    extract_clip_frames as _spa_extract_clip_frames,
    build_contact_sheet_image as _spa_build_contact_sheet,
    probe_video_fps as _spa_probe_fps,
    probe_video_resolution as _spa_probe_resolution,
    PASS_TYPES as _SPA_PASS_TYPES,
)
from .utils.proxy_cache import ensure_source_profile_proxy as _ensure_proxy
from .utils.proxy_cache import ensure_bundle_video_proxy as _ensure_bundle_proxy
from .utils.scene_templates import (
    SceneTemplate,
    load_template as _load_scene_template,
    scan_templates as _scan_scene_templates,
    template_ids as _scene_template_ids,
    format_template_list as _format_template_list,
    dir_fingerprint as _templates_dir_fingerprint,
)
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
from .utils.libber_resolve import resolve_libber_refs, extract_libber_names, apply_slot_dialogue
from .utils.outfit_registry import (
    OutfitRegistry,
    load_outfit_registry as _load_outfit_registry,
    save_outfit_registry as _save_outfit_registry,
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
import time





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


# ── Custom type: CONCEPT_REGISTRY ────────────────────────────────────────────

CONCEPT_REGISTRY_TYPE = "CONCEPT_REGISTRY"


@io.comfytype(io_type=CONCEPT_REGISTRY_TYPE)
class ConceptRegistryIOType:
    """Carries a ConceptRegistry instance between Load → Define → Resolve nodes."""
    Type = object  # ConceptRegistry

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "registry", **kwargs):
            super().__init__(name, **kwargs)


def _concept_get_ids() -> list[str]:
    """Read concept_registry.json and return concept IDs for combo widgets."""
    try:
        registry = _load_concept_registry(default_registry_path())
        ids = sorted(registry.concepts.keys())
        return ["None"] + ids if ids else ["None"]
    except Exception:
        return ["None"]


# ── Node: ConceptRegistryLoad ─────────────────────────────────────────────────

class ConceptRegistryLoad(io.ComfyNode):
    """Load the concept registry from disk.

    Connects to one or more ConceptDefine or ConceptResolve nodes.
    Use the "Reload Registry" button (added by JS) to force re-execution
    after the file has been edited externally.
    """
    node_id = prefixed_node_id("ConceptRegistryLoad")
    display_name = "Concept Registry Load"
    category = "🧊 frost-byte/lora"
    is_output_node = True  # allow standalone execution to preview available concepts

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.String.Input(
                    "registry_file",
                    display_name="Registry File (leave empty for default)",
                    default="",
                    tooltip=(
                        "Absolute path to a concept_registry.json file. "
                        "Leave empty to use the default user-data location."
                    ),
                    multiline=False,
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Concept registry to wire into ConceptDefine or ConceptResolve nodes.",
                ),
                io.String.Output(
                    "available_concepts",
                    display_name="Available Concepts",
                    tooltip="Human-readable list of all defined concepts.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, registry_file: str = "", **_):
        path = registry_file.strip() or default_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("concept"))

    @classmethod
    def execute(cls, registry_file: str = ""):
        path = registry_file.strip() or default_registry_path()
        registry = _load_concept_registry(path)
        available = registry.list_concepts()
        return io.NodeOutput(registry, available, ui={"available_concepts": available})


# ── Node: ConceptDefine ───────────────────────────────────────────────────────

class ConceptDefine(io.ComfyNode):
    """Define (or update) one concept entry for a specific model type.

    Chain multiple ConceptDefine nodes to build a complete registry before
    passing it to ConceptResolve.  If auto_save is enabled the updated
    registry is written back to its source file (with backup).
    """
    node_id = prefixed_node_id("ConceptDefine")
    display_name = "Concept Define"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or a previous ConceptDefine.",
                ),
                io.String.Input(
                    "concept_id",
                    display_name="Concept ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or style_cinematic.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Display Name",
                    default="",
                    tooltip="Human-readable label shown in ConceptList.",
                    multiline=False,
                ),
                io.String.Input(
                    "description",
                    display_name="Description",
                    default="",
                    multiline=True,
                    tooltip="Internal notes about this concept — not included in assembled prompts.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Target model family for this LoRA entry.",
                ),
                io.Combo.Input(
                    "lora",
                    display_name="LoRA (or High LoRA for split models)",
                    options=_lora_get_list(),
                    default="None",
                    tooltip="LoRA file. For split models (Wan 2.2, BerniniR) this is the HIGH model LoRA.",
                ),
                io.Combo.Input(
                    "lora_low",
                    display_name="Low LoRA (split models only)",
                    options=_lora_get_list(),
                    default="None",
                    optional=True,
                    tooltip="Low-model LoRA for Wan 2.2 / BerniniR. Hidden by JS for single-model types.",
                ),
                io.Float.Input(
                    "weight",
                    display_name="Weight (or High Weight)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="LoRA strength. For split models this applies to the HIGH LoRA.",
                ),
                io.Float.Input(
                    "weight_low",
                    display_name="Low Weight (split models only)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    optional=True,
                    tooltip="LoRA strength for the LOW model LoRA. Hidden by JS for single-model types.",
                ),
                io.String.Input(
                    "trigger",
                    display_name="Trigger Words",
                    default="",
                    multiline=False,
                    tooltip="Trigger text appended/prepended to the prompt by ConceptResolve.",
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=False,
                    tooltip="If enabled, persist the updated registry to disk after each execution.",
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Updated registry with this concept entry added or merged.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, lora="None", lora_low="None", **kwargs):
        return True

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concept_id: str,
        name: str,
        description: str,
        model_type: str,
        lora: str = "None",
        lora_low: str = "None",
        weight: float = 1.0,
        weight_low: float = 1.0,
        trigger: str = "",
        auto_save: bool = False,
    ):
        model_entry = _build_model_entry(model_type, lora, lora_low, weight, weight_low, trigger)
        updated = registry.define(concept_id.strip(), name.strip(), description, model_type, model_entry)
        if auto_save:
            save_path = registry.file_path or default_registry_path()
            try:
                _save_concept_registry(updated, save_path, backup=True)
            except Exception as exc:
                logger.warning("ConceptDefine: auto_save failed: %s", exc)
        return io.NodeOutput(updated)


# ── Node: ConceptResolve ──────────────────────────────────────────────────────

class ConceptResolve(io.ComfyNode):
    """Resolve concept IDs against the registry and apply their LoRAs.

    For split-model types (Wan 2.2, BerniniR) the HIGH LoRA is applied to
    *model* and the LOW LoRA is applied to *model_low*.  For single-model
    types only *model* is used.

    Trigger words from each concept are collected and assembled into the
    output prompt according to the trigger_position setting.
    """
    node_id = prefixed_node_id("ConceptResolve")
    display_name = "Concept Resolve"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or ConceptDefine chain.",
                ),
                io.String.Input(
                    "concepts",
                    display_name="Concepts",
                    default="",
                    multiline=True,
                    tooltip="Concept IDs to resolve — one per line or comma-separated.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Select the model family so the correct LoRAs are chosen.",
                ),
                io.Model.Input(
                    "model",
                    display_name="Model",
                    tooltip="Primary model (or HIGH model for split types).",
                ),
                io.Model.Input(
                    "model_low",
                    display_name="Model (Low)",
                    optional=True,
                    tooltip="Low model for split types (Wan 2.2 / BerniniR). Leave unconnected for single-model types.",
                ),
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP encoder. Both high and low LoRAs are applied to CLIP for split models.",
                ),
                io.String.Input(
                    "base_prompt",
                    display_name="Base Prompt",
                    default="",
                    multiline=True,
                    tooltip="Starting prompt text. Concept triggers are merged in via trigger_position.",
                ),
                io.Combo.Input(
                    "trigger_position",
                    display_name="Trigger Position",
                    options=["prepend", "append"],
                    default="prepend",
                    tooltip="Where to place concept trigger words relative to base_prompt.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    "model",
                    display_name="Model",
                    tooltip="Primary model with HIGH (or single) LoRAs applied.",
                ),
                io.Model.Output(
                    "model_low",
                    display_name="Model (Low)",
                    tooltip="Low model with LOW LoRAs applied (passthrough for single-model types).",
                ),
                io.Clip.Output(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP with all concept LoRAs applied.",
                ),
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Base prompt with concept trigger words merged in.",
                ),
                io.String.Output(
                    "resolved_info",
                    display_name="Resolved Info",
                    tooltip="Summary of which concepts were resolved and which LoRAs were applied.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concepts: str,
        model_type: str,
        model,
        clip,
        base_prompt: str = "",
        trigger_position: str = "prepend",
        model_low=None,
    ):
        import comfy.sd as _comfy_sd
        import comfy.utils as _comfy_utils

        concept_ids = _parse_concept_ids(concepts)
        resolved = _resolve_concepts(registry, concept_ids, model_type)
        is_split = _CONCEPT_MODEL_PROFILES.get(model_type, {}).get("split", False)

        cur_model = model
        cur_model_low = model_low
        cur_clip = clip
        triggers: list[str] = []

        for r in resolved:
            if r.error:
                logger.warning("ConceptResolve [%s]: %s", r.concept_id, r.error)
                continue

            if r.trigger:
                triggers.append(r.trigger)

            # Apply high (or single) LoRA to primary model + CLIP
            if r.lora_high:
                path = folder_paths.get_full_path("loras", r.lora_high)
                if path:
                    weights = _comfy_utils.load_torch_file(path, safe_load=True)
                    cur_model, cur_clip = _comfy_sd.load_lora_for_models(
                        cur_model, cur_clip, weights, r.weight_high, r.weight_high
                    )
                else:
                    logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_high)
                    r.error = f"file missing: {r.lora_high}"

            # Apply low LoRA for split model types
            if is_split and r.lora_low:
                if cur_model_low is not None:
                    path = folder_paths.get_full_path("loras", r.lora_low)
                    if path:
                        weights = _comfy_utils.load_torch_file(path, safe_load=True)
                        cur_model_low, cur_clip = _comfy_sd.load_lora_for_models(
                            cur_model_low, cur_clip, weights, r.weight_low, r.weight_low
                        )
                    else:
                        logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_low)
                        r.error = f"file missing: {r.lora_low}"
                else:
                    logger.warning(
                        "ConceptResolve [%s]: model_low not connected for split type %s — "
                        "low LoRA '%s' skipped",
                        r.concept_id, model_type, r.lora_low,
                    )

        prompt = _assemble_concept_prompt(triggers, base_prompt, trigger_position)
        resolved_info = _format_resolved_info(model_type, resolved, prompt)

        return io.NodeOutput(cur_model, cur_model_low, cur_clip, prompt, resolved_info)


# ── Node: ConceptList ─────────────────────────────────────────────────────────

class ConceptList(io.ComfyNode):
    """Display a summary of all concepts in the registry.

    Optionally filter by model_type to show only concepts defined for that
    target.  Useful for quickly reviewing which concepts are available before
    wiring up ConceptResolve.
    """
    node_id = prefixed_node_id("ConceptList")
    display_name = "Concept List"
    category = "🧊 frost-byte/lora"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry to inspect.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Filter by Model Type",
                    options=["all"] + _CONCEPT_MODEL_TYPE_IDS,
                    default="all",
                    tooltip="Show only concepts that have an entry for this model type, or 'all'.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "concept_list",
                    display_name="Concept List",
                    tooltip="Formatted list of matching concepts.",
                ),
                io.Int.Output(
                    "concept_count",
                    display_name="Count",
                    tooltip="Number of matching concepts.",
                ),
            ],
        )

    @classmethod
    def execute(cls, registry: ConceptRegistry, model_type: str = "all"):
        filter_type = None if model_type == "all" else model_type
        listing = registry.list_concepts(filter_type)
        count = len([
            cid for cid, c in registry.concepts.items()
            if filter_type is None or filter_type in c.get("models", {})
        ])
        return io.NodeOutput(listing, count, ui={"concept_list": listing, "concept_count": count})


# ── Subject Profile helpers ───────────────────────────────────────────────────

def _subject_get_ids() -> list[str]:
    """Read subject_profiles.json and return subject IDs for combo widgets."""
    try:
        reg = _load_subject_registry(default_subject_profiles_path())
        ids = reg.subject_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]






def _load_subject_images(filenames: "list[str | dict]") -> "torch.Tensor | None":
    """Load character sheet images from the ComfyUI input directory.

    Accepts both legacy list[str] and new list[{file, role}] formats.
    Returns a [N, H, W, 3] float32 tensor (batch), or None if no images load.
    Images that fail to load are silently skipped.
    Images with different sizes are resized to match the first loaded image.
    """
    tensors: list = []
    target_h = target_w = None
    input_dir  = get_input_directory()
    output_dir = get_output_directory()
    for entry in filenames:
        fname = entry["file"] if isinstance(entry, dict) else entry
        if not fname:
            continue
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            alt = os.path.join(output_dir, fname)
            if os.path.exists(alt):
                path = alt
        if not os.path.exists(path):
            logger.debug("Subject image not found: %s", path)
            continue
        try:
            img, _ = load_image_comfyui(path, include_mask=False)  # [1, H, W, 3]
            h, w = img.shape[1], img.shape[2]
            if target_h is None:
                target_h, target_w = h, w
            if h != target_h or w != target_w:
                img = normalize_image_tensor(img, target_h, target_w)
            tensors.append(img)
        except Exception as exc:
            logger.warning("Failed to load subject image %s: %s", fname, exc)
    if not tensors:
        return None
    import torch as _torch
    return _torch.cat(tensors, dim=0)  # [N, H, W, 3]


def _load_subject_audio(filename: str) -> "dict | None":
    """Load an audio reference file from the ComfyUI input directory."""
    if not filename:
        return None
    path = os.path.join(get_input_directory(), filename)
    if not os.path.exists(path):
        logger.debug("Subject audio not found: %s", path)
        return None
    try:
        import torchaudio as _torchaudio
        waveform, sample_rate = _torchaudio.load(path)
        # ComfyUI AUDIO dict: waveform is [batch, channels, samples]
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as exc:
        logger.warning("Failed to load subject audio %s: %s", filename, exc)
        return None


# ── Custom type: SUBJECT_PROFILE ──────────────────────────────────────────────

SUBJECT_PROFILE_TYPE = "SUBJECT_PROFILE"


@io.comfytype(io_type=SUBJECT_PROFILE_TYPE)
class SubjectProfileIOType:
    """Carries a subject profile dict between Load/Define → SceneCompose nodes."""
    Type = object  # dict with name, appearance, voice, character_sheet_images, concept_id

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "subject_profile", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: SubjectProfileLoad ──────────────────────────────────────────────────

class SubjectProfileLoad(io.ComfyNode):
    """Load a subject profile from disk and expose its fields as outputs.

    Character sheet images are loaded from the ComfyUI input directory and
    stacked into an IMAGE batch.  Audio reference is loaded if defined.
    Use the Reload button in the REST endpoint to force re-execution after
    editing subject_profiles.json externally.
    """
    node_id = prefixed_node_id("SubjectProfileLoad")
    display_name = "Subject Profile Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        subject_ids = _subject_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "subject_id",
                    options=subject_ids,
                    display_name="Subject ID",
                    tooltip="Subject to load. Press R to refresh the list after adding new subjects.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Full subject profile dict for wiring into SceneCompose.",
                ),
                io.String.Output("name", display_name="Name"),
                io.String.Output("appearance_summary", display_name="Appearance Summary"),
                io.Image.Output(
                    "character_sheet_images",
                    display_name="Character Sheet Images",
                    tooltip="Batch of all character sheet images (N×H×W×3). None if no images defined.",
                ),
                io.Audio.Output(
                    "audio_reference",
                    display_name="Audio Reference",
                    tooltip="Voice reference audio, or None if not defined.",
                ),
                io.String.Output("concept_id", display_name="Concept ID"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, subject_id: str = "", **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, subject_id, mtime, reload_counter("subject"))

    @classmethod
    def execute(cls, subject_id: str = "") -> io.NodeOutput:
        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)

        subject = registry.get_subject(subject_id) if subject_id and subject_id != "(none)" else None
        if subject is None:
            logger.warning("SubjectProfileLoad: subject_id %r not found in %s", subject_id, path)
            return io.NodeOutput(None, "", "", None, None, "")

        name = subject.get("name", subject_id)
        appearance = subject.get("appearance", {})
        appearance_summary = appearance.get("summary", "")
        voice = subject.get("voice", {})
        audio_file = voice.get("audio_reference_file", "")
        concept_id = subject.get("concept_id", "")
        sheet_files = subject.get("character_sheet_images", [])

        images = _load_subject_images(sheet_files)
        audio = _load_subject_audio(audio_file)

        # Inject subject_id so downstream nodes (SceneCompose) can reference it
        subject_with_id = dict(subject)
        subject_with_id["subject_id"] = subject_id

        send_status_update(
            cls.node_id,
            f"Loaded: {name} | {len(sheet_files)} sheet images | audio: {'yes' if audio else 'no'}",
        )
        return io.NodeOutput(subject_with_id, name, appearance_summary, images, audio, concept_id)


# ── Node: SubjectProfileDefine ────────────────────────────────────────────────

class SubjectProfileDefine(io.ComfyNode):
    """Create or update a subject profile entry.

    When auto_save is enabled the updated profiles file is written back to
    subject_profiles.json immediately.  Character sheet images are managed
    separately (edit the JSON directly to update the file list).
    """
    node_id = prefixed_node_id("SubjectProfileDefine")
    display_name = "Subject Profile Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "subject_id",
                    display_name="Subject ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or narrator.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Name",
                    default="",
                    multiline=False,
                    tooltip="Full name of the subject as used in prompts and scene summaries.",
                ),
                io.String.Input(
                    "appearance_summary",
                    display_name="Appearance Summary",
                    default="",
                    tooltip="One-sentence description used in compact prompts and as the baseline for detailed descriptions.",
                    multiline=True,
                ),
                io.String.Input(
                    "face",
                    display_name="Face",
                    default="",
                    multiline=True,
                    tooltip="Face shape, eye colour, skin tone, and any distinguishing facial features.",
                ),
                io.String.Input(
                    "hair",
                    display_name="Hair",
                    default="",
                    multiline=False,
                    tooltip="Hair style, length, and colour.",
                ),
                io.String.Input(
                    "body",
                    display_name="Body",
                    default="",
                    multiline=False,
                    tooltip="Height, build, and other physical characteristics.",
                ),
                io.String.Input(
                    "default_outfit",
                    display_name="Default Outfit",
                    default="",
                    multiline=True,
                    tooltip="Default clothing and accessories used when no outfit override is specified in SceneCompose.",
                ),
                io.String.Input(
                    "voice_description",
                    display_name="Voice Description",
                    default="",
                    multiline=True,
                    tooltip="Textual description of vocal quality for use in prompts referencing audio.",
                ),
                io.Combo.Input(
                    "audio_reference_file",
                    display_name="Audio Reference File",
                    options=_audio_get_list(),
                    default="None",
                    tooltip="Voice reference clip from the ComfyUI input directory. Press R to refresh the list after adding new files.",
                ),
                io.Combo.Input(
                    "language",
                    options=_SUBJECT_LANGUAGES,
                    display_name="Language",
                    tooltip="BCP-47 language tag for dialogue tags in H3 prompts.",
                ),
                io.Combo.Input(
                    "concept_id",
                    display_name="Concept ID",
                    options=_concept_get_ids(),
                    default="None",
                    tooltip="Links to a concept registry entry for LoRA resolution. Press R to refresh after adding concepts.",
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=True,
                    tooltip="Write subject_profiles.json immediately after defining this subject.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Defined subject profile dict.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        subject_id: str,
        name: str = "",
        appearance_summary: str = "",
        face: str = "",
        hair: str = "",
        body: str = "",
        default_outfit: str = "",
        voice_description: str = "",
        audio_reference_file: str = "",
        language: str = "en-us",
        concept_id: str = "",
        auto_save: bool = True,
    ) -> io.NodeOutput:
        if not subject_id.strip():
            raise ValueError("SubjectProfileDefine: subject_id cannot be empty")

        audio_reference_file = "" if audio_reference_file == "None" else audio_reference_file
        concept_id = "" if concept_id == "None" else concept_id

        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)
        registry = registry.define(
            subject_id=subject_id.strip(),
            name=name,
            appearance_summary=appearance_summary,
            face=face,
            hair=hair,
            body=body,
            default_outfit=default_outfit,
            voice_description=voice_description,
            audio_reference_file=audio_reference_file,
            language=language,
            concept_id=concept_id,
        )

        if auto_save:
            _save_subject_registry(registry, path, backup=True)
            logger.info("SubjectProfileDefine: saved %r to %s", subject_id, path)
            send_status_update(cls.node_id, f"Saved subject: {subject_id}")

        sid = subject_id.strip()
        result = dict(registry.get_subject(sid))
        result["subject_id"] = sid
        return io.NodeOutput(result)


# ── Node: SubjectProfileList ──────────────────────────────────────────────────

class SubjectProfileList(io.ComfyNode):
    """Display all defined subject profiles.

    Useful for quickly reviewing what subjects are available without opening
    the JSON file.
    """
    node_id = prefixed_node_id("SubjectProfileList")
    display_name = "Subject Profile List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "subject_list",
                    display_name="Subject List",
                    tooltip="Formatted list of all defined subjects.",
                ),
                io.Int.Output("subject_count", display_name="Subject Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("subject"))

    @classmethod
    def execute(cls) -> io.NodeOutput:
        registry = _load_subject_registry(default_subject_profiles_path())
        listing = registry.list_subjects()
        count = len(registry.subjects)
        return io.NodeOutput(listing, count, ui={"subject_list": listing, "subject_count": count})


# ── Source Profile helpers ────────────────────────────────────────────────────

def _source_profile_get_names() -> list[str]:
    """Read source_profiles.json and return profile names for combo widgets."""
    try:
        reg = _load_source_registry(default_source_profiles_path())
        names = reg.profile_names()
        return names if names else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Custom type: SOURCE_PROFILE ───────────────────────────────────────────────

SOURCE_PROFILE_TYPE = "SOURCE_PROFILE"


@io.comfytype(io_type=SOURCE_PROFILE_TYPE)
class SourceProfileIOType:
    """Carries a full source profile (media ref + subjects list) between nodes."""
    Type = object  # SourceProfileRegistry profile dict

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "source_profile", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: SourceProfileLoad ───────────────────────────────────────────────────

class SourceProfileLoad(io.ComfyNode):
    """Load a source profile from disk and expose it for wiring into CastBuild.

    A source profile is a media-first subject catalog: one video or image
    annotated with the identifiable subjects it contains. Connect the
    source_profile output to a CastBuild node to make all subjects in this
    profile available as cast slot options.

    Use POST /fbtools/source_profiles/reload to force re-execution after
    editing source_profiles.json externally.
    """
    node_id = prefixed_node_id("SourceProfileLoad")
    display_name = "Source Profile Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        profile_names = _source_profile_get_names()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "profile_name",
                    options=profile_names,
                    display_name="Source Profile",
                    tooltip="Source profile to load. Press R to refresh the list after adding new profiles.",
                ),
            ],
            outputs=[
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Full source profile dict for wiring into CastBuild.",
                ),
                io.String.Output(
                    "subject_info",
                    display_name="Subject Info",
                    tooltip="Formatted summary of subjects in this profile.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, profile_name: str = "", **_):
        path = default_source_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, profile_name, mtime, reload_counter("source_profile"))

    @classmethod
    def execute(cls, profile_name: str = "") -> io.NodeOutput:
        path = default_source_profiles_path()
        registry = _load_source_registry(path)

        profile = None
        if profile_name and profile_name != "(none)":
            profile = registry.get_profile_by_name(profile_name)
            # Compat: old saved workflows stored the profile ID in this widget before
            # it was renamed from profile_id → profile_name.  Fall back to ID lookup
            # so those workflows continue to work without manual intervention.
            if profile is None:
                profile = registry.get_profile(profile_name)
        if profile is None:
            logger.warning("SourceProfileLoad: profile %r not found in %s", profile_name, path)
            return io.NodeOutput(None, "")

        subjects = profile.get("subjects", [])
        lines = [f"[{s.get('entity_type','?')}] {s.get('label', s.get('id',''))} — {s.get('role_description','')}"
                 for s in subjects]
        subject_info = "\n".join(lines) if lines else "(no subjects defined)"

        send_status_update(
            cls.node_id,
            f"Loaded: {profile_name} | {len(subjects)} subject(s)",
        )
        return io.NodeOutput(profile, subject_info)


# ── Node: SourceProfileDefine ─────────────────────────────────────────────────

class SourceProfileDefine(io.ComfyNode):
    """Create or update a source profile entry.

    Defines the profile metadata (name, media file) only. Use the Source
    Profiles UI panel or SourceProfileDefine nodes chained together to add
    subjects. When auto_save is enabled the registry is written to
    source_profiles.json immediately.
    """
    node_id = prefixed_node_id("SourceProfileDefine")
    display_name = "Source Profile Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "profile_id",
                    display_name="Profile ID",
                    default="",
                    multiline=False,
                    tooltip="Unique snake_case identifier, e.g. theater_scene_video1.",
                ),
                io.String.Input(
                    "name",
                    display_name="Name",
                    default="",
                    multiline=False,
                    tooltip="Human-readable name for this source profile.",
                ),
                io.String.Input(
                    "media_filename",
                    display_name="Media Filename",
                    default="",
                    multiline=False,
                    tooltip="Filename of the source video or image in the ComfyUI input or output directory.",
                ),
                io.Combo.Input(
                    "media_dir",
                    options=_SOURCE_MEDIA_DIRS,
                    display_name="Media Dir",
                    tooltip="Whether the source file lives in the input or output directory.",
                ),
                io.Combo.Input(
                    "media_type",
                    options=_SOURCE_MEDIA_TYPES,
                    display_name="Media Type",
                    tooltip="video or image.",
                ),
                io.String.Input(
                    "subjects_json",
                    display_name="Subjects JSON",
                    default="[]",
                    multiline=True,
                    tooltip='Optional JSON array of subject entries to add/update. Each entry: {"id","label","role_description","entity_type","notes"}.',
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=True,
                    tooltip="Write source_profiles.json immediately after defining this profile.",
                ),
            ],
            outputs=[
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="The created or updated source profile dict.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        profile_id: str,
        name: str = "",
        media_filename: str = "",
        media_dir: str = "input",
        media_type: str = "video",
        subjects_json: str = "[]",
        auto_save: bool = True,
    ) -> io.NodeOutput:
        if not profile_id.strip():
            raise ValueError("SourceProfileDefine: profile_id cannot be empty")

        pid = profile_id.strip()
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        registry = registry.define_profile(
            profile_id=pid,
            name=name,
            media_filename=media_filename,
            media_dir=media_dir,
            media_type=media_type,
        )

        try:
            subjects = json.loads(subjects_json) if subjects_json.strip() not in ("", "[]") else []
        except Exception:
            subjects = []
            logger.warning("SourceProfileDefine: invalid subjects_json, skipping subjects")

        for s in subjects:
            sid = s.get("id", "").strip()
            if sid:
                registry = registry.define_subject(
                    profile_id=pid,
                    subject_id=sid,
                    label=s.get("label", ""),
                    role_description=s.get("role_description", ""),
                    entity_type=s.get("entity_type", "person"),
                    notes=s.get("notes", ""),
                )

        if auto_save:
            _save_source_registry(registry, path, backup=True)
            logger.info("SourceProfileDefine: saved %r to %s", pid, path)
            send_status_update(cls.node_id, f"Saved source profile: {pid}")

        profile = registry.get_profile(pid)
        return io.NodeOutput(profile)


# ── Node: SourceProfileList ───────────────────────────────────────────────────

class SourceProfileList(io.ComfyNode):
    """Display all defined source profiles and their subjects.

    Useful for quickly reviewing the source catalog without opening the
    JSON file.  Filter by media type to narrow the listing.
    """
    node_id = prefixed_node_id("SourceProfileList")
    display_name = "Source Profile List"
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
                    "filter_type",
                    options=["all"] + _SOURCE_MEDIA_TYPES,
                    display_name="Filter by Type",
                    tooltip="Show all profiles, or only video/image sources.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "profile_list",
                    display_name="Profile List",
                    tooltip="Formatted listing of all source profiles and their subjects.",
                ),
                io.Int.Output("profile_count", display_name="Profile Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, filter_type: str = "all", **_):
        path = default_source_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, filter_type, mtime, reload_counter("source_profile"))

    @classmethod
    def execute(cls, filter_type: str = "all") -> io.NodeOutput:
        registry = _load_source_registry(default_source_profiles_path())
        listing = registry.list_profiles(filter_type=filter_type)
        count = len(registry.profiles)
        return io.NodeOutput(listing, count, ui={"profile_list": listing, "profile_count": count})


# ── Node: SourceProfileClipPrompt ─────────────────────────────────────────────

class SourceProfileClipPrompt(io.ComfyNode):
    """Build a model-specific prompt from a Source Profile clip.

    For H3 models: outputs a full 6-section H3 prompt with task_flags=["video editing"],
    so the model treats <Video 1> as the source material to edit. Preserves
    {A}/{B}/{C}/{D} → <Subject 1>/<Subject 2>/… mapping from clip.subjects order.

    clip_duration_frames = ceil(duration_s × 24) — the H3 output length in frames.
    LoRAs defined on the clip are emitted as LORA_STACK_DATA for direct connection
    to LoraStackApply.
    """
    node_id = prefixed_node_id("SourceProfileClipPrompt")
    display_name = "Source Profile Clip Prompt"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                SourceProfileIOType.Input(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Source profile from SourceProfileLoad. Provides subjects and clip library.",
                ),
                io.String.Input(
                    "clip_id",
                    display_name="Clip ID",
                    default="",
                    tooltip="ID of the clip to process (e.g. 'clip_1'). Leave empty to use the first clip.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_PROMPT_MODEL_TYPES,
                    display_name="Model Type",
                    tooltip="Target model. h3_ref2va emits a 6-section structured brief with video editing task flag.",
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    default="",
                    tooltip=(
                        "Optional prefix prepended to 'profile_name/clip_label' for the filename_prefix output. "
                        "Include a trailing '/' to place outputs in a subfolder (e.g. 'video/' → 'video/office_work/clip_1'). "
                        "Wire the output into a VHS_VideoCombine filename_prefix input."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    optional=True,
                    tooltip=(
                        "Optional cast from SceneCastBuild. When provided, bundle appearance data "
                        "and images replace raw source-profile role descriptions for matched subjects."
                    ),
                ),
                io.Boolean.Input(
                    "include_original_subject_tags",
                    display_name="Tag Replaced Subjects",
                    default=False,
                    tooltip=(
                        "When enabled, each source-profile subject being replaced by a SceneCastBuild bundle "
                        "gets its own <Subject N> tag in subject_definitions (minimal description) and an "
                        "attribute_transfer entry in retention_analysis that explicitly scopes motion transfer "
                        "to the replacement while stating that face, hair, and clothing are NOT copied. "
                        "Use to A/B test whether explicit original-subject tagging improves swap quality."
                    ),
                    optional=True,
                ),
                io.Int.Input(
                    "clip_duration_multiplier",
                    display_name="Duration Multiplier",
                    default=1,
                    min=1,
                    max=4,
                    tooltip=(
                        "Scale factor applied to native clip duration for the clip_duration_frames output. "
                        "1 = native duration, 2 = double, up to 4×. "
                        "Wire from SceneCastBuild.clip_duration_multiplier."
                    ),
                    optional=True,
                ),
                io.Int.Input(
                    "max_clip_frames",
                    display_name="Max Clip Frames",
                    default=360,
                    min=0,
                    max=9999,
                    tooltip=(
                        "Hard ceiling on clip_duration_frames (after the duration multiplier is applied). "
                        "0 = unclamped. Default 360 = 15 s × 24 fps — the practical H3 output limit. "
                        "Lower this on memory-constrained systems or at high output resolution."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Model-specific assembled prompt text.",
                ),
                H3RefplanType.Output(
                    "h3_refplan",
                    display_name="H3 Ref Plan",
                    tooltip=(
                        "Ordered reference descriptor bundle (FBTOOLS_H3_REFPLAN) for "
                        "CompositionToH3Conditioning. Contains the source video reference "
                        "with clip load params. Empty when model_type is not an H3 variant."
                    ),
                ),
                LoraStackData.Output(
                    "lora_stack_data",
                    display_name="LoRA Stack",
                    tooltip="LoRAs defined on this clip. Feed into LoraStackApply.",
                ),
                io.String.Output(
                    "concept_ids",
                    display_name="Concept IDs",
                    tooltip="Comma-separated concept IDs from assigned subjects. Wire into ConceptResolve.",
                ),
                io.Int.Output(
                    "clip_duration_frames",
                    display_name="Clip Frames",
                    tooltip=(
                        "Clip duration converted to H3 output frames at 24 fps: ceil(duration_s × 24). "
                        "Use as the 'length' input for CompositionToH3Conditioning."
                    ),
                ),
                io.Int.Output(
                    "width",
                    display_name="Width",
                    tooltip="Source video width in pixels (probed via ffprobe). 0 if probe failed.",
                ),
                io.Int.Output(
                    "height",
                    display_name="Height",
                    tooltip="Source video height in pixels (probed via ffprobe). 0 if probe failed.",
                ),
                io.String.Output(
                    "clip_summary",
                    display_name="Clip Summary",
                    tooltip="Human-readable description of the assembled clip and its subject assignments.",
                ),
                io.String.Output(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="prefix + profile_name/clip_label (e.g. 'video/office_work/segment_1'). Wire into VHS_VideoCombine filename_prefix.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_profile=None,
        clip_id: str = "",
        model_type: str = "h3_ref2va",
        filename_prefix: str = "",
        scene_cast=None,
        include_original_subject_tags: bool = False,
        clip_duration_multiplier: int = 1,
        max_clip_frames: int = 360,
    ) -> io.NodeOutput:
        if source_profile is None:
            return io.NodeOutput("", None, [], "", 0, 0, 0, "No source profile connected.", "")

        # ── Resolve clip ───────────────────────────────────────────────────────
        profile_id     = source_profile.get("id", "")
        profile_name   = source_profile.get("name", profile_id)
        clips          = source_profile.get("clips", [])
        subjects_all   = source_profile.get("subjects", [])
        media_filename = source_profile.get("media_filename", "")
        media_dir      = source_profile.get("media_dir", "input")

        # scene_cast.clip_ids takes priority — wiring scene_cast is the primary
        # path and must override the serialized widget value, which can be stale.
        # The widget/link value is only consulted when scene_cast is absent, to
        # support standalone use without a SceneCastBuild node.
        effective_clip_id = ""
        if isinstance(scene_cast, dict):
            effective_clip_id = scene_cast.get("clip_ids", {}).get(profile_id, "")
        if not effective_clip_id:
            effective_clip_id = clip_id.strip() if clip_id else ""

        clip = None
        if effective_clip_id:
            for c in clips:
                if c.get("id") == effective_clip_id:
                    clip = c
                    break
        if clip is None:
            clip = clips[0] if clips else None
        if clip is None:
            return io.NodeOutput("", None, [], "", 0, 0, 0, f"No clips found in profile '{profile_name}'.", "")

        clip_id_used = clip.get("id", "")
        clip_label   = clip.get("label", clip_id_used)

        # ── Resolve clip subjects ──────────────────────────────────────────────
        # ALL entity types (people, environments, objects) go into slots so that
        # {A}/{B}/{C}/{D} placeholders in the clip action text resolve correctly.
        # Stale IDs (removed from the profile) are silently skipped; cap at 4.
        # Fallback: if the clip has no subjects tagged (e.g. manually created
        # clips or clips auto-partitioned before subjects were added), use all
        # profile subjects in definition order so nothing is silently dropped.
        subject_index = {s["id"]: s for s in subjects_all}
        source_subject_ids: list[str] = [
            sid for sid in clip.get("subjects", [])
            if sid in subject_index
        ][:10]
        if not source_subject_ids:
            source_subject_ids = [s["id"] for s in subjects_all if s.get("id")][:10]

        SOURCE_SLOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]   # preserve placeholder mapping
        BUNDLE_SLOTS = ["K", "L", "M", "N"]   # replacement subjects added after source slots

        # ── Build cast lookup for bundle substitution ──────────────────────────
        # Hybrid cast entries (source_profile_id + source_subject_id + bundle_id)
        # mean a bundle is replacing a source subject. Source subject stays in its
        # original slot (so action placeholders resolve); bundle replacement gets an
        # additional slot from BUNDLE_SLOTS.
        cast_lookup: dict[str, dict] = {}   # source_subject_id → cast entry
        bundle_reg = None
        subject_reg = None
        if isinstance(scene_cast, dict):
            for entry in scene_cast.get("entries", []):
                if (entry.get("source_profile_id") == profile_id
                        and entry.get("source_subject_id")
                        and entry.get("bundle_id")):
                    cast_lookup[entry["source_subject_id"]] = entry
            if cast_lookup:
                try:
                    bundle_reg = _load_bundle_registry(default_bundle_registry_path())
                except Exception:
                    bundle_reg = None
                try:
                    subject_reg = _load_subject_registry(default_subject_profiles_path())
                except Exception:
                    subject_reg = None

        # ── Build slot_assignments ─────────────────────────────────────────────
        slot_assignments: dict = {}
        bundle_slot_pairs: list[tuple[str, str]] = []   # (bundle_slot, source_slot)
        bundle_slot_idx = 0
        bundle_video_entries: list[dict] = []           # video refs for bundle slots

        _ENTITY_PRONOUN_DEFAULTS = {
            "location":   "location",
            "object":     "object",
            "soundscape": "object",
        }  # anything else (person, animal, …) → "neutral"

        for i, sid in enumerate(source_subject_ids):
            src_slot = SOURCE_SLOTS[i]
            subj      = subject_index[sid]
            label     = subj.get("label", sid)
            etype     = subj.get("entity_type", "person").lower()
            role_desc = subj.get("role_description", "")

            # Pronoun style: prefer explicit field on the subject; fall back to
            # entity_type-based default so location/object subjects say "its"/"the X's"
            # without needing manual configuration.
            pronoun_style = (subj.get("pronoun_style")
                             or _ENTITY_PRONOUN_DEFAULTS.get(etype, "neutral"))
            short_name    = subj.get("short_name", "")

            # Source subject entry — always added; provides the video reference and
            # preserves {A}/{B}/… placeholder mapping in the clip action text.
            # _cast_retention and _transfer_to_slot are filled in below if a bundle
            # replacement is found for this subject.
            slot_assignments[src_slot] = {
                "subject_id":             sid,
                "name":                   label,
                "concept_id":             None,
                "character_sheet_images": [],
                "appearance": {
                    "summary":        role_desc or label,
                    "hair":           "",
                    "face":           "",
                    "body":           "",
                    "default_outfit": "",
                },
                "voice":           {},
                "_cast_retention": "fully_preserved",
                "_pronoun_style":  pronoun_style,
                "_short_name":     short_name,
                "entity_type":     etype,
            }

            # If a bundle is assigned to this source subject, add a replacement slot.
            cast_entry = cast_lookup.get(sid)
            if cast_entry and bundle_reg is not None and bundle_slot_idx < len(BUNDLE_SLOTS):
                bundle = bundle_reg.get(cast_entry["bundle_id"])
                if bundle:
                    bun_slot = BUNDLE_SLOTS[bundle_slot_idx]
                    bundle_slot_idx += 1
                    bundle_slot_pairs.append((bun_slot, src_slot))
                    # Mark the source slot as a motion donor for the assembler
                    slot_assignments[src_slot]["_cast_retention"] = "replaced"
                    slot_assignments[src_slot]["_transfer_to_slot"] = bun_slot

                    bun_name   = bundle.get("name") or label
                    # Look up the Subject linked to this bundle for pronoun/short_name/appearance.
                    bun_subject_id = bundle.get("subject_id", "")
                    bun_subj = (subject_reg.get_subject(bun_subject_id)
                                if subject_reg and bun_subject_id else None) or {}
                    # Merge bundle appearance over subject appearance (bundle wins if set).
                    bun_app_dict = bundle.get("appearance") or {}
                    if isinstance(bun_app_dict, str):   # legacy appearance_override string
                        bun_app_dict = {"summary": bun_app_dict}
                    subj_app_dict = bun_subj.get("appearance", {}) if bun_subj else {}
                    def _merge(key):
                        return bun_app_dict.get(key) or subj_app_dict.get(key, "")
                    bun_appear = (
                        bun_app_dict.get("summary")
                        or bundle.get("appearance_override", "")  # legacy fallback
                        or subj_app_dict.get("summary", "")
                    )
                    visual = bundle.get("visual", {})
                    # Resolve mode: cast entry overrides the bundle's default mode.
                    # "both" = use images AND video simultaneously.
                    bun_visual_mode = cast_entry.get("visual_mode") or visual.get("type", "images")

                    # Images path — character sheet images (images mode OR both mode)
                    bun_images: list = []
                    if bun_visual_mode != "video":
                        raw_files = visual.get("files", [])
                        bun_images = [
                            {"file": f.get("file", ""), "role": f.get("role", "character sheet")}
                            if isinstance(f, dict)
                            else {"file": f, "role": "character sheet"}
                            for f in raw_files
                            if (f.get("file", "") if isinstance(f, dict) else f)
                        ]
                        # image_selection: None=all, list[int]=specific indices, int=legacy single
                        img_sel = cast_entry.get("image_selection")
                        if img_sel is not None:
                            if isinstance(img_sel, list):
                                bun_images = [bun_images[i] for i in img_sel if isinstance(i, int) and 0 <= i < len(bun_images)]
                            else:
                                try:
                                    idx = int(img_sel)
                                    bun_images = [bun_images[idx]] if 0 <= idx < len(bun_images) else []
                                except (TypeError, ValueError):
                                    pass

                    audio        = bundle.get("audio", {})
                    audio_source = audio.get("source", "none")

                    # Video path — add a video_entry for the bundle slot so the
                    # assembler emits a <Video N> reference for this subject.
                    # Runs when mode is "video" OR "both".
                    if bun_visual_mode in ("video", "both"):
                        vfile = visual.get("file", "")
                        if vfile:
                            vdir = visual.get("video_dir", "input")
                            bun_base = (get_output_directory() if vdir == "output"
                                        else get_input_directory())
                            bun_load = {
                                "start_time":        float(visual.get("start_time", 0.0)),
                                "duration":          float(visual.get("duration", 0.0)),
                                "force_rate":        visual.get("force_rate", 0),
                                "frame_load_cap":    visual.get("frame_load_cap", 96),
                                "skip_first_frames": visual.get("skip_first_frames", 0),
                                "select_every_nth":  visual.get("select_every_nth", 1),
                            }
                            # extract_from_visual: audio extracted from this same video.
                            # extract_from_video / file: separate source — handled as
                            # bun_voice below; this entry has no audio.
                            # use_audio on the cast entry: treat bundle video's audio
                            # track as the voice-timbre reference for this subject.
                            # allows_dialogue=False on the clip means no audio
                            # involvement for this shot at all, regardless of where
                            # the bundle's audio would otherwise come from.
                            ve_audio_src = (
                                "extract_from_visual"
                                if (clip.get("allows_dialogue", True)
                                    and (audio_source == "extract_from_visual"
                                         or cast_entry.get("use_audio")))
                                else "none"
                            )
                            bundle_video_entries.append({
                                "subject_id":       cast_entry["bundle_id"],
                                "subject_ids":      [cast_entry["bundle_id"]],
                                "video_file":       os.path.join(bun_base, vfile),
                                "load_params":      bun_load,
                                "audio_source":     ve_audio_src,
                                "audio_path":       "",
                                "audio_start_time": 0.0,
                                "audio_duration":   0.0,
                                "audio_retention":  audio.get("retention", "timbre"),
                                "audio_role":       audio.get("role", ""),
                                "audio_cache":      audio.get("audio_cache", ""),
                            })

                    # extract_from_video: a SEPARATE video whose audio track is the
                    # voice reference.  Add an audio_only video_entry linked to the
                    # bundle slot so the assembler emits <Audio N> without also
                    # assigning a spurious <Video N> visual reference to the slot.
                    # Gated on allows_dialogue like the other audio paths above —
                    # this is an independent file, but still audio for this clip's
                    # shot, so the clip's "no audio" setting must still apply.
                    if audio_source == "extract_from_video" and clip.get("allows_dialogue", True):
                        aud_vfile = audio.get("video_file", "")
                        if aud_vfile:
                            aud_vdir = audio.get("video_dir", "input")
                            aud_base = (get_output_directory() if aud_vdir == "output"
                                        else get_input_directory())
                            aud_load = {
                                "start_time":        float(audio.get("start_time", 0.0)),
                                "duration":          float(audio.get("duration", 0.0)),
                                "force_rate":        audio.get("force_rate", 0),
                                "frame_load_cap":    audio.get("frame_load_cap", 0) or 4,
                                "skip_first_frames": audio.get("skip_first_frames", 0),
                                "select_every_nth":  audio.get("select_every_nth", 1),
                            }
                            # When visual mode is images, link to the bundle slot via
                            # bundle_id so the assembler can resolve soundtrack_num.
                            # audio_only=True tells the assembler to assign <Audio N>
                            # only (no <Video N> added to the slot's visual refs).
                            # When visual mode is video or both the visual entry already
                            # holds bundle_id; use a distinct id to avoid collisions.
                            has_vid_entry = bun_visual_mode in ("video", "both")
                            aud_sid = (cast_entry["bundle_id"] + "_audvid" if has_vid_entry
                                       else cast_entry["bundle_id"])
                            bundle_video_entries.append({
                                "subject_id":       aud_sid,
                                "subject_ids":      [aud_sid],
                                "video_file":       os.path.join(aud_base, aud_vfile),
                                "load_params":      aud_load,
                                "audio_source":     "extract_from_visual",
                                "audio_only":       not has_vid_entry,
                                "audio_path":       "",
                                "audio_start_time": 0.0,
                                "audio_duration":   0.0,
                                "audio_retention":  audio.get("retention", "timbre"),
                                "audio_role":       audio.get("role", ""),
                                "audio_cache":      audio.get("audio_cache", ""),
                            })

                    # Standalone audio voice reference (<Audio N>) — emitted when the
                    # bundle carries a separate audio file (source == "file").
                    # extract_from_visual / extract_from_video are handled via
                    # video_entries above. Same allows_dialogue gate: a bundle's own
                    # dedicated audio file is unrelated to the clip's footage, but
                    # it's still audio attached to this clip's shot.
                    bun_voice: dict = {}
                    if audio_source == "file" and audio.get("file") and clip.get("allows_dialogue", True):
                        bun_voice = {
                            "audio_reference_file": os.path.join(
                                get_input_directory(), audio["file"]
                            ),
                            "audio_start_time": float(audio.get("start_time", 0.0)),
                            "audio_duration":   float(audio.get("duration", 0.0)),
                            "audio_retention":  audio.get("retention", "timbre"),
                            "audio_role":       audio.get("role", ""),
                            "audio_cache":      audio.get("audio_cache", ""),
                            "description":      audio.get("description", ""),
                            "language":         audio.get("language", "en-us"),
                        }

                    bun_pronoun = (bun_subj.get("pronoun_style")
                                   or _ENTITY_PRONOUN_DEFAULTS.get(
                                       bundle.get("entity_type", "person"), "neutral"))
                    bun_short_name = bun_subj.get("short_name", "")
                    slot_assignments[bun_slot] = {
                        "subject_id":             cast_entry["bundle_id"],
                        "name":                   bun_name,
                        "concept_id":             None,
                        "character_sheet_images": bun_images,
                        "appearance": {
                            "summary":        bun_appear,
                            "hair":           _merge("hair"),
                            "face":           _merge("face"),
                            "body":           _merge("body"),
                            "default_outfit": _merge("default_outfit"),
                        },
                        "voice":             bun_voice,
                        "_cast_retention":   "attribute_transfer",
                        "_transfer_to_slot": src_slot,
                        "_pronoun_style":    bun_pronoun,
                        "_short_name":       bun_short_name,
                    }

        # ── Task flags and scene synopsis ──────────────────────────────────────
        if bundle_slot_pairs:
            task_flags = ["video editing", "reference generation"]
            # Auto-generate a replacement synopsis.  Bundle slot placeholders {b}
            # are resolved by _bare() to <Subject N>.  Source slot placeholders
            # {s} resolve the same way ONLY when include_original_subject_tags
            # is set — that's what gives a replaced source subject a Subject N
            # label in the first place (see _pre_subject_nums above); otherwise
            # {s} would resolve to an empty string, so the source is named
            # literally instead.
            repl_parts = []
            for _b, _s in bundle_slot_pairs:
                _s_idx = SOURCE_SLOTS.index(_s)
                _src_sid = source_subject_ids[_s_idx]
                if include_original_subject_tags:
                    repl_parts.append(
                        f"Replace {{{_s}}} in <Video 1> with {{{_b}}}, adopting "
                        f"{{{_b}}}'s full appearance and outfit while retaining "
                        f"{{{_s}}}'s original motion, pose and screen position "
                        f"throughout the shot"
                    )
                else:
                    _src_name = subject_index[_src_sid].get("label", _src_sid)
                    repl_parts.append(
                        f"{{{_b}}} takes the place of {_src_name}, "
                        f"replicating their pose, movement, and screen position"
                    )
            retained = [
                SOURCE_SLOTS[i] for i, sid in enumerate(source_subject_ids)
                if sid not in cast_lookup
            ]
            if retained:
                if len(retained) == 1:
                    ret_str = f"{{{retained[0]}}} performs the same actions from <Video 1>"
                else:
                    parts = [f"{{{s}}}" for s in retained]
                    ret_str = (", ".join(parts[:-1]) + " and " + parts[-1]
                               + " perform the same actions from <Video 1>")
                scene_synopsis = ". ".join(repl_parts) + ". " + ret_str + "."
            else:
                scene_synopsis = ". ".join(repl_parts) + "."
        else:
            task_flags    = ["video editing"]
            scene_synopsis = ""

        # ── Collect per-slot dialogue from cast entries ────────────────────────
        # slot_dialogue maps slot key (e.g. "E", "A") → raw dialogue string.
        # Hybrid entries: dialogue belongs to the bundle replacement slot.
        # Source-only entries: dialogue belongs to the original source slot.
        slot_dialogue: dict[str, str] = {}
        if isinstance(scene_cast, dict):
            _src_dlg_lookup: dict[str, str] = {}  # source_subject_id → dialogue
            for _ce in scene_cast.get("entries", []):
                if _ce.get("source_profile_id") != profile_id:
                    continue
                _d = str(_ce.get("dialogue", "") or "").strip()
                if not _d:
                    continue
                _ssid = _ce.get("source_subject_id", "")
                if _ssid and not _ce.get("bundle_id"):
                    _src_dlg_lookup[_ssid] = _d

            for _i2, _sid2 in enumerate(source_subject_ids):
                _d2 = _src_dlg_lookup.get(_sid2, "")
                if _d2:
                    slot_dialogue[SOURCE_SLOTS[_i2]] = _d2

            for _bslot, _sslot in bundle_slot_pairs:
                _s_idx2 = SOURCE_SLOTS.index(_sslot)
                _src_sid2 = source_subject_ids[_s_idx2]
                _ce2 = cast_lookup.get(_src_sid2)
                if _ce2:
                    _d3 = str(_ce2.get("dialogue", "") or "").strip()
                    if _d3:
                        slot_dialogue[_bslot] = _d3

        # ── Build scene_instance ────────────────────────────────────────────────
        _shot_id = f"{clip_id_used}_shot_1"
        scene_instance = {
            "template_id":                  profile_id,
            "template_name":                profile_name,
            "task_flags":                   task_flags,
            "scene_synopsis":               scene_synopsis,
            "slot_assignments":             slot_assignments,
            "dialogue":                     {},
            "outfit_overrides":             {},
            "include_original_subject_tags": include_original_subject_tags,
            "template": {
                "shots": [
                    {
                        "id":       _shot_id,
                        "action":   clip.get("action", ""),
                        "camera":   "",
                        "dialogue": None,
                    }
                ],
                "environment":         {},
                "style":               clip.get("style", ""),
                "overall_soundscape":  clip.get("overall_soundscape", ""),
                "non_diegetic_music":  clip.get("non_diegetic_music", ""),
            },
        }

        # ── Resolve dialogue and populate shot ─────────────────────────────────
        if not clip.get("allows_dialogue", True):
            slot_dialogue = {}
        if slot_dialogue:
            _libber_mgr = LibberStateManager.instance()
            _all_raw = " ".join(slot_dialogue.values())
            _lib_registry: dict = {}
            for _lname in extract_libber_names(_all_raw):
                _lb = _libber_mgr.ensure_libber(_lname)
                if _lb:
                    _lib_registry[_lname] = _lb.libs

            _dlg_map, _shot_patch = apply_slot_dialogue(slot_dialogue, _lib_registry, _shot_id)
            scene_instance["dialogue"].update(_dlg_map)
            _shot = scene_instance["template"]["shots"][0]
            _shot.update(_shot_patch)

        # ── Build video_entries ────────────────────────────────────────────────
        # ONE entry shared by all clip subjects: `subject_ids` (list) causes
        # _build_ref_map to assign the same <Video 1> ordinal to every subject,
        # while `subject_id` (primary) feeds _build_h3_refplan's singular lookup.
        base_dir  = get_output_directory() if media_dir == "output" else get_input_directory()
        video_abs = os.path.join(base_dir, media_filename) if media_filename else ""

        clip_start  = clip.get("start_time", 0.0)
        clip_end    = clip.get("end_time", 0.0)
        clip_dur    = max(0.0, clip_end - clip_start)

        load_params = {
            "start_time":        clip_start,
            "duration":          clip_dur,
            "force_rate":        24,  # H3 requires 24fps reference video
            "frame_load_cap":    clip.get("frame_load_cap", 120),
            "skip_first_frames": 0,
            "select_every_nth":  clip.get("select_every_nth", 2),
        }

        # ── Per-segment proxy ──────────────────────────────────────────────────
        # Trim + downscale once so CompositionToH3Conditioning doesn't need to
        # seek inside a large source file on every generation run.
        video_file_for_entry = video_abs
        if video_abs:
            try:
                proxy_path = _ensure_proxy(
                    source_path=video_abs,
                    profile_id=profile_id,
                    clip_id=clip_id_used,
                    start_time=clip_start,
                    end_time=clip_end,
                    short_edge=source_profile.get("proxy_short_edge", 768),
                    base_dir=str(user_data_dir()),
                )
                if proxy_path:
                    video_file_for_entry = proxy_path
                    load_params = dict(load_params)
                    load_params["start_time"] = 0.0
                    load_params["duration"]   = clip_dur
            except Exception as _proxy_exc:
                logger.warning(
                    "SourceProfileClipPrompt: proxy generation failed for %s: %s",
                    os.path.basename(video_abs), _proxy_exc,
                )

        video_entries = [{
            "subject_id":       source_subject_ids[0] if source_subject_ids else "",
            "subject_ids":      source_subject_ids,
            "video_file":       video_file_for_entry,
            "load_params":      load_params,
            "audio_source":     "none",
            "audio_path":       "",
            "audio_start_time": 0.0,
            "audio_duration":   0.0,
            "audio_retention":  "timbre",
            "audio_role":       "",
            "audio_cache":      "",
        }] if source_subject_ids else []

        # Bundle slots that have visual_mode="video" contribute their own video
        # reference entries (separate from the source profile's motion-donor video).
        video_entries.extend(bundle_video_entries)

        # ── use_audio: voice-timbre reference from the subject's video clip ──────
        # Cast entries with use_audio=True request an <Audio N> voice-timbre
        # reference for that subject.
        # • VIDEO mode bundles: audio_source is already set to "extract_from_visual"
        #   on the bundle video entry above — no extra entry needed.
        # • IMAGE mode bundles / source-only: add an audio_only entry pointing to
        #   the source profile video (the motion-donor clip).
        # allows_dialogue=False is this clip's "no audio involvement" setting —
        # it must also suppress use_audio extraction, not just dialogue text,
        # or a Scene Cast Audio checkbox silently pulls the clip's own audio
        # back in regardless of what's set here.
        if isinstance(scene_cast, dict) and video_file_for_entry and clip.get("allows_dialogue", True):
            _audio_load = dict(load_params, frame_load_cap=4, select_every_nth=1)
            for _ce in scene_cast.get("entries", []):
                if _ce.get("source_profile_id") != profile_id:
                    continue
                if not _ce.get("use_audio"):
                    continue
                _bid  = _ce.get("bundle_id", "")
                _ssid = _ce.get("source_subject_id", "")
                # VIDEO/BOTH mode bundle: audio handled by bundle video entry — skip.
                if _bid and _ce.get("visual_mode") in ("video", "both"):
                    continue
                _audio_sid = _bid if _bid else _ssid
                if not _audio_sid:
                    continue
                video_entries.append({
                    "subject_id":       _audio_sid,
                    "subject_ids":      [_audio_sid],
                    "video_file":       video_file_for_entry,
                    "load_params":      _audio_load,
                    "audio_source":     "extract_from_visual",
                    "audio_only":       True,
                    "audio_path":       "",
                    "audio_start_time": 0.0,
                    "audio_duration":   0.0,
                    "audio_retention":  "timbre",
                    "audio_role":       "",
                    "audio_cache":      "",
                })

        # ── Assemble prompt ────────────────────────────────────────────────────
        try:
            result = _assemble_prompt(scene_instance, model_type, video_entries)
        except Exception as exc:
            logger.error("SourceProfileClipPrompt: prompt assembly failed: %s", exc)
            return io.NodeOutput("", None, [], "", 0, 0, 0, f"Prompt assembly error: {exc}", "")

        prompt       = result["prompt"]
        concept_ids  = result.get("concept_ids", [])

        # ── Build h3_refplan (H3 models only) ─────────────────────────────────
        h3_refplan = None
        if model_type in ("h3_ref2va", "h3_fl2va"):
            rp = _build_h3_refplan(scene_instance, video_entries)
            rp["prompt"]         = prompt
            rp["model_type"]     = model_type
            rp["ref_image_size"] = "match"
            rp["has_turbo_lora"] = any(
                "turbo" in (e.get("name", "") or "").lower()
                for e in clip.get("loras", [])
            )
            h3_refplan = rp

        # ── Build LORA_STACK_DATA from clip loras ──────────────────────────────
        lora_stack_data: list = []
        for entry in clip.get("loras", []):
            name = entry.get("name", "")
            if name:
                lora_stack_data.append({
                    "lora":           name,
                    "strength_model": entry.get("strength_model", 1.0),
                    "strength_clip":  entry.get("strength_clip", 1.0),
                    "enabled":        True,
                    "model_target":   "both",
                    "audio_enabled":  False,
                })

        # ── Clip frame count + resolution ──────────────────────────────────────
        # clip_frames is the H3 output length: how many frames the model should
        # generate.  frame_load_cap and select_every_nth are VHS reference-loading
        # constraints and must NOT affect the output frame count.
        duration_s  = max(0.0, clip.get("end_time", 0.0) - clip.get("start_time", 0.0))
        clip_frames = math.ceil(duration_s * max(1, int(clip_duration_multiplier or 1)) * 24)
        _max = int(max_clip_frames or 0)
        if _max > 0:
            clip_frames = min(clip_frames, _max)
        vid_w, vid_h = _spa_probe_resolution(video_abs) if video_abs else (0, 0)

        # ── Summary ────────────────────────────────────────────────────────────
        slot_desc = ", ".join(
            f"{SOURCE_SLOTS[i]}={subject_index[sid].get('label', sid)}"
            for i, sid in enumerate(source_subject_ids)
        )
        concept_ids_str = ", ".join(c for c in concept_ids if c)

        # Bundle substitution summary — one line per matched entry
        bundle_lines: list[str] = []
        if cast_lookup and bundle_reg is None:
            bundle_lines.append("  WARNING: bundle registry failed to load — no substitutions applied")
        for src_sid, ce in cast_lookup.items():
            src_label = subject_index.get(src_sid, {}).get("label", src_sid)
            bun_id    = ce.get("bundle_id", "")
            bun       = bundle_reg.get(bun_id) if bundle_reg else None
            if bun is None:
                bundle_lines.append(f"  {src_label} → bundle '{bun_id}' NOT FOUND in registry")
            else:
                vm      = ce.get("visual_mode") or bun.get("visual", {}).get("type", "images")
                nimgs   = len(bun.get("visual", {}).get("files", []))
                has_vid = bool(bun.get("visual", {}).get("file", ""))
                aud     = bun.get("audio", {})
                aud_src = aud.get("source", "none")
                if aud_src == "extract_from_visual":
                    audio_str = f"audio=extract_from_visual:{'ok' if has_vid else 'NO VIDEO'}"
                elif aud_src == "extract_from_video":
                    aud_vf = aud.get("video_file", "")
                    audio_str = f"audio=extract_from_video:{aud_vf or 'NO FILE'}"
                elif aud_src == "file":
                    audio_str = f"audio=file:{aud.get('file', '') or 'NO FILE'}"
                else:
                    audio_str = "audio=none"
                if vm == "video":
                    media_str = f"video={'yes' if has_vid else 'NO'}"
                elif vm == "both":
                    media_str = f"images={nimgs}+video={'yes' if has_vid else 'NO'}"
                else:
                    media_str = f"images={nimgs}"
                bundle_lines.append(
                    f"  {src_label} → '{bun.get('name', bun_id)}' "
                    f"[mode={vm}, {media_str}, {audio_str}]"
                )

        # Slot map — shows which slots actually reached the assembler
        slot_map_parts = []
        for sl, sa in sorted(slot_assignments.items()):
            nimgs_sl = len(sa.get("character_sheet_images", []))
            has_aud  = bool(sa.get("voice", {}).get("audio_reference_file"))
            marker   = "📷" if nimgs_sl else ""
            marker  += "🔊" if has_aud else ""
            slot_map_parts.append(f"{sl}={sa.get('name', sl)}{marker}")
        slot_map_str = ", ".join(slot_map_parts)
        bundle_section = ("\nBundles:\n" + "\n".join(bundle_lines)) if bundle_lines else ""

        res_str = f"{vid_w}×{vid_h}" if vid_w and vid_h else "unknown"
        proxy_used = video_file_for_entry != video_abs and video_file_for_entry
        proxy_note = f"\nProxy: {os.path.basename(video_file_for_entry)}" if proxy_used else ""
        clip_summary = (
            f"Profile: {profile_name} | Clip: {clip_label} ({clip_id_used})\n"
            f"Duration: {duration_s:.1f}s → {clip_frames} frames | Resolution: {res_str}\n"
            f"Slots: {slot_map_str}\n"
            f"Model: {model_type} | LoRAs: {len(lora_stack_data)}\n"
            f"Concept IDs: {concept_ids_str or '(none)'}"
            f"{bundle_section}"
            f"{proxy_note}"
        )

        # ── Filename prefix ────────────────────────────────────────────────────
        def _slug(s: str) -> str:
            import re as _re
            return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

        filename_prefix_out = f"{filename_prefix}{_slug(profile_name)}/{_slug(clip_label)}"

        send_status_update(
            cls.node_id,
            f"Clip: {clip_label} | {duration_s:.0f}s → {clip_frames}fr | {model_type} | {len(prompt)} chars",
        )

        return io.NodeOutput(prompt, h3_refplan, lora_stack_data, concept_ids_str, clip_frames, vid_w, vid_h, clip_summary, filename_prefix_out)


# ── Subject REST API endpoints ────────────────────────────────────────────────





# ── Source Profile REST API endpoints ─────────────────────────────────────────

@routes.post("/fbtools/source_profiles/reload")
async def _source_profiles_reload(request):
    """Increment reload counter so SourceProfileLoad/List nodes re-execute."""
    _source_profile_reload_counter = bump_reload("source_profile")
    logger.info("Source profiles reload requested (counter=%d)", _source_profile_reload_counter)
    return web.json_response({"success": True, "counter": _source_profile_reload_counter})


@routes.get("/fbtools/source_profiles/list")
async def _source_profiles_list(request):
    """Return [{id, name, media_type, media_filename, media_dir, subject_count}] sorted by name."""
    try:
        registry = _load_source_registry(default_source_profiles_path())
        items = []
        for pid, p in registry.profiles.items():
            items.append({
                "id":             pid,
                "name":           p.get("name", pid),
                "media_type":     p.get("media_type", "video"),
                "media_filename": p.get("media_filename", ""),
                "media_dir":      p.get("media_dir", "input"),
                "subject_count":  len(p.get("subjects", [])),
            })
        items.sort(key=lambda x: x["name"].lower())
        return web.json_response({"profiles": items})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/get")
async def _source_profiles_get_one(request):
    """Return a single source profile by ?id=<profile_id> or ?name=<profile_name>."""
    pid  = request.rel_url.query.get("id", "").strip()
    name = request.rel_url.query.get("name", "").strip()
    if not pid and not name:
        return web.json_response({"error": "id or name parameter required"}, status=400)
    try:
        registry = _load_source_registry(default_source_profiles_path())
        if pid:
            profile = registry.get_profile(pid)
        else:
            profile = registry.get_profile_by_name(name)
        if profile is None:
            key = pid or name
            return web.json_response({"error": f"Source profile '{key}' not found"}, status=404)
        return web.json_response(profile)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/save")
async def _source_profiles_save(request):
    """Create or update a source profile. Body: full profile dict with 'id'."""
    try:
        data = await request.json()
        pid = data.get("id", "").strip()
        if not pid:
            return web.json_response({"error": "Profile 'id' is required"}, status=400)
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        seg_dur_raw = data.get("default_segment_duration")
        registry = registry.define_profile(
            profile_id=pid,
            name=data.get("name", pid),
            media_filename=data.get("media_filename", ""),
            media_dir=data.get("media_dir", "input"),
            media_type=data.get("media_type", "video"),
            default_segment_duration=float(seg_dur_raw) if seg_dur_raw is not None else None,
        )
        if "subjects" in data:
            registry = registry.set_subjects(pid, data["subjects"])
        if "clips" in data:
            registry = registry.set_clips(pid, data["clips"])
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"success": True, "id": pid})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/source_profiles/delete")
async def _source_profiles_delete(request):
    """Delete a source profile by ?id=<profile_id>."""
    pid = request.rel_url.query.get("id", "")
    if not pid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        if registry.get_profile(pid) is None:
            return web.json_response({"error": f"Source profile '{pid}' not found"}, status=404)
        registry = registry.remove_profile(pid)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)










@routes.post("/fbtools/source_profiles/analyze")
async def _source_profiles_analyze(request: web.Request) -> web.Response:
    """Run a focused VLM pass on a source profile's media.

    JSON body:
        profile_id        str   — profile to analyze
        pass_type         str   — one of PASS_TYPES ("people", "setting", …)
        prompt_override   str   — optional; replaces the template body
        captioner_type    str   — "gemini_flash" uses Gemini API (reads GEMINI_API_KEY
                                  env var); everything else routes through the LLM panel model
        start_time        float — clip start in seconds (omit for single-frame mode)
        end_time          float — clip end in seconds   (omit for single-frame mode)
        select_every_nth  int   — frame sampling stride (default: clip's own value or 1)
        max_frames        int   — hard cap on frames sent to the VLM (default: 20)

    Returns:
        {
          "candidates": [ {label, role_description, entity_type, notes}, … ],
          "pass_type":  str,
          "prompt":     str,
          "frame_mode": "clip_multi" | "single",
          "frame_count": int,
        }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        prompt_override  = str(body.get("prompt_override", "")).strip()
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        api_key          = os.environ.get("GEMINI_API_KEY", "")
        _raw_start       = body.get("start_time")
        _raw_end         = body.get("end_time")
        start_time       = float(_raw_start) if _raw_start is not None else None
        end_time         = float(_raw_end)   if _raw_end   is not None else None
        select_every_nth     = int(body.get("select_every_nth", 1)) or 1
        max_frames           = int(body.get("max_frames", 20)) or 20
        video_duration       = float(body.get("video_duration", 0.0)) or 0.0
        batch_window_seconds = max(30.0, float(body.get("batch_window_seconds", 60.0)))
        # Accept either pass_types (list) or legacy pass_type (string)
        _raw_pass_types = body.get("pass_types")
        if isinstance(_raw_pass_types, list) and _raw_pass_types:
            pass_types = [str(pt).strip() for pt in _raw_pass_types if str(pt).strip() in _SPA_PASS_TYPES]
        else:
            _single = str(body.get("pass_type", "people")).strip()
            pass_types = [_single] if _single in _SPA_PASS_TYPES else ["people"]
        pass_type = pass_types[0]  # primary type for single-pass history label
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not pass_types:
        return web.json_response(
            {"error": f"pass_type must be one of {_SPA_PASS_TYPES}"}, status=400
        )

    use_clip_mode = (start_time is not None and end_time is not None and end_time > start_time)

    # Load the profile to get the media file
    try:
        registry = _load_source_registry(default_source_profiles_path())
    except Exception as exc:
        return web.json_response({"error": f"Could not load source profiles: {exc}"}, status=500)

    profile = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    media_type     = profile.get("media_type", "video")

    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    # Resolve absolute path
    base_dir = get_output_directory() if media_dir == "output" else get_input_directory()
    abs_media = os.path.join(base_dir, media_filename)
    if not os.path.exists(abs_media):
        return web.json_response({"error": f"Media file not found: {abs_media}"}, status=404)

    # Build the analysis prompt — combined for multiple pass types, single otherwise
    prompt = _spa_build_multi_prompt(pass_types, prompt_override)

    pass_label = "+".join(pass_types)
    send_status_update(
        _SPA_STATUS_ID,
        f"Source analysis: preparing {media_type} {'clip' if use_clip_mode else 'frame'} "
        f"for '{profile_id}' ({pass_label} pass)",
        source="source_profile_analysis",
    )

    frame_count = 1

    if use_clip_mode and media_type == "video":
        # Multi-frame clip path
        try:
            raw_fps = _spa_probe_fps(abs_media)
            pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                abs_media,
                start_time=start_time,
                end_time=end_time,
                max_frames=max_frames,
                select_every_nth=select_every_nth,
                raw_fps=raw_fps,
            )
            frame_count = len(pil_frames)
        except Exception as exc:
            return web.json_response({"error": f"Could not extract clip frames: {exc}"}, status=500)

        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: {frame_count} frames extracted, calling LLM ({pass_type} pass)",
            source="source_profile_analysis",
        )

        try:
            if captioner_type == "gemini_flash":
                # Gemini API doesn't support video_frames — use a contact sheet
                sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    sheet.save(tmp.name, quality=85)
                    tmp.close()
                    raw_response = await asyncio.to_thread(
                        _run_vision_inference,
                        tmp.name, prompt, "gemini_flash",
                        "auto", None, False, profile_id, pass_type,
                    )
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
            else:
                raw_response = await asyncio.to_thread(
                    _run_vision_inference_clip,
                    pil_frames, timestamps, sample_fps, raw_fps, prompt,
                    captioner_type, profile_id, pass_type,
                )

        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        frame_mode = "clip_multi"

    elif media_type == "video" and video_duration > 0:
        # Windowed whole-video analysis: split into batch_window_seconds windows,
        # run the clip-mode path on each, then deduplicate candidates by label.
        if video_duration <= batch_window_seconds:
            analyze_windows: list[tuple[float, float]] = [(0.0, video_duration)]
        else:
            analyze_windows = []
            w = 0.0
            while w < video_duration:
                analyze_windows.append((w, min(w + batch_window_seconds, video_duration)))
                w += batch_window_seconds

        n_windows = len(analyze_windows)
        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: {n_windows} window(s) × {batch_window_seconds:.0f}s "
            f"over {video_duration:.0f}s video ({pass_type} pass)",
            source="source_profile_analysis",
        )

        raw_fps = _spa_probe_fps(abs_media)
        all_candidates_raw: list[dict] = []
        seen_labels: set[str] = set()
        total_frames = 0

        for win_idx, (w_start, w_end) in enumerate(analyze_windows):
            if n_windows > 1:
                send_status_update(
                    _SPA_STATUS_ID,
                    f"Window {win_idx + 1}/{n_windows}: {w_start:.0f}s – {w_end:.0f}s ({pass_type}) ...",
                    source="source_profile_analysis",
                )
            try:
                pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                    abs_media,
                    start_time=w_start,
                    end_time=w_end,
                    max_frames=max_frames,
                    select_every_nth=select_every_nth,
                    raw_fps=raw_fps,
                )
            except Exception as exc:
                logger.warning("analyze: frame extraction failed for window %s–%s: %s",
                               w_start, w_end, exc)
                continue

            total_frames += len(pil_frames)
            send_status_update(
                _SPA_STATUS_ID,
                f"Window {win_idx + 1}/{n_windows}: {len(pil_frames)} frame(s) extracted, calling VLM...",
                source="source_profile_analysis",
            )

            try:
                if captioner_type == "gemini_flash":
                    sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                    import tempfile as _tf2
                    tmp2 = _tf2.NamedTemporaryFile(suffix=".jpg", delete=False)
                    try:
                        sheet.save(tmp2.name, quality=85)
                        tmp2.close()
                        win_raw = await asyncio.to_thread(
                            _run_vision_inference,
                            tmp2.name, prompt, "gemini_flash",
                            "auto", None, False, profile_id, pass_type,
                        )
                    finally:
                        try: os.unlink(tmp2.name)
                        except Exception: pass
                else:
                    win_raw = await asyncio.to_thread(
                        _run_vision_inference_clip,
                        pil_frames, timestamps, sample_fps, raw_fps, prompt,
                        captioner_type, profile_id, pass_type,
                    )
            except Exception as exc:
                logger.warning("analyze: VLM call failed for window %s–%s: %s",
                               w_start, w_end, exc)
                continue

            for cand in _spa_parse_response(win_raw, pass_type):
                label_key = cand.get("label", "").strip().lower()
                if label_key and label_key not in seen_labels:
                    seen_labels.add(label_key)
                    all_candidates_raw.append(cand)

            if n_windows > 1:
                send_status_update(
                    _SPA_STATUS_ID,
                    f"Window {win_idx + 1}/{n_windows}: {len(all_candidates_raw)} unique candidate(s) so far",
                    source="source_profile_analysis",
                )

        candidates = all_candidates_raw
        frame_count = total_frames
        raw_response = ""
        frame_mode = "windowed_multi"

    else:
        # Single-frame path (image media, or video with no duration provided)
        frame_path = abs_media
        _tmp_frame = None
        if media_type == "video":
            try:
                _tmp_frame = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                _tmp_frame.close()
                frame_path = _spa_extract_frame(abs_media, _tmp_frame.name, position_frac=0.10)
            except Exception as exc:
                if _tmp_frame and os.path.exists(_tmp_frame.name):
                    os.unlink(_tmp_frame.name)
                return web.json_response(
                    {"error": f"Could not extract video frame: {exc}"}, status=500
                )

        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: calling LLM for {pass_type} pass",
            source="source_profile_analysis",
        )

        try:
            raw_response = await asyncio.to_thread(
                _run_vision_inference,
                str(frame_path), prompt, captioner_type,
                "auto", None, False, profile_id, pass_type,
            )
        finally:
            if _tmp_frame and os.path.exists(_tmp_frame.name):
                try: os.unlink(_tmp_frame.name)
                except Exception: pass

        frame_mode = "single"

    # Windowed mode builds candidates directly; other paths produce raw_response to parse.
    if frame_mode != "windowed_multi":
        candidates = _spa_parse_response(raw_response, pass_type)

    # Persist to history — use joined label for multi-pass runs, store pass_types list
    history_pass_type = "+".join(pass_types) if len(pass_types) > 1 else pass_type
    _spa_append_history(
        data_dir   = user_data_dir(),
        profile_id = profile_id,
        media_file = media_filename,
        pass_type  = history_pass_type,
        prompt     = prompt,
        candidates = candidates,
        pass_types = pass_types,
    )

    send_status_update(
        _SPA_STATUS_ID,
        f"Source analysis: {len(candidates)} candidate(s) found ({history_pass_type} pass, "
        f"{frame_count} frame(s))",
        source="source_profile_analysis",
    )

    return web.json_response({
        "candidates":  candidates,
        "pass_type":   history_pass_type,
        "pass_types":  pass_types,
        "prompt":      prompt,
        "frame_mode":  frame_mode,
        "frame_count": frame_count,
    })


@routes.get("/fbtools/source_profiles/frame_at")
async def _source_profiles_frame_at(request: web.Request) -> web.Response:
    """Return a single JPEG frame at an exact timestamp — used by the clip
    editor's Start/End boundary thumbnails. Returns raw image bytes (not JSON)
    so it can be used directly as an <img> src.

    Query params:
        profile_id  str   — required
        t           float — timestamp in seconds — required
        w           int   — thumbnail width in pixels (default 160, clamped 32-640)
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    t_raw      = request.rel_url.query.get("t", "")
    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    try:
        t = float(t_raw)
    except ValueError:
        return web.json_response({"error": "t must be a number"}, status=400)
    try:
        w = max(32, min(640, int(request.rel_url.query.get("w", "160"))))
    except ValueError:
        w = 160

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir   = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _spa_extract_frame_at_time, video_path, tmp.name, t, w)
        with open(tmp.name, "rb") as f:
            data = f.read()
        return web.Response(
            body=data, content_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except Exception as exc:
        logger.warning("frame_at extraction failed for profile %r @ %.2fs: %s", profile_id, t, exc)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


@routes.get("/fbtools/source_profiles/analysis_history")
async def _source_profiles_analysis_history(request: web.Request) -> web.Response:
    """Return analysis history entries for a source profile.

    Query params:
        profile_id  str   — required
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    if not profile_id:
        return web.json_response({"error": "profile_id parameter required"}, status=400)
    try:
        entries = _spa_history_for_profile(user_data_dir(), profile_id)
        return web.json_response({"entries": entries})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/segment_prompt_preview")
async def _source_profiles_segment_prompt_preview(request: web.Request) -> web.Response:
    """Return the exact VLM prompt Detect boundaries would send for the given
    flags/override, without running any detection. Lets the editor UI show a
    live preview as the user toggles flags or edits the override — sourced
    from the same builder the real request uses, so it can never drift.

    JSON body:
        prompt_override  str   — optional; when non-empty, flags are ignored
        flags             dict  — camera_cuts / subject_changes / lower_threshold

    Returns:
        { "prompt": str }
    """
    try:
        body            = await request.json()
        prompt_override = str(body.get("prompt_override", "")).strip()
        flags           = body.get("flags") if isinstance(body.get("flags"), dict) else None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    prompt = _spa_build_segment_prompt(prompt_override, flags)
    return web.json_response({"prompt": prompt})


@routes.post("/fbtools/source_profiles/detect_segments")
async def _source_profiles_detect_segments(request: web.Request) -> web.Response:
    """Run VLM boundary detection on a source profile's video.

    Extracts one frame per interval_seconds across the full duration,
    composites them into a labelled contact sheet, then asks the VLM
    to identify meaningful action transitions.

    JSON body:
        profile_id          str   — required
        video_duration      float — required (caller probes via /fbtools/media/info)
        interval_seconds      float — seconds between sampled frames — this is the
                                      single "precision" knob the UI exposes
                                      (default: 3.0)
        batch_window_seconds  float — split video into windows of this many seconds;
                                      each window is a separate VLM call. Normally
                                      OMITTED and auto-derived as interval_seconds * 20
                                      so every window uses the full 20-frame-per-call
                                      budget evenly (no wasted capacity, no silent
                                      truncation). Pass explicitly only to override.
        prompt_override       str   — optional full prompt replacement (bypasses flags)
        flags               dict  — optional flag overrides for prompt construction:
                                    camera_cuts (bool, default True)
                                    subject_changes (bool, default False)
                                    lower_threshold (bool, default False)
        captioner_type      str   — "qwen_vl" | "qwen_omni" | "gemini_flash"
        device              str   — "auto" | "cpu" | "cuda"
        use_8bit            bool

    Returns:
        { "segments": [{start_time, end_time, label, action}, …], "raw_response": str }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        video_duration   = float(body.get("video_duration", 0.0))
        interval_seconds = float(body.get("interval_seconds", 0.0)) or 0.0
        prompt_override  = str(body.get("prompt_override", "")).strip()
        flags            = body.get("flags") if isinstance(body.get("flags"), dict) else None
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        device           = str(body.get("device", "auto")).strip()
        _use_8bit_raw    = body.get("use_8bit")
        use_8bit         = bool(_use_8bit_raw) if _use_8bit_raw is not None else None
        api_key          = os.environ.get("GEMINI_API_KEY", "")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if video_duration <= 0:
        return web.json_response({"error": "video_duration must be > 0"}, status=400)

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir    = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path  = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    if not interval_seconds:
        interval_seconds = 3.0

    # Split video into fixed-size windows; each window is processed separately so
    # the VLM sees a focused contact sheet rather than one frame per many seconds.
    # Window length is derived from interval_seconds so every window lands on
    # exactly 20 frames (the per-call cap below) — full utilization of the frame
    # budget at the requested precision, with no silent truncation.
    _bw_raw = body.get("batch_window_seconds")
    batch_window_seconds = max(20.0, float(_bw_raw)) if _bw_raw else max(20.0, interval_seconds * 20.0)

    _tmp_frames: list[str] = []

    async def _run_window(win_start: float, win_end: float) -> tuple[list[dict], str]:
        """Build a contact sheet for [win_start, win_end) and run the VLM.

        Frames carry absolute timestamps, so the VLM returns absolute
        start_time/end_time values that can be concatenated across windows
        without any offset adjustment.
        """
        win_ts: list[float] = []
        t = win_start + interval_seconds / 2.0
        while t < win_end:
            win_ts.append(t)
            t += interval_seconds
        if not win_ts:
            win_ts = [(win_start + win_end) / 2.0]

        frame_paths: list[str] = []
        for ts in win_ts[:20]:  # cap at 20 frames per window
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            _tmp_frames.append(tmp.name)
            try:
                _spa_extract_frame(video_path, tmp.name, position_frac=ts / video_duration)
                frame_paths.append(tmp.name)
            except Exception:
                pass

        if not frame_paths:
            return [], ""

        contact_path = tempfile.mktemp(suffix=".jpg")
        _tmp_frames.append(contact_path)
        try:
            from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw
            imgs = [_PIL_Image.open(p).convert("RGB") for p in frame_paths]
            # Smaller thumbnails = fewer input tokens; readable enough for boundary detection
            thumb_w, thumb_h = 200, 112
            imgs = [img.resize((thumb_w, thumb_h)) for img in imgs]
            n_cols = min(len(imgs), 6)
            n_rows = (len(imgs) + n_cols - 1) // n_cols
            sheet = _PIL_Image.new("RGB", (thumb_w * n_cols, (thumb_h + 20) * n_rows), (30, 30, 30))
            draw  = _PIL_Draw.Draw(sheet)
            for i, (img, ts) in enumerate(zip(imgs, win_ts[:len(imgs)])):
                row, col = divmod(i, n_cols)
                x, y = col * thumb_w, row * (thumb_h + 20)
                sheet.paste(img, (x, y))
                draw.text((x + 4, y + thumb_h + 2), f"{ts:.1f}s", fill=(200, 200, 200))
            sheet.save(contact_path, quality=85)
        except Exception:
            import shutil as _shutil
            _shutil.copy2(frame_paths[0], contact_path)

        send_status_update(
            _SPA_STATUS_ID,
            f"Extracted {len(frame_paths)} frame(s) from {win_start:.0f}s–{win_end:.0f}s — querying VLM...",
            source="source_profile_analysis",
        )

        window_note = (
            f"\n\nNOTE: These frames cover {win_start:.1f}s – {win_end:.1f}s of a "
            f"{video_duration:.1f}s video. Return all start_time/end_time values as "
            f"absolute timestamps (seconds from the start of the full video) within "
            f"this range."
        )
        win_prompt = _spa_build_segment_prompt(prompt_override, flags) + window_note

        raw = await asyncio.to_thread(
            _run_vision_inference,
            contact_path, win_prompt, captioner_type, device, use_8bit,
            False, profile_id, "detect_segments",
            2048,  # max_tokens: segment lists can be long; 512 truncates JSON mid-object
        )
        # video_duration=0 suppresses the auto-fallback — caller handles empty windows
        segs = _spa_parse_segments(raw, 0.0)
        return segs, raw or ""

    # Build non-overlapping windows covering the full video
    if video_duration <= batch_window_seconds:
        windows: list[tuple[float, float]] = [(0.0, video_duration)]
    else:
        windows = []
        w = 0.0
        while w < video_duration:
            windows.append((w, min(w + batch_window_seconds, video_duration)))
            w += batch_window_seconds

    try:
        n_windows = len(windows)
        run_started_at = time.time()
        send_status_update(
            _SPA_STATUS_ID,
            (f"Detect segments: {n_windows} window(s) × {batch_window_seconds:.0f}s "
             f"over {video_duration:.0f}s video — sending to VLM..."),
            source="source_profile_analysis",
            extra={
                "phase":       "start",
                "profile_id":  profile_id,
                "n_windows":   n_windows,
                "windows":     [[round(s, 2), round(e, 2)] for s, e in windows],
                "video_duration": round(video_duration, 2),
            },
        )

        all_segments: list[dict] = []
        all_raw:      list[str]  = []

        for idx, (w_start, w_end) in enumerate(windows):
            win_started_at = time.time()
            start_text = (
                f"Window {idx + 1}/{n_windows}: {w_start:.0f}s – {w_end:.0f}s ..."
                if n_windows > 1 else "Analyzing video for boundaries..."
            )
            send_status_update(
                _SPA_STATUS_ID, start_text, source="source_profile_analysis",
                extra={
                    "phase": "window_start", "profile_id": profile_id,
                    "window_idx": idx, "n_windows": n_windows,
                },
            )
            segs, raw_text = await _run_window(w_start, w_end)
            all_segments.extend(segs)
            if raw_text:
                prefix = f"[{w_start:.0f}s–{w_end:.0f}s]\n" if n_windows > 1 else ""
                all_raw.append(f"{prefix}{raw_text}")
            win_elapsed = time.time() - win_started_at
            done_text = (
                f"Window {idx + 1}/{n_windows}: {len(segs)} segment(s) found"
                if n_windows > 1 else f"{len(segs)} segment(s) found"
            )
            _win_frames = min(20, math.ceil((w_end - w_start) / interval_seconds)) if interval_seconds > 0 else 0
            send_status_update(
                _SPA_STATUS_ID, done_text, source="source_profile_analysis",
                extra={
                    "phase": "window_done", "profile_id": profile_id,
                    "window_idx": idx, "n_windows": n_windows,
                    "segments_found": len(segs),
                    "elapsed_s": round(win_elapsed, 2),
                    "frames": _win_frames,
                },
            )

        total_elapsed = time.time() - run_started_at
        send_status_update(
            _SPA_STATUS_ID,
            (f"Detect segments complete: {len(all_segments)} segment(s) "
             f"from {n_windows} window(s)"),
            source="source_profile_analysis",
            extra={
                "phase": "complete", "profile_id": profile_id,
                "n_windows": n_windows, "total_segments": len(all_segments),
                "elapsed_s": round(total_elapsed, 2),
            },
        )

        if not all_segments:
            all_segments.append({
                "start_time": 0.0,
                "end_time":   round(video_duration, 3),
                "label":      "Full video",
                "action":     "",
            })

        # --- Infer subjects from action descriptions via text-only LLM ---
        inferred_subjects: list[dict] = []
        action_texts = [s.get("action", "") for s in all_segments if s.get("action", "").strip()]
        if len(action_texts) >= 2:
            send_status_update(
                _SPA_STATUS_ID,
                f"Inferring subjects from {len(action_texts)} action description(s)...",
                source="source_profile_analysis",
            )
            try:
                existing_labels = [
                    s.get("label", "") for s in profile.get("subjects", [])
                    if s.get("label", "").strip()
                ]
                subj_prompt = _spa_build_subject_inference_prompt(action_texts, existing_labels)
                raw_subjects = await asyncio.to_thread(
                    _run_text_inference,
                    # 1024 was too tight for thinking-mode models (e.g. reasoning_effort=xhigh):
                    # the <think> block alone can exhaust the budget before any JSON is emitted,
                    # silently yielding an empty response with zero error surfaced. Match the
                    # 2048 already used for segment detection for the same reason.
                    subj_prompt, captioner_type, profile_id, "subject_inference", 2048,
                )
                if not raw_subjects:
                    logger.warning(
                        "Subject inference returned empty text for profile %r "
                        "(model may be thinking-only-truncated or backend unavailable)",
                        profile_id,
                    )
                if raw_subjects:
                    inferred_subjects = _spa_parse_inferred_subjects(raw_subjects)
            except Exception as _subj_exc:
                logger.warning("Subject inference from detect segments failed: %s", _subj_exc)
            send_status_update(
                _SPA_STATUS_ID,
                f"Detect segments done: {len(all_segments)} segment(s), "
                f"{len(inferred_subjects)} inferred subject(s)",
                source="source_profile_analysis",
            )

        return web.json_response({
            "segments":          all_segments,
            "raw_response":      "\n\n".join(all_raw),
            "inferred_subjects": inferred_subjects,
        })

    except Exception as exc:
        logger.exception("detect_segments failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        for p in _tmp_frames:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


@routes.post("/fbtools/source_profiles/describe_clip")
async def _source_profiles_describe_clip(request: web.Request) -> web.Response:
    """Run VLM to generate an action description for a clip.

    Extracts up to max_frames frames spread across [start_time, end_time] and
    sends them to the VLM as a contact sheet (or natively for video-capable
    backends).  Falls back to a single midpoint frame for non-video media.

    JSON body:
        profile_id        str
        start_time        float   — clip start (seconds)
        end_time          float   — clip end (seconds)
        prompt_override   str     — optional
        existing_action   str     — optional; VLM refines rather than generating fresh
        captioner_type    str
        device            str
        use_8bit          bool
        max_frames        int     — max frames to sample (default 5)
        select_every_nth  int     — frame stride before applying max_frames cap (default 1)

    Returns:
        { "action": "1-2 sentence action description", "frame_count": int }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        start_time       = float(body.get("start_time", 0.0))
        end_time         = float(body.get("end_time", 0.0))
        prompt_override  = str(body.get("prompt_override", "")).strip()
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        device           = str(body.get("device", "auto")).strip()
        _use_8bit_raw    = body.get("use_8bit")
        use_8bit         = bool(_use_8bit_raw) if _use_8bit_raw is not None else None
        existing_action  = str(body.get("existing_action", "")).strip()
        max_frames       = int(body.get("max_frames", 5)) or 5
        select_every_nth = int(body.get("select_every_nth", 1)) or 1
        raw_subjects     = body.get("subjects") or []
        subjects: list[tuple[str, str, str]] = [
            (str(s.get("slot", "")).strip(),
             str(s.get("name", "")).strip(),
             str(s.get("appearance", "")).strip())
            for s in raw_subjects
            if isinstance(s, dict) and s.get("slot") and s.get("name")
        ]
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    media_type     = profile.get("media_type", "video")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir   = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    prompt = _spa_build_clip_desc_prompt(
        prompt_override,
        subjects=subjects or None,
        existing_action=existing_action,
    )

    send_status_update(
        _SPA_STATUS_ID,
        f"Describe clip: extracting frames for '{profile_id}' ({start_time:.1f}–{end_time:.1f}s)",
        source="source_profile_analysis",
    )

    frame_count = 1
    try:
        if media_type == "video" and end_time > start_time:
            # Multi-frame path — mirrors the analyze endpoint
            raw_fps = _spa_probe_fps(video_path)
            pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                video_path,
                start_time=start_time,
                end_time=end_time,
                max_frames=max_frames,
                select_every_nth=select_every_nth,
                raw_fps=raw_fps,
            )
            frame_count = len(pil_frames)
            send_status_update(
                _SPA_STATUS_ID,
                f"Describe clip: {frame_count} frames extracted, calling LLM",
                source="source_profile_analysis",
            )
            if captioner_type == "gemini_flash":
                sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    sheet.save(tmp.name, quality=85)
                    tmp.close()
                    raw = await asyncio.to_thread(
                        _run_vision_inference,
                        tmp.name, prompt, "gemini_flash",
                        device, use_8bit, False, profile_id, "describe_clip",
                    )
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
            else:
                raw = await asyncio.to_thread(
                    _run_vision_inference_clip,
                    pil_frames, timestamps, sample_fps, raw_fps,
                    prompt, captioner_type, profile_id, "describe_clip",
                )
        else:
            # Non-video or zero-duration: single midpoint frame fallback
            midpoint = (start_time + end_time) / 2.0 if end_time > start_time else start_time
            _dur: float = 0.0
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True, timeout=10,
                )
                _dur = float(result.stdout.strip())
            except Exception:
                pass
            frac = (midpoint / _dur) if _dur > 0 else 0.1
            _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            _tmp.close()
            try:
                _spa_extract_frame(video_path, _tmp.name, position_frac=max(0.0, min(1.0, frac)))
                raw = await asyncio.to_thread(
                    _run_vision_inference,
                    _tmp.name, prompt, captioner_type, device, use_8bit,
                    False, profile_id, "describe_clip",
                )
            finally:
                try: os.unlink(_tmp.name)
                except Exception: pass

        action = _spa_parse_clip_desc(raw)

        _spa_append_history(
            data_dir    = user_data_dir(),
            profile_id  = profile_id,
            media_file  = media_filename,
            pass_type   = "describe_clip",
            prompt      = prompt,
            candidates  = [],
            clip_start  = start_time,
            clip_end    = end_time,
            action      = action,
            frame_count = frame_count,
        )

        return web.json_response({"action": action, "frame_count": frame_count})

    except Exception as exc:
        logger.exception("describe_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        pass  # temp files cleaned up inside each branch above


@routes.post("/fbtools/source_profiles/auto_partition")
async def _source_profiles_auto_partition(request: web.Request) -> web.Response:
    """Auto-partition a source profile's video into equal-duration clips.

    JSON body:
        profile_id       str
        video_duration   float   — total video length in seconds
        segment_duration float   — segment length (0 = use profile default or 10s)

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        video_duration   = float(body.get("video_duration", 0.0))
        segment_duration = float(body.get("segment_duration", 0.0)) or None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if video_duration <= 0:
        return web.json_response({"error": "video_duration must be > 0"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.auto_partition(
            profile_id, video_duration, segment_duration=segment_duration
        )
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("auto_partition failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/set_clips")
async def _source_profiles_set_clips(request: web.Request) -> web.Response:
    """Replace a profile's entire clip list.

    Used to apply VLM-detected segment suggestions as clips in one shot.

    JSON body:
        profile_id  str
        clips       [{id, label, start_time, end_time, action, subjects, ...}, ...]

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clips      = body.get("clips")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(clips, list):
        return web.json_response({"error": "clips must be a list"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.set_clips(profile_id, clips)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("set_clips failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/merge_subjects")
async def _source_profiles_merge_subjects(request: web.Request) -> web.Response:
    """Add inferred subjects to a profile without duplicating existing ones.

    Subjects are deduplicated by label (case-insensitive).  New subjects get
    an auto-generated ID of the form ``subj_<8hex>``.

    JSON body:
        profile_id  str
        subjects    [{label, role_description, entity_type}, ...]

    Returns: { "profile": <updated profile dict>, "added": N }
    """
    import uuid as _uuid

    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        subjects   = body.get("subjects")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(subjects, list):
        return web.json_response({"error": "subjects must be a list"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if profile is None:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)

        existing_labels = {
            s.get("label", "").strip().lower()
            for s in profile.get("subjects", [])
        }

        added = 0
        for entry in subjects:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            if not label or label.lower() in existing_labels:
                continue
            subject_id = f"subj_{_uuid.uuid4().hex[:8]}"
            registry = registry.define_subject(
                profile_id,
                subject_id,
                label=label,
                role_description=str(entry.get("role_description", "")).strip(),
                entity_type=str(entry.get("entity_type", "person")).strip(),
            )
            existing_labels.add(label.lower())
            added += 1

        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id), "added": added})
    except Exception as exc:
        logger.exception("merge_subjects failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/upsert_clip")
async def _source_profiles_upsert_clip(request: web.Request) -> web.Response:
    """Add or update a single clip within a source profile.

    JSON body:
        profile_id  str
        clip        { id, label, start_time, end_time, select_every_nth,
                      frame_load_cap, subjects, action }

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clip       = body.get("clip")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(clip, dict) or not clip.get("id"):
        return web.json_response({"error": "clip must be an object with an id field"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.upsert_clip(profile_id, clip)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("upsert_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/remove_clip")
async def _source_profiles_remove_clip(request: web.Request) -> web.Response:
    """Remove a clip from a source profile.

    JSON body:
        profile_id  str
        clip_id     str

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clip_id    = str(body.get("clip_id", "")).strip()
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not clip_id:
        return web.json_response({"error": "clip_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.remove_clip(profile_id, clip_id)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("remove_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/proxy_status")
async def _source_profiles_proxy_status(request: web.Request) -> web.Response:
    """Return proxy freshness for every clip in a profile.

    Query params:
        profile_id  str

    Returns: { "clips": [ { "clip_id", "fresh": bool, "proxy_path": str|null } ] }
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if not profile:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)

        media_filename = profile.get("media_filename", "")
        media_dir      = profile.get("media_dir", "input")
        base_dir       = get_input_directory() if media_dir != "output" else get_output_directory()
        video_abs      = os.path.join(base_dir, media_filename) if media_filename else ""
        short_edge     = profile.get("proxy_short_edge", 768)

        results = []
        for clip in profile.get("clips", []):
            clip_id    = clip.get("id", "")
            start_time = float(clip.get("start_time", 0.0))
            end_time   = float(clip.get("end_time", 0.0))

            # Re-derive the expected proxy path without generating it
            from .utils.proxy_cache import _proxy_dir, _proxy_stem, _is_fresh
            stem       = _proxy_stem(profile_id, clip_id, start_time, end_time, short_edge)
            proxy_path = _proxy_dir(str(user_data_dir())) / f"{stem}.mp4"
            fresh      = _is_fresh(proxy_path, video_abs) if video_abs else False

            results.append({
                "clip_id":    clip_id,
                "fresh":      fresh,
                "proxy_path": str(proxy_path) if fresh else None,
            })

        return web.json_response({"clips": results})
    except Exception as exc:
        logger.exception("proxy_status failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/proxy_stream")
async def _source_profiles_proxy_stream(request: web.Request) -> web.Response:
    """Stream a cached source-profile proxy file for playback.

    ?path=<absolute_path> — must be inside user_data_dir()/proxies/source_profiles/.
    Supports HTTP Range requests so browsers can seek. Mirrors
    /fbtools/bundles/audio_cache/stream's allow-listed-root pattern. Never generates a proxy
    (that's prebuild_proxies' job, as a background job — ffmpeg here can take minutes) — this
    only ever serves one that already exists.
    """
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="path required")
    allowed_root = os.path.realpath(os.path.join(str(user_data_dir()), "proxies", "source_profiles"))
    real_path = os.path.realpath(path)
    if not real_path.startswith(allowed_root + os.sep):
        return web.Response(status=403, text="Forbidden")
    if not os.path.isfile(real_path):
        return web.Response(status=404, text="Not found")
    return web.FileResponse(real_path)


@routes.post("/fbtools/source_profiles/prebuild_proxies")
async def _source_profiles_prebuild_proxies(request: web.Request) -> web.Response:
    """Pre-generate proxies for all clips in a profile.

    JSON body:
        profile_id  str
        clip_id     str|null  (optional — build only this one clip)

    Progress is broadcast via websocket as fbtools.status events with
    source="proxy_build".  Returns immediately with { "started": true };
    the caller polls proxy_status or listens to websocket events.
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        only_clip  = (body.get("clip_id") or "").strip() or None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if not profile:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    base_dir       = get_input_directory() if media_dir != "output" else get_output_directory()
    video_abs      = os.path.join(base_dir, media_filename) if media_filename else ""
    short_edge     = profile.get("proxy_short_edge", 768)
    clips          = profile.get("clips", [])
    if only_clip:
        clips = [c for c in clips if c.get("id") == only_clip]

    if not video_abs or not os.path.exists(video_abs):
        return web.json_response({"error": "Source video file not found"}, status=404)

    def _build_all():
        total = len(clips)
        for i, clip in enumerate(clips):
            clip_id    = clip.get("id", "")
            start_time = float(clip.get("start_time", 0.0))
            end_time   = float(clip.get("end_time", 0.0))
            label      = clip.get("label", clip_id)
            send_status_update(
                "proxy_build",
                f"Building proxy {i + 1}/{total}: {label} ({start_time:.1f}–{end_time:.1f}s)",
                source="proxy_build",
            )
            try:
                result = _ensure_proxy(
                    source_path=video_abs,
                    profile_id=profile_id,
                    clip_id=clip_id,
                    start_time=start_time,
                    end_time=end_time,
                    short_edge=short_edge,
                    base_dir=str(user_data_dir()),
                )
                status = "ready" if result else "failed"
            except Exception as exc:
                status = f"error: {exc}"
                logger.warning("prebuild_proxies: clip %r failed: %s", clip_id, exc)
            send_status_update(
                "proxy_build",
                f"Proxy {i + 1}/{total} {status}: {label}",
                source="proxy_build",
            )
        send_status_update(
            "proxy_build",
            f"Proxy build complete ({total} clip{'s' if total != 1 else ''})",
            source="proxy_build",
        )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build_all)
    return web.json_response({"started": True, "clip_count": len(clips)})


# ── Custom type: SCENE_TEMPLATE ──────────────────────────────────────────────

SCENE_TEMPLATE_TYPE = "SCENE_TEMPLATE"


@io.comfytype(io_type=SCENE_TEMPLATE_TYPE)
class SceneTemplateIOType:
    """Carries a SceneTemplate instance between Load → SceneCompose nodes."""
    Type = object  # SceneTemplate

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "template", **kwargs):
            super().__init__(name, **kwargs)


# ── Scene Template helpers ─────────────────────────────────────────────────────

def _template_get_ids() -> list[str]:
    """Return available template IDs for combo population at schema time."""
    try:
        ids = _scene_template_ids(default_scene_templates_dir())
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: SceneTemplateLoad ───────────────────────────────────────────────────

class SceneTemplateLoad(io.ComfyNode):
    """Load a scene template from the scene_templates directory.

    The combo is populated at extension load time from the user's
    scene_templates/ directory (seeded with bundled examples on first use).
    Press R to refresh the list after adding new templates.
    """
    node_id = prefixed_node_id("SceneTemplateLoad")
    display_name = "Scene Template Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        template_id_options = _template_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "template_id",
                    options=template_id_options,
                    display_name="Template ID",
                    tooltip="Scene template to load. Press R to refresh the list after adding new templates.",
                ),
            ],
            outputs=[
                SceneTemplateIOType.Output(
                    "template",
                    display_name="Scene Template",
                    tooltip="Template object for wiring into SceneCompose.",
                ),
                io.String.Output(
                    "slot_info",
                    display_name="Slot Info",
                    tooltip="Formatted summary of slot requirements.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, template_id: str = "", **_):
        templates_dir = default_scene_templates_dir()
        path = os.path.join(templates_dir, f"{template_id}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("scene_template"))

    @classmethod
    def execute(cls, template_id: str = "") -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        if not template_id or template_id == "(none)":
            logger.warning("SceneTemplateLoad: no template_id selected")
            return io.NodeOutput(None, "")
        path = os.path.join(templates_dir, f"{template_id}.json")
        if not os.path.exists(path):
            logger.warning("SceneTemplateLoad: template not found: %s", path)
            return io.NodeOutput(None, f"Template not found: {template_id}")
        template = _load_scene_template(path)
        slot_info = template.format_slot_info()
        send_status_update(
            cls.node_id,
            f"Loaded: {template.name} | {template.slot_count} slot(s) | {len(template.shots)} shots",
        )
        return io.NodeOutput(
            template,
            slot_info,
            ui={"slot_info": slot_info},
        )


# ── Node: SceneTemplateList ───────────────────────────────────────────────────

class SceneTemplateList(io.ComfyNode):
    """List all available scene templates.

    Scans the scene_templates/ directory and returns a formatted summary.
    Useful for quickly reviewing available templates.
    """
    node_id = prefixed_node_id("SceneTemplateList")
    display_name = "Scene Template List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "template_list",
                    display_name="Template List",
                    tooltip="Formatted list of all available templates.",
                ),
                io.Int.Output("template_count", display_name="Template Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        templates_dir = default_scene_templates_dir()
        return (_templates_dir_fingerprint(templates_dir), reload_counter("scene_template"))

    @classmethod
    def execute(cls) -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        listing = _format_template_list(templates_dir)
        count = len(_scan_scene_templates(templates_dir))
        return io.NodeOutput(listing, count, ui={"template_list": listing, "template_count": count})


# ── Scene Template REST API endpoints ─────────────────────────────────────────





# ── Custom type: OUTFIT_REGISTRY ─────────────────────────────────────────────

OUTFIT_REGISTRY_TYPE = "OUTFIT_REGISTRY"


@io.comfytype(io_type=OUTFIT_REGISTRY_TYPE)
class OutfitRegistryIOType:
    """Carries an OutfitRegistry instance between Load → Define nodes."""
    Type = object  # OutfitRegistry instance

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "outfit_registry", **kwargs):
            super().__init__(name, **kwargs)


# ── Outfit Registry helpers ───────────────────────────────────────────────────

def _outfit_get_ids() -> list[str]:
    """Read outfit_registry.json and return outfit IDs for combo widgets."""
    try:
        reg = _load_outfit_registry(default_outfit_registry_path())
        ids = reg.outfit_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: OutfitRegistryLoad ──────────────────────────────────────────────────

class OutfitRegistryLoad(io.ComfyNode):
    """Load the outfit registry from disk.

    Outputs an OUTFIT_REGISTRY that can be chained through OutfitDefine nodes
    or wired directly into SceneCompose for slot-based outfit lookups.
    """
    node_id = prefixed_node_id("OutfitRegistryLoad")
    display_name = "Outfit Registry Load"
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
                io.String.Input(
                    "registry_file",
                    display_name="Registry File (leave empty for default)",
                    default="",
                    tooltip=(
                        "Absolute path to an outfit_registry.json file. "
                        "Leave empty to use the default user-data location."
                    ),
                    multiline=False,
                ),
            ],
            outputs=[
                OutfitRegistryIOType.Output(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip="Outfit registry to wire into OutfitDefine or SceneCompose.",
                ),
                io.String.Output(
                    "available_outfits",
                    display_name="Available Outfits",
                    tooltip="Human-readable list of all defined outfits.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, registry_file: str = "", **_):
        path = registry_file.strip() or default_outfit_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("outfit"))

    @classmethod
    def execute(cls, registry_file: str = ""):
        path = registry_file.strip() or default_outfit_registry_path()
        registry = _load_outfit_registry(path)
        available = registry.list_outfits()
        return io.NodeOutput(registry, available, ui={"available_outfits": available})


# ── Node: OutfitDefine ────────────────────────────────────────────────────────

class OutfitDefine(io.ComfyNode):
    """Define (or update) one outfit entry in the registry.

    Chain multiple OutfitDefine nodes to build up a registry inline.
    If auto_save is enabled the registry is written back to its source file.
    """
    node_id = prefixed_node_id("OutfitDefine")
    display_name = "Outfit Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                OutfitRegistryIOType.Input(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip="Registry from OutfitRegistryLoad or a previous OutfitDefine.",
                ),
                io.String.Input(
                    "outfit_id",
                    display_name="Outfit ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. casual_summer or formal_black.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Display Name",
                    default="",
                    tooltip="Human-readable label shown in OutfitList.",
                    multiline=False,
                ),
                io.String.Input(
                    "description",
                    display_name="Description",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Outfit description used in prompts. "
                        "Describe garments, colors, materials, and accessories."
                    ),
                ),
                io.String.Input(
                    "tags",
                    display_name="Tags (comma-separated)",
                    default="",
                    multiline=False,
                    tooltip="Optional tags for filtering, e.g. casual, formal, summer.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=False,
                    tooltip="If enabled, persist the updated registry to disk after each execution.",
                ),
            ],
            outputs=[
                OutfitRegistryIOType.Output(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip="Updated registry with this outfit entry added or replaced.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        outfit_registry: OutfitRegistry,
        outfit_id: str = "",
        name: str = "",
        description: str = "",
        tags: str = "",
        auto_save: bool = False,
    ) -> io.NodeOutput:
        outfit_id = outfit_id.strip()
        if not outfit_id:
            send_status_update(cls.node_id, "outfit_id is empty — skipped", level="warn")
            return io.NodeOutput(outfit_registry)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        updated = outfit_registry.define(outfit_id, name, description, tag_list)
        if auto_save:
            try:
                updated.save()
                send_status_update(cls.node_id, f"Saved outfit '{outfit_id}'")
            except Exception as exc:
                logger.warning("OutfitDefine: auto_save failed: %s", exc)
                send_status_update(cls.node_id, f"Save failed: {exc}", level="warn")
        else:
            send_status_update(cls.node_id, f"Defined outfit '{outfit_id}' (not saved)")
        return io.NodeOutput(updated)


# ── Node: OutfitList ──────────────────────────────────────────────────────────

class OutfitList(io.ComfyNode):
    """Display a summary of all outfits in the registry, optionally filtered by tag."""
    node_id = prefixed_node_id("OutfitList")
    display_name = "Outfit List"
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
                OutfitRegistryIOType.Input(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip="Registry to inspect.",
                ),
                io.String.Input(
                    "tag_filter",
                    display_name="Filter by Tag",
                    default="",
                    multiline=False,
                    tooltip="Show only outfits that include this tag. Leave empty for all.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "outfit_list",
                    display_name="Outfit List",
                    tooltip="Formatted list of matching outfits.",
                ),
                io.Int.Output(
                    "outfit_count",
                    display_name="Count",
                    tooltip="Number of matching outfits.",
                ),
            ],
        )

    @classmethod
    def execute(cls, outfit_registry: OutfitRegistry, tag_filter: str = ""):
        flt = tag_filter.strip() or None
        listing = outfit_registry.list_outfits(flt)
        count = len([
            oid for oid, e in outfit_registry.outfits.items()
            if flt is None or flt in e.get("tags", [])
        ])
        return io.NodeOutput(listing, count, ui={"outfit_list": listing, "outfit_count": count})


# ── Custom type: SCENE_INSTANCE ──────────────────────────────────────────────

SCENE_INSTANCE_TYPE = "SCENE_INSTANCE"


@io.comfytype(io_type=SCENE_INSTANCE_TYPE)
class SceneInstanceIOType:
    """Carries a composed scene dict between SceneCompose → PromptAssemble nodes."""
    Type = object  # dict with template, slot_assignments, dialogue, outfit_overrides

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "scene_instance", **kwargs):
            super().__init__(name, **kwargs)


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

# ── Custom type: SCENE_CAST ───────────────────────────────────────────────────

SCENE_CAST_TYPE = "SCENE_CAST"


@io.comfytype(io_type=SCENE_CAST_TYPE)
class CastIOType:
    """Carries a scene cast dict between SceneCastLoad → PromptCompositionLoader nodes."""
    Type = object  # dict: {id, name, entries: [{subject_id, bundle_id, visual_mode, use_audio}], ...}

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)


H3_REFPLAN_TYPE = "FBTOOLS_H3_REFPLAN"


@io.comfytype(io_type=H3_REFPLAN_TYPE)
class H3RefplanType:
    """Ordered reference descriptor bundle from PromptCompositionLoader → CompositionToH3Conditioning.

    Carries descriptors (paths + params) for all references in native node order:
      images → [soundtrack_audio + video] pairs → standalone_audios
    Terminal node decodes media and delegates to MiniMaxH3ReferenceToVideo.execute.
    """
    Type = object  # dict: {prompt, model_type, ref_image_size, references: [...]}

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "scene_cast", **kwargs):
            super().__init__(name, **kwargs)


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


# ── Custom type + loader: PROMPT_COMPOSITION ─────────────────────────────────

COMPOSITION_TYPE = "PROMPT_COMPOSITION"


@io.comfytype(io_type=COMPOSITION_TYPE)
class CompositionIOType:
    """Carries a full saved Prompt Composition dict between nodes."""
    Type = object  # composition dict (see utils/prompt_compositions.py schema)

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "prompt_composition", **kwargs):
            super().__init__(name, **kwargs)


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