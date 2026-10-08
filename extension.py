from __future__ import annotations

from typing_extensions import override
from .nodes.shared import user_data_dir
from .nodes.dataset_caption import DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader
from .nodes.libber import LibberManager, LibberApply
from .nodes.lora_stacks import (
    LoraStackBuilder, LoraStackApply, LoraEntryDefine, LoraStackCollect, WanVidLoraStack,
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
from .nodes.run_tracking import RunMetaCapture, JobCompleteNotifier
from .nodes.concepts import ConceptRegistryLoad, ConceptDefine, ConceptResolve, ConceptList
from .nodes.subjects import SubjectProfileLoad, SubjectProfileDefine, SubjectProfileList
from .nodes.scene_templates import SceneTemplateLoad, SceneTemplateList
from .nodes.outfits import OutfitRegistryLoad, OutfitDefine, OutfitList
from .nodes.source_profiles import SourceProfileLoad, SourceProfileDefine, SourceProfileList, SourceProfileClipPrompt
from .nodes.compose import SceneCompose, PromptAssemble
from .nodes.scene_casts import SceneCastLoad, SceneCastBuild
from .nodes.compositions import CompositionLoad, PromptCompositionLoader, CompositionToH3Conditioning
from .nodes.marker_frame_split import MarkerFrameSplit
from .nodes.h3_source_guides import H3SourceGuides
from .nodes.image_text_overlay import ImageTextOverlay
from .nodes.prompt_shorthand import PromptShorthandExpander
from .nodes import kdenlive_archive as _kdenlive_archive_routes  # noqa: F401  (registers /fbtools/kdenlive/* routes on import)
# Route-only modules: importing them registers their /fbtools/* handlers on the PromptServer routes.
from .nodes import backgrounds_presets as _backgrounds_presets_routes  # noqa: F401
from .nodes import registry_api as _registry_api_routes  # noqa: F401
from .nodes import lora_info as _lora_info_routes  # noqa: F401
from .nodes import media as _media_routes  # noqa: F401
from .nodes import prompt_collections as _prompt_collections_routes  # noqa: F401
from .nodes import llm_assistant as _llm_assistant_routes  # noqa: F401
from .nodes.bundles import BundleAudioReferenceLoad
from .nodes.h3_reference_summary import H3ReferenceSummary
from .nodes import h3_character_sheet as _h3_character_sheet_routes  # noqa: F401
from .nodes import qwen21_photo_restore as _qwen21_photo_restore_routes  # noqa: F401
from .utils.util import (
    draw_pose_json,
    draw_pose,
    extend_scalelist,
    pose_normalized,
)

from comfy_api.latest import ComfyExtension, io
import torch
import numpy as np
import json
from .utils.logging_utils import get_logger


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
# SceneCompose/PromptAssemble moved to nodes/compose.py (Plan 29)
# ── Outfit REST API endpoints ─────────────────────────────────────────────────

















# ── Concept REST API endpoints ─────────────────────────────────────────────────





# ── Subject CRUD routes (editor-facing) ───────────────────────────────────────









# Reference Bundle routes moved to nodes/bundles.py (Plan 29); SceneCastLoad/SceneCastBuild
# (+ CastIOType/H3RefplanType consumers, _cast_get_ids) moved to nodes/scene_casts.py (Plan 29)


# ── Scene Cast reload endpoint ────────────────────────────────────────────────




# Prompt Composition routes + Composition system settings moved to nodes/compositions.py (Plan 29)
# ── LLM assistant routes ──────────────────────────────────────────────────────

from .utils import unsloth_client as _unsloth_client
from .utils import modal_deploy as _modal_deploy

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













# CompositionLoad/PromptCompositionLoader/CompositionToH3Conditioning (+ helpers)
# moved to nodes/compositions.py (Plan 29)
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
            H3SourceGuides,
            MarkerFrameSplit,
            ImageTextOverlay,
            PromptShorthandExpander,
            # Scene Cast nodes
            SceneCastLoad,
            SceneCastBuild,
            # Reference Bundle nodes
            BundleAudioReferenceLoad,
            # H3 reference reporting
            H3ReferenceSummary,
            # Run tracking
            RunMetaCapture,
            # Notifications
            JobCompleteNotifier,
        ]