"""Story domain: node classes + /fbtools/story/* routes, extracted from extension.py (Plan 22).

Sibling to nodes/narrative/scene.py in the same nodes/narrative/ subpackage — imports SceneInfo/
MaskType/etc. from it directly via `.scene` rather than through extension.py.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

import torch
from aiohttp import web
from comfy_api.latest import io, ui
from folder_paths import get_output_directory

from ..libber import LibberStateManager
from ..lora_stacks import LoraStackData, LORA_MODEL_TARGETS
from ..shared import (
    prefixed_node_id,
    default_scenes_dir,
    default_stories_dir,
    get_subdirectories,
    _directory_fingerprint,
    routes,
)
from .scene import (
    SceneInfo,
    load_masks_json,
    resolve_mask_key,
    default_depth_options,
    default_pose_options,
    load_lora_stack,
)
from ...prompt_models import PromptCollection
from ...story_models import SceneInStory, StoryInfo, save_story, load_story
from ...utils.util import update_ui_widget
from ...utils.io import load_prompt_json
from ...utils.images import make_empty_image
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_available_stories():
    stories_dir = default_stories_dir() if callable(globals().get('default_stories_dir')) else os.path.join(get_output_directory(), "stories")
    if not os.path.isdir(stories_dir):
        return ["default_story"]
    story_names = []
    for entry in os.listdir(stories_dir):
        entry_path = os.path.join(stories_dir, entry)
        if os.path.isdir(entry_path):
            story_names.append(entry)
    return story_names if story_names else ["default_story"]


def build_positive_prompt(prompt_type: str, prompt_data: dict, custom_prompt: str = "") -> str:
    """Select the prompt text for a scene based on prompt_type and available prompt data."""
    prompt_data = prompt_data or {}
    girl = prompt_data.get("girl_pos", "") or ""
    male = prompt_data.get("male_pos", "") or ""
    four = prompt_data.get("four_image_prompt", "") or ""
    wan_hi = prompt_data.get("wan_prompt", "") or ""
    wan_low = prompt_data.get("wan_low_prompt", "") or ""

    if prompt_type == "custom":
        return custom_prompt or ""
    if prompt_type == "combined":
        return " ".join([p for p in [girl, male] if p]).strip()
    if prompt_type == "girl_pos":
        return girl
    if prompt_type == "male_pos":
        return male
    if prompt_type == "four_image_prompt":
        return four
    if prompt_type == "wan_prompt":
        return wan_hi
    if prompt_type == "wan_low_prompt":
        return wan_low
    return girl or male or four or wan_hi or wan_low

class StoryCreate(io.ComfyNode):
    """Create a new story with an initial scene"""
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_stories_dir_path = default_stories_dir()
        default_scenes_dir_path = default_scenes_dir()
        
        # Get available scenes
        scenes_subdir_dict = get_subdirectories(default_scenes_dir_path)
        available_scenes = sorted(scenes_subdir_dict.keys()) if scenes_subdir_dict else ["default_scene"]
        
        # Placeholder for prompt keys (dynamically populated in UI)
        prompt_key_options = ["(select a key from scene)"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryCreate"),
            display_name="Story Create",
            category="🧊 frost-byte/Story",
            inputs=[
                io.String.Input(id="story_name", display_name="story_name", default="my_story", tooltip="Name of the story"),
                io.String.Input(id="story_dir", display_name="story_dir", default=default_stories_dir_path, tooltip="Directory to save the story"),
                io.Combo.Input(id="initial_scene", display_name="initial_scene", options=available_scenes, default=available_scenes[0], tooltip="First scene to add to the story"),
                io.String.Input(id="mask_name", display_name="mask_name", default="", tooltip="Name of mask from the scene (leave empty to skip)"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Include background in mask"),
                io.Combo.Input(id="prompt_source", display_name="prompt_source", options=["prompt", "composition", "custom"], default="prompt", tooltip="Source of the prompt: 'prompt' (from prompt_dict), 'composition' (from composition_dict), or 'custom'"),
                io.String.Input(id="prompt_key", display_name="prompt_key", default="", tooltip="Key from scene's prompt_dict or composition_dict (leave empty for custom)"),
                io.String.Input(id="custom_prompt", display_name="custom_prompt", default="", multiline=True, tooltip="Custom prompt (only used if prompt_source is 'custom')"),
                io.Combo.Input(id="depth_type", display_name="depth_type", options=list(default_depth_options.keys()), default="depth", tooltip="Depth image type"),
                io.Combo.Input(id="pose_type", display_name="pose_type", options=list(default_pose_options.keys()), default="open", tooltip="Pose image type"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info", display_name="story_info", tooltip="Story information"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        story_name="my_story",
        story_dir="",
        initial_scene="default_scene",
        mask_name="",
        mask_background=True,
        prompt_source="prompt",
        prompt_key="",
        custom_prompt="",
        depth_type="depth",
        pose_type="open",
    ) -> io.NodeOutput:
        if not story_dir:
            story_dir = default_stories_dir()
        
        # Create story directory if it doesn't exist
        story_path = Path(story_dir) / story_name
        os.makedirs(story_path, exist_ok=True)
        
        # Create initial scene
        initial_scene_obj = SceneInStory(
            scene_name=initial_scene,
            scene_order=0,
            mask_name=mask_name,
            mask_background=mask_background,
            prompt_source=prompt_source,
            prompt_key=prompt_key,
            custom_prompt=custom_prompt,
            depth_type=depth_type,
            pose_type=pose_type,
        )
        
        story_info = StoryInfo(
            version=2,
            story_name=story_name,
            story_dir=str(story_path),
            scenes=[initial_scene_obj]
        )
        
        logger.info(
            "StoryCreate: Created story '%s' (v2) with initial scene '%s' using %s:%s",
            story_name,
            initial_scene,
            prompt_source,
            prompt_key or "custom",
        )
        
        return io.NodeOutput(story_info)


class StoryEdit(io.ComfyNode):
    """View and preview a story with scene selection. CRUD operations handled via frontend REST API."""
    @classmethod
    def define_schema(cls):
        available_stories = get_available_stories() if callable(globals().get('get_available_stories')) else ["default_story"]
        
        # Get scene names from first story for initial preview_scene options
        default_scene_options = [""]  # Empty option means "use first scene"
        if available_stories:
            first_story_info = cls._load_story_info(available_stories[0])
            if first_story_info and hasattr(first_story_info, 'scenes') and first_story_info.scenes:
                sorted_scenes = sorted(first_story_info.scenes, key=lambda s: s.scene_order)
                default_scene_options.extend([scene.scene_name for scene in sorted_scenes])
        
        return io.Schema(
            node_id=prefixed_node_id("StoryEdit"),
            display_name="Story Edit",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_select", display_name="Story", options=available_stories, default=available_stories[0], tooltip="Select a story to view/edit"),
                io.Combo.Input(id="preview_scene_name", display_name="Preview Scene", options=default_scene_options, default=default_scene_options[0], tooltip="Scene within the story to preview (empty selects the first scene)"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info_out", display_name="story_info", tooltip="Loaded story information"),
                io.Image.Output(id="base_image", display_name="base_image", tooltip="Base/upscale image for preview scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask image for preview scene"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask for preview scene"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Pose image for preview scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth image for preview scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def validate_inputs(cls, story_select: str = "default_story", preview_scene_name: str = ""):
        """Validate that story_select exists in the stories directory."""
        if not story_select:
            return "Story selection is required"
        
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_select / "story.json"
        
        if not story_json_path.exists():
            return f"Story '{story_select}' not found at {story_json_path}"
        
        # Validate preview_scene_name if provided
        if preview_scene_name:
            story_info = cls._load_story_info(story_select)
            if story_info and hasattr(story_info, 'scenes'):
                scene_names = [scene.scene_name for scene in story_info.scenes]
                if preview_scene_name not in scene_names:
                    return f"Scene '{preview_scene_name}' not found in story '{story_select}'"
        
        return True
    
    @classmethod
    def fingerprint_inputs(cls, story_select: str = "default_story", preview_scene_name: str = ""):
        """Generate fingerprint based on stories directory modification time to trigger combo refresh."""
        try:
            stories_dir = default_stories_dir()
            stories_path = Path(stories_dir)
            
            # Collect all story.json modification times and sizes
            story_fingerprints = []
            if stories_path.exists():
                for story_dir in stories_path.iterdir():
                    if story_dir.is_dir():
                        story_json = story_dir / "story.json"
                        if story_json.exists():
                            st = os.stat(story_json)
                            story_fingerprints.append((story_dir.name, int(st.st_mtime), int(st.st_size)))
            
            # Sort by name for consistent fingerprinting
            story_fingerprints.sort()
            
            logger.debug("StoryEdit: Fingerprint includes %d stories", len(story_fingerprints))
            return tuple(story_fingerprints) if story_fingerprints else None
            
        except Exception as e:
            logger.warning("StoryEdit: Failed to generate fingerprint: %s", e)
            return None
    
    @classmethod
    def execute(
        cls,
        story_select="default_story",
        preview_scene_name="",
    ) -> io.NodeOutput:
        # Load story from file system
        story_info = cls._load_story_info(story_select)
        if story_info is None:
            logger.error("StoryEdit: Story '%s' could not be loaded", story_select)
            return io.NodeOutput(None, None, None, None, None, None)
        
        # Resolve which scene to preview
        preview_scene = cls._resolve_preview_scene(story_info, preview_scene_name)
        
        # Initialize preview outputs
        base_image = None
        mask_image = None
        mask = None
        pose_image = None
        depth_image = None
        selected_prompt_text = ""
        preview_image_ui = None
        
        # Load preview assets if we have a scene
        if preview_scene:
            assets = cls._load_scene_assets(preview_scene)
            base_image = assets.get("base_image")
            mask_image = assets.get("mask_image")
            mask = assets.get("mask")
            pose_image = assets.get("pose_image")
            depth_image = assets.get("depth_image")
            selected_prompt_text = cls._load_prompt_text(
                preview_scene.scene_name,
                preview_scene.prompt_source,
                preview_scene.prompt_key,
                preview_scene.custom_prompt,
            )
            
            # Build preview image UI
            preview_batch = assets.get("preview_batch", [])
            if preview_batch:
                try:
                    preview_image_ui = ui.PreviewImage(image=torch.cat(preview_batch, dim=0))
                except Exception as exc:
                    logger.exception("StoryEdit: Failed to build preview image UI")
        
        # Build summary text and metadata
        summary_text = cls._build_summary_text(story_info, preview_scene)
        meta_payload = cls._build_meta_payload(story_info, preview_scene)
        
        # Combine UI elements
        ui_payload = {
            "text": [summary_text, selected_prompt_text, meta_payload],
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        return io.NodeOutput(
            story_info,
            base_image,
            mask_image,
            mask,
            pose_image,
            depth_image,
            ui=ui_payload
        )
    
    @staticmethod
    def _load_story_info(story_select: str) -> Optional[StoryInfo]:
        """Load story from filesystem"""
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_select / "story.json"
        if not story_json_path.exists():
            logger.warning("StoryEdit: Story file not found at '%s'", story_json_path)
            return None
        return load_story(str(story_json_path))
    
    @staticmethod
    def _resolve_preview_scene(story_info: StoryInfo, preview_scene_name: str) -> Optional[SceneInStory]:
        """Determine which scene to preview"""
        if not story_info or not getattr(story_info, "scenes", None):
            logger.warning("StoryEdit: Story has no scenes to preview")
            return None
        
        # If a specific scene name is provided, find it
        if preview_scene_name:
            for scene in story_info.scenes:
                if scene.scene_name == preview_scene_name:
                    return scene
        
        # Default to first scene by order
        return sorted(story_info.scenes, key=lambda s: s.scene_order)[0]
    
    @staticmethod
    def _load_scene_assets(scene: SceneInStory) -> dict:
        """Load preview assets for a scene"""
        scenes_dir = default_scenes_dir()
        scene_dir = os.path.join(scenes_dir, scene.scene_name)
        if not os.path.isdir(scene_dir):
            logger.warning("StoryEdit: Scene directory '%s' missing for preview", scene_dir)
            return {}
        
        depth_attr = default_depth_options.get(scene.depth_type, "depth_image")
        pose_attr = default_pose_options.get(scene.pose_type, "pose_open_image")
        
        try:
            assets = SceneInfo.load_preview_assets(
                scene_dir,
                depth_attr=depth_attr,
                pose_attr=pose_attr,
                mask_name=scene.mask_name,
                mask_background=scene.mask_background,
                include_upscale=True,
                include_canny=False,
            )
            assets["scene_dir"] = scene_dir
            return assets
        except Exception as exc:
            logger.exception("StoryEdit: Failed to load preview assets for '%s'", scene.scene_name)
            return {}
    
    @staticmethod
    def _load_prompt_text(scene_name: str, prompt_source: str, prompt_key: str, custom_prompt: str) -> str:
        """Load prompt text for preview"""
        if prompt_source == "custom":
            return custom_prompt or ""
        
        scene_dir = os.path.join(default_scenes_dir(), scene_name)
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_data_raw = load_prompt_json(prompt_json_path) or {}
        
        if prompt_data_raw.get("version") == 2:            
            prompt_collection = PromptCollection.from_dict(prompt_data_raw)
            libber_manager = LibberStateManager.instance()
            
            # Build individual prompts
            prompt_dict = {}
            for key, metadata in prompt_collection.prompts.items():
                value = metadata.value
                if metadata.processing_type == "libber" and metadata.libber_name:
                    libber = libber_manager.ensure_libber(metadata.libber_name)
                    if libber:
                        value = libber.substitute(value)
                prompt_dict[key] = value
            
            # Build compositions
            compositions = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager) if prompt_collection.compositions else {}
            
            if prompt_source == "prompt" and prompt_key:
                return prompt_dict.get(prompt_key, "")
            if prompt_source == "composition" and prompt_key:
                return compositions.get(prompt_key, "")
            return ""
        
        # Legacy format fallback
        if prompt_key:
            return prompt_data_raw.get(prompt_key, "")
        return ""
    
    @staticmethod
    def _build_summary_text(story_info: StoryInfo, preview_scene: Optional[SceneInStory]) -> str:
        """Build text summary of story and scenes"""
        selected_id = preview_scene.scene_id if preview_scene else ""
        lines = []
        for scene in sorted(getattr(story_info, "scenes", []), key=lambda s: s.scene_order):
            marker = "▶ " if selected_id and scene.scene_id == selected_id else "  "
            mask_suffix = "" if scene.mask_background else " (no bg)"
            prompt_display = f"{scene.prompt_source}:{scene.prompt_key}" if scene.prompt_key else scene.prompt_source
            lines.append(
                f"{marker}{scene.scene_order}: {scene.scene_name} | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={prompt_display} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
        
        summary_header = (
            f"Story: {story_info.story_name}\n"
            f"Dir: {story_info.story_dir}\n"
            f"Scenes: {len(getattr(story_info, 'scenes', []))}\n"
            f"Preview: {preview_scene.scene_name if preview_scene else '(none)'}\n\n"
            "Scenes:\n"
        )
        return summary_header + ("\n".join(lines) if lines else "No scenes available")
    
    @staticmethod
    def _build_meta_payload(story_info: StoryInfo, preview_scene: Optional[SceneInStory]) -> str:
        """Build JSON metadata for frontend"""
        # Include full scene data for frontend table
        scenes_dir = default_scenes_dir()
        scenes_data = []
        for scene in getattr(story_info, "scenes", []):
            # Load available masks for this scene
            available_masks = ["none"]
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if os.path.isdir(scene_dir):
                try:
                    # Load new mask system masks
                    masks_dict = load_masks_json(scene_dir)
                    available_masks.extend(masks_dict.keys())
                    
                    # Add legacy masks if they exist
                    legacy_mask_names = ["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"]
                    for legacy_name in legacy_mask_names:
                        mask_file = f"{legacy_name.replace('_no_bg', '_mask_no_bkgd' if '_no_bg' in legacy_name else '_mask_bkgd')}.png"
                        mask_path = os.path.join(scene_dir, mask_file)
                        if os.path.exists(mask_path) and legacy_name not in available_masks:
                            available_masks.append(legacy_name)
                except Exception as e:
                    logger.debug(f"StoryEdit: Could not load masks for scene '{scene.scene_name}': {e}")
            
            scenes_data.append({
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key or "",
                "custom_prompt": scene.custom_prompt or "",
                "video_prompt_source": getattr(scene, "video_prompt_source", "auto"),
                "video_prompt_key": getattr(scene, "video_prompt_key", ""),
                "video_custom_prompt": getattr(scene, "video_custom_prompt", ""),
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
                "use_depth": getattr(scene, "use_depth", False),
                "use_mask": getattr(scene, "use_mask", False),
                "use_pose": getattr(scene, "use_pose", False),
                "use_canny": getattr(scene, "use_canny", False),
                "available_masks": available_masks,
            })
        
        payload = {
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scene_count": len(getattr(story_info, "scenes", [])),
            "preview_scene": preview_scene.scene_name if preview_scene else None,
            "scenes": scenes_data,
        }
        return json.dumps(payload)


class StoryView(io.ComfyNode):
    """View and select scenes from a story with preview capabilities"""
    @classmethod
    def define_schema(cls):
        # Get default scene options for when no story is loaded
        default_scenes_dir_path = default_scenes_dir()
        scenes_subdir_dict = get_subdirectories(default_scenes_dir_path)
        default_scene_options = sorted(scenes_subdir_dict.keys()) if scenes_subdir_dict else ["default_scene"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryView"),
            display_name="Story View",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to view"),
                io.Combo.Input(id="selected_scene", display_name="selected_scene", options=default_scene_options, default=default_scene_options[0], tooltip="Select a scene from the story"),
                io.String.Input(id="prompt_in", display_name="prompt_in", multiline=True, default="", tooltip="Editable prompt text"),
                io.Combo.Input(id="prompt_action", display_name="prompt_action", options=["use_file", "use_edit"], default="use_file", tooltip="Use file prompt or edited prompt"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info_out", display_name="story_info", tooltip="Story information (pass-through for chaining to StorySave)"),
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information for selected scene"),
                io.String.Output(id="story_name", display_name="story_name", tooltip="Name of the story"),
                io.String.Output(id="story_dir", display_name="story_dir", tooltip="Directory of the story"),
                io.Int.Output(id="scene_count", display_name="scene_count", tooltip="Number of scenes in the story"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="selected_prompt", display_name="selected_prompt", tooltip="The selected prompt text"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Pose image for selected scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask image for selected scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth image for selected scene"),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo 
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        story_info=None,
        selected_scene="default_scene",
        prompt_in="",
        prompt_action="use_file",
    ) -> io.NodeOutput:
        className = cls.__name__
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        
        if story_info is None:
            logger.error("StoryView: story_info is None")
            return io.NodeOutput(None, None, "", "", 0, "", "", None, None, None)
        
        # Find the selected scene configuration in the story
        scene_config = None
        for scene in story_info.scenes:
            if scene.scene_name == selected_scene:
                scene_config = scene
                break
        
        if scene_config is None and story_info.scenes:
            logger.warning(
                "StoryView: Scene '%s' not found in story, defaulting to first scene '%s'",
                selected_scene,
                story_info.scenes[0].scene_name,
            )
            scene_config = story_info.scenes[0]
            selected_scene = scene_config.scene_name

        # If scene not found in story, create a default configuration
        if scene_config is None:
            logger.warning("StoryView: Scene '%s' not found in story, using defaults", selected_scene)
            scene_config = SceneInStory(
                scene_name=selected_scene,
                scene_order=0,
                mask_name="",
                mask_background=True,
                prompt_source="prompt",
                prompt_key="",
                custom_prompt="",
                depth_type="depth",
                pose_type="open",
            )
        
        # Load scene data from scene directory
        scenes_dir = default_scenes_dir()
        scene_dir = os.path.join(scenes_dir, selected_scene)
        
        if not os.path.isdir(scene_dir):
            logger.error("StoryView: scene_dir '%s' is not a valid directory", scene_dir)
            return io.NodeOutput(story_info, None, story_info.story_name, story_info.story_dir, len(story_info.scenes), selected_scene, "", None, None, None)
        
        try:
            scene_info, assets, selected_prompt, prompt_data, prompt_widget_text = SceneInfo.from_story_scene(
                scene_config,
                scenes_dir=scenes_dir,
                prompt_in=prompt_in,
                prompt_action=prompt_action,
                include_upscale=False,
                include_canny=False,
            )
        except Exception as e:
            logger.exception("StoryView: failed to build SceneInfo for '%s'", selected_scene)
            return io.NodeOutput(story_info, None, story_info.story_name, story_info.story_dir, len(story_info.scenes), selected_scene, "", None, None, None)

        if prompt_widget_text is not None:
            input_types = cls.INPUT_TYPES()
            inputs = input_types.get('required', {}) if isinstance(input_types, dict) else {}
            update_ui_widget(className, unique_id, extra_pnginfo, prompt_widget_text, "prompt_in", inputs)

        selected_depth_image = assets.get("depth_image")
        selected_pose_image = assets.get("pose_image")
        selected_mask_image = assets.get("mask_image")
        mask = assets.get("mask")
        
        # Create preview UI combining pose, mask, and depth
        preview_batch = assets.get("preview_batch", [])
        preview_image_ui = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        # Create text preview with scene IDs
        scene_list_lines = []
        for scene in sorted(story_info.scenes, key=lambda s: s.scene_order):
            marker = "▶ " if scene.scene_name == selected_scene else "  "
            mask_suffix = "" if scene.mask_background else " (no bg)"
            
            # Display prompt_source:prompt_key or custom
            prompt_display = f"{scene.prompt_source}:{scene.prompt_key}" if scene.prompt_key else scene.prompt_source
            
            scene_line = (
                f"{marker}{scene.scene_order}: {scene.scene_name} [{scene.scene_id[:8]}] | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={prompt_display} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
            if scene.prompt_source == "custom" and scene.custom_prompt:
                scene_line += f" | custom='{scene.custom_prompt[:30]}...'"
            scene_list_lines.append(scene_line)
        
        scene_list_text = "\n".join(scene_list_lines) if scene_list_lines else "No scenes"
        
        prompt_display = f"{scene_config.prompt_source}:{scene_config.prompt_key}" if scene_config.prompt_key else scene_config.prompt_source
        
        preview_text = (
            f"Story: {story_info.story_name}\n"
            f"Dir: {story_info.story_dir}\n"
            f"Scenes: {len(story_info.scenes)}\n"
            f"Selected: {selected_scene} (order {scene_config.scene_order})\n"
            f"Prompt: {prompt_display}\n"
            f"Prompt Text: {selected_prompt}\n\n"
            f"All Scenes:\n{scene_list_text}"
        )
        text_ui = ui.PreviewText(value=preview_text)
        
        # Combine UI elements
        combined_ui = {
            "text": text_ui.as_dict().get("text", []),
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        logger.info(
            "StoryView: Story '%s' - Selected scene '%s' with prompt '%s'",
            story_info.story_name,
            selected_scene,
            prompt_display,
        )
        
        return io.NodeOutput(
            story_info,
            scene_info,
            story_info.story_name,
            story_info.story_dir,
            len(story_info.scenes),
            selected_scene,
            selected_prompt,
            selected_pose_image,
            selected_mask_image,
            selected_depth_image,
            ui=combined_ui
        )


class StorySceneBatch(io.ComfyNode):
    """Create an ordered list of scene descriptors for iteration."""

    @classmethod
    def define_schema(cls):
        # Get available stories for dropdown
        stories_dir = default_stories_dir()
        available_stories = get_subdirectories(stories_dir)
        story_names = list(available_stories.keys()) if available_stories else [""]
        
        # Job ID options will be populated dynamically by frontend when story_name changes
        # Empty string means auto-generate a new unique job_id
        job_id_options = [""]
        
        return io.Schema(
            node_id=prefixed_node_id("StorySceneBatch"),
            display_name="Story Scene Batch",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_name", display_name="story_name", options=story_names, default=story_names[0] if story_names else "", tooltip="Select story to batch process"),
                io.Combo.Input(id="job_id", display_name="job_id", options=job_id_options, default="", tooltip="Select existing job_id or leave empty to auto-generate. Options update when story changes."),
            ],
            outputs=[
                io.Int.Output(id="scene_count", display_name="scene_count", tooltip="Total number of scenes"),
                io.Custom("SCENE_BATCH").Output(id="scene_batch", display_name="scene_batch", tooltip="Ordered list of scene dictionaries"),
                io.String.Output(id="job_id_out", display_name="job_id", tooltip="Job id used for this batch"),
                io.String.Output(id="job_root_dir_out", display_name="job_root_dir", tooltip="Resolved job root directory"),
            ],
        )

    @classmethod
    def validate_inputs(cls, story_name: str = "", job_id: str = ""):
        """Validate that job_id is valid for the selected story."""
        if not story_name:
            return "Story name is required"
        
        if not job_id:
            # Empty job_id is allowed - will auto-generate
            return True
        
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            return f"Story '{story_name}' not found"
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return f"Failed to load story '{story_name}'"
        
        available_jobs = list_job_ids(story_info.story_dir)
        if job_id not in available_jobs:
            return f"Job ID '{job_id}' not found in story '{story_name}'. Available jobs: {', '.join(available_jobs) if available_jobs else '(none)'}"
        
        return True

    @classmethod
    def fingerprint_inputs(cls, story_name: str = "", job_id: str = ""):
        """Generate fingerprint based on story.json and all referenced scene directory content."""
        if story_name:
            logger.debug("StorySceneBatch: Generating fingerprint for story '%s'", story_name)
            stories_dir = default_stories_dir()
            story_json_path = Path(stories_dir) / story_name / "story.json"
            if story_json_path.exists():
                try:
                    story_stat = os.stat(story_json_path)
                    story_info = load_story(str(story_json_path))

                    if not story_info or not getattr(story_info, "scenes", None):
                        return (
                            str(story_json_path),
                            int(story_stat.st_mtime_ns),
                            int(story_stat.st_size),
                            (),
                            job_id.strip(),
                        )

                    scenes_dir = Path(default_scenes_dir())
                    scene_fingerprints: list[tuple[str, str, int, int]] = []

                    for scene in sorted(story_info.scenes, key=lambda s: (s.scene_order, s.scene_name)):
                        scene_dir = scenes_dir / scene.scene_name
                        scene_hash, dir_count, file_count = _directory_fingerprint(scene_dir)
                        scene_fingerprints.append((scene.scene_name, scene_hash, dir_count, file_count))

                    return (
                        str(story_json_path),
                        int(story_stat.st_mtime_ns),
                        int(story_stat.st_size),
                        tuple(scene_fingerprints),
                        job_id.strip(),
                    )
                except Exception as e:
                    logger.warning("StorySceneBatch: Failed to stat story.json for fingerprinting: %s", e)
        # Return None to use default fingerprinting behavior
        return None

    @classmethod
    def execute(
        cls,
        story_name: str = "",
        job_id: str = "",
    ) -> io.NodeOutput:
        if not story_name:
            logger.warning("StorySceneBatch: story_name is empty")
            return io.NodeOutput(0, [], "", "")
        
        # Load story from filesystem
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            logger.error("StorySceneBatch: Story '%s' not found at '%s'", story_name, story_json_path)
            return io.NodeOutput(0, [], "", "")
        
        story_info = load_story(str(story_json_path))
        if not story_info or not getattr(story_info, "scenes", None):
            logger.error("StorySceneBatch: Failed to load story '%s' or story has no scenes", story_name)
            return io.NodeOutput(0, [], "", "")

        # Auto-generate unique job_id - check for collisions with existing jobs
        jobs_dir = Path(story_info.story_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided job_id or generate a new unique one
        resolved_job_id = job_id.strip() if job_id else ""
        
        if resolved_job_id:
            # User provided job_id - validate and use it
            logger.info("StorySceneBatch: Using user-provided job_id='%s'", resolved_job_id)
            job_root = jobs_dir / resolved_job_id
            
            # Check if this job already exists
            if job_root.exists():
                logger.warning(
                    "StorySceneBatch: Job directory '%s' already exists, will reuse it",
                    job_root
                )
        else:
            # Auto-generate unique job_id
            existing_job_ids = set()
            if jobs_dir.exists():
                existing_job_ids = {d.name for d in jobs_dir.iterdir() if d.is_dir()}
            
            # Generate unique job_id (should succeed on first try, but be safe)
            max_attempts = 100
            for _ in range(max_attempts):
                candidate_id = uuid.uuid4().hex[:12]
                if candidate_id not in existing_job_ids:
                    resolved_job_id = candidate_id
                    break
            
            if not resolved_job_id:
                logger.error("StorySceneBatch: Failed to generate unique job_id after %d attempts", max_attempts)
                return io.NodeOutput(0, [], "", "")
            
            logger.info("StorySceneBatch: Auto-generated unique job_id='%s'", resolved_job_id)
            job_root = jobs_dir / resolved_job_id
        
        # Create job root directory if it doesn't exist
        job_root.mkdir(parents=True, exist_ok=True)

        scenes_dir = default_scenes_dir()
        batch: list[dict] = []

        scenes_sorted = sorted(story_info.scenes, key=lambda s: s.scene_order)
        logger.info(
            "StorySceneBatch: Preparing batch for story '%s' with %d scenes under job_id='%s' at '%s'",
            story_info.story_name,
            len(scenes_sorted),
            resolved_job_id,
            job_root,
        )
        for scene in scenes_sorted:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            prompt_path = os.path.join(scene_dir, "prompts.json")
            prompt_data_raw = load_prompt_json(prompt_path) or {}
            
            logger.debug(
                "StorySceneBatch: Processing scene '%s' (order %s)",
                scene.scene_name,
                scene.scene_order,
            )
            
            # Log scene configuration for debugging
            logger.info(
                "StorySceneBatch: Scene '%s' - depth_type='%s', pose_type='%s', mask_type='%s', use_pose=%s, use_depth=%s",
                scene.scene_name,
                scene.depth_type,
                scene.pose_type,
                scene.mask_type,
                scene.use_pose,
                scene.use_depth,
            )
            
            # Load PromptCollection and compose prompts using the new system
            if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
                logger.debug("StorySceneBatch: Detected v2 prompt format for scene '%s'", scene.scene_name)
                prompt_collection = PromptCollection.from_dict(prompt_data_raw)
                # Use shared LibberStateManager so any loaded libbers are applied across nodes
                libber_manager = LibberStateManager.instance()
                
                # Build prompt_dict: individual prompts (not composed)
                prompt_dict = {}
                for key, metadata in prompt_collection.prompts.items():
                    value = metadata.value
                    # Process libber substitution if needed
                    if metadata.processing_type == "libber" and metadata.libber_name:
                        libber = libber_manager.ensure_libber(metadata.libber_name)
                        if libber:
                            value = libber.substitute(value)
                    prompt_dict[key] = value
                
                # Build composition_dict: composed prompts from compositions
                composition_dict = {}
                if prompt_collection.compositions:
                    composition_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
                # Determine positive_prompt based on prompt_source and prompt_key
                if scene.prompt_source == "custom":
                    positive_prompt = scene.custom_prompt
                elif scene.prompt_source == "prompt" and scene.prompt_key:
                    positive_prompt = prompt_dict.get(scene.prompt_key, "")
                elif scene.prompt_source == "composition" and scene.prompt_key:
                    positive_prompt = composition_dict.get(scene.prompt_key, "")
                else:
                    positive_prompt = ""
                    logger.warning("StorySceneBatch: No valid prompt configuration for scene '%s'", scene.scene_name)
                
                # Warn if prompt is empty
                if not positive_prompt:
                    logger.warning(
                        "StorySceneBatch: Scene '%s' order=%d has EMPTY positive_prompt!",
                        scene.scene_name, scene.scene_order
                    )
                
                # For backwards compatibility, keep old prompt fields
                prompt_data = {
                    "girl_pos": prompt_dict.get("girl_pos", ""),
                    "male_pos": prompt_dict.get("male_pos", ""),
                    "four_image_prompt": prompt_dict.get("four_image_prompt", ""),
                    "wan_prompt": prompt_dict.get("wan_prompt", ""),
                    "wan_low_prompt": prompt_dict.get("wan_low_prompt", ""),
                }
            else:
                logger.debug("StorySceneBatch: Detected legacy prompt format for scene '%s'", scene.scene_name)
                # Legacy format
                prompt_data = prompt_data_raw
                # Use old build_positive_prompt for backwards compatibility if needed
                # But we should still respect the new fields if they exist
                if hasattr(scene, 'prompt_source') and scene.prompt_source:
                    if scene.prompt_source == "custom":
                        positive_prompt = scene.custom_prompt
                    elif scene.prompt_key:
                        if scene.prompt_source == "prompt":
                            positive_prompt = prompt_data.get(scene.prompt_key, "")
                        elif scene.prompt_source == "composition":
                            positive_prompt = prompt_data
                        else:
                            positive_prompt = prompt_data.get(scene.prompt_key, "")
                    else:
                        positive_prompt = ""
                else:
                    # Very old data - fallback
                    positive_prompt = build_positive_prompt(getattr(scene, 'prompt_type', 'girl_pos'), prompt_data, scene.custom_prompt)

            mask_key = resolve_mask_key(scene.mask_name, scene.mask_background)
            depth_key = default_depth_options.get(scene.depth_type, "depth_image")
            pose_key = default_pose_options.get(scene.pose_type, "pose_open_image")

            # Use flat structure: job_root/input/ for all scene images
            job_input_dir = job_root / "input"
            job_output_dir = job_root / "output"
            job_input_dir.mkdir(parents=True, exist_ok=True)
            job_output_dir.mkdir(parents=True, exist_ok=True)

            source_input_dir = Path(scene_dir) / "input"
            first_input_image = None
            for ext in ["png", "jpg", "jpeg", "webp"]:
                matches = sorted(source_input_dir.glob(f"*.{ext}"))
                if matches:
                    first_input_image = str(matches[0])
                    break

            descriptor = {
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_name": scene.mask_name,
                "mask_background": scene.mask_background,
                "mask_key": mask_key,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key,
                "custom_prompt": scene.custom_prompt,
                # Legacy fields for backwards compatibility
                "prompt_type": getattr(scene, 'prompt_type', ''),
                "depth_type": scene.depth_type,
                "depth_key": depth_key,
                "pose_type": scene.pose_type,
                "pose_key": pose_key,
                # Control flags for which inputs to use
                "use_depth": scene.use_depth,
                "use_mask": scene.use_mask,
                "use_pose": scene.use_pose,
                "use_canny": scene.use_canny,
                "scene_dir": scene_dir,
                "story_dir": story_info.story_dir,
                "job_id": resolved_job_id,
                "job_root": str(job_root),
                "job_input_dir": str(job_input_dir),
                "job_output_dir": str(job_output_dir),
                "source_input_dir": str(source_input_dir),
                "source_output_dir": str(Path(scene_dir) / "output"),
                "positive_prompt": positive_prompt,
                "wan_prompt": prompt_data.get("wan_prompt", ""),
                "wan_low_prompt": prompt_data.get("wan_low_prompt", ""),
                "four_image_prompt": prompt_data.get("four_image_prompt", ""),
                "girl_pos": prompt_data.get("girl_pos", ""),
                "male_pos": prompt_data.get("male_pos", ""),
                "input_image_path": first_input_image,
                "prompt_data": prompt_data,
            }

            logger.debug("StorySceneBatch: Added descriptor for scene '%s'", scene.scene_name)
            logger.info(
                "StorySceneBatch: Descriptor for '%s' has positive_prompt: '%s...'",
                scene.scene_name,
                descriptor.get("positive_prompt", "")[:100]
            )
            batch.append(descriptor)

        logger.info(
            "StorySceneBatch: Prepared %d scenes with job_id=%s at %s",
            len(batch),
            resolved_job_id,
            job_root,
        )

        return io.NodeOutput(
            len(batch),
            batch,
            resolved_job_id,
            str(job_root),
        )


class StoryScenePick(io.ComfyNode):
    """Select one scene descriptor by index and load the assets for generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StoryScenePick"),
            display_name="Story Scene Pick",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("SCENE_BATCH").Input(id="scene_batch", display_name="scene_batch", tooltip="Scene descriptor list from StorySceneBatch"),
                io.Int.Input(id="scene_index", display_name="scene_index", default=0, tooltip="Index into scene_batch (0-based)"),
            ],
            outputs=[
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Selected mask image"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Single-channel mask"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected depth image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected pose image"),
                io.Image.Output(id="canny_image", display_name="canny_image", tooltip="Canny edge image"),
                io.String.Output(id="prompt", display_name="prompt", tooltip="Selected prompt for this scene (composition/custom/prompt)"),
                io.Boolean.Output(id="use_pose", display_name="use_pose", tooltip="Whether the pose image should be used for this scene"),
                io.Boolean.Output(id="use_depth", display_name="use_depth", tooltip="Whether the depth image should be used for this scene"),
                io.Boolean.Output(id="use_canny", display_name="use_canny", tooltip="Whether the canny image should be used for this scene"),
                io.Boolean.Output(id="use_mask", display_name="use_mask", tooltip="Whether the mask image should be used for this scene"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Scene name"),
                io.Int.Output(id="scene_order", display_name="scene_order", tooltip="Scene order"),
                io.String.Output(id="scene_id", display_name="scene_id", tooltip="Scene id"),
                io.String.Output(id="job_id", display_name="job_id", tooltip="Job id"),
                io.String.Output(id="job_input_dir", display_name="job_input_dir", tooltip="Job input directory (where images are saved)"),
                io.String.Output(id="input_image_path", display_name="input_image_path", tooltip="Path to first input image (if any)"),
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Fully-loaded SceneInfo for the selected scene"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        scene_batch=None,
        scene_index: int = 0,
    ) -> io.NodeOutput:
        if not scene_batch:
            logger.warning("StoryScenePick: scene_batch is empty")
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, "", 0, "", "", "", "", None)

        try:
            scenes_sorted = sorted(scene_batch, key=lambda d: d.get("scene_order", 0))
        except Exception:
            scenes_sorted = scene_batch

        safe_index = max(0, min(len(scenes_sorted) - 1, scene_index))
        descriptor = scenes_sorted[safe_index]

        scene_dir = descriptor.get("scene_dir", "")
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("StoryScenePick: scene_dir '%s' is invalid", scene_dir)
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_input_dir", ""), descriptor.get("input_image_path", ""), None)
        prompt_key = descriptor.get("prompt_key", "")
        scene_config = SceneInStory(
            scene_id=descriptor.get("scene_id", ""),
            scene_name=descriptor.get("scene_name", ""),
            scene_order=descriptor.get("scene_order", 0),
            mask_name=descriptor.get("mask_name", descriptor.get("mask_type", "")),  # Support both new and legacy
            mask_background=descriptor.get("mask_background", True),
            prompt_source=descriptor.get("prompt_source", "prompt"),
            prompt_key=prompt_key,
            custom_prompt=descriptor.get("custom_prompt", ""),
            # Include legacy prompt_type for backwards compatibility
            prompt_type=descriptor.get("prompt_type", ""),
            depth_type=descriptor.get("depth_type", "depth"),
            pose_type=descriptor.get("pose_type", "open"),
            use_depth=descriptor.get("use_depth", False),
            use_mask=descriptor.get("use_mask", False),
            use_pose=descriptor.get("use_pose", False),
            use_canny=descriptor.get("use_canny", False),
        )
        
        # Log the pose configuration for debugging
        pose_attr = default_pose_options.get(scene_config.pose_type, "pose_open_image")
        logger.info(
            "StoryScenePick: Scene '%s' - pose_type='%s' -> pose_attr='%s' (from descriptor: '%s')",
            scene_config.scene_name,
            scene_config.pose_type,
            pose_attr,
            descriptor.get("pose_type", "NOT_IN_DESCRIPTOR")
        )
        logger.debug("StoryScenePick: Processing scene '%s'", scene_config.scene_name)

        # Use the pre-computed positive_prompt from StorySceneBatch
        # It's already been processed with compositions and libbers applied
        prompt = descriptor.get("positive_prompt", "")
        
        if not prompt:
            logger.warning(
                "StoryScenePick: Scene '%s' order=%d - No positive_prompt in descriptor; available keys: %s",
                descriptor.get("scene_name", "unknown"), descriptor.get("scene_order", -1), list(descriptor.keys())
            )

        try:
            # Use the descriptor's pre-computed prompt - don't let from_story_scene override it
            scene_info, assets, selected_prompt, prompt_data, _ = SceneInfo.from_story_scene(
                scene_config,
                scene_dir_override=scene_dir,
                include_upscale=False,
                include_canny=True,
                prompt_override=prompt,  # Use the descriptor's positive_prompt
            )
        except Exception as e:
            logger.error("StoryScenePick: failed to build SceneInfo for '%s': %s", scene_config.scene_name, e)
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_input_dir", ""), descriptor.get("input_image_path", ""), None)

        empty_image = make_empty_image()
        canny_image = assets.get("canny_image", empty_image)
        mask_image = assets.get("mask_image")
        mask = assets.get("mask")
        depth_image = assets.get("depth_image", empty_image)
        pose_image = assets.get("pose_image", empty_image)
        
        logger.debug(
            "StoryScenePick: Scene '%s' (order %s) - prompt length: %d",
            scene_config.scene_name,
            scene_config.scene_order,
            len(prompt),
        )

        return io.NodeOutput(
            mask_image,
            mask,
            depth_image,
            pose_image,
            canny_image,
            prompt,
            scene_config.use_pose,
            scene_config.use_depth,
            scene_config.use_canny,
            scene_config.use_mask,
            descriptor.get("scene_name", ""),
            descriptor.get("scene_order", 0),
            descriptor.get("scene_id", ""),
            descriptor.get("job_id", ""),
            descriptor.get("job_input_dir", ""),
            descriptor.get("input_image_path", ""),
            scene_info,
        )


class StorySave(io.ComfyNode):
    """Save the story configuration to a JSON file"""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StorySave"),
            display_name="Story Save",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to save"),
                io.String.Input(id="filename", display_name="filename", default="story.json", tooltip="Filename for the story JSON"),
            ],
            outputs=[
                io.String.Output(id="save_path", display_name="save_path", tooltip="Path where story was saved"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        story_info=None,
        filename="story.json",
    ) -> io.NodeOutput:
        if story_info is None:
            logger.warning("StorySave: story_info is None")
            return io.NodeOutput("")
        
        # Ensure story directory exists
        os.makedirs(story_info.story_dir, exist_ok=True)
        
        # Build save path
        save_path = Path(story_info.story_dir) / filename
        
        # Save the story
        save_story(story_info, str(save_path))
        
        logger.info("StorySave: Saved story to '%s'", save_path)
        
        preview_ui = ui.PreviewText(value=f"Story saved to: {save_path}\nScenes: {len(story_info.scenes)}")
        
        return io.NodeOutput(
            str(save_path),
            ui=preview_ui.as_dict()
        )

class StoryLoad(io.ComfyNode):
    """Load a story from a JSON file"""
    @classmethod
    def define_schema(cls):
        default_stories_dir_path = default_stories_dir()
        stories_subdir_dict = get_subdirectories(default_stories_dir_path)
        available_stories = sorted(stories_subdir_dict.keys()) if stories_subdir_dict else ["default_story"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryLoad"),
            display_name="Story Load",
            category="🧊 frost-byte/Story",
            inputs=[
                io.String.Input(id="stories_dir", display_name="stories_dir", default=default_stories_dir_path, tooltip="Directory containing stories"),
                io.Combo.Input(id="story_name", display_name="story_name", options=available_stories, default=available_stories[0], tooltip="Story to load"),
                io.String.Input(id="filename", display_name="filename", default="story.json", tooltip="Filename of the story JSON"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info", display_name="story_info", tooltip="Loaded story information"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        stories_dir="",
        story_name="default_story",
        filename="story.json",
    ) -> io.NodeOutput:
        if not stories_dir:
            stories_dir = default_stories_dir()
        
        story_path = Path(stories_dir) / story_name / filename
        
        if not story_path.exists():
            logger.warning("StoryLoad: Story file not found at '%s'", story_path)
            return io.NodeOutput(None)
        
        story_info = load_story(str(story_path))
        
        if story_info is None:
            logger.error("StoryLoad: Failed to load story from '%s'", story_path)
        
        return io.NodeOutput(story_info)

# ============================================================================
# TESTABLE IMAGE SAVE HELPERS - imported from utils module
# ============================================================================

from ...utils.scene_image_save import (
    SceneImageSaveConfig,
    ImageSaver,
    select_scene_descriptor,
    generate_preview_text
)


class StorySceneImageSave(io.ComfyNode):
    """Save generated image for a story scene with automatic naming and path management"""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StorySceneImageSave"),
            display_name="Story Scene Image Save",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Image.Input(id="image", display_name="image", tooltip="Generated image to save"),
                io.Custom("SCENE_BATCH").Input(id="scene_batch", display_name="scene_batch", tooltip="Scene batch from StorySceneBatch"),
                io.Int.Input(id="scene_index", display_name="scene_index", default=0, tooltip="Index into scene_batch (0-based); images saved to job_root/input/"),
                io.Combo.Input(id="image_format", display_name="image_format", options=["png", "jpg", "jpeg", "webp"], default="png", tooltip="Output image format"),
                io.Int.Input(id="quality", display_name="quality", default=95, min=1, max=100, tooltip="JPEG/WebP quality (1-100)"),
            ],
            outputs=[
                io.Image.Output(id="image_out", display_name="image", tooltip="Pass-through of the input image"),
                io.String.Output(id="filename", display_name="filename", tooltip="Name of the saved file"),
                io.String.Output(id="filepath", display_name="filepath", tooltip="Full path to the saved file"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        image=None,
        scene_batch=None,
        scene_index: int = 0,
        image_format: str = "png",
        quality: int = 95,
    ) -> io.NodeOutput:
        """Main execution - thin orchestration layer over testable components"""
        if image is None:
            logger.warning("StorySceneImageSave: No image provided")
            return io.NodeOutput(None, "", "")

        if scene_batch is None or not isinstance(scene_batch, list) or not scene_batch:
            logger.warning("StorySceneImageSave: scene_batch is missing or invalid")
            return io.NodeOutput(image, "", "")

        # Use testable functions
        descriptor = select_scene_descriptor(scene_batch, scene_index)
        if descriptor is None:
            logger.warning("StorySceneImageSave: Could not select descriptor")
            return io.NodeOutput(image, "", "")
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, scene_index, image_format, quality)
        if config is None:
            logger.warning("StorySceneImageSave: Invalid configuration from descriptor")
            return io.NodeOutput(image, "", "")

        filepath = config.generate_filepath()
        
        try:
            # Use injected/mockable ImageSaver
            ImageSaver.ensure_directory(config.target_dir)
            pil_image = ImageSaver.tensor_to_pil(image)
            ImageSaver.save_pil_image(pil_image, filepath, config.image_format, config.quality)
            
            logger.info("StorySceneImageSave: Saved image to '%s'", filepath)
            
            preview_text = generate_preview_text(config, filepath)
            preview_ui = ui.PreviewText(value=preview_text)
            
            return io.NodeOutput(
                image,
                config.generate_filename(),
                filepath,
                ui=preview_ui.as_dict()
            )
        except Exception as e:
            logger.exception("StorySceneImageSave: Error saving image: %s", e)
            return io.NodeOutput(image, "", "")


# Import video generation utilities
from ...utils.story_video import (
    list_job_ids,
    find_scene_image,
    pair_consecutive_scenes,
    generate_video_filename,
    resolve_video_prompt,
    build_video_descriptor,
)


class StoryVideoBatch(io.ComfyNode):
    """Generate video prompts and aggregate LoRAs for story scene transitions.

    Self-contained node with story and job selection via combo widgets.
    Use `lora_target_filter` to select which model pipeline will be used;
    the aggregated `lora_stack_data` output can be fed directly into
    `LoraStackApply` (which handles High/Low routing internally for Wan2.2).

    Outputs:
    1. input_folder_path - Path to the job's input folder with ordered scene images
    2. video_prompts     - Multiline string, one prompt per scene transition
    3. lora_stack_data   - Aggregated unique LoRA stack filtered by lora_target_filter
    4. video_count       - Total number of video transitions
    5. story_name_out    - Selected story name
    """
    
    @classmethod
    def define_schema(cls):
        # Get available stories for combo widget
        available_stories = get_available_stories()
        default_story = available_stories[0] if available_stories else "default_story"
        
        # Get job IDs for the first story as default options
        default_jobs = [""]
        if available_stories:
            stories_dir = default_stories_dir()
            first_story_path = os.path.join(stories_dir, default_story)
            if os.path.isdir(first_story_path):
                story_info = load_story(os.path.join(first_story_path, "story.json"))
                if story_info:
                    jobs = list_job_ids(story_info.story_dir)
                    default_jobs = jobs if jobs else [""]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryVideoBatch"),
            display_name="Story Video Batch",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_name", display_name="story_name", options=available_stories, default=default_story, tooltip="Select story from available stories"),
                io.Combo.Input(id="job_id", display_name="job_id", options=default_jobs, default=default_jobs[0] if default_jobs else "", tooltip="Select job ID from available jobs (leave empty to use most recent)"),
                io.Combo.Input(
                    id="lora_target_filter",
                    display_name="LoRA Target Filter",
                    options=["All"] + LORA_MODEL_TARGETS,
                    default="All",
                    tooltip=(
                        "Filter the aggregated LoRA stack to a specific model target. "
                        "'All' includes every entry regardless of target. "
                        "Select e.g. 'Wan2.2-Wrapper-High' to output only entries for that pass, "
                        "or 'LTX2.3' for LTX inference. "
                        "Feed the output into LoraStackApply."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(id="input_folder_path", display_name="input_folder_path", tooltip="Path to job input folder with ordered scene images"),
                io.String.Output(id="video_prompts", display_name="video_prompts", tooltip="Multiline string with one prompt per transition"),
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Aggregated unique LoRA stack across all story scenes, filtered by lora_target_filter. Feed into LoraStackApply."),
                io.Int.Output(id="video_count", display_name="video_count", tooltip="Total number of video transitions"),
                io.String.Output(id="story_name_out", display_name="story_name", tooltip="Selected story name"),
            ],
        )
    
    @classmethod
    def validate_inputs(cls, story_name: str = "default_story", job_id: str = ""):
        """Validate that job_id is valid for the selected story."""
        if not job_id:
            # Empty job_id is allowed - will use most recent
            return True
        
        stories_dir = default_stories_dir()
        story_json_path = os.path.join(stories_dir, story_name, "story.json")
        
        if not os.path.isfile(story_json_path):
            return f"Story '{story_name}' not found"
        
        story_info = load_story(story_json_path)
        if not story_info:
            return f"Failed to load story '{story_name}'"
        
        available_jobs = list_job_ids(story_info.story_dir)
        if job_id not in available_jobs:
            return f"Job ID '{job_id}' not found in story '{story_name}'. Available jobs: {', '.join(available_jobs)}"
        
        return True
    
    @classmethod
    def fingerprint_inputs(cls, story_name: str = "default_story", job_id: str = ""):
        """Generate fingerprint based on story.json and all referenced scene directory content."""
        if story_name:
            stories_dir = default_stories_dir()
            story_json_path = os.path.join(stories_dir, story_name, "story.json")
            if os.path.isfile(story_json_path):
                try:
                    story_stat = os.stat(story_json_path)
                    story_info = load_story(story_json_path)

                    if not story_info or not getattr(story_info, "scenes", None):
                        return (
                            str(story_json_path),
                            int(story_stat.st_mtime_ns),
                            int(story_stat.st_size),
                            (),
                            job_id.strip(),
                        )

                    scenes_dir = Path(default_scenes_dir())
                    scene_fingerprints: list[tuple[str, str, int, int]] = []

                    for scene in sorted(story_info.scenes, key=lambda s: (s.scene_order, s.scene_name)):
                        scene_dir = scenes_dir / scene.scene_name
                        scene_hash, dir_count, file_count = _directory_fingerprint(scene_dir)
                        scene_fingerprints.append((scene.scene_name, scene_hash, dir_count, file_count))

                    return (
                        str(story_json_path),
                        int(story_stat.st_mtime_ns),
                        int(story_stat.st_size),
                        tuple(scene_fingerprints),
                        job_id.strip(),
                    )
                except Exception as e:
                    logger.warning("StoryVideoBatch: Failed to stat story.json for fingerprinting: %s", e)
        # Return None to use default fingerprinting behavior
        return None
    
    @classmethod
    def execute(
        cls,
        story_name: str = "default_story",
        job_id: str = "",
        lora_target_filter: str = "All",
    ) -> io.NodeOutput:
        # Debug logging to track what parameters are being received
        logger.info("StoryVideoBatch.execute called with story_name='%s', job_id='%s', lora_target_filter='%s'", story_name, job_id, lora_target_filter)
        
        # Load story from story_name
        stories_dir = default_stories_dir()
        story_json_path = os.path.join(stories_dir, story_name, "story.json")
        
        if not os.path.isfile(story_json_path):
            logger.warning("StoryVideoBatch: Story file not found: '%s'", story_json_path)
            return io.NodeOutput("", "", None, 0, story_name)
        
        story_info = load_story(story_json_path)
        if story_info is None or not getattr(story_info, "scenes", None):
            logger.warning("StoryVideoBatch: Failed to load story or story has no scenes")
            return io.NodeOutput("", "", None, 0, story_name)
        
        # List available job IDs
        available_jobs = list_job_ids(story_info.story_dir)
        if not available_jobs:
            logger.warning("StoryVideoBatch: No jobs found in story directory '%s'", story_info.story_dir)
            return io.NodeOutput("", "", None, 0, story_name)
        
        logger.info("StoryVideoBatch: Available job_ids for story '%s': %s", story_name, available_jobs)
        
        # Select job ID (use first/newest if not specified or not found)
        if job_id and job_id in available_jobs:
            selected_job = job_id
            logger.info("StoryVideoBatch: Using specified job_id='%s'", selected_job)
        else:
            selected_job = available_jobs[0]
            if job_id:
                logger.warning("StoryVideoBatch: Specified job_id='%s' not found, using most recent: '%s'", job_id, selected_job)
            else:
                logger.info("StoryVideoBatch: No job_id specified, using most recent: '%s'", selected_job)
        
        job_root = Path(story_info.story_dir) / "jobs" / selected_job
        job_input_dir = str(job_root / "input")
        
        if not Path(job_input_dir).exists():
            logger.warning("StoryVideoBatch: Job input directory does not exist: '%s'", job_input_dir)
            return io.NodeOutput("", "", None, 0, story_name)
        
        scenes_dir = default_scenes_dir()
        scenes_sorted = sorted(story_info.scenes, key=lambda s: s.scene_order)
        
        logger.info(
            "StoryVideoBatch: Preparing video prompts for story '%s' with %d scenes from job_id='%s'",
            story_info.story_name,
            len(scenes_sorted),
            selected_job,
        )
        
        # Dict keyed by (lora, model_target) to aggregate unique entries across scenes
        loras_stack_dict: dict[tuple[str, str], dict] = {}
        
        # Build scene descriptors with processed prompts
        scene_descriptors = []
        libber_manager = LibberStateManager.instance()
        
        for scene in scenes_sorted:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            prompt_path = os.path.join(scene_dir, "prompts.json")
            prompt_data_raw = load_prompt_json(prompt_path) or {}
            
            # Process prompts using the v2 system with compositions
            prompt_dict = {}
            composition_dict = {}
            
            if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
                prompt_collection = PromptCollection.from_dict(prompt_data_raw)
                
                # Process individual prompts
                for key, metadata in prompt_collection.prompts.items():
                    value = metadata.value
                    if metadata.processing_type == "libber" and metadata.libber_name:
                        libber = libber_manager.get_libber(metadata.libber_name)
                        if libber:
                            value = libber.substitute(value)
                    prompt_dict[key] = value
                
                # Process compositions
                if prompt_collection.compositions:
                    composition_dict = prompt_collection.compose_prompts(
                        prompt_collection.compositions,
                        libber_manager
                    )
            else:
                # Legacy format
                prompt_dict = prompt_data_raw
            
            # Load LoRA stack and aggregate unique entries
            scene_lora_stack = load_lora_stack(scene_dir) or []
            for entry in scene_lora_stack:
                key = (entry.get("lora", ""), entry.get("model_target", ""))
                if key[0] and key[0].lower() != "none":
                    loras_stack_dict[key] = entry  # last scene's value wins

            scene_descriptors.append({
                "scene": scene,
                "prompt_dict": prompt_dict,
                "composition_dict": composition_dict,
            })
        
        # Generate video prompts for consecutive scene transitions
        video_prompts = []
        scene_pairs = pair_consecutive_scenes(scene_descriptors)
        
        for current_desc, next_desc in scene_pairs:
            current_scene = current_desc["scene"]
            prompt_dict = current_desc["prompt_dict"]
            composition_dict = current_desc["composition_dict"]
            
            # Resolve video prompt based on video_prompt_source
            video_prompt = ""
            
            if current_scene.video_prompt_source == "auto":
                # Use the image prompt based on prompt_source and prompt_key
                if current_scene.prompt_source == "custom":
                    video_prompt = current_scene.custom_prompt
                elif current_scene.prompt_source == "prompt" and current_scene.prompt_key:
                    video_prompt = prompt_dict.get(current_scene.prompt_key, "")
                elif current_scene.prompt_source == "composition" and current_scene.prompt_key:
                    video_prompt = composition_dict.get(current_scene.prompt_key, "")
            
            elif current_scene.video_prompt_source == "custom":
                video_prompt = current_scene.video_custom_prompt
            
            elif current_scene.video_prompt_source == "prompt" and current_scene.video_prompt_key:
                video_prompt = prompt_dict.get(current_scene.video_prompt_key, "")
            
            elif current_scene.video_prompt_source == "composition" and current_scene.video_prompt_key:
                video_prompt = composition_dict.get(current_scene.video_prompt_key, "")
            
            video_prompts.append(video_prompt)
            
            logger.debug(
                "StoryVideoBatch: Added video prompt for transition '%s' -> '%s': %s",
                current_scene.scene_name,
                next_desc["scene"].scene_name if next_desc else "end",
                video_prompt[:50] + "..." if len(video_prompt) > 50 else video_prompt,
            )
        
        # Build the aggregated stack, applying the target filter
        aggregated_stack = list(loras_stack_dict.values())
        if lora_target_filter and lora_target_filter != "All":
            aggregated_stack = [e for e in aggregated_stack if e.get("model_target") == lora_target_filter]
        lora_stack_out = aggregated_stack if aggregated_stack else None
        
        # Join video prompts into multiline string
        video_prompts_multiline = "\n".join(video_prompts)
        
        logger.info(
            "StoryVideoBatch: Generated %d video prompts, %d unique LoRA entries (filter=%s) for story '%s'",
            len(video_prompts),
            len(aggregated_stack),
            lora_target_filter,
            story_name,
        )
        
        return io.NodeOutput(
            job_input_dir,
            video_prompts_multiline,
            lora_stack_out,
            len(video_prompts),
            story_name,
        )


# ── /fbtools/story/* routes — moved from extension.py (Plan 22) ───────────────

@routes.get("/fbtools/story/load/{story_name}")
async def story_load(request):
    """
    Load story data from filesystem.
    Returns: {"story_name": str, "story_dir": str, "scenes": [...]}
    """
    try:
        story_name = request.match_info.get("story_name")
        
        if not story_name:
            return web.json_response({"error": "story_name required"}, status=400)
        
        # Load story from filesystem
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            return web.json_response({"error": f"Story '{story_name}' not found"}, status=404)
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return web.json_response({"error": f"Failed to load story '{story_name}'"}, status=500)
        
        # Convert scenes to dict format for frontend
        scenes_dir = default_scenes_dir()
        scenes_data = []
        for scene in getattr(story_info, "scenes", []):
            # Load available masks for this scene
            available_masks = ["none"]
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if os.path.isdir(scene_dir):
                try:
                    # Load new mask system masks
                    masks_dict = load_masks_json(scene_dir)
                    available_masks.extend(masks_dict.keys())
                    
                    # Add legacy masks if they exist
                    legacy_mask_names = ["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"]
                    for legacy_name in legacy_mask_names:
                        mask_file = f"{legacy_name.replace('_no_bg', '_mask_no_bkgd' if '_no_bg' in legacy_name else '_mask_bkgd')}.png"
                        mask_path = os.path.join(scene_dir, mask_file)
                        if os.path.exists(mask_path) and legacy_name not in available_masks:
                            available_masks.append(legacy_name)
                except Exception as e:
                    logger.debug(f"story_load API: Could not load masks for scene '{scene.scene_name}': {e}")
            
            scenes_data.append({
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_name": getattr(scene, "mask_name", getattr(scene, "mask_type", "")),  # Use mask_name, fall back to mask_type for old data
                "mask_background": scene.mask_background,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key or "",
                "custom_prompt": scene.custom_prompt or "",
                "video_prompt_source": getattr(scene, "video_prompt_source", "auto"),
                "video_prompt_key": getattr(scene, "video_prompt_key", ""),
                "video_custom_prompt": getattr(scene, "video_custom_prompt", ""),
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
                "use_depth": getattr(scene, "use_depth", False),
                "use_mask": getattr(scene, "use_mask", False),
                "use_pose": getattr(scene, "use_pose", False),
                "use_canny": getattr(scene, "use_canny", False),
                "available_masks": available_masks,
            })
        
        return web.json_response({
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scene_count": len(scenes_data),
            "scenes": scenes_data,
        })
    
    except Exception as e:
        logger.exception("Error loading story")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/story/job_ids")
async def story_get_job_ids(request):
    """
    Get list of job IDs for a specific story.
    Query params: story_name (required)
    Returns: {"job_ids": [str]} - List of job IDs sorted by modification time (newest first)
    """
    try:
        story_name = request.query.get('story_name', '')
        
        if not story_name:
            return web.json_response({'error': 'story_name parameter required'}, status=400)
        
        stories_dir = default_stories_dir()
        story_dir = os.path.join(stories_dir, story_name)
        
        if not os.path.isdir(story_dir):
            logger.warning("fbTools API: story_dir '%s' not found for story_name='%s'", story_dir, story_name)
            return web.json_response({'job_ids': []})
        
        job_ids = list_job_ids(story_dir)
        logger.info("fbTools API: story_name='%s' has %d job_ids: %s", story_name, len(job_ids), job_ids)
        
        return web.json_response({'job_ids': job_ids})
    except Exception as e:
        logger.exception("fbTools API: Error getting job_ids for story_name='%s'", story_name)
        return web.json_response({'error': str(e)}, status=500)


@routes.get("/fbtools/story/list")
async def story_list(request):
    """
    Get list of available story names.
    Returns: {"stories": [str]}
    """
    try:
        stories_dir = default_stories_dir()
        available_stories = get_subdirectories(stories_dir)
        story_names = sorted(available_stories.keys()) if available_stories else []
        
        return web.json_response({'stories': story_names})
    except Exception as e:
        logger.exception("fbTools API: Error listing stories")
        return web.json_response({'error': str(e)}, status=500)


@routes.post("/fbtools/story/regenerate_thumbnails")
async def story_regenerate_thumbnails(request):
    """
    Regenerate thumbnails for all scenes in a story that don't have them.
    Body: {"story_name": str}
    Returns: {"success": bool, "regenerated": int, "message": str}
    """
    try:
        data = await request.json()
        story_name = data.get("story_name")
        
        if not story_name:
            return web.json_response({'success': False, 'error': 'story_name is required'}, status=400)
        
        stories_dir = default_stories_dir()
        story_dir = os.path.join(stories_dir, story_name)
        
        if not os.path.isdir(story_dir):
            return web.json_response({'success': False, 'error': f'Story "{story_name}" not found'}, status=404)
        
        # Load story
        story_json_path = os.path.join(story_dir, "story.json")
        story_info = load_story(story_json_path)
        if not story_info:
            return web.json_response({'success': False, 'error': f'Failed to load story "{story_name}"'}, status=500)
        
        regenerated_count = 0
        scenes_dir = default_scenes_dir()
        
        # Regenerate thumbnails for each scene (force=True to regenerate all)
        for scene in story_info.scenes:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if not os.path.isdir(scene_dir):
                logger.warning("Scene directory not found: %s", scene_dir)
                continue
            
            thumbnail_path = os.path.join(scene_dir, "thumbnail.png")
            
            # Load scene info and regenerate thumbnail (force=True)
            scene_info = SceneInfo.from_scene_directory(scene_dir, scene.scene_name)
            scene_info.regenerate_thumbnail(scene_dir, force=True)
            
            # Check if thumbnail was actually created
            if os.path.exists(thumbnail_path):
                regenerated_count += 1
                logger.info("Generated thumbnail for scene '%s'", scene.scene_name)
            else:
                logger.warning("Failed to generate thumbnail for scene '%s'", scene.scene_name)
        
        return web.json_response({
            'success': True,
            'regenerated': regenerated_count,
            'message': f'Regenerated {regenerated_count} thumbnails for story "{story_name}"'
        })
        
    except Exception as e:
        logger.exception("fbTools API: Error regenerating thumbnails")
        return web.json_response({'error': str(e)}, status=500)


@routes.post("/fbtools/story/save")
async def story_save(request):
    """
    Save story data to filesystem.
    Body: {"story_name": str, "scenes": [...]}
    Returns: {"success": bool, "message": str}
    """
    try:
        data = await request.json()
        story_name = data.get("story_name")
        scenes_data = data.get("scenes", [])
        
        logger.info(
            "fb_tools -> StoryEdit: Received save request for story '%s' with %d scenes",
            story_name,
            len(scenes_data),
        )
        
        if not story_name:
            return web.json_response({"error": "story_name required"}, status=400)
        
        # Load existing story
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            logger.warning("fb_tools -> StoryEdit: Story not found at %s", story_json_path)
            return web.json_response({"error": f"Story '{story_name}' not found"}, status=404)
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return web.json_response({"error": f"Failed to load story '{story_name}'"}, status=500)
        
        # Update scenes from received data
        updated_scenes = []
        for scene_data in scenes_data:
            scene = SceneInStory(
                scene_id=scene_data.get("scene_id", ""),
                scene_name=scene_data.get("scene_name", ""),
                scene_order=scene_data.get("scene_order", 0),
                mask_name=scene_data.get("mask_name", scene_data.get("mask_type", "")),  # Support both new and legacy
                mask_background=scene_data.get("mask_background", True),
                prompt_source=scene_data.get("prompt_source", "prompt"),
                prompt_key=scene_data.get("prompt_key", ""),
                custom_prompt=scene_data.get("custom_prompt", ""),
                video_prompt_source=scene_data.get("video_prompt_source", "auto"),
                video_prompt_key=scene_data.get("video_prompt_key", ""),
                video_custom_prompt=scene_data.get("video_custom_prompt", ""),
                depth_type=scene_data.get("depth_type", "depth"),
                pose_type=scene_data.get("pose_type", "open"),
                use_depth=scene_data.get("use_depth", False),
                use_mask=scene_data.get("use_mask", False),
                use_pose=scene_data.get("use_pose", False),
                use_canny=scene_data.get("use_canny", False),
            )
            updated_scenes.append(scene)
        
        # Update story info with new scenes
        story_info.scenes = updated_scenes
        
        # Save to disk
        save_story(story_info, str(story_json_path))
        
        return web.json_response({
            "success": True,
            "message": f"Saved story '{story_name}' with {len(updated_scenes)} scenes"
        })

    except Exception as e:
        logger.exception("Error saving story")
        return web.json_response({"error": str(e)}, status=500)
