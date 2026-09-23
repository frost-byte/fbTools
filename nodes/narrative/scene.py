"""SceneInfo / mask-system core, extracted from extension.py (Plan 20), plus the
10 Scene CRUD node classes and their 5 /fbtools/scene/* routes (Plan 21).

Story and lora-presets stay in extension.py and import SceneInfo/MaskType/etc.
(and now the Scene CRUD classes) back from here.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from enum import Enum
from inspect import cleandoc
from pathlib import Path
from typing import Dict, Optional, Tuple

import folder_paths
import numpy as np
import torch
from aiohttp import web
from comfy_api.latest import io, ui
from folder_paths import get_output_directory
from nodes import ImageScaleBy
from pydantic import BaseModel, ConfigDict

from ..libber import LibberStateManager
from ..lora_stacks import LoraStackData, _lora_entries_for_target, _lora_build_wanvid, _lora_json_to_stack
from ..shared import (
    default_scenes_dir,
    default_libber_dir,
    prefixed_node_id,
    get_subdirectories,
    _directory_fingerprint,
    send_status_update,
    routes,
)
from ...prompt_models import PromptCollection
from ...story_models import SceneInStory
from ...utils.io import load_json_file, load_prompt_json, save_json_file
from ...utils.util import select_text_by_action
from ...utils.images import (
    load_image_comfyui,
    save_image_comfyui,
    make_empty_image,
    normalize_image_tensor,
    generate_thumbnail,
    image_resize_ess,
)
from ...utils.pose import estimate_dwpose, dense_pose, depth_anything, depth_anything_v2, zoe, zoe_any, openpose, midas, canny
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# MASK SYSTEM - Generic User-Definable Masks
# ============================================================================

RGB = Tuple[int, int, int]

class MaskType(str, Enum):
    """Type of mask definition"""
    TRANSPARENT = "transparent"  # alpha / transparency-based
    COLOR = "color"              # color-defined regions


@dataclass
class MaskDefinition:
    """Definition of a scene mask with arbitrary name and properties"""
    name: str
    type: MaskType
    has_background: bool = True
    color: Optional[RGB] = None  # Only used when type == COLOR

    def validate(self) -> None:
        """Validate mask definition constraints"""
        if self.type == MaskType.TRANSPARENT:
            if self.color is not None:
                raise ValueError(
                    f"Transparent mask '{self.name}' must not define a color."
                )
        elif self.type == MaskType.COLOR:
            if self.color is None:
                raise ValueError(
                    f"Color mask '{self.name}' must define an RGB color."
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "has_background": self.has_background,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaskDefinition":
        """Create from dictionary"""
        mask_type = MaskType(data["type"]) if isinstance(data["type"], str) else data["type"]
        color = tuple(data["color"]) if data.get("color") and isinstance(data["color"], (list, tuple)) else data.get("color")
        return cls(
            name=data["name"],
            type=mask_type,
            has_background=data.get("has_background", True),
            color=color
        )

    def get_filename(self) -> str:
        """Generate filename for this mask"""
        suffix = "_bkgd" if self.has_background else "_no_bkgd"
        return f"{self.name}_mask{suffix}.png"


def load_masks_json(scene_dir: str) -> Dict[str, MaskDefinition]:
    """Load mask definitions from masks.json"""
    masks_path = Path(scene_dir) / "masks.json"
    if not masks_path.exists():
        return {}
    
    try:
        with open(masks_path, 'r') as f:
            data = json.load(f)
        
        masks = {}
        for mask_data in data.get("masks", []):
            mask_def = MaskDefinition.from_dict(mask_data)
            mask_def.validate()
            masks[mask_def.name] = mask_def
        
        return masks
    except Exception as e:
        logger.error(f"Failed to load masks.json from {scene_dir}: {e}")
        return {}


def save_masks_json(scene_dir: str, masks: Dict[str, MaskDefinition]) -> None:
    """Save mask definitions to masks.json"""
    masks_path = Path(scene_dir) / "masks.json"
    
    try:
        data = {
            "version": 1,
            "masks": [mask.to_dict() for mask in masks.values()]
        }
        
        with open(masks_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(masks)} mask definitions to {masks_path}")
    except Exception as e:
        logger.error(f"Failed to save masks.json to {scene_dir}: {e}")


class SceneInfo(BaseModel):
    #metadata
    scene_dir: str
    scene_name: str
    
    # Legacy individual prompt fields - maintained for backward compatibility
    girl_pos: str = ""
    male_pos: str = ""
    wan_prompt: str = ""
    wan_low_prompt: str = ""
    four_image_prompt: str = ""
    
    # V2 PromptCollection - new flexible prompt system
    prompts: Optional[PromptCollection] = None
    
    pose_json: str
    resolution: int

    # Image Tensors (ComfyUI uses torch.Tensor with shape [B,H,W,C] for IMAGE)
    depth_image: Optional[torch.Tensor] = None
    depth_any_image: Optional[torch.Tensor] = None
    depth_midas_image: Optional[torch.Tensor] = None
    depth_zoe_image: Optional[torch.Tensor] = None
    depth_zoe_any_image: Optional[torch.Tensor] = None
    pose_dense_image: Optional[torch.Tensor] = None
    pose_dw_image: Optional[torch.Tensor] = None
    pose_dwpose_json: Optional[str] = None
    pose_edit_image: Optional[torch.Tensor] = None
    pose_face_image: Optional[torch.Tensor] = None
    pose_open_image: Optional[torch.Tensor] = None
    pose_nlf_image: Optional[torch.Tensor] = None  # NLF pose rendering
    canny_image: Optional[torch.Tensor] = None
    
    # Image hierarchy: base_image → upscale_image → derived images (pose, depth, canny)
    base_image: Optional[torch.Tensor] = None  # Original input image (saved as base.png)
    upscale_image: Optional[torch.Tensor] = None  # Scaled version of base_image (source for derived images)
    
    lora_stack: Optional[list] = None  # list of LORA_ENTRY dicts (LoraStackCollect format)

    # Mask system - generic user-definable masks
    masks: Optional[Dict[str, MaskDefinition]] = None  # Mask definitions by name
    mask_images: Optional[Dict[str, torch.Tensor]] = None  # Mask image tensors by name
    
    # Legacy mask fields - maintained for backward compatibility with existing scenes
    girl_mask_bkgd_image: Optional[torch.Tensor] = None
    male_mask_bkgd_image: Optional[torch.Tensor] = None
    combined_mask_bkgd_image: Optional[torch.Tensor] = None
    girl_mask_no_bkgd_image: Optional[torch.Tensor] = None
    male_mask_no_bkgd_image: Optional[torch.Tensor] = None
    combined_mask_no_bkgd_image: Optional[torch.Tensor] = None
    
    # Backward compatibility properties - delegate to PromptCollection if present
    def get_prompt_field(self, field_name: str, legacy_value: str) -> str:
        """Get prompt from PromptCollection if available, else return legacy field."""
        if self.prompts:
            value = self.prompts.get_prompt_value(field_name)
            return value if value is not None else legacy_value
        return legacy_value

    def three_image_prompt(self) -> str:
        return f"{self.girl_pos} {self.male_pos}"

    def input_img_glob(self) -> str:
        return os.path.join(self.scene_dir, "input") + "/*.png"

    def input_img_dir(self) -> str:
        return f"scenes/{self.scene_name}/input/img"

    def output_dir(self) -> str:
        return f"scenes/{self.scene_name}/output"

    @classmethod
    def load_depth_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load depth images from a scene directory, optionally filtering by keys."""
        mapping = {
            'depth_image': "depth.png",
            'depth_any_image': "depth_any.png",
            'depth_midas_image': "depth_midas.png",
            'depth_zoe_image': "depth_zoe.png",
            'depth_zoe_any_image': "depth_zoe_any.png",
        }

        def _img(path: str):
            img, _ = load_image_comfyui(path, include_mask=False)
            return img

        selected_keys = list(mapping.keys()) if keys is None else list(keys)
        images = {}
        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            images[key] = _img(os.path.join(scene_dir, filename))
        return images

    @classmethod
    def load_pose_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load pose images from a scene directory, optionally filtering by keys."""
        mapping = {
            'base_image': "base.png",  # Original input image
            'pose_dense_image': "pose_dense.png",
            'pose_dw_image': "pose_dw.png",
            'pose_edit_image': "pose_edit.png",
            'pose_face_image': "pose_face.png",
            'pose_open_image': "pose_open.png",
            'pose_nlf_image': "pose_nlf.png",  # NLF pose rendering
            'canny_image': "canny.png",
            'upscale_image': "upscale.png",
        }

        def _img(path: str):
            img, _ = load_image_comfyui(path, include_mask=False)
            return img

        selected_keys = list(mapping.keys()) if keys is None else list(keys)
        images = {}
        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            images[key] = _img(os.path.join(scene_dir, filename))
        return images

    @classmethod
    def load_mask_images(cls, scene_dir: str, mask_names: Optional[list[str]] = None) -> tuple[dict, dict]:
        """Load mask images and their alpha masks from a scene directory.

        Returns a tuple `(images, masks)` where images maps mask names to IMAGE tensors,
        and masks maps the same names to [B,H,W] float masks (1.0 means masked-out).
        
        Args:
            scene_dir: Path to scene directory
            mask_names: Optional list of mask names to load. If None, loads all from masks.json
        
        Returns:
            (images_dict, masks_dict) tuple
        """
        # Load mask definitions
        mask_defs = load_masks_json(scene_dir)
        
        # If no masks.json exists, try legacy format
        if not mask_defs:
            logger.debug(f"No masks.json found in {scene_dir}, trying legacy mask format")
            return cls._load_legacy_mask_images(scene_dir, mask_names)
        
        images = {}
        masks = {}
        
        # Determine which masks to load
        names_to_load = mask_names if mask_names else list(mask_defs.keys())
        
        for name in names_to_load:
            if name not in mask_defs:
                # Only warn if it's not the "combined" fallback
                if name != "combined":
                    logger.warning(f"Mask '{name}' not found in mask definitions")
                continue
            
            mask_def = mask_defs[name]
            filename = mask_def.get_filename()
            filepath = os.path.join(scene_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Mask file not found: {filepath}")
                continue
            
            try:
                image, mask = load_image_comfyui(filepath, include_mask=True)
                images[name] = image
                masks[name] = mask
            except Exception as e:
                logger.error(f"Failed to load mask '{name}' from {filepath}: {e}")
        
        return images, masks

    @classmethod
    def _load_legacy_mask_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> tuple[dict, dict]:
        """Load legacy hardcoded mask images for backward compatibility.
        
        Returns a tuple `(images, masks)` where images maps mask keys to IMAGE tensors,
        and masks maps the same keys to [B,H,W] float masks (1.0 means masked-out).
        """
        mapping = {
            "girl": "girl_mask_bkgd.png",
            "male": "male_mask_bkgd.png",
            "combined": "combined_mask_bkgd.png",
            "girl_no_bg": "girl_mask_no_bkgd.png",
            "male_no_bg": "male_mask_no_bkgd.png",
            "combined_no_bg": "combined_mask_no_bkgd.png",
        }

        images = {}
        masks = {}
        selected_keys = list(mapping.keys()) if keys is None else list(keys)

        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            filepath = os.path.join(scene_dir, filename)
            if not os.path.exists(filepath):
                continue
            
            try:
                image, mask = load_image_comfyui(filepath, include_mask=True)
                images[key] = image
                masks[key] = mask
            except Exception as e:
                logger.debug(f"Could not load legacy mask {filename}: {e}")

        return images, masks

    @classmethod
    def load_all_images(cls, scene_dir: str) -> dict:
        """Load all images (depth, pose, mask) from a scene directory"""
        all_images = {}
        all_images.update(cls.load_depth_images(scene_dir))
        all_images.update(cls.load_pose_images(scene_dir))
        mask_images, _ = cls.load_mask_images(scene_dir)
        all_images.update(mask_images)
        return all_images

    @classmethod
    def load_preview_assets(
            cls,
            scene_dir: str,
            depth_attr: str,
            pose_attr: str,
            mask_name: str,
            mask_background: Optional[bool] = None,  # None = use mask name directly (new system), True/False = legacy behavior
            include_upscale: bool = False,
            include_canny: bool = False,
    ) -> dict:
        """Load a minimal, normalized bundle for preview/output (depth, pose, mask, base_image, optional canny).

        Returns dict keys:
            depth_image, pose_image, mask_image, mask (B,H,W,1), mask_preview (B,H,W,3),
            base_image, canny_image, preview_batch (list of tensors), H, W, resolution,
            plus raw dictionaries depth_images/pose_images/mask_images for downstream SceneInfo population.
        
        Args:
            scene_dir: Path to scene directory
            depth_attr: Depth image attribute name to load
            pose_attr: Pose image attribute name to load
            mask_name: Name of mask to load
            mask_background: For legacy masks - whether to include background. If None, uses mask_name directly
            include_upscale: Whether to include upscale image (kept for compatibility)
            include_canny: Whether to include canny image
        
        Note: include_upscale parameter is kept for compatibility but base_image is always loaded for previews.
        """
        mask_key = resolve_mask_key(mask_name, mask_background)

        depth_keys = {depth_attr, "depth_image"}
        pose_keys = {pose_attr, "pose_open_image", "base_image"}  # Always load base_image for preview
        if include_canny:
            pose_keys.add("canny_image")
        
        # For masks, try to load the requested mask
        # For legacy compatibility, also try to load "combined" as fallback (only if mask_key is not empty)
        mask_names_to_load = []
        if mask_key:  # Only load masks if mask_key is not empty
            mask_names_to_load.append(mask_key)
            if mask_key != "combined":
                mask_names_to_load.append("combined")

        depth_images = cls.load_depth_images(scene_dir, keys=list(depth_keys))
        pose_images = cls.load_pose_images(scene_dir, keys=list(pose_keys))
        mask_images, mask_tensors = cls.load_mask_images(scene_dir, mask_names=mask_names_to_load) if mask_names_to_load else ({}, {})

        # Determine spatial size from available images
        empty_image = make_empty_image(1, 512, 512)
        base_image = pose_images.get("base_image")  # Always load base for preview
        depth_image_raw = depth_images.get("depth_image")
        pose_image_raw = pose_images.get(pose_attr, pose_images.get("pose_open_image", empty_image))
        mask_image_raw = mask_images.get(mask_key, mask_images.get("combined", empty_image))

        if depth_image_raw is not None:
            H, W = depth_image_raw.shape[1], depth_image_raw.shape[2]
        elif pose_image_raw is not None:
            H, W = pose_image_raw.shape[1], pose_image_raw.shape[2]
        elif mask_image_raw is not None:
            H, W = mask_image_raw.shape[1], mask_image_raw.shape[2]
        elif base_image is not None:
            H, W = base_image.shape[1], base_image.shape[2]
        else:
            H, W = 512, 512

        # Normalize images to a consistent size
        depth_image = normalize_image_tensor(depth_images.get(depth_attr, depth_images.get("depth_image", empty_image)), H, W)
        pose_image = normalize_image_tensor(pose_image_raw, H, W)
        base_image = normalize_image_tensor(base_image, H, W) if base_image is not None else None
        mask_image = normalize_image_tensor(mask_image_raw, H, W)

        # Build mask output (single-channel) and preview (3-channel)
        mask = None
        mask_tensor = mask_tensors.get(mask_key)
        unsqueeze_me = False
        if mask_tensor is not None:
            logger.debug("SceneInfo.load_preview_assets: using mask tensor for key '%s'", mask_key)
            mask = mask_tensor
            unsqueeze_me = True
        elif mask_image is not None:
            logger.debug("SceneInfo.load_preview_assets: building empty mask matching mask_image shape")
            b, hh, ww, _ = mask_image.shape
            mask = torch.zeros((b, hh, ww, 1), device=mask_image.device, dtype=torch.float32)
        else:
            logger.debug(
                "SceneInfo.load_preview_assets: building empty mask of size (1,%s,%s,1)",
                H,
                W,
            )
            mask = torch.zeros((1, H, W, 1), dtype=torch.float32)

        if mask is not None and mask.dtype != torch.float32:
            logger.debug("SceneInfo.load_preview_assets: converting mask to float32")
            mask = mask.float()

        mask_preview = None
        if mask is not None:
            preview_mask = mask
            if unsqueeze_me:
                preview_mask = mask.unsqueeze(-1)
            if preview_mask.shape[-1] == 1:
                preview_mask = preview_mask.repeat(1, 1, 1, 3)
            mask_preview = normalize_image_tensor(preview_mask, H, W)

        canny_image = None
        if include_canny:
            canny_image = normalize_image_tensor(pose_images.get("canny_image"), H, W)

        preview_batch = []
        if base_image is not None:
            preview_batch.append(base_image)
        if mask_image is not None:
            preview_batch.append(mask_image)
        if pose_image is not None:
            preview_batch.append(pose_image)
        if depth_image is not None:
            preview_batch.append(depth_image)
        if mask_preview is not None:
            preview_batch.append(mask_preview)

        resolution = max(H, W)

        return {
            "depth_image": depth_image,
            "pose_image": pose_image,
            "mask_image": mask_image,
            "mask": mask,
            "mask_preview": mask_preview,
            "base_image": base_image,
            "canny_image": canny_image,
            "preview_batch": preview_batch,
            "H": H,
            "W": W,
            "resolution": resolution,
            "depth_images": depth_images,
            "pose_images": pose_images,
            "mask_images": mask_images,
        }


    @classmethod
    def from_story_scene(
            cls,
            scene: "SceneInStory",
            scenes_dir: Optional[str] = None,
            prompt_in: str = "",
            prompt_action: str = "use_file",
            include_upscale: bool = False,
            include_canny: bool = False,
            prompt_override: Optional[str] = None,
            scene_dir_override: Optional[str] = None,
    ) -> tuple["SceneInfo", dict, str, dict, Optional[str]]:
        """Build SceneInfo + assets from a SceneInStory configuration.

        Returns (scene_info, assets, selected_prompt, prompt_data, prompt_widget_text).
        """

        scenes_dir = scenes_dir or default_scenes_dir()
        scene_dir = scene_dir_override if scene_dir_override else os.path.join(scenes_dir, scene.scene_name)

        if not os.path.isdir(scene_dir):
            raise ValueError(f"from_story_scene: scene_dir '{scene_dir}' is invalid")

        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_data_raw = load_prompt_json(prompt_json_path) or {}
        
        # Load the scene's PromptCollection to get prompt_dict and composition_dict
        if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
            prompt_collection = PromptCollection.from_dict(prompt_data_raw)
        else:
            # Legacy format - migrate
            prompt_collection = PromptCollection.from_legacy_dict(prompt_data_raw)
        
        logger.debug(
            "SceneInfo.from_story_scene: Loaded PromptCollection with %d prompts and %d compositions",
            len(prompt_collection.prompts),
            len(prompt_collection.compositions),
        )
        if prompt_collection.compositions:
            logger.debug("  -> compositions: %s", list(prompt_collection.compositions.keys()))
        else:
            logger.debug("  -> compositions: None/Empty")
        
        # Use shared LibberStateManager so loaded libbers (e.g., story_libber) are applied
        libber_manager = LibberStateManager.instance()
        
        # Build prompt_dict: just the raw individual prompts (not composed)
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
        # compositions is dict[str, List[str]] where key is output name, value is list of prompt keys
        composition_dict = {}
        if prompt_collection.compositions:
            composition_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
            logger.debug("  -> Composed %d compositions: %s", len(composition_dict), list(composition_dict.keys()))
        else:
            logger.debug("  -> No compositions to compose")
        
        # Determine the selected prompt based on prompt_source and prompt_key
        prompt_file_text = ""
        if scene.prompt_source == "custom":
            prompt_file_text = scene.custom_prompt
        elif scene.prompt_source == "prompt" and scene.prompt_key:
            prompt_file_text = prompt_dict.get(scene.prompt_key, "")
        elif scene.prompt_source == "composition" and scene.prompt_key:
            prompt_file_text = composition_dict.get(scene.prompt_key, "")
        
        logger.debug(
            "SceneInfo.from_story_scene: scene=%s, prompt_source=%s, prompt_key=%s",
            scene.scene_name,
            scene.prompt_source,
            scene.prompt_key,
        )
        logger.debug("  -> prompt_dict has %d keys", len(prompt_dict))
        logger.debug("  -> composition_dict has %d keys", len(composition_dict))
        logger.debug("  -> prompt_file_text length: %d", len(prompt_file_text))
        
        class_name = f"{cls.__name__}.from_story_scene"
        selected_prompt, prompt_widget_text = select_text_by_action(
            prompt_in,
            prompt_file_text,
            prompt_action,
            class_name,
        )
        if prompt_override:
            selected_prompt = prompt_override

        pose_json_path = os.path.join(scene_dir, "pose.json")
        pose_json_obj = load_json_file(pose_json_path)
        pose_json = json.dumps(pose_json_obj) if pose_json_obj else "[]"

        lora_stack = load_lora_stack(scene_dir)

        depth_attr = default_depth_options.get(scene.depth_type, "depth_image")
        pose_attr = default_pose_options.get(scene.pose_type, "pose_open_image")

        assets = cls.load_preview_assets(
            scene_dir,
            depth_attr=depth_attr,
            pose_attr=pose_attr,
            mask_name=scene.mask_name,
            mask_background=scene.mask_background,
            include_upscale=include_upscale,
            include_canny=include_canny,
        )

        depth_images = assets.get("depth_images", {})
        pose_images = assets.get("pose_images", {})
        mask_images = assets.get("mask_images", {})
        
        # For backwards compatibility, keep the old fields but they'll be empty
        # since we no longer use them in the new system
        scene_info = cls(
            scene_dir=scene_dir,
            scene_name=scene.scene_name,
            girl_pos="",  # Deprecated
            male_pos="",  # Deprecated
            four_image_prompt="",  # Deprecated
            wan_prompt="",  # Deprecated
            wan_low_prompt="",  # Deprecated
            pose_json=pose_json,
            resolution=assets.get("resolution", 0),
            prompts=prompt_collection,  # Now using PromptCollection
            lora_stack=lora_stack,
            **depth_images,
            **pose_images,
            **mask_images,
        )
        
        # Return prompt_dict in the prompt_data for compatibility
        return_prompt_data = {
            "prompt_dict": prompt_dict,
            "composition_dict": composition_dict,
        }

        return scene_info, assets, selected_prompt or "", return_prompt_data, prompt_widget_text

    @classmethod
    def from_scene_directory(cls, scene_dir: str, scene_name: str, prompt_data: Optional[dict] = None,
                           pose_json: str = "", lora_stack: Optional[list] = None):
        """Create a SceneInfo instance by loading all data from a scene directory"""
        if prompt_data is None:
            prompt_json_path = os.path.join(scene_dir, "prompts.json")
            prompt_data = load_prompt_json(prompt_json_path)
        
        # Migrate legacy prompts to PromptCollection
        prompt_collection = None
        if prompt_data:
            # Check if it's v2 format (has "version" field)
            if "version" in prompt_data and prompt_data.get("version") == 2:
                prompt_collection = PromptCollection.from_dict(prompt_data)
            else:
                # Legacy format - migrate
                prompt_collection = PromptCollection.from_legacy_dict(prompt_data)
                logger.info(
                    "SceneInfo.from_scene_directory: Migrated %d legacy prompts",
                    len(prompt_collection.prompts),
                )
        else:
            # No prompts file - create empty collection
            prompt_collection = PromptCollection()
        
        # Load all images
        all_images = cls.load_all_images(scene_dir)
        
        # Load mask definitions and separate out mask images
        mask_defs = load_masks_json(scene_dir)
        mask_images_dict = {}
        
        # Extract mask images from all_images based on mask definitions
        for mask_name in list(mask_defs.keys()):
            if mask_name in all_images:
                mask_images_dict[mask_name] = all_images.pop(mask_name)
        
        # Determine resolution from depth_image
        depth_image = all_images.get('depth_image')
        if depth_image is not None:
            H, W = depth_image.shape[1], depth_image.shape[2]
            resolution = max(H, W)
        else:
            resolution = 512
        
        return cls(
            scene_dir=scene_dir,
            scene_name=scene_name,
            prompts=prompt_collection,
            pose_json=pose_json,
            resolution=resolution,
            lora_stack=lora_stack,
            masks=mask_defs if mask_defs else None,
            mask_images=mask_images_dict if mask_images_dict else None,
            **all_images
        )

    def save_all_images(self, scene_dir: Optional[str] = None):
        """Save all images to the scene directory"""
        from pathlib import Path
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        
        # Save depth images
        if self.depth_image is not None:
            save_image_comfyui(self.depth_image, scene_path / "depth.png")
        if self.depth_any_image is not None:
            save_image_comfyui(self.depth_any_image, scene_path / "depth_any.png")
        if self.depth_midas_image is not None:
            save_image_comfyui(self.depth_midas_image, scene_path / "depth_midas.png")
        if self.depth_zoe_image is not None:
            save_image_comfyui(self.depth_zoe_image, scene_path / "depth_zoe.png")
        if self.depth_zoe_any_image is not None:
            save_image_comfyui(self.depth_zoe_any_image, scene_path / "depth_zoe_any.png")
        
        # Handle base.webp conversion to base.png
        base_webp_path = scene_path / "base.webp"
        base_png_path = scene_path / "base.png"
        if base_webp_path.exists() and not base_png_path.exists():
            try:
                from PIL import Image
                webp_img = Image.open(base_webp_path)
                webp_img.save(base_png_path, format='PNG')
                logger.info("SceneInfo: Converted base.webp to base.png")
            except Exception as e:
                logger.error("SceneInfo: Failed to convert base.webp to base.png: %s", e)
        
        # Save pose images
        if self.base_image is not None:
            save_image_comfyui(self.base_image, base_png_path)
            # Generate thumbnail from base image
            generate_thumbnail(self.base_image, scene_path / "thumbnail.png", size=(128, 128))
        elif self.upscale_image is not None:
            # Check if we should create base.png from upscale.png using depth.png dimensions
            depth_png_path = scene_path / "depth.png"
            if depth_png_path.exists() and not base_png_path.exists():
                try:
                    # Load depth to get target dimensions
                    depth_img, _ = load_image_comfyui(str(depth_png_path), include_mask=False)
                    _, depth_h, depth_w, _ = depth_img.shape
                    
                    # Resize upscale image to match depth dimensions
                    from PIL import Image
                    upscale_np = (self.upscale_image[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    upscale_pil = Image.fromarray(upscale_np)
                    resized_pil = upscale_pil.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
                    
                    # Convert back to tensor and save
                    import numpy as np
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    base_tensor = torch.from_numpy(resized_np).unsqueeze(0)
                    save_image_comfyui(base_tensor, base_png_path)
                    logger.info("SceneInfo: Created base.png from upscale.png at depth.png resolution")
                except Exception as e:
                    logger.error("SceneInfo: Failed to create base.png from upscale: %s", e)
            
            # Fallback: use upscale image for thumbnail if base doesn't exist
            generate_thumbnail(self.upscale_image, scene_path / "thumbnail.png", size=(128, 128))
        
        if self.pose_dense_image is not None:
            save_image_comfyui(self.pose_dense_image, scene_path / "pose_dense.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_dense_image")
        if self.pose_dw_image is not None:
            save_image_comfyui(self.pose_dw_image, scene_path / "pose_dw.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_dw_image")
        if self.pose_edit_image is not None:
            save_image_comfyui(self.pose_edit_image, scene_path / "pose_edit.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_edit_image")
        if self.pose_face_image is not None:
            save_image_comfyui(self.pose_face_image, scene_path / "pose_face.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_face_image")
        if self.pose_open_image is not None:
            save_image_comfyui(self.pose_open_image, scene_path / "pose_open.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_open_image")
        if self.pose_nlf_image is not None:
            save_image_comfyui(self.pose_nlf_image, scene_path / "pose_nlf.png")
            logger.info("SceneInfo.save_all_images: Saved pose_nlf_image to pose_nlf.png")
        else:
            logger.debug("SceneInfo.save_all_images: pose_nlf_image is None, skipping")
        if self.canny_image is not None:
            save_image_comfyui(self.canny_image, scene_path / "canny.png")
        if self.upscale_image is not None:
            save_image_comfyui(self.upscale_image, scene_path / "upscale.png")
        
        # Save new mask system images and definitions
        if self.masks and self.mask_images:
            # Save mask definitions
            save_masks_json(str(scene_path), self.masks)
            
            # Save mask image files
            for mask_name, mask_tensor in self.mask_images.items():
                if mask_tensor is not None and mask_name in self.masks:
                    mask_def = self.masks[mask_name]
                    filename = mask_def.get_filename()
                    save_image_comfyui(mask_tensor, scene_path / filename)
                    logger.debug(f"Saved mask '{mask_name}' to {filename}")
        
        # Save legacy mask images (for backward compatibility)
        if self.girl_mask_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_bkgd_image, scene_path / "girl_mask_bkgd.png")
        if self.male_mask_bkgd_image is not None:
            save_image_comfyui(self.male_mask_bkgd_image, scene_path / "male_mask_bkgd.png")
        if self.combined_mask_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_bkgd_image, scene_path / "combined_mask_bkgd.png")
        if self.girl_mask_no_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_no_bkgd_image, scene_path / "girl_mask_no_bkgd.png")
        if self.male_mask_no_bkgd_image is not None:
            save_image_comfyui(self.male_mask_no_bkgd_image, scene_path / "male_mask_no_bkgd.png")
        if self.combined_mask_no_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_no_bkgd_image, scene_path / "combined_mask_no_bkgd.png")

    def save_prompts(self, scene_dir: Optional[str] = None):
        """Save prompts to prompts.json in v2 format with v1_backup"""
        from pathlib import Path
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        prompts_path = scene_path / "prompts.json"
        
        # If using PromptCollection, save v2 format
        if self.prompts:
            save_json_file(prompts_path, self.prompts.to_dict())
        else:
            # Legacy mode: save v1 format but wrap in v2 structure for migration
            legacy_data = {
                "girl_pos": self.girl_pos if self.girl_pos else "",
                "male_pos": self.male_pos if self.male_pos else "",
                "wan_prompt": self.wan_prompt if self.wan_prompt else "",
                "wan_low_prompt": self.wan_low_prompt if self.wan_low_prompt else "",
                "four_image_prompt": self.four_image_prompt if self.four_image_prompt else "",
            }
            # Auto-migrate to v2 format on save
            prompt_collection = PromptCollection.from_legacy_dict(legacy_data)
            save_json_file(prompts_path, prompt_collection.to_dict())

    def save_pose_json(self, scene_dir: Optional[str] = None):
        """Save pose_json to pose.json in the pose directory"""
        from pathlib import Path
        import json
        
        if not self.pose_json:
            return
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        pose_json_path = scene_path / "pose.json"
        save_json_file(pose_json_path, json.loads(self.pose_json))

    def save_loras(self, scene_dir: Optional[str] = None):
        """Save LoRA stack to lora_stack.json in the scene directory."""
        from pathlib import Path

        if self.lora_stack is None:
            return

        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        lora_stack_path = scene_path / "lora_stack.json"
        save_json_file(str(lora_stack_path), self.lora_stack)

    def ensure_directories(self, scene_dir: Optional[str] = None):
        """Ensure scene directory and input/output subdirectories exist"""
        import os
        
        scene_path = scene_dir if scene_dir else self.scene_dir
        
        if not os.path.exists(scene_path):
            os.makedirs(scene_path, exist_ok=True)
            logger.info("SceneInfo: Created scene_dir='%s'", scene_path)
        
        input_dir = os.path.join(scene_path, "input")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            logger.info("SceneInfo: Created input_dir='%s'", input_dir)
        
        output_dir = os.path.join(scene_path, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info("SceneInfo: Created output_dir='%s'", output_dir)

    def save_all(self, scene_dir: Optional[str] = None):
        """Save all scene data (images, prompts, pose_json, loras) to the scene directory"""
        target_dir = scene_dir if scene_dir else self.scene_dir
        self.ensure_directories(target_dir)
        self.save_all_images(target_dir)
        self.save_prompts(target_dir)
        self.save_pose_json(target_dir)
        self.save_loras(target_dir)
    
    def regenerate_thumbnail(self, scene_dir: Optional[str] = None, force: bool = False):
        """Regenerate thumbnail if missing or base/upscale image changed
        
        Args:
            scene_dir: Target scene directory (uses self.scene_dir if None)
            force: If True, regenerate thumbnail even if it already exists
        """
        from pathlib import Path
        from PIL import Image
        import numpy as np
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        thumbnail_path = scene_path / "thumbnail.png"
        
        # Check if thumbnail already exists (skip if not forcing)
        if thumbnail_path.exists() and not force:
            logger.debug("SceneInfo: Thumbnail already exists at '%s'", thumbnail_path)
            return
        
        # Define paths for disk-based operations
        base_png_path = scene_path / "base.png"
        base_webp_path = scene_path / "base.webp"
        upscale_path = scene_path / "upscale.png"
        depth_path = scene_path / "depth.png"
        
        # Priority 1: Check for base.webp and convert to base.png
        if base_webp_path.exists() and not base_png_path.exists():
            try:
                webp_img = Image.open(base_webp_path)
                webp_img.save(base_png_path, format='PNG')
                logger.info("SceneInfo: Converted base.webp to base.png at '%s'", base_png_path)
            except Exception as e:
                logger.error("SceneInfo: Failed to convert base.webp to base.png: %s", e)
        
        # Priority 2: Use base_image from memory if available
        if self.base_image is not None:
            generate_thumbnail(self.base_image, thumbnail_path, size=(128, 128))
            logger.info("SceneInfo: Generated thumbnail from base_image in memory at '%s'", thumbnail_path)
            return
        
        # Priority 3: Use base.png from disk
        if base_png_path.exists():
            try:
                img, _ = load_image_comfyui(str(base_png_path), include_mask=False)
                generate_thumbnail(img, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from base.png at '%s'", thumbnail_path)
                return
            except Exception as e:
                logger.error("SceneInfo: Failed to generate thumbnail from base.png: %s", e)
        
        # Priority 4: Use upscale_image from memory if available
        if self.upscale_image is not None:
            generate_thumbnail(self.upscale_image, thumbnail_path, size=(128, 128))
            logger.info("SceneInfo: Generated thumbnail from upscale_image in memory at '%s'", thumbnail_path)
            return
        
        # Priority 5: Create base.png from upscale.png if base.png doesn't exist
        if upscale_path.exists() and not base_png_path.exists():
            try:
                img, _ = load_image_comfyui(str(upscale_path), include_mask=False)
                
                # Determine target resolution for base.png
                target_width, target_height = 1024, 1024  # Default resolution
                
                # Check if depth.png exists and is not empty (64x64)
                if depth_path.exists():
                    try:
                        depth_img, _ = load_image_comfyui(str(depth_path), include_mask=False)
                        _, depth_h, depth_w, _ = depth_img.shape
                        
                        # Only use depth dimensions if not the empty 64x64 size
                        if depth_w != 64 or depth_h != 64:
                            target_width, target_height = depth_w, depth_h
                            logger.info("SceneInfo: Using depth.png resolution for base.png: %dx%d", target_width, target_height)
                        else:
                            logger.info("SceneInfo: depth.png is 64x64 (empty), using default 1024x1024 for base.png")
                    except Exception as e:
                        logger.warning("SceneInfo: Failed to read depth.png dimensions, using default 1024x1024: %s", e)
                else:
                    logger.info("SceneInfo: No depth.png found, using default 1024x1024 for base.png")
                
                # Resize upscale to target dimensions and save as base.png
                upscale_np = (img[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                upscale_pil = Image.fromarray(upscale_np)
                resized_pil = upscale_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Convert back to tensor and save
                resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                base_tensor = torch.from_numpy(resized_np).unsqueeze(0)
                save_image_comfyui(base_tensor, base_png_path)
                logger.info("SceneInfo: Created base.png from upscale.png at %dx%d resolution", target_width, target_height)
                
                # Now generate thumbnail from the newly created base.png
                generate_thumbnail(base_tensor, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from newly created base.png at '%s'", thumbnail_path)
                return
                
            except Exception as e:
                logger.error("SceneInfo: Failed to create base.png and thumbnail from upscale.png: %s", e)
        
        # Priority 6: If upscale.png exists but base.png already exists, use upscale for thumbnail
        if upscale_path.exists():
            try:
                img, _ = load_image_comfyui(str(upscale_path), include_mask=False)
                generate_thumbnail(img, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from upscale.png at '%s'", thumbnail_path)
                return
            except Exception as e:
                logger.error("SceneInfo: Failed to generate thumbnail from upscale.png: %s", e)
        
        # Priority 7: Create empty/default thumbnail
        try:
            # Create a small empty gray image as default
            default_img = Image.new('RGB', (128, 128), color=(64, 64, 64))
            default_img.save(thumbnail_path, format='PNG')
            logger.info("SceneInfo: Created default empty thumbnail at '%s'", thumbnail_path)
        except Exception as e:
            logger.error("SceneInfo: Failed to create default thumbnail: %s", e)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


def _migrate_loras_json_to_stack(loras_json_path: str) -> list:
    """Convert a legacy loras.json (high/low WANVIDLORA format) to a lora_stack list.

    Each 'high' entry becomes a Wan2.2-Wrapper-High LORA_ENTRY dict;
    each 'low' entry becomes a Wan2.2-Wrapper-Low entry.
    Per-entry blocks, layer_filter, low_mem_load, and merge_loras are preserved
    so WANVIDLORA output remains bit-for-bit identical to the old output.
    """
    data = load_json_file(loras_json_path)
    if not data or not isinstance(data, dict):
        return []

    entries: list[dict] = []
    for target, lora_type in [("Wan2.2-Wrapper-High", "high"), ("Wan2.2-Wrapper-Low", "low")]:
        for item in data.get(lora_type, []):
            lora_name = item.get("lora_name", "")
            strength  = item.get("strength", 1.0)
            if not lora_name or lora_name.lower() == "none":
                continue
            entries.append({
                "lora":           lora_name,
                "model_target":   target,
                "strength_model": strength,
                "strength_clip":  1.0,
                "enabled":        True,
                # Preserve WanVideoWrapper-specific fields verbatim
                "blocks":         item.get("blocks", {}),
                "layer_filter":   item.get("layer_filter", ""),
                "low_mem_load":   item.get("low_mem_load", False),
                "merge_loras":    item.get("merge_loras", False),
            })
    return entries


def load_lora_stack(scene_dir: str) -> Optional[list]:
    """Load the LoRA stack for a scene.

    Checks for lora_stack.json first (new format).
    Falls back to migrating loras.json (old Wan-only WANVIDLORA format) if not found.
    Returns None if neither file exists.
    """
    lora_stack_path = os.path.join(scene_dir, "lora_stack.json")
    if os.path.isfile(lora_stack_path):
        data = load_json_file(lora_stack_path)
        return data if isinstance(data, list) else None

    # Legacy migration path — does not write anything; StorySceneBatch/SceneSelect
    # will transparently use migrated data.  Run the migration script to persist.
    loras_json_path = os.path.join(scene_dir, "loras.json")
    if os.path.isfile(loras_json_path):
        return _migrate_loras_json_to_stack(loras_json_path)

    return None


default_depth_options = {
    "depth": "depth_image",
    "depth_any": "depth_any_image",
    "midas": "depth_midas_image",
    "zoe": "depth_zoe_image",
    "zoe_any": "depth_zoe_any_image",
}

default_pose_options = {
    "dense": "pose_dense_image",
    "dw": "pose_dw_image",
    "edit": "pose_edit_image",
    "face": "pose_face_image",
    "open": "pose_open_image",
    "nlf": "pose_nlf_image",
}

default_mask_options = {
    "girl": "girl_mask_bkgd",
    "male": "male_mask_bkgd",
    "combined": "combined_mask_bkgd",
    "girl_no_bg": "girl_mask_no_bkgd",
    "male_no_bg": "male_mask_no_bkgd",
    "combined_no_bg": "combined_mask_no_bkgd",
}

def resolve_mask_key(mask_name: str, mask_background: Optional[bool] = None) -> str:
    """Return the mask key to use. 
    
    For new mask system: just returns mask_name
    For legacy system: handles _no_bg suffix based on mask_background flag
    
    Args:
        mask_name: Name of the mask
        mask_background: Optional flag for legacy masks. If provided, adds/removes _no_bg suffix
    
    Returns:
        Mask key/name to use for lookups
    """
    # If mask_background is not specified, return name as-is (new system)
    if mask_background is None:
        return mask_name
    
    # Legacy behavior: add/remove _no_bg suffix
    key = mask_name or "combined"
    if not mask_background and not key.endswith("_no_bg"):
        key = f"{key}_no_bg"
    elif mask_background and key.endswith("_no_bg"):
        # Remove _no_bg if background is requested
        key = key.replace("_no_bg", "")
    return key


# ============================================================================
# SCENE CRUD NODES — moved from extension.py (Plan 21)
# ============================================================================

@io.comfytype(io_type="DICT")
class DictType:
    Type = dict
    
    class Output(io.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class Input(io.Input):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class SceneSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_dir = os.path.join(output_dir, "scenes")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            os.makedirs(os.path.join(default_dir, "default_scene"), exist_ok=True)

        subdir_dict = get_subdirectories(default_dir)
        default_options = sorted(subdir_dict.keys()) if subdir_dict else ["default_scene"]
        default_scene = default_options[0]
        pose_options = list(default_pose_options.keys())
        depth_options = list(default_depth_options.keys())
        
        # Load mask names from default scene
        default_scene_dir = os.path.join(default_dir, default_scene)
        masks_dict = load_masks_json(default_scene_dir)
        mask_options = ["(none)"] + sorted(masks_dict.keys()) if masks_dict else ["(none)"]

        return io.Schema(
            node_id=prefixed_node_id("SceneSelect"),
            display_name="Scene Select",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("scenes_dir", default=default_dir, tooltip="Directory containing scene subdirectories"),
                io.Combo.Input('selected_scene', options=default_options, default=default_scene, tooltip="Select a scene name"),
                io.Combo.Input(id="depth_image_type", display_name="depth_image_type", options=depth_options, default="depth", tooltip="Type of depth image to use from the scene"),
                io.Combo.Input(id="pose_image_type", display_name="pose_image_type", options=pose_options, default="open", tooltip="Type of pose image to use from the scene"),
                io.Combo.Input(id="mask_name", display_name="mask_name", options=mask_options, default="(none)", tooltip="Name of the mask to use from the scene (select '(none)' to skip mask selection)"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Whether to use the background variant of the mask (True=with background, False=no background)"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images with PromptCollection"),
                DictType.Output(id="prompt_dict", display_name="prompt_dict", tooltip="Dictionary of composed prompts from the scene"),
                DictType.Output(id="comp_dict", display_name="comp_dict", tooltip="Dictionary of composition names to their fully processed prompt values"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory of the selected scene"),
                io.String.Output(id="input_img_glob", display_name="input_img_glob", tooltip="Input image glob pattern for the scene"),
                io.String.Output(id="output_image_prefix", display_name="output_image_prefix", tooltip="Output image prefix for the scene"),
                io.String.Output(id="output_video_prefix", display_name="output_video_prefix", tooltip="Output video prefix for the scene"),
                io.Image.Output(id="base_image", display_name="base_image", tooltip="Base IMAGE from the scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth IMAGE from the scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask IMAGE from the scene"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from the selected mask image"),
                io.Image.Output(id='canny_image', display_name='canny_image', tooltip='Canny IMAGE from the scene'),
                io.Image.Output(id='pose_image', display_name='pose_image', tooltip='Pose IMAGE from the scene'),
                io.Image.Output(id='upscale_image', display_name='upscale_image', tooltip='Upscaled base IMAGE from the scene'),
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Full multi-target LoRA stack (LORA_STACK_DATA). Feed into LoraStackApply."),
                io.Custom("WANVIDLORA").Output(id="loras_high_out", display_name="loras_high", tooltip="WanVideoWrapper WANVIDLORA list for the Wan2.2-Wrapper-High (first-pass) model"),
                io.Custom("WANVIDLORA").Output(id="loras_low_out",  display_name="loras_low",  tooltip="WanVideoWrapper WANVIDLORA list for the Wan2.2-Wrapper-Low (second-pass) model"),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo 
            ],
            is_output_node=True,
        )
    
    @classmethod
    def fingerprint_inputs(
        cls,
        scenes_dir: str = "",
        selected_scene: str = "",
        depth_image_type: str = "",
        pose_image_type: str = "",
        mask_name: str = "",
        mask_background: bool = True,
        **_,
    ):
        """Invalidate cache whenever any file inside the selected scene folder changes."""
        resolved_dir = scenes_dir if scenes_dir else default_scenes_dir()
        if not resolved_dir or not selected_scene:
            return None
        scene_path = Path(resolved_dir) / selected_scene
        scene_hash, dir_count, file_count = _directory_fingerprint(scene_path)
        return (
            str(scene_path),
            scene_hash,
            dir_count,
            file_count,
            depth_image_type,
            pose_image_type,
            mask_name,
            mask_background,
        )

    @classmethod
    def execute(
        cls,
        scenes_dir="",
        selected_scene="default_scene",
        depth_image_type="depth",
        pose_image_type="open",
        mask_name="",
        mask_background=True,
    ) -> io.NodeOutput:
        className = cls.__name__
        input_types = cls.INPUT_TYPES()
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        logger.debug("%s: unique_id='%s'; extra_pnginfo='%s'", className, unique_id, extra_pnginfo)
        logger.debug("%s: selected_scene input='%s'", className, selected_scene)

        if not scenes_dir:
            scenes_dir = default_scenes_dir()

        if not scenes_dir or not selected_scene:
            logger.warning("%s: scenes_dir or selected_scene is empty", className)
            return io.NodeOutput(None)
        
        scene_dir = os.path.join(scenes_dir, selected_scene)
        logger.debug("%s: using scene_dir='%s' for selected_scene='%s'", className, scene_dir, selected_scene)

        if not os.path.isdir(scene_dir):
            logger.error("%s: scene_dir '%s' is not a valid directory", className, scene_dir)
            return io.NodeOutput(None)
        
        # Load prompts.json for PromptCollection
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_collection = PromptCollection.load_from_json(prompt_json_path)
        
        # Load pose.json
        pose_json_path = os.path.join(scene_dir, "pose.json")
        pose_json = load_json_file(pose_json_path)
        if not pose_json:
            pose_json = "[]"
        else:
            pose_json = json.dumps(pose_json)

        # Load LoRA stack (new format, with automatic legacy migration)
        lora_stack = load_lora_stack(scene_dir)
        if lora_stack is None:
            logger.warning("%s: no lora_stack.json or loras.json found in '%s'", className, scene_dir)

        # Derive WANVIDLORA outputs for backward-compatible workflow wiring
        wan_high_entries = _lora_entries_for_target(lora_stack or [], "Wan2.2-Wrapper-High")
        wan_low_entries  = _lora_entries_for_target(lora_stack or [], "Wan2.2-Wrapper-Low")
        loras_high, _ = _lora_build_wanvid(wan_high_entries, None, False, True)
        loras_low,  _ = _lora_build_wanvid(wan_low_entries,  None, False, True)

        # Load selected/normalized assets (and mask preview/output separation)
        selected_depth_attr = default_depth_options.get(depth_image_type, "depth_image")
        selected_pose_attr = default_pose_options.get(pose_image_type, "pose_open_image")
        
        # Load assets with mask_name (supports both new and legacy mask systems)
        # Treat "(none)" as empty string to skip mask loading
        actual_mask_name = "" if mask_name == "(none)" else mask_name
        logger.info(
            "%s: Loading assets from scene_dir='%s'; mask_name='%s', mask_background=%s",
            className,
            scene_dir,
            actual_mask_name,
            mask_background,
        )
        assets = SceneInfo.load_preview_assets(
            scene_dir,
            depth_attr=selected_depth_attr,
            pose_attr=selected_pose_attr,
            mask_name=actual_mask_name,
            mask_background=mask_background,
            include_upscale=True,
            include_canny=True,
        )

        # Also load full images for SceneInfo completeness (depth variants, masks, canny)
        depth_images_full = SceneInfo.load_depth_images(scene_dir)
        pose_images_full = SceneInfo.load_pose_images(scene_dir)
        
        # Load new mask system if available
        masks_dict = load_masks_json(scene_dir)
        mask_images_dict = {}
        if masks_dict:
            # Load images for all masks in masks.json
            mask_names = list(masks_dict.keys())
            mask_images_full, _ = SceneInfo.load_mask_images(scene_dir, mask_names=mask_names)
            mask_images_dict = mask_images_full
            logger.info(f"%s: Loaded {len(masks_dict)} masks from new system", className)
        else:
            # Fall back to legacy mask loading
            mask_images_full, _ = SceneInfo.load_mask_images(scene_dir)
            logger.info(f"%s: Loaded masks from legacy system", className)
        
        # Ensure canny present even if missing on disk
        canny_image = pose_images_full.get("canny_image")

        base_image = assets["base_image"]
        selected_depth_image = assets["depth_image"]
        pose_image = assets["pose_image"]
        mask_image = assets["mask_image"]
        mask = assets["mask"]
        preview_mask = assets["mask_preview"]
        H, W = assets["H"], assets["W"]
        resolution = assets["resolution"]

        # Normalize canny to match preview size
        canny_image = normalize_image_tensor(canny_image, H, W)

        logger.debug(
            "%s: depth_image shape: %s",
            className,
            selected_depth_image.shape if selected_depth_image is not None else "None",
        )
        logger.debug(
            "%s: upscale_image shape: %s",
            className,
            base_image.shape if base_image is not None else "None",
        )

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None

        ui_data = {
            "images": preview_image.as_dict().get("images", []) if preview_image else None,
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
        }

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=selected_scene,
            pose_json=pose_json,
            resolution=resolution,
            prompts=prompt_collection,
            masks=masks_dict,
            mask_images=mask_images_dict,
            base_image=pose_images_full.get("base_image"),  # Load base image from scene
            depth_image=depth_images_full.get("depth_image"),
            depth_any_image=depth_images_full.get("depth_any_image"),
            depth_midas_image=depth_images_full.get("depth_midas_image"),
            depth_zoe_image=depth_images_full.get("depth_zoe_image"),
            depth_zoe_any_image=depth_images_full.get("depth_zoe_any_image"),
            pose_dense_image=pose_images_full.get("pose_dense_image"),
            pose_dw_image=pose_images_full.get("pose_dw_image"),
            pose_edit_image=pose_images_full.get("pose_edit_image"),
            pose_face_image=pose_images_full.get("pose_face_image"),
            pose_open_image=pose_images_full.get("pose_open_image"),
            canny_image=canny_image,
            upscale_image=pose_images_full.get("upscale_image"),
            girl_mask_bkgd_image=mask_images_full.get('girl') if not masks_dict else None,
            male_mask_bkgd_image=mask_images_full.get('male') if not masks_dict else None,
            combined_mask_bkgd_image=mask_images_full.get('combined') if not masks_dict else None,
            girl_mask_no_bkgd_image=mask_images_full.get('girl_no_bg') if not masks_dict else None,
            male_mask_no_bkgd_image=mask_images_full.get('male_no_bg') if not masks_dict else None,
            combined_mask_no_bkgd_image=mask_images_full.get('combined_no_bg') if not masks_dict else None,
            lora_stack=lora_stack,
        )

        # Build prompt_dict and comp_dict from PromptCollection
        prompt_dict = {}  # Individual prompts processed
        comp_dict = {}    # Compositions processed
        
        if prompt_collection:
            libber_manager = LibberStateManager.instance()
            
            # Process individual prompts
            for key, metadata in prompt_collection.prompts.items():
                value = metadata.value
                
                # Apply libber substitution if needed
                if metadata.processing_type == "libber" and metadata.libber_name and libber_manager:
                    libber = libber_manager.ensure_libber(metadata.libber_name)
                    if libber:
                        value = libber.substitute(value)
                
                prompt_dict[key] = value
            
            # Process compositions
            if prompt_collection.compositions:
                comp_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
        
        return io.NodeOutput(
            scene_info,
            prompt_dict,
            comp_dict,
            selected_scene,
            scene_dir,
            scene_info.input_img_glob(),
            scene_info.input_img_dir(),
            os.path.join(scene_info.output_dir(), "vid_"),
            base_image,
            selected_depth_image,
            mask_image,
            mask,
            canny_image,
            pose_image,
            base_image,
            lora_stack,   # LORA_STACK_DATA — full multi-target stack
            loras_high,   # WANVIDLORA — Wan2.2-Wrapper-High entries (backward compat)
            loras_low,    # WANVIDLORA — Wan2.2-Wrapper-Low entries (backward compat)
            ui=ui_data
        )

def load_loras(loras_json_path: str) -> tuple[list, list] | tuple[None, None]:
    """DEPRECATED: use load_lora_stack(scene_dir) instead.
    Retained so any external callers don't break immediately."""
    entries = _migrate_loras_json_to_stack(loras_json_path) if os.path.isfile(loras_json_path) else []
    high = [e for e in entries if e.get("model_target") == "Wan2.2-Wrapper-High"]
    low  = [e for e in entries if e.get("model_target") == "Wan2.2-Wrapper-Low"]
    # Re-shape back to old path/strength/blocks/layer_filter/low_mem_load/merge_loras dicts
    def _to_wanvid(entry: dict) -> dict:
        try:
            path = folder_paths.get_full_path_or_raise("loras", entry["lora"])
        except Exception:
            path = folder_paths.get_full_path("loras", entry["lora"]) or entry["lora"]
        return {
            "path": path,
            "strength": entry["strength_model"],
            "name": os.path.splitext(entry["lora"])[0],
            "blocks": entry.get("blocks", {}),
            "layer_filter": entry.get("layer_filter", ""),
            "low_mem_load": entry.get("low_mem_load", False),
            "merge_loras": entry.get("merge_loras", False),
        }
    loras_high = [_to_wanvid(e) for e in high]
    loras_low  = [_to_wanvid(e) for e in low]
    return (loras_high or None, loras_low or None)


def save_loras(loras_high: list, loras_low: list, loras_json_path: str):
    """DEPRECATED: use save_lora_stack(scene_dir, lora_stack) instead.
    Retained so SceneWanVideoLoraMultiSave continues to function unchanged."""
    high, low = [], []
    for lora in (loras_high or []):
        high.append({
            "lora_name":    os.path.basename(lora["path"]),
            "strength":     lora["strength"],
            "blocks":       lora.get("blocks", {}),
            "layer_filter": lora.get("layer_filter", ""),
            "low_mem_load": lora.get("low_mem_load", False),
            "merge_loras":  lora.get("merge_loras", False),
        })
    for lora in (loras_low or []):
        low.append({
            "lora_name":    os.path.basename(lora["path"]),
            "strength":     lora["strength"],
            "blocks":       lora.get("blocks", {}),
            "layer_filter": lora.get("layer_filter", ""),
            "low_mem_load": lora.get("low_mem_load", False),
            "merge_loras":  lora.get("merge_loras", False),
        })
    save_json_file(loras_json_path, {"high": high, "low": low})

class SceneWanVideoLoraMultiSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneWanVideoLoraMultiSave"),
            display_name="SceneWanVideoLoraMultiSave",
            category="🧊 frost-byte/Scene",
            description=cleandoc("""
                                 Saves the name, weights, layer filter and blocks of multiple LoRA models using the
                                 output from WanVideoWrapper's WanVideoLoraSelectMulti node to the directory for the given scene.
            """),
            inputs=[
                io.Custom("SCENE_INFO").Input(id="info_in", display_name="scene_info", tooltip="SceneInfo, which provides the path to save the LoRA information to"),
                io.Custom("WANVIDLORA").Input(id="loras_high", display_name="lora", tooltip="WanVideoSelectMulti output with multiple High LoRA entries"),
                io.Custom("WANVIDLORA").Input(id="loras_low", display_name="lora", tooltip="WanVideoSelectMulti output with multiple Low LoRA entries"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="info_out", display_name="scene_info", tooltip="Save operation information"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        info_in,
        loras_high=None,
        loras_low=None,
    ) -> io.NodeOutput:
        className = cls.__name__

        if info_in is None or loras_high is None or loras_low is None:
            return io.NodeOutput(None)

        scene_dir = info_in.scene_dir
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("%s: Invalid scene_dir '%s' in SceneInfo", className, scene_dir)
            return io.NodeOutput(None)

        if not loras_high is None:
            logger.info("%s: Saving %d High LoRA entries to scene_dir '%s'", className, len(loras_high), scene_dir)
            loras_high_path = os.path.join(scene_dir, "loras_high.json")
        else:
            loras_high = []
        if not loras_low is None:
            logger.info("%s: Saving %d Low LoRA entries to scene_dir '%s'", className, len(loras_low), scene_dir)
            loras_low_path = os.path.join(scene_dir, "loras_low.json")
        else:
            loras_low = []

        loras_path = os.path.join(scene_dir, "loras.json")
        save_loras(loras_high, loras_low, loras_path)
        logger.info("%s: Saved LoRA preset to: %s", className, loras_path)

        return io.NodeOutput(info_in)


def save_lora_stack(scene_dir: str, lora_stack: list) -> None:
    """Persist a lora_stack list to lora_stack.json in the given scene directory."""
    lora_stack_path = os.path.join(scene_dir, "lora_stack.json")
    save_json_file(lora_stack_path, lora_stack)


class SceneLoraStackSave(io.ComfyNode):
    """Save a LoRA stack (from LoraStackCollect) to the given scene directory.

    Writes lora_stack.json — the new multi-target format that replaces the
    Wan-only loras.json.  Connect the lora_stack_data output of LoraStackCollect
    here, or supply a raw stack_json STRING if you prefer text storage.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneLoraStackSave"),
            display_name="Scene LoRA Stack Save",
            category="🧊 frost-byte/Scene",
            description=(
                "Save a LoRA stack to the scene directory as lora_stack.json. "
                "Connect LoraStackCollect's lora_stack_data output here. "
                "SceneSelect will load this file automatically."
            ),
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info",
                    display_name="scene_info",
                    tooltip="SceneInfo providing the scene directory path.",
                ),
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from LoraStackCollect. Takes priority over stack_json.",
                ),
                io.String.Input(
                    "stack_json",
                    display_name="Stack JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip="Raw JSON string (stack_json output of LoraStackCollect). Used when Stack Data is not connected.",
                ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(
                    id="scene_info_out",
                    display_name="scene_info",
                    tooltip="Pass-through scene_info.",
                ),
                io.Int.Output("entry_count", display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info,
        lora_stack_data: Optional[list] = None,
        stack_json: str = "[]",
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneLoraStackSave: scene_info is None")
            return io.NodeOutput(None, 0)

        scene_dir = scene_info.scene_dir
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("SceneLoraStackSave: invalid scene_dir '%s'", scene_dir)
            return io.NodeOutput(scene_info, 0)

        stack = lora_stack_data if lora_stack_data is not None else _lora_json_to_stack(stack_json)

        save_lora_stack(scene_dir, stack)
        logger.info("SceneLoraStackSave: saved %d entries to '%s/lora_stack.json'", len(stack), scene_dir)

        return io.NodeOutput(scene_info, len(stack))


class SceneCreate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneCreate"),
            display_name="Scene Create",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="scenes_dir", display_name="scenes_dir", tooltip="Root Directory where all scene subdirectories are saved"),
                io.String.Input(id="scene_name", display_name="scene_name", tooltip="Name of the pose"),
                io.Int.Input(id="resolution", display_name="resolution", tooltip="Resolution for the pose, depth and other images", default=512),
                io.Combo.Input(
                    id="upscale_method",
                    display_name="upscale_method",
                    options=["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],
                    default="nearest-exact",
                    tooltip="Method to use for upscaling the base image"
                ),
                io.Float.Input(
                    id="upscale_factor", 
                    display_name="upscale_factor", 
                    tooltip="Factor to upscale the base image by", 
                    default=1.0, min=0.1, max=10.0, step=0.1
                ),
                io.Combo.Input(
                    id="densepose_model", 
                    display_name="densepose_model",
                    options=["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], 
                    default="densepose_r50_fpn_dl.torchscript", 
                    tooltip="DensePose model to use"
                ),
                io.Combo.Input(
                    id="densepose_cmap",
                    display_name="densepose_cmap",
                    options=["viridis", "parula"],
                    default="viridis",
                    tooltip="Color map to use for DensePose visualization"
                ),
                io.Combo.Input(
                    id="depth_any_ckpt",
                    display_name="depth_any_ckpt",
                    options=["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"],
                    default="depth_anything_vitl14.pth",
                    tooltip="Checkpoint for Depth Any model"
                ),
                io.Combo.Input(
                    id="depth_any_v2_ckpt",
                    display_name="depth_any_v2_ckpt",
                    options=["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                    default="depth_anything_v2_vitl.pth",
                    tooltip="Checkpoint for Depth Any v2 model"
                ),
                io.Float.Input(
                    id="midas_a",
                    display_name="midas_a",
                    tooltip="MiDas parameter A for depth scaling",
                    default=np.pi * 2.0, min=0.0, max=np.pi * 5.0, step=0.1
                ),
                io.Float.Input(
                    id="midas_bg_thresh",
                    display_name="midas_bg_thresh",
                    tooltip="MiDas parameter Bg threshold for depth scaling",
                    default=0.1, min=0.1, max=np.pi * 5.0, step=0.1
                ),
                io.Combo.Input(
                    id="zoe_environment",
                    display_name="zoe_environment",
                    options=["indoor", "outdoor"],
                    default="indoor",
                    tooltip="Environment setting for Zoe Any model"
                ),
                io.Int.Input(
                    id="canny_low_threshold",
                    display_name="canny_low_threshold",
                    tooltip="Canny edge detector low threshold",
                    default=100, min=0, max=255, step=1
                ),
                io.Int.Input(
                    id="canny_high_threshold",
                    display_name="canny_high_threshold",
                    tooltip="Canny edge detector high threshold",
                    default=200, min=0, max=255, step=1
                ),
                io.Boolean.Input(
                    id="generate_nlf_pose",
                    display_name="generate_nlf_pose",
                    default=False,
                    tooltip="Generate NLF (Neural Lifting Framework) pose from base image"
                ),
                io.Combo.Input(
                    id="nlf_model",
                    display_name="nlf_model",
                    options=["nlf_l_multi_0.3.2.torchscript", "nlf_l_multi_0.2.2.torchscript"],
                    default="nlf_l_multi_0.3.2.torchscript",
                    tooltip="NLF model to use (will auto-download if not present)"
                ),
                io.Boolean.Input(
                    id="nlf_draw_face",
                    display_name="nlf_draw_face",
                    default=True,
                    tooltip="Draw face keypoints in NLF pose rendering"
                ),
                io.Boolean.Input(
                    id="nlf_draw_hands",
                    display_name="nlf_draw_hands",
                    default=True,
                    tooltip="Draw hand keypoints in NLF pose rendering"
                ),
                io.Combo.Input(
                    id="nlf_render_device",
                    display_name="nlf_render_device",
                    options=["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"],
                    default="gpu",
                    tooltip="Device to use for NLF pose rendering (Taichi backend)"
                ),
                io.Boolean.Input(
                    id="nlf_scale_hands",
                    display_name="nlf_scale_hands",
                    default=True,
                    tooltip="Scale hand keypoints in NLF pose rendering"
                ),
                io.Combo.Input(
                    id="nlf_render_backend",
                    display_name="nlf_render_backend",
                    options=["torch", "taichi"],
                    default="torch",
                    tooltip="Rendering backend for NLF poses (torch=more compatible, taichi=faster if installed)"
                ),
                io.Image.Input(id="base_image", display_name="base_image", tooltip="Base image for the scene"),
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="Optional LoRA stack to assign to this scene (from LoraStackCollect)."),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene Information"),
                io.String.Output(id="scene_name_out", display_name="scene_name", tooltip="Name of the created scene")
            ],
        )

    @classmethod
    async def execute(
        cls,
        scenes_dir="",
        scene_name="default_scene",
        resolution=512,
        upscale_method="nearest-exact",
        upscale_factor=1.0,
        densepose_model="densepose_r50_fpn_dl.torchscript",
        densepose_cmap="viridis",
        depth_any_ckpt="depth_anything_vitl14.pth",
        depth_any_v2_ckpt="depth_anything_v2_vitl.pth",
        midas_a=np.pi * 2.0,
        midas_bg_thresh=0.1,
        zoe_environment="indoor",
        canny_low_threshold=100,
        canny_high_threshold=200,
        generate_nlf_pose=False,
        nlf_model="nlf_l_multi_0.3.2.torchscript",
        nlf_draw_face=True,
        nlf_draw_hands=True,
        nlf_render_device="gpu",
        nlf_scale_hands=True,
        nlf_render_backend="torch",
        base_image=None,
        lora_stack_data=None,
    ) -> io.NodeOutput:
        if base_image is None:
            logger.error("SceneCreate: base_image is None")
            return io.NodeOutput(None)
        
        if not scenes_dir:
            scenes_dir = default_scenes_dir()
        
        if not scene_name:
            scene_name = "default_scene"

        scene_dir = os.path.join(scenes_dir, scene_name)

        # Create upscale_image from base_image
        upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
        logger.info(
            "SceneCreate: Created upscale_image from base_image - shape %s",
            upscale_image.shape if torch.is_tensor(upscale_image) else "N/A",
        )

        # DensePose
        dense_pose_image = dense_pose(upscale_image, densepose_model, densepose_cmap, resolution)

        # Depth Anything
        depth_any_image = depth_anything(upscale_image, ckpt=depth_any_ckpt, resolution=resolution)
        
        # Depth Anything V2
        depth_image = depth_anything_v2(upscale_image, ckpt=depth_any_v2_ckpt, resolution=resolution)

        # MiDas
        midas_depth_image = midas(upscale_image, a=midas_a, bg_thresh=midas_bg_thresh)

        # Zoe
        depth_zoe_image = zoe(upscale_image, resolution=resolution)
        
        # Zoe Any
        depth_zoe_any_image = zoe_any(upscale_image, environment=zoe_environment, resolution=resolution)

        if type(depth_any_image) is not torch.Tensor:
            H = 512
            W = 512
        elif not depth_any_image is None and type(depth_any_image) is torch.Tensor:
            H, W = depth_any_image.shape[1], depth_any_image.shape[2]
        pose_dw_image, pose_json = estimate_dwpose(upscale_image, detect_face=False, resolution=resolution)
        pose_face_image = openpose(upscale_image, include_hand=False, include_face=True, include_body=False, resolution=resolution)
        normalized_upscale_image = image_resize_ess(upscale_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)
        base_image_normalized = image_resize_ess(base_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)

        pose_open_image = openpose(normalized_upscale_image, include_face=False, resolution=resolution)
        canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)

        # NLF Pose Generation
        pose_nlf_image = None
        if generate_nlf_pose:
            try:
                from ...utils.nlf_pose import load_nlf_model, predict_nlf_pose, render_nlf_pose, nlfpred_to_pose_keypoint
                
                logger.info("SceneCreate: Generating NLF pose...")
                
                # Load NLF model
                nlf_model = load_nlf_model(nlf_model, warmup=True)
                
                # Predict poses from upscale image
                nlf_pred, bboxes = predict_nlf_pose(nlf_model, upscale_image)
                logger.info(f"SceneCreate: NLF detected {len(bboxes)} person(s)")
                
                # Render NLF poses
                pose_nlf_image, nlf_mask = render_nlf_pose(
                    nlf_pred, W, H,
                    draw_face=nlf_draw_face,
                    draw_hands=nlf_draw_hands,
                    render_device=nlf_render_device,
                    scale_hands=nlf_scale_hands,
                    render_backend=nlf_render_backend
                )
                
                # Convert to POSE_KEYPOINT format for potential editing
                pose_keypoints = nlfpred_to_pose_keypoint(nlf_pred, W, H)
                
                # Update pose_json with NLF-derived keypoints
                # This allows users to edit the pose with OpenposeEditorNode
                if pose_keypoints:
                    pose_json = pose_keypoints  # Keep as list for now, will be converted to JSON string below
                    logger.info("SceneCreate: Updated pose.json with NLF-derived keypoints")
                
                logger.info("SceneCreate: NLF pose generation complete")
                
            except ImportError as e:
                logger.error(f"SceneCreate: Failed to import NLF utilities: {e}")
                logger.error("Make sure utils/nlf_pose.py exists and dependencies are installed")
            except Exception as e:
                logger.error(f"SceneCreate: NLF pose generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"
        
        # Convert pose_json (list of dicts) to JSON string for storage
        if isinstance(pose_json, list):
            pose_dwpose_json = json.dumps({'people': pose_json})
        else:
            pose_dwpose_json = json.dumps(pose_json)

        # Create empty PromptCollection for new scenes
        # Users will add prompts via ScenePromptManager
        prompt_collection = PromptCollection()

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=scene_name,
            resolution=resolution,
            prompts=prompt_collection,
            base_image=base_image_normalized,
            upscale_image=upscale_image,
            depth_image=depth_image,
            depth_any_image=depth_any_image,
            depth_midas_image=midas_depth_image,
            depth_zoe_image=depth_zoe_image,
            depth_zoe_any_image=depth_zoe_any_image,
            pose_dense_image=dense_pose_image,
            pose_dw_image=pose_dw_image,
            pose_edit_image=pose_dw_image,
            pose_dwpose_json=pose_dwpose_json,
            pose_open_image=pose_open_image,
            pose_face_image=pose_face_image,
            pose_nlf_image=pose_nlf_image,
            pose_json=pose_dwpose_json,  # Store as JSON string
            canny_image=canny_image,
            lora_stack=lora_stack_data,
        )
        
        # Save all scene data using the helper method
        scene_info.save_all(scene_dir)
        logger.info("SceneCreate: Saved all scene data to '%s'", scene_dir)
        
        return io.NodeOutput(
            scene_info,
            scene_name,
        )

class SceneUpdate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneUpdate"),
            display_name="Scene Update",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info_in", display_name="scene_info", tooltip="Scene Information" ),
                io.Image.Input(id="base_image", display_name="base_image", tooltip="New base image (if update_base=True)", optional=True),
                io.Boolean.Input(id="update_base", display_name="update_base", tooltip="If true, will replace base_image and regenerate upscale_image and all derived images", default=False),
                io.Boolean.Input(id="update_zoe", display_name="update_zoe", tooltip="If true, will update the Zoe depth images in the scene_info", default=False),
                io.Boolean.Input(id="update_depth", display_name="update_depth", tooltip="If true, will update the Depth Anything images in the scene_info", default=False),
                io.Boolean.Input(id="update_densepose", display_name="update_densepose", tooltip="If true, will update the DensePose image in the scene_info", default=False),
                io.Boolean.Input(id="update_openpose", display_name="update_openpose", tooltip="If true, will update the OpenPose image in the scene_info", default=False),
                io.Boolean.Input(id="update_midas", display_name="update_midas", tooltip="If true, will update the MiDas depth image in the scene_info", default=False),
                io.Boolean.Input(id="update_canny", display_name="update_canny", tooltip="If true, will update the Canny edge image in the scene_info", default=False),
                io.Boolean.Input(id="update_upscale", display_name="update_upscale", tooltip="If true, will update the Upscale image in the scene_info", default=False),
                io.Boolean.Input(id="update_pose_json", display_name="update_pose_json", tooltip="If true, will update the pose_json in the scene_info", default=False),
                io.Boolean.Input(id="update_facepose", display_name="update_facepose", tooltip="If true, will update the Face Pose image in the scene_info", default=False),
                io.Boolean.Input(id="update_editpose", display_name="update_editpose", tooltip="If true, will update the Edit Pose image in the scene_info", default=False),
                io.Boolean.Input(id="update_dwpose", display_name="update_dwpose", tooltip="If true, will update the DensePose image in the scene_info", default=False),
                io.Boolean.Input(id="update_nlf_pose", display_name="update_nlf_pose", tooltip="If true, will update the NLF pose in the scene_info", default=False),
                io.Image.Input(id="pose_image", display_name="pose_image", tooltip="Custom pose image to use (if provided with pose_keypoint, skips generation)", optional=True),
                io.Custom("POSE_KEYPOINT").Input(id="pose_keypoint", display_name="pose_keypoint", tooltip="Custom pose keypoints (if provided with pose_image, skips generation)", optional=True),
                io.Boolean.Input(id="nlf_draw_face", display_name="nlf_draw_face", default=True, tooltip="Draw face keypoints in NLF pose rendering"),
                io.Boolean.Input(id="nlf_draw_hands", display_name="nlf_draw_hands", default=True, tooltip="Draw hand keypoints in NLF pose rendering"),
                io.Combo.Input(
                    id="nlf_render_device",
                    display_name="nlf_render_device",
                    options=["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"],
                    default="gpu",
                    tooltip="Device to use for NLF pose rendering (Taichi backend)"
                ),
                io.Boolean.Input(id="nlf_scale_hands", display_name="nlf_scale_hands", default=True, tooltip="Scale hand keypoints in NLF pose rendering"),
                io.Combo.Input(
                    id="nlf_render_backend",
                    display_name="nlf_render_backend",
                    options=["torch", "taichi"],
                    default="torch",
                    tooltip="Rendering backend for NLF poses (torch=more compatible, taichi=faster if installed)"
                ),
                io.Combo.Input(
                    id="nlf_model",
                    display_name="nlf_model",
                    options=["nlf_l_multi_0.3.2.torchscript", "nlf_l_multi_0.2.2.torchscript"],
                    default="nlf_l_multi_0.3.2.torchscript",
                    tooltip="NLF model to use (will auto-download if not present)"
                ),
                io.Boolean.Input(id="update_loras", display_name="update_loras", tooltip="If true, replaces the scene's LoRA stack with the provided lora_stack_data", default=False),
                io.String.Input(id="pose_json", display_name="pose_json", tooltip="JSON string for the pose keypoints"),
                io.Int.Input(id="resolution", display_name="resolution", tooltip="Resolution for the pose, depth and other images", default=512),
                io.Combo.Input(
                    id="upscale_method",
                    display_name="upscale_method",
                    options=["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],
                    default="nearest-exact",
                    tooltip="Method to use for upscaling the base image"
                ),
                io.Float.Input(
                    id="upscale_factor", 
                    display_name="upscale_factor", 
                    tooltip="Factor to upscale the base image by", 
                    default=1.0, min=0.1, max=10.0, step=0.1
                ),
                io.Combo.Input(
                    id="densepose_model", 
                    display_name="densepose_model",
                    options=["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], 
                    default="densepose_r50_fpn_dl.torchscript", 
                    tooltip="DensePose model to use"
                ),
                io.Combo.Input(
                    id="densepose_cmap",
                    display_name="densepose_cmap",
                    options=["viridis", "parula"],
                    default="viridis",
                    tooltip="Color map to use for DensePose visualization"
                ),
                io.Combo.Input(
                    id="depth_any_ckpt",
                    display_name="depth_any_ckpt",
                    options=["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"],
                    default="depth_anything_vitl14.pth",
                    tooltip="Checkpoint for Depth Any model"
                ),
                io.Combo.Input(
                    id="depth_any_v2_ckpt",
                    display_name="depth_any_v2_ckpt",
                    options=["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                    default="depth_anything_v2_vitl.pth",
                    tooltip="Checkpoint for Depth Any v2 model"
                ),
                io.Float.Input(
                    id="midas_a",
                    display_name="midas_a",
                    tooltip="MiDas parameter A for depth scaling",
                    default=np.pi * 2.0, min=0.0, max=np.pi * 5.0, step=0.1
                ),
                io.Float.Input(
                    id="midas_bg_thresh",
                    display_name="midas_bg_thresh",
                    tooltip="MiDas parameter Bg threshold for depth scaling",
                    default=0.1, min=0.1, max=np.pi * 5.0, step=0.1
                ),
                io.Combo.Input(
                    id="zoe_environment",
                    display_name="zoe_environment",
                    options=["indoor", "outdoor"],
                    default="indoor",
                    tooltip="Environment setting for Zoe Any model"
                ),
                io.Int.Input(
                    id="canny_low_threshold",
                    display_name="canny_low_threshold",
                    tooltip="Canny edge detector low threshold",
                    default=100, min=0, max=255, step=1
                ),
                io.Int.Input(
                    id="canny_high_threshold",
                    display_name="canny_high_threshold",
                    tooltip="Canny edge detector high threshold",
                    default=200, min=0, max=255, step=1
                ),
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="New LoRA stack to replace the scene's existing stack (requires update_loras=True)."),
            ],
            hidden=[
                io.Hidden.unique_id,
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info_out", display_name="scene_info", tooltip="Updated Scene Information"),
            ],
        )
        
    @classmethod
    async def execute(
        cls,
        scene_info_in=None,
        base_image=None,
        update_base=False,
        update_zoe=False,
        update_depth=False,
        update_densepose=False,
        update_openpose=False,
        update_midas=False,
        update_canny=False,
        update_upscale=False,
        update_pose_json=False,
        update_facepose=False,
        update_editpose=False,
        update_dwpose=False,
        update_nlf_pose=False,
        pose_image=None,
        pose_keypoint=None,
        nlf_draw_face=True,
        nlf_draw_hands=True,
        nlf_render_device="gpu",
        nlf_scale_hands=True,
        nlf_render_backend="torch",
        nlf_model="nlf_l_multi_0.3.2.torchscript",
        update_loras=False,
        pose_json="[]",
        resolution=512,
        upscale_method="nearest-exact",
        upscale_factor=1.0,
        densepose_model="densepose_r50_fpn_dl.torchscript",
        densepose_cmap="viridis",
        depth_any_ckpt="depth_anything_vitl14.pth",
        depth_any_v2_ckpt="depth_anything_v2_vitl.pth",
        midas_a=np.pi * 2.0,
        midas_bg_thresh=0.1,
        zoe_environment="indoor",
        canny_low_threshold=100,
        canny_high_threshold=200,
        lora_stack_data=None,
    ):
        # Get node ID for status updates
        node_id = cls.hidden.unique_id
        
        send_status_update(node_id, "Starting scene update...")
        logger.info("="*60)
        logger.info("SceneUpdate: Node execution started")
        logger.info("SceneUpdate: update_nlf_pose=%s, update_base=%s, update_upscale=%s", 
                   update_nlf_pose, update_base, update_upscale)
        
        if scene_info_in is None:
            logger.error("SceneUpdate: scene_info is None")
            return io.NodeOutput(None)

        scene_info_out = scene_info_in
        
        # Handle base_image update first (triggers full regeneration)
        if update_base:
            if base_image is None:
                logger.warning("SceneUpdate: update_base=True but base_image is None - attempting to use existing base_image")
                logger.debug("SceneUpdate: scene_info_in.base_image is None: %s", scene_info_in.base_image is None)
                logger.debug("SceneUpdate: scene_info_in.scene_dir: %s", scene_info_in.scene_dir)
                base_image = scene_info_in.base_image
                if base_image is None:
                    logger.error("SceneUpdate: Cannot update - both input base_image and scene_info.base_image are None")
            else:
                logger.info("SceneUpdate: Replacing base_image with new input")
                logger.debug("SceneUpdate: New base_image shape: %s", base_image.shape if hasattr(base_image, 'shape') else 'N/A')
                scene_info_out.base_image = base_image
            
            # Regenerate upscale_image from base_image
            if base_image is not None:
                logger.info(
                    "SceneUpdate: Regenerating upscale_image from base_image using factor %s",
                    upscale_factor,
                )
                upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
                scene_info_out.upscale_image = upscale_image
                # Force regeneration of all derived images
                update_upscale = True
            else:
                logger.error("SceneUpdate: base_image is None, cannot regenerate upscale_image")
                upscale_image = scene_info_in.upscale_image
        else:
            # Start with existing upscale_image from scene
            upscale_image = scene_info_in.upscale_image
            
            # If user wants to refresh upscale_image, prefer regenerating from base_image.
            # This ensures updates to base.png are reflected even when update_base is False.
            if update_upscale:
                source_base = base_image if base_image is not None else scene_info_in.base_image

                if source_base is not None:
                    logger.info(
                        "SceneUpdate: Regenerating upscale_image from base_image using factor %s and method %s",
                        upscale_factor,
                        upscale_method,
                    )
                    if base_image is not None:
                        logger.info("SceneUpdate: Applying provided base_image while update_base=False")
                        scene_info_out.base_image = base_image
                    base_image = source_base
                    upscale_image, = ImageScaleBy().upscale(source_base, upscale_method=upscale_method, scale_by=upscale_factor)
                    scene_info_out.upscale_image = upscale_image
                elif upscale_image is not None:
                    logger.warning(
                        "SceneUpdate: base_image unavailable; falling back to rescaling existing upscale_image by factor %s using %s",
                        upscale_factor,
                        upscale_method,
                    )
                    upscale_image, = ImageScaleBy().upscale(upscale_image, upscale_method=upscale_method, scale_by=upscale_factor)
                    scene_info_out.upscale_image = upscale_image
                else:
                    logger.error("SceneUpdate: Cannot update upscale_image - no base_image or existing upscale_image available")
        
        if upscale_image is None:
            logger.error("SceneUpdate: upscale_image is None, cannot regenerate derived images")
            return io.NodeOutput(scene_info_out)
        
        # upscale_image is now the source for regenerating all other images

        if update_facepose:
            pose_face_image = openpose(upscale_image, include_hand=False, include_face=True, include_body=False, resolution=resolution)
            scene_info_out.pose_face_image = pose_face_image
            scene_info_out.pose_json = pose_json
        if update_densepose:
            send_status_update(node_id, f"Generating DensePose ({densepose_model})...")
            scene_info_out.pose_dense_image = dense_pose(upscale_image, densepose_model, densepose_cmap, resolution)

        if update_depth:
            send_status_update(node_id, f"Generating depth maps ({depth_any_v2_ckpt})...")
            # Depth Anything
            scene_info_out.depth_any_image = depth_anything(upscale_image, ckpt=depth_any_ckpt, resolution=resolution)
            scene_info_out.depth_image = depth_anything_v2(upscale_image, ckpt=depth_any_v2_ckpt, resolution=resolution)

        # MiDas
        if update_midas:
            send_status_update(node_id, "Generating Midas depth map...")
            scene_info_out.depth_midas_image = midas(upscale_image, a=midas_a, bg_thresh=midas_bg_thresh)

        # Zoe
        if update_zoe:
            depth_zoe_image = zoe(upscale_image, resolution=resolution)
            scene_info_out.depth_zoe_image = depth_zoe_image
            depth_zoe_any_image = zoe_any(upscale_image, environment=zoe_environment, resolution=resolution)
            scene_info_out.depth_zoe_any_image = depth_zoe_any_image

        # Pose Json
        if update_pose_json:
            scene_info_out.pose_json = pose_json
        
        if update_canny:
            send_status_update(node_id, "Generating Canny edges...")
            canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)
            scene_info_out.canny_image = canny_image

        if update_dwpose:
            send_status_update(node_id, "Generating DWPose...")
            pose_dw_image, pose_json = estimate_dwpose(upscale_image, detect_face=False, resolution=resolution)
            scene_info_out.pose_dw_image = pose_dw_image
            #scene_info_out.pose_json = pose_json

        # Update NLF pose
        if update_nlf_pose:
            send_status_update(node_id, "Processing NLF pose...")
            logger.info("SceneUpdate: NLF pose update requested")
            logger.info("SceneUpdate: pose_image provided: %s", pose_image is not None)
            logger.info("SceneUpdate: pose_keypoint provided: %s", pose_keypoint is not None)
            logger.info("SceneUpdate: base_image available: %s", base_image is not None)
            logger.info("SceneUpdate: upscale_image available: %s", 'upscale_image' in locals())
            
            from ...utils.nlf_pose import (
                load_nlf_model,
                predict_nlf_pose,
                render_nlf_pose,
                nlfpred_to_pose_keypoint
            )
            
            # Check if custom pose image and keypoint were provided (edited workflow)
            if pose_image is not None and pose_keypoint is not None:
                logger.info("SceneUpdate: Using provided pose_image and pose_keypoint for NLF pose")
                logger.info("SceneUpdate: pose_image shape: %s", pose_image.shape if hasattr(pose_image, 'shape') else 'unknown')
                logger.info("SceneUpdate: pose_keypoint type: %s, length: %s", 
                           type(pose_keypoint).__name__, 
                           len(pose_keypoint) if isinstance(pose_keypoint, list) else 'N/A')
                scene_info_out.pose_nlf_image = pose_image
                # Update pose.json with custom keypoints for editing support
                # pose_keypoint is a list of dicts in OpenPose format
                if isinstance(pose_keypoint, list):
                    # Store as JSON string
                    import json
                    scene_info_out.pose_json = json.dumps({
                        'people': pose_keypoint
                    })
                    logger.info("SceneUpdate: Updated pose.json with custom pose keypoints")
            else:
                # Regenerate NLF pose from base_image (or upscale_image if base not available)
                logger.debug("SceneUpdate: Checking source images for NLF generation")
                logger.debug("SceneUpdate: base_image is None: %s", base_image is None)
                logger.debug("SceneUpdate: upscale_image defined: %s", 'upscale_image' in locals())
                logger.debug("SceneUpdate: scene_info_in.base_image is None: %s", scene_info_in.base_image is None)
                logger.debug("SceneUpdate: scene_info_in.upscale_image is None: %s", scene_info_in.upscale_image is None)
                
                # Try to get source image from multiple sources
                if base_image is None:
                    base_image = scene_info_in.base_image
                    logger.debug("SceneUpdate: Using scene_info_in.base_image as source")
                
                if 'upscale_image' not in locals():
                    upscale_image = scene_info_in.upscale_image
                    logger.debug("SceneUpdate: Using scene_info_in.upscale_image as fallback")
                
                source_image = base_image if base_image is not None else upscale_image
                logger.info("SceneUpdate: Source image selection - using base_image: %s", base_image is not None)
                
                if source_image is None:
                    logger.error("SceneUpdate: Cannot generate NLF pose - no source image available")
                    logger.error("SceneUpdate: base_image is None: %s", base_image is None)
                    logger.error("SceneUpdate: upscale_image is None: %s", upscale_image is None if 'upscale_image' in locals() else 'not defined')
                else:
                    logger.info("SceneUpdate: Regenerating NLF pose from source image")
                    logger.info("SceneUpdate: Source image shape: %s", source_image.shape)
                    logger.info("SceneUpdate: NLF model: %s", nlf_model)
                    logger.info("SceneUpdate: NLF config - draw_face=%s, draw_hands=%s, render_device=%s, render_backend=%s",
                               nlf_draw_face, nlf_draw_hands, nlf_render_device, nlf_render_backend)
                    try:
                        # Load NLF model
                        send_status_update(node_id, f"Loading NLF model ({nlf_model})...")
                        logger.info("SceneUpdate: Loading NLF model...")
                        nlf_model_obj = load_nlf_model(nlf_model, warmup=True)
                        send_status_update(node_id, "Running NLF prediction...")
                        logger.info("SceneUpdate: NLF model loaded successfully")
                        
                        # Generate NLF predictions (returns tuple of dict and list)
                        logger.info("SceneUpdate: Generating NLF predictions...")
                        nlf_pred_dict, nlf_pred_list = predict_nlf_pose(nlf_model_obj, source_image, per_batch=1)
                        logger.info("SceneUpdate: NLF predictions generated - dict keys: %s, list length: %s",
                                   list(nlf_pred_dict.keys()) if nlf_pred_dict else 'None',
                                   len(nlf_pred_list) if nlf_pred_list else 'None')
                        
                        # Debug NLF prediction structure
                        if nlf_pred_dict and 'joints3d_nonparam' in nlf_pred_dict:
                            joints = nlf_pred_dict['joints3d_nonparam']
                            logger.debug("SceneUpdate: joints3d_nonparam type: %s", type(joints))
                            logger.debug("SceneUpdate: joints3d_nonparam length: %s", len(joints) if hasattr(joints, '__len__') else 'N/A')
                            if isinstance(joints, list) and len(joints) > 0:
                                logger.debug("SceneUpdate: joints[0] type: %s", type(joints[0]))
                                logger.debug("SceneUpdate: joints[0] length: %s", len(joints[0]) if hasattr(joints[0], '__len__') else 'N/A')
                                if len(joints[0]) > 0:
                                    logger.debug("SceneUpdate: joints[0][0] shape: %s", joints[0][0].shape if hasattr(joints[0][0], 'shape') else 'N/A')
                        
                        # Check if any persons were detected
                        num_detections = len(nlf_pred_list) if nlf_pred_list else 0
                        logger.info("SceneUpdate: NLF detected %d person(s)", num_detections)
                        
                        if num_detections == 0:
                            logger.warning("SceneUpdate: No persons detected by NLF - pose_nlf will be black")
                            send_status_update(node_id, "⚠️ NLF: No persons detected in image")
                        else:
                            send_status_update(node_id, f"✓ NLF: Detected {num_detections} person(s)")
                        
                        # Get dimensions from source image
                        h, w = source_image.shape[1], source_image.shape[2]
                        logger.info("SceneUpdate: Target dimensions - width=%s, height=%s", w, h)
                        
                        # Render NLF pose (returns tuple of image tensor and mask tensor)
                        send_status_update(node_id, f"Rendering NLF pose ({nlf_render_backend})...")
                        logger.info("SceneUpdate: Rendering NLF pose...")
                        pose_nlf_image, nlf_mask = render_nlf_pose(
                            nlf_pred_dict,
                            w, h,
                            draw_face=nlf_draw_face,
                            draw_hands=nlf_draw_hands,
                            render_device=nlf_render_device,
                            scale_hands=nlf_scale_hands,
                            render_backend=nlf_render_backend
                        )
                        
                        scene_info_out.pose_nlf_image = pose_nlf_image
                        logger.info("SceneUpdate: NLF pose rendered successfully")
                        logger.info("SceneUpdate: pose_nlf_image shape: %s", pose_nlf_image.shape)
                        logger.info("SceneUpdate: Generated NLF pose image (%sx%s)", w, h)
                        logger.info("SceneUpdate: pose_nlf_image tensor id: %s", id(pose_nlf_image))
                        
                        # Verify it's not being aliased to other pose fields
                        if scene_info_out.pose_dense_image is not None:
                            logger.warning("SceneUpdate: pose_dense_image is also set (tensor id: %s)", id(scene_info_out.pose_dense_image))
                        if scene_info_out.pose_dw_image is not None:
                            logger.warning("SceneUpdate: pose_dw_image is also set (tensor id: %s)", id(scene_info_out.pose_dw_image))
                        if scene_info_out.pose_edit_image is not None:
                            logger.warning("SceneUpdate: pose_edit_image is also set (tensor id: %s)", id(scene_info_out.pose_edit_image))
                        
                        # Convert NLF prediction to POSE_KEYPOINT format for editing
                        # nlfpred_to_pose_keypoint expects just the dict, not the tuple
                        logger.info("SceneUpdate: Converting NLF predictions to POSE_KEYPOINT format...")
                        
                        # Only convert if we have detections
                        if num_detections > 0:
                            try:
                                pose_keypoint_list = nlfpred_to_pose_keypoint(nlf_pred_dict, w, h)
                                logger.info("SceneUpdate: Converted to %s pose keypoint entries", len(pose_keypoint_list) if pose_keypoint_list else 0)
                            except Exception as e:
                                logger.error("SceneUpdate: Failed to convert NLF to POSE_KEYPOINT: %s", str(e))
                                logger.debug("SceneUpdate: Conversion error details:", exc_info=True)
                                pose_keypoint_list = []
                        else:
                            logger.info("SceneUpdate: Skipping POSE_KEYPOINT conversion - no detections")
                            pose_keypoint_list = []
                        
                        # Store as JSON string
                        import json
                        scene_info_out.pose_json = json.dumps({
                            'people': pose_keypoint_list
                        })
                        logger.info("SceneUpdate: Updated pose.json with NLF-derived keypoints for editing")
                        logger.info("SceneUpdate: NLF pose update completed successfully")
                        
                    except ImportError as e:
                        error_msg = (
                            "NLF pose generation failed: ComfyUI-SCAIL-Pose not found. "
                            "Install via ComfyUI-Manager or from https://github.com/kijai/ComfyUI-SCAIL-Pose"
                        )
                        logger.error("SceneUpdate: %s", error_msg)
                        logger.debug("SceneUpdate: Import error details: %s", str(e))
                        send_status_update(node_id, f"⚠ {error_msg}")
                    except Exception as e:
                        logger.error("SceneUpdate: Failed to generate NLF pose: %s", str(e), exc_info=True)
                        send_status_update(node_id, f"⚠ NLF pose generation failed: {str(e)}")

        # Determine target dimensions from reference images
        # Use upscale_image dimensions as the reference since it's the source
        ref_h, ref_w = upscale_image.shape[1], upscale_image.shape[2]
        logger.debug("SceneUpdate: Using upscale_image dimensions as reference: %sx%s", ref_w, ref_h)
        
        # Normalize midas image to match reference dimensions (typically half size)
        if scene_info_out.depth_midas_image is not None and torch.is_tensor(scene_info_out.depth_midas_image):
            midas_h, midas_w = scene_info_out.depth_midas_image.shape[1], scene_info_out.depth_midas_image.shape[2]
            if midas_h != ref_h or midas_w != ref_w:
                logger.debug(
                    "SceneUpdate: Normalizing midas image from %sx%s to %sx%s",
                    midas_w,
                    midas_h,
                    ref_w,
                    ref_h,
                )
                scene_info_out.depth_midas_image = image_resize_ess(
                    scene_info_out.depth_midas_image, ref_w, ref_h,
                    method="keep proportion", interpolation="nearest", multiple_of=16
                )
        
        # Normalize all depth images to reference dimensions
        for depth_attr in ['depth_image', 'depth_any_image', 'depth_zoe_image', 'depth_zoe_any_image']:
            img = getattr(scene_info_out, depth_attr, None)
            if img is not None and torch.is_tensor(img):
                img_h, img_w = img.shape[1], img.shape[2]
                if img_h != ref_h or img_w != ref_w:
                    logger.debug(
                        "SceneUpdate: Normalizing %s from %sx%s to %sx%s",
                        depth_attr,
                        img_w,
                        img_h,
                        ref_w,
                        ref_h,
                    )
                    setattr(scene_info_out, depth_attr, image_resize_ess(
                        img, ref_w, ref_h,
                        method="keep proportion", interpolation="nearest", multiple_of=16
                    ))
        
        # Normalize all pose images to reference dimensions
        for pose_attr in ['pose_dense_image', 'pose_dw_image', 'pose_edit_image', 'pose_face_image', 'pose_open_image', 'pose_nlf_image']:
            img = getattr(scene_info_out, pose_attr, None)
            if img is not None and torch.is_tensor(img):
                img_h, img_w = img.shape[1], img.shape[2]
                if img_h != ref_h or img_w != ref_w:
                    logger.debug(
                        "SceneUpdate: Normalizing %s from %sx%s to %sx%s",
                        pose_attr,
                        img_w,
                        img_h,
                        ref_w,
                        ref_h,
                    )
                    setattr(scene_info_out, pose_attr, image_resize_ess(
                        img, ref_w, ref_h,
                        method="keep proportion", interpolation="nearest", multiple_of=16
                    ))

        normalized_upscale_image = image_resize_ess(upscale_image, ref_w, ref_h, method="keep proportion", interpolation="nearest", multiple_of=16)

        if update_openpose or update_editpose:
            pose_open_image = openpose(normalized_upscale_image, include_face=False, resolution=resolution)
            scene_info_out.pose_open_image = pose_open_image

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"

        # Resize existing masks if dimensions changed
        if scene_info_out.masks and scene_info_out.mask_images:
            # Get new dimensions from depth, pose, or base image
            new_H, new_W = None, None
            if scene_info_out.depth_image is not None:
                new_H, new_W = scene_info_out.depth_image.shape[1], scene_info_out.depth_image.shape[2]
            elif scene_info_out.pose_dense_image is not None:
                new_H, new_W = scene_info_out.pose_dense_image.shape[1], scene_info_out.pose_dense_image.shape[2]
            elif scene_info_out.base_image is not None:
                new_H, new_W = scene_info_out.base_image.shape[1], scene_info_out.base_image.shape[2]
            
            if new_H and new_W:
                for mask_name, mask_image in scene_info_out.mask_images.items():
                    old_H, old_W = mask_image.shape[1], mask_image.shape[2]
                    if old_H != new_H or old_W != new_W:
                        logger.info(f"SceneUpdate: Resizing mask '{mask_name}' from {old_W}x{old_H} to {new_W}x{new_H}")
                        scene_info_out.mask_images[mask_name] = normalize_image_tensor(mask_image, new_H, new_W)

        # Update LoRA stack
        if update_loras and lora_stack_data is not None:
            scene_info_out.lora_stack = lora_stack_data

        if update_loras:
            scene_info_out.save_loras()
            logger.info(
                "SceneUpdate: Saved LoRA stack (%d entries) to: %s/lora_stack.json",
                len(scene_info_out.lora_stack or []),
                scene_info_in.scene_dir,
            )

        # Log which pose images are set for debugging
        pose_status = {
            "pose_dense": scene_info_out.pose_dense_image is not None,
            "pose_dw": scene_info_out.pose_dw_image is not None,
            "pose_edit": scene_info_out.pose_edit_image is not None,
            "pose_face": scene_info_out.pose_face_image is not None,
            "pose_open": scene_info_out.pose_open_image is not None,
            "pose_nlf": scene_info_out.pose_nlf_image is not None,
        }
        logger.info("SceneUpdate: Final pose image status: %s", pose_status)

        # Save all updated scene data to disk
        send_status_update(node_id, "Saving scene data...")
        scene_info_out.save_all(scene_info_out.scene_dir)
        logger.info("SceneUpdate: Saved all scene data to '%s'", scene_info_out.scene_dir)

        send_status_update(node_id, "✓ Scene update completed")
        logger.info("SceneUpdate: Node execution completed successfully")
        logger.info("="*60)
        return io.NodeOutput(
            scene_info_out,
        )

class SceneView(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneView"),
            display_name="Scene View",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information" ),
                io.Combo.Input(
                    id="depth_type", options=list(default_depth_options.keys())
                ),
                io.Combo.Input(
                    id="pose_type", options=list(default_pose_options.keys())
                ),
                io.Combo.Input(
                    id="mask_name", 
                    display_name="mask_name", 
                    options=["none"], 
                    default="none", 
                    tooltip="Name of mask to preview (dynamically updated from scene)"
                ),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected Depth Image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected Pose Image"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Selected Mask Image"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from selected mask image"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory of the selected scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    async def execute(
        cls,
        scene_info=Optional[SceneInfo],
        depth_type="depth",
        pose_type="dense",
        mask_name="none",
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneView: scene_info is None")
            return io.NodeOutput(None, None, None, None, None, None)
        
        if not isinstance(scene_info, SceneInfo):
            logger.error("SceneView: scene_info is not of type SceneInfo")
            return io.NodeOutput(None, None, None, None, None, None)

        # Auto-select first mask if mask_name is "none" and masks are available
        if (mask_name == "none" or not mask_name) and scene_info.masks:
            available_masks = sorted(scene_info.masks.keys())
            if available_masks:
                mask_name = available_masks[0]
                logger.info(f"SceneView: Auto-selected first available mask: {mask_name}")

        # Determine include_mask_bg from mask definition
        include_mask_bg = True
        if mask_name and mask_name != "none" and scene_info.masks and mask_name in scene_info.masks:
            mask_def = scene_info.masks[mask_name]
            include_mask_bg = mask_def.has_background
        
        assets = scene_info.load_preview_assets(
            scene_info.scene_dir,
            depth_attr=depth_type,
            pose_attr=pose_type,
            mask_name=mask_name if mask_name != "none" else "",
            mask_background=include_mask_bg,
            include_canny=True,
        )

        mask_image = assets["mask_image"]
        mask = assets["mask"]
        depth_image = assets["depth_image"]
        pose_image = assets["pose_image"]
        girl_pos = getattr(scene_info, "girl_pos", "")
        male_pos = getattr(scene_info, "male_pos", "")
        scene_name = getattr(scene_info, "scene_name", "")
        scene_dir = getattr(scene_info, "scene_dir", "")

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        # Show scene info instead of deprecated prompts
        info_text = f"Scene: {scene_name}\nDepth: {depth_type}\nPose: {pose_type}"
        if mask_name and mask_name != "none":
            info_text += f"\nMask: {mask_name}"
        text_ui = ui.PreviewText(value=info_text)
 
        ui_data = {
            "text": text_ui.as_dict().get("text", ''),
            "images": preview_image.as_dict().get("images", []) if preview_image else [],
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
        }

        return io.NodeOutput(
            depth_image,
            pose_image,
            mask_image,
            mask,
            scene_name,
            scene_dir,
            ui=ui_data
        )

class SceneMaskDefinition(io.ComfyNode):
    """
    Define and generate masks for scenes using SAM3 segmentation.
    Outputs an updated scene_info with the mask definition added.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneMaskDefinition"),
            display_name="Scene Mask Definition",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info", 
                    display_name="scene_info", 
                    tooltip="Scene Information (mask will be generated from scene base_image)"
                ),
                io.String.Input(
                    id="mask_name", 
                    display_name="mask_name", 
                    default="mask_1",
                    tooltip="Name of the mask (must be unique within the scene)"
                ),
                io.Combo.Input(
                    id="mask_type",
                    display_name="mask_type",
                    options=["transparent", "color"],
                    default="transparent",
                    tooltip="Type of mask: transparent (alpha-based) or color (colored regions)"
                ),
                io.Boolean.Input(
                    id="has_background",
                    display_name="has_background",
                    default=True,
                    tooltip="Whether the mask includes background (True) or masks it out (False)"
                ),
                io.String.Input(
                    id="mask_color",
                    display_name="mask_color",
                    default="255,255,255",
                    tooltip="RGB color for the mask as r,g,b (e.g., '255,0,0' for red). Only used if mask_type is 'color'"
                ),
                # SAM3 Segmentation parameters
                io.String.Input(
                    id="sam3_prompt",
                    display_name="sam3_prompt",
                    default="",
                    tooltip="Text prompt describing what to segment (e.g., 'person', 'face', 'clothing')"
                ),
                io.Float.Input(
                    id="confidence_threshold",
                    display_name="confidence_threshold",
                    default=0.30,
                    min=0.00,
                    max=1.00,
                    step=0.01,
                    tooltip="Confidence threshold for segmentation (lower = more permissive)"
                ),
                io.Combo.Input(
                    id="background_mode",
                    display_name="background_mode",
                    options=["Alpha", "Color"],
                    default="Alpha",
                    tooltip="Background mode: Alpha (transparent) or Color (solid color)"
                ),
                io.String.Input(
                    id="background_color",
                    display_name="background_color",
                    default="#222222",
                    tooltip="Background color as hex code (only used if background_mode is 'Color')"
                ),
                io.Int.Input(
                    id="max_segments",
                    display_name="max_segments",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Maximum number of segments to keep (0 = no limit)"
                ),
                io.Int.Input(
                    id="segment_pick",
                    display_name="segment_pick",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Pick a specific segment by index (0 = use all segments)"
                ),
                io.Int.Input(
                    id="mask_blur",
                    display_name="mask_blur",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Amount of blur to apply to mask edges"
                ),
                io.Int.Input(
                    id="mask_offset",
                    display_name="mask_offset",
                    default=0,
                    min=-64,
                    max=64,
                    step=1,
                    tooltip="Offset to grow (+) or shrink (-) the mask"
                ),
                io.Combo.Input(
                    id="device",
                    display_name="device",
                    options=["Auto", "CPU", "GPU"],
                    default="Auto",
                    tooltip="Device to run segmentation on"
                ),
                io.Boolean.Input(
                    id="invert_output",
                    display_name="invert_output",
                    default=False,
                    tooltip="Invert the mask output"
                ),
                io.Boolean.Input(
                    id="unload_model",
                    display_name="unload_model",
                    default=False,
                    tooltip="Unload the model after processing to free memory"
                ),
                # MaskProcessor parameters
                io.Int.Input(
                    id="min_hole_size",
                    display_name="min_hole_size",
                    default=10,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Minimum hole size (in pixels) to fill. Holes smaller than this will be filled."
                ),
                io.Int.Input(
                    id="grow_amount",
                    display_name="grow_amount",
                    default=5,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Amount to grow (dilate) the mask borders in pixels"
                ),
                io.Int.Input(
                    id="smooth_iterations",
                    display_name="smooth_iterations",
                    default=0,
                    min=0,
                    max=10,
                    step=1,
                    tooltip="Number of morphological smoothing iterations (can shrink mask)"
                ),
                io.Boolean.Input(
                    id="enable_region_smooth",
                    display_name="enable_region_smooth",
                    default=True,
                    tooltip="Enable region smoothing (Gaussian filter with thresholding - maintains mask size)"
                ),
                io.Int.Input(
                    id="region_smooth_sigma",
                    display_name="region_smooth_sigma",
                    default=128,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Sigma for region smoothing (only used if enabled)"
                ),
                io.Float.Input(
                    id="blur_radius",
                    display_name="blur_radius",
                    default=5.0,
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    tooltip="Gaussian blur radius (sigma value) for edge softening"
                ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(
                    id="scene_info_out", 
                    display_name="scene_info", 
                    tooltip="Updated Scene Information with mask definition added"
                ),
                io.Image.Output(
                    id="image_out", 
                    display_name="IMAGE", 
                    tooltip="Segmented image with background applied"
                ),
                io.Mask.Output(
                    id="mask_out", 
                    display_name="MASK", 
                    tooltip="Binary mask of segmented region"
                ),
                io.Image.Output(
                    id="mask_image_out", 
                    display_name="MASK_IMAGE", 
                    tooltip="Grayscale visualization of the mask"
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        scene_info=None,
        mask_name="mask_1",
        mask_type="transparent",
        has_background=True,
        mask_color="255,255,255",
        sam3_prompt="",
        confidence_threshold=0.30,
        background_mode="Alpha",
        background_color="#222222",
        max_segments=0,
        segment_pick=0,
        mask_blur=0,
        mask_offset=0,
        device="Auto",
        invert_output=False,
        unload_model=False,
        min_hole_size=10,
        grow_amount=5,
        smooth_iterations=0,
        enable_region_smooth=True,
        region_smooth_sigma=128,
        blur_radius=5.0,
    ) -> io.NodeOutput:
        """Execute mask definition and segmentation"""
        
        if scene_info is None:
            logger.error("SceneMaskDefinition: scene_info is required")
            return io.NodeOutput(None, None, None, None)
        
        if not isinstance(scene_info, SceneInfo):
            logger.error("SceneMaskDefinition: scene_info is not of type SceneInfo")
            return io.NodeOutput(None, None, None, None)
        
        # Get base_image from scene_info
        original_image = scene_info.base_image
        if original_image is None:
            logger.error("SceneMaskDefinition: scene_info.base_image is None - base_image is required for segmentation")
            return io.NodeOutput(None, None, None, None)
        
        # Clone and convert base_image to RGB for SAM3
        # SAM3 requires RGB format (no alpha channel)
        logger.info(f"SceneMaskDefinition: Converting base_image to RGB for SAM3 (original shape: {original_image.shape})")
        
        # Image tensor is [B, H, W, C] in ComfyUI format
        if original_image.shape[-1] == 4:
            # Has alpha channel - extract RGB only
            image_rgb = original_image[..., :3].clone()
            logger.info(f"SceneMaskDefinition: Extracted RGB channels from RGBA image")
        elif original_image.shape[-1] == 3:
            # Already RGB
            image_rgb = original_image.clone()
            logger.info(f"SceneMaskDefinition: Image already in RGB format")
        else:
            logger.error(f"SceneMaskDefinition: Unexpected image channel count: {original_image.shape[-1]}")
            return io.NodeOutput(None, None, None, None)
        
        # Import SAM3 segmentation from ComfyUI-RMBG
        try:
            from ...utils.util import import_virtual_package, add_custom_node_to_syspath
            candidates = ["ComfyUI-RMBG", "comfyui-rmbg"]
            rmbg_path = add_custom_node_to_syspath(candidates)
            
            if rmbg_path is None:
                logger.error(
                    "SceneMaskDefinition: ComfyUI-RMBG not found. "
                    "Install from ComfyUI-Manager or https://github.com/AInsert/ComfyUI-RMBG"
                )
                return io.NodeOutput(None, None, None, None)
            
            import_virtual_package("rmbg", rmbg_path)
            from rmbg.py.AILab_SAM3Segment import SAM3Segment # type: ignore
            
            logger.info("SceneMaskDefinition: Successfully imported SAM3Segment")
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to import SAM3Segment: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Run SAM3 segmentation on RGB image
        try:
            logger.info(f"SceneMaskDefinition: Running SAM3 segmentation for mask '{mask_name}'")
            logger.info(f"SceneMaskDefinition: Prompt: '{sam3_prompt}', Confidence: {confidence_threshold}")
            
            sam3_node = SAM3Segment()
            result = sam3_node.segment(
                image=image_rgb,
                prompt=sam3_prompt or "object",
                output_mode="Merged",
                confidence_threshold=confidence_threshold,
                max_segments=max_segments,
                segment_pick=segment_pick,
                mask_blur=mask_blur,
                mask_offset=mask_offset,
                device=device,
                invert_output=invert_output,
                unload_model=unload_model,
                background=background_mode,
                background_color=background_color,
            )
            
            segmented_image, sam3_mask, mask_image = result
            logger.info(f"SceneMaskDefinition: SAM3 segmentation complete")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Segmentation failed: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Process mask using MaskProcessor with original image (can be RGBA)
        try:
            from ...utils.images import (
                mask_remove_holes, 
                mask_grow, 
                mask_gaussian_blur, 
                mask_smooth, 
                create_mask_overlay_image, 
                smooth_masks_region_was
            )
            
            logger.info(f"SceneMaskDefinition: Processing mask with MaskProcessor")
            
            # Handle batch: select first mask
            if sam3_mask.dim() == 3:  # [B, H, W]
                mask_single = sam3_mask[0]  # [H, W]
            elif sam3_mask.dim() == 2:  # [H, W]
                mask_single = sam3_mask
            else:
                logger.error(f"SceneMaskDefinition: Unexpected mask shape: {sam3_mask.shape}")
                return io.NodeOutput(None, None, None, None)
            
            # Apply MaskProcessor operations in sequence
            processed_mask = mask_single
            operations = []
            
            # 1. Remove holes
            if min_hole_size > 0:
                processed_mask = mask_remove_holes(processed_mask, min_hole_size=min_hole_size)
                operations.append(f"remove_holes(min_size={min_hole_size})")
            
            # 2. Grow (dilate)
            if grow_amount > 0:
                processed_mask = mask_grow(processed_mask, grow_amount=grow_amount)
                operations.append(f"grow(amount={grow_amount})")
            
            # 3. Smooth (morphological cleanup)
            if smooth_iterations > 0:
                processed_mask = mask_smooth(processed_mask, smooth_iterations=smooth_iterations)
                operations.append(f"smooth(iterations={smooth_iterations})")
            
            # 4. Region smooth (Gaussian with thresholding - WAS method)
            if enable_region_smooth:
                # Need to add batch dim temporarily for smooth_masks_region_was
                if processed_mask.dim() == 2:
                    processed_mask_batch = processed_mask.unsqueeze(0)
                else:
                    processed_mask_batch = processed_mask
                processed_mask_batch = smooth_masks_region_was(processed_mask_batch, sigma=region_smooth_sigma)
                # Extract single mask again
                processed_mask = processed_mask_batch[0] if processed_mask_batch.dim() == 3 else processed_mask_batch
                operations.append(f"region_smooth(sigma={region_smooth_sigma})")
            
            # 5. Gaussian blur (LAST - creates soft edges for blending)
            if blur_radius > 0.0:
                processed_mask = mask_gaussian_blur(processed_mask, blur_radius=blur_radius)
                operations.append(f"gaussian_blur(radius={blur_radius})")
            
            # Ensure output is 3D [B, H, W] for compatibility
            if processed_mask.dim() == 2:
                processed_mask = processed_mask.unsqueeze(0)
            
            operations_str = " -> ".join(operations) if operations else "no operations"
            logger.info(f"SceneMaskDefinition: Applied MaskProcessor operations: {operations_str}")
            
            # Create overlay image using original_image (can be RGBA)
            overlay_image = create_mask_overlay_image(processed_mask, original_image)
            logger.info(f"SceneMaskDefinition: Created overlay_image with shape {overlay_image.shape}")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: MaskProcessor failed: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Parse mask color
        parsed_color: Optional[RGB] = None
        try:
            if mask_type == "color":
                color_parts = [int(x.strip()) for x in mask_color.split(',')]
                if len(color_parts) != 3:
                    raise ValueError("Color must be in format 'r,g,b'")
                parsed_color = (color_parts[0], color_parts[1], color_parts[2])
            else:
                parsed_color = None
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Invalid mask_color format '{mask_color}': {e}")
            return io.NodeOutput(None, None, None, None)
        
        # Create MaskDefinition
        try:
            mask_def = MaskDefinition(
                name=mask_name,
                type=MaskType(mask_type),
                has_background=has_background,
                color=parsed_color
            )
            mask_def.validate()
            logger.info(f"SceneMaskDefinition: Created mask definition for '{mask_name}'")
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to create mask definition: {e}")
            return io.NodeOutput(None, None, None, None)
        
        # Save mask and overlay image to scene directory
        try:
            import numpy as np
            from PIL import Image
            
            scene_dir = scene_info.scene_dir
            if not scene_dir or not os.path.exists(scene_dir):
                logger.error(f"SceneMaskDefinition: Invalid scene_dir: {scene_dir}")
                return io.NodeOutput(None, None, None, None)
            
            # Build save path based on mask definition filename
            mask_filename = mask_def.get_filename()
            save_path = os.path.join(scene_dir, mask_filename)
            
            logger.info(f"SceneMaskDefinition: Saving mask to {save_path}")
            
            # Use PathSaveImageRGBA logic to save the overlay image with mask as alpha
            # Extract first image and mask from batch
            img_tensor = overlay_image[0].cpu().numpy()
            mask_tensor = processed_mask[0].cpu()
            
            # Convert to alpha channel (0-255)
            # Note: invert_mask=False, so we don't invert
            alpha_np = (255.0 * (1.0 - mask_tensor.numpy())).astype(np.uint8)
            
            # Convert to uint8 format for PIL
            img_np = (img_tensor * 255).astype(np.uint8)
            
            # Create PIL image - handle both RGB and RGBA input
            if img_np.shape[-1] == 4:
                # Already RGBA - extract RGB only
                pil_img = Image.fromarray(img_np[..., :3])
            elif img_np.shape[-1] == 3:
                # RGB
                pil_img = Image.fromarray(img_np)
            else:
                logger.error(f"SceneMaskDefinition: Unexpected image channel count: {img_np.shape[-1]}")
                return io.NodeOutput(None, None, None, None)
            
            # Create alpha channel image
            alpha_img = Image.fromarray(alpha_np, mode='L')
            
            # Convert to RGBA and add alpha channel
            pil_img_rgba = pil_img.convert("RGBA")
            pil_img_rgba.putalpha(alpha_img)
            
            # Save the image (format=png, quality=95, create_dirs=False per requirements)
            pil_img_rgba.save(save_path, format="PNG")
            
            logger.info(f"SceneMaskDefinition: Successfully saved mask image to {save_path}")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to save mask image: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Add mask definition to scene_info
        scene_info_out = copy.deepcopy(scene_info)
        if scene_info_out.masks is None:
            scene_info_out.masks = {}
        if scene_info_out.mask_images is None:
            scene_info_out.mask_images = {}
        
        # Add or update the mask definition
        scene_info_out.masks[mask_name] = mask_def
        scene_info_out.mask_images[mask_name] = overlay_image
        
        logger.info(f"SceneMaskDefinition: Added mask '{mask_name}' to scene_info")
        logger.info(f"SceneMaskDefinition: Scene now has {len(scene_info_out.masks)} mask(s)")
        
        return io.NodeOutput(
            scene_info_out,
            overlay_image,
            processed_mask,
            overlay_image  # Return overlay_image as MASK_IMAGE output
        )
 
class SceneOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneOutput"),
            display_name="Scene Output",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information"),
            ],
            outputs=[
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory where the scene is saved"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the pose"),
                io.String.Output(id="girl_pos", display_name="girl_pos", tooltip="Girl Positive Prompt"),
                io.String.Output(id="male_pos", display_name="male_pos", tooltip="Male Positive Prompt"),
                io.String.Output(id="four_image_prompt", display_name="four_image_prompt", tooltip="Four Image Prompt"),
                io.String.Output(id="wan_prompt", display_name="wan_prompt", tooltip="Wan High Positive Prompt"),
                io.String.Output(id="wan_low_prompt", display_name="wan_low_prompt", tooltip="Wan Low Positive Prompt"),
                io.String.Output(id="pose_json", display_name="pose_json", tooltip="Pose JSON data"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth Image"),
                io.Image.Output(id="depth_any_image", display_name="depth_any_image", tooltip="Depth Any Image"),
                io.Image.Output(id="depth_midas_image", display_name="depth_midas_image", tooltip="Depth Midas Image"),
                io.Image.Output(id="depth_zoe_image", display_name="depth_zoe_image", tooltip="Depth Zoe Image"),
                io.Image.Output(id="depth_zoe_any_image", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image"),
                io.Image.Output(id="pose_dense_image", display_name="pose_dense_image", tooltip="Pose Dense Image"),
                io.Image.Output(id="pose_dw_image", display_name="pose_dw_image", tooltip="Pose DW Image"),
                io.Image.Output(id="pose_edit_image", display_name="pose_edit_image", tooltip="Pose Edit Image"),
                io.Image.Output(id="pose_face_image", display_name="pose_face_image", tooltip="Pose Face Image"),
                io.Image.Output(id="pose_open_image", display_name="pose_open_image", tooltip="Pose Open Image"),
                io.Image.Output(id="canny_image", display_name="canny_image", tooltip="Canny Image"),
                io.Image.Output(id="upscale_image", display_name="upscale_image", tooltip="Upscale Image"),
                io.Image.Output(id="girl_mask_image", display_name="girl_mask_image", tooltip="Girl Mask Image, with background"),
                io.Image.Output(id="male_mask_image", display_name="male_mask_image", tooltip="Male Mask Image, with background"),
                io.Image.Output(id="combined_mask_image", display_name="combined_mask_image", tooltip="Combined Mask Image, with background"),
                io.Image.Output(id="girl_mask_nobg_image", display_name="girl_mask_nobg_image", tooltip="Girl Mask Image, no background"),
                io.Image.Output(id="male_mask_nobg_image", display_name="male_mask_nobg_image", tooltip="Male Mask Image, no background"),
                io.Image.Output(id="combined_mask_nobg_image", display_name="combined_mask_nobg_image", tooltip="Combined Mask Image, no background"),
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Multi-target LoRA stack for this scene. Feed into LoraStackApply."),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info=None,
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneOutput: scene_info is None")
            return io.NodeOutput((
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ))
        
        logger.info(
            "SceneOutput: scene_dir='%s', scene_name='%s', girl_pos='%s', male_pos='%s', wan_prompt='%s', wan_low_prompt='%s', depth_image shape=%s",
            scene_info.scene_dir,
            scene_info.scene_name,
            scene_info.girl_pos[:32],
            scene_info.male_pos[:32],
            scene_info.wan_prompt[:32],
            scene_info.wan_low_prompt[:32],
            scene_info.depth_image.shape if scene_info.depth_image is not None else "None",
        )
        return io.NodeOutput(
            scene_info.scene_dir,
            scene_info.scene_name,
            scene_info.girl_pos,
            scene_info.male_pos,
            scene_info.four_image_prompt,
            scene_info.wan_prompt,
            scene_info.wan_low_prompt,
            scene_info.pose_json,
            scene_info.depth_image,
            scene_info.depth_any_image,
            scene_info.depth_midas_image,
            scene_info.depth_zoe_image,
            scene_info.depth_zoe_any_image,
            scene_info.pose_dense_image,
            scene_info.pose_dw_image,
            scene_info.pose_edit_image,
            scene_info.pose_face_image,
            scene_info.pose_open_image,
            scene_info.canny_image,
            scene_info.upscale_image,
            scene_info.girl_mask_bkgd_image,
            scene_info.male_mask_bkgd_image,
            scene_info.combined_mask_bkgd_image,
            scene_info.girl_mask_no_bkgd_image,
            scene_info.male_mask_no_bkgd_image,
            scene_info.combined_mask_no_bkgd_image,
            scene_info.lora_stack,
        )

class SceneSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneSave"),
            display_name="Scene Save",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Info Input"),
                io.String.Input(id="scene_dir", display_name="scene_dir", optional=True, tooltip="The Pose directory for the scene, overrides the scene_info", multiline=False, default=""),
            ],
            outputs=[],
            is_output_node=True,
        )        

    @classmethod
    def execute(
        cls,
        scene_info=None,
        scene_dir="",
    ) -> io.NodeOutput:
        if scene_info is None or not scene_info.scene_name:
            logger.error("SaveScene: scene_info is None or scene_name is empty")
            return io.NodeOutput(None)

        # Use provided scene_dir or fall back to scene_info's scene_dir
        target_dir = scene_dir if scene_dir else scene_info.scene_dir
        if not target_dir:
            target_dir = str(Path(default_scenes_dir()) / scene_info.scene_name)

        logger.info("SaveScene: scene_name='%s'; dest_dir='%s'", scene_info.scene_name, target_dir)
        
        # Use the unified save_all method
        scene_info.save_all(target_dir)

        return io.NodeOutput(
            ui=ui.PreviewText(f"Scene saved to '{target_dir}' with prompt='The girl {scene_info.girl_pos}, The male {scene_info.male_pos}'"),
        )

class SceneInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneInput"),
            display_name="Scene Input",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="scene_dir", display_name="scene_dir", tooltip="Directory where the scene is saved", multiline=False, default=""),
                io.String.Input(id="scene_name", display_name="scene_name", tooltip="Name of the pose", multiline=False, default=""),
                io.String.Input(id="girl_pos", display_name="girl_pos", tooltip="The prompt for the girl in the scene", multiline=True, default=""),
                io.String.Input(id="male_pos", display_name="male_pos", tooltip="The prompt for the male(s) in the scene", multiline=True, default=""),
                io.String.Input(id="four_image_prompt", display_name="four_image_prompt", tooltip="The Four Image prompt for the scene", multiline=True, default=""),
                io.String.Input(id="wan_prompt", display_name="wan_prompt", tooltip="The Wan High positive prompt for the scene", multiline=True, default=""),
                io.String.Input(id="wan_low_prompt", display_name="wan_low_prompt", tooltip="The Wan Low positive prompt for the scene", multiline=True, default=""),
                io.String.Input(id="pose_json", display_name="pose_json", tooltip="Pose JSON data", multiline=True, default=""),
                io.Image.Input(id="depth_image", display_name="depth_image", tooltip="Depth Image", optional=True),
                io.Image.Input(id="depth_any_image", display_name="depth_any_image", tooltip="Depth Any Image", optional=True),
                io.Image.Input(id="depth_midas_image", display_name="depth_midas_image", tooltip="Depth Midas Image", optional=True),
                io.Image.Input(id="depth_zoe_image", display_name="depth_zoe_image", tooltip="Depth Zoe Image", optional=True),
                io.Image.Input(id="depth_zoe_any_image", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image", optional=True),
                io.Image.Input(id="pose_dense_image", display_name="pose_dense_image", tooltip="Pose Dense Image", optional=True),
                io.Image.Input(id="pose_dw_image", display_name="pose_dw_image", tooltip="Pose DW Image", optional=True),
                io.Image.Input(id="pose_edit_image", display_name="pose_edit_image", tooltip="Pose Edit Image", optional=True),
                io.Image.Input(id="pose_face_image", display_name="pose_face_image", tooltip="Pose Face Image", optional=True),
                io.Image.Input(id="pose_open_image", display_name="pose_open_image", tooltip="Pose Open Image", optional=True),
                io.Image.Input(id="canny_image", display_name="canny_image", tooltip="Canny Image", optional=True),
                io.Image.Input(id="upscale_image", display_name="upscale_image", tooltip="Upscale Image", optional=True),
                io.Image.Input(id="girl_mask_image", display_name="girl_mask_image", tooltip="Girl Mask Image, with background", optional=True),
                io.Image.Input(id="male_mask_image", display_name="male_mask_image", tooltip="Male Mask Image, with background", optional=True),
                io.Image.Input(id="combined_mask_image", display_name="combined_mask_image", tooltip="Combined Mask Image, with background", optional=True),
                io.Image.Input(id="girl_mask_nobg_image", display_name="girl_mask_nobg_image", tooltip="Girl Mask Image, no background", optional=True),
                io.Image.Input(id="male_mask_nobg_image", display_name="male_mask_nobg_image", tooltip="Male Mask Image, no background", optional=True),
                io.Image.Input(id="combined_mask_nobg_image", display_name="combined_mask_nobg_image", tooltip="Combined Mask Image, no background", optional=True),
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="Multi-target LoRA stack. If omitted, loaded from scene directory on disk."),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_dir="",
        scene_name="",
        girl_pos="",
        male_pos="",
        four_image_prompt="",
        wan_prompt="",
        wan_low_prompt="",
        pose_json="",
        depth_image=None,
        depth_any_image=None,
        depth_midas_image=None,
        depth_zoe_image=None,
        depth_zoe_any_image=None,
        pose_dense_image=None,
        pose_dw_image=None,
        pose_edit_image=None,
        pose_face_image=None,
        pose_open_image=None,
        canny_image=None,
        upscale_image=None,
        girl_mask_image=None,
        male_mask_image=None,
        combined_mask_image=None,
        girl_mask_no_bkgd_image=None,
        male_mask_no_bkgd_image=None,
        combined_mask_no_bkgd_image=None,
        lora_stack_data=None,
    ) -> io.NodeOutput:
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("SceneInput: scene_dir '%s' is invalid", scene_dir)
            return io.NodeOutput(None)

        logger.info("SceneInput: scene_dir='%s'; scene_name='%s'", scene_dir, scene_name)
        resolution = min(depth_image.shape[1], depth_image.shape[2]) if depth_image is not None else 512

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=scene_name,
            girl_pos=girl_pos,
            male_pos=male_pos,
            four_image_prompt=four_image_prompt,
            wan_prompt=wan_prompt,
            wan_low_prompt=wan_low_prompt,
            pose_json=pose_json,
            depth_image=depth_image,
            depth_any_image=depth_any_image,
            depth_midas_image=depth_midas_image,
            depth_zoe_image=depth_zoe_image,
            depth_zoe_any_image=depth_zoe_any_image,
            pose_dense_image=pose_dense_image,
            pose_dw_image=pose_dw_image,
            pose_edit_image=pose_edit_image,
            pose_face_image=pose_face_image,
            pose_open_image=pose_open_image,
            girl_mask_bkgd_image=girl_mask_image,
            male_mask_bkgd_image=male_mask_image,
            combined_mask_bkgd_image=combined_mask_image,
            girl_mask_no_bkgd_image=girl_mask_no_bkgd_image,
            male_mask_no_bkgd_image=male_mask_no_bkgd_image,
            combined_mask_no_bkgd_image=combined_mask_no_bkgd_image,
            canny_image=canny_image,
            upscale_image=upscale_image,
            lora_stack=lora_stack_data if lora_stack_data is not None else load_lora_stack(scene_dir),
            resolution=resolution,
        )

        return io.NodeOutput(
            scene_info
        )


# ── /fbtools/scene/* routes — moved from extension.py (Plan 21) ───────────────

@routes.post("/fbtools/scene/process_compositions")
async def scene_process_compositions(request):
    """
    Process compositions from a prompt collection and return composed prompts.
    Body: {"collection": dict}
    Returns: {"prompt_dict": dict, "status": str}
    """
    try:
        data = await request.json()
        collection_data = data.get("collection")
        
        if not collection_data:
            return web.json_response({"error": "collection data required"}, status=400)
        
        # Parse collection data
        try:
            collection = PromptCollection.from_dict(collection_data)
        except Exception as e:
            return web.json_response({"error": f"Invalid collection data: {str(e)}"}, status=400)
        
        # Get libber manager for substitutions
        libber_manager = LibberStateManager.instance()
        
        # Compose prompts
        prompt_dict = collection.compose_prompts(collection.compositions, libber_manager)
        
        return web.json_response({
            "prompt_dict": prompt_dict,
            "status": f"Processed {len(prompt_dict)} compositions"
        })
    
    except Exception as e:
        logger.exception("Error processing compositions")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/scene/get_scene_prompts")
async def scene_get_prompts(request):
    """
    Get prompts and compositions from a scene's prompts.json file.
    Query param: scene_dir
    Returns: {"prompts": [...], "compositions": {...}}
    """
    try:
        scene_dir = request.query.get("scene_dir")
        
        if not scene_dir:
            return web.json_response({"error": "scene_dir parameter required"}, status=400)
        
        if not os.path.isdir(scene_dir):
            return web.json_response({"error": f"scene_dir '{scene_dir}' is not a valid directory"}, status=400)
        
        # Load prompts.json
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        if not os.path.isfile(prompt_json_path):
            return web.json_response({"prompts": [], "compositions": {}})
        
        try:
            collection = PromptCollection.load_from_json(prompt_json_path)
        except Exception as e:
            return web.json_response({"error": f"Failed to load prompts.json: {str(e)}"}, status=500)
        
        # Convert prompts to list format for UI
        prompts_list = [
            {
                "key": key,
                "value": prompt.value,
                "category": prompt.category,
                "processing_type": prompt.processing_type,
                "libber_name": prompt.libber_name
            }
            for key, prompt in collection.prompts.items()
        ]
        
        # Get available libbers (merge in-memory and on-disk)
        libbers_set = set()
        try:
            manager = LibberStateManager.instance()
            libbers_set.update(manager.list_libbers())

            libbers_dir = default_libber_dir()
            if os.path.isdir(libbers_dir):
                for filename in os.listdir(libbers_dir):
                    filepath = os.path.join(libbers_dir, filename)
                    if os.path.isfile(filepath) and filename.endswith('.json'):
                        libbers_set.add(filename[:-5])
        except Exception as e:
            logger.warning("Warning: Could not load libbers list: %s", e)

        libbers_list = ["none"] + sorted(libbers_set)
        
        # Get scene_flags from collection if present
        scene_flags = {}
        collection_dict = collection.to_dict()
        if 'scene_flags' in collection_dict:
            scene_flags = collection_dict['scene_flags']
        
        # Load masks from masks.json
        masks_dict = load_masks_json(scene_dir)
        masks_data = {}
        if masks_dict:
            # Convert MaskDefinition objects to dict format
            masks_data = {name: mask.to_dict() for name, mask in masks_dict.items()}
        
        # Return compositions as dict with scene_flags and masks
        return web.json_response({
            "prompts": prompts_list,
            "compositions": collection.compositions,
            "scene_flags": scene_flags,
            "libbers": libbers_list,
            "masks": masks_data
        })
    
    except Exception as e:
        logger.exception("Error getting scene prompts")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/scene/save_scene_prompts")
async def scene_save_prompts(request):
    """
    Save prompts and compositions to a scene's prompts.json file.
    Body: {"scene_dir": str, "collection": dict}
    Returns: {"success": bool, "message": str}
    """
    try:
        data = await request.json()
        scene_dir = data.get("scene_dir")
        collection_data = data.get("collection")
        
        logger.info("ScenePromptManager API: Received save request for scene_dir='%s'", scene_dir)
        
        if not scene_dir:
            return web.json_response({"error": "scene_dir required"}, status=400)
        
        if not collection_data:
            return web.json_response({"error": "collection data required"}, status=400)
        
        if not os.path.isdir(scene_dir):
            logger.error("ScenePromptManager API: scene_dir '%s' is not a valid directory", scene_dir)
            return web.json_response({"error": f"scene_dir '{scene_dir}' is not a valid directory"}, status=400)
        
        # Parse and validate collection
        try:
            logger.debug("ScenePromptManager API: Incoming collection_data keys: %s", list(collection_data.keys()) if isinstance(collection_data, dict) else "not a dict")
            logger.debug("ScenePromptManager API: collection_data type: %s", type(collection_data))
            
            # Check if collection_data has the expected structure
            if not isinstance(collection_data, dict):
                raise ValueError(f"collection_data must be a dict, got {type(collection_data)}")
            
            # Log the structure
            if 'prompts' in collection_data:
                logger.debug("ScenePromptManager API: prompts type: %s, count: %d", 
                           type(collection_data['prompts']), 
                           len(collection_data['prompts']) if isinstance(collection_data['prompts'], (dict, list)) else 0)
            if 'compositions' in collection_data:
                logger.debug("ScenePromptManager API: compositions type: %s, count: %d",
                           type(collection_data['compositions']),
                           len(collection_data['compositions']) if isinstance(collection_data['compositions'], (dict, list)) else 0)
            if 'scene_flags' in collection_data:
                logger.debug("ScenePromptManager API: scene_flags: %s", collection_data['scene_flags'])
            
            collection = PromptCollection.from_dict(collection_data)
            logger.info(
                "ScenePromptManager API: Parsed collection with %d prompts and %d compositions",
                len(collection.prompts),
                len(collection.compositions),
            )
        except Exception as e:
            logger.exception("ScenePromptManager API: Error parsing collection data: %s", str(e))
            logger.error("ScenePromptManager API: collection_data content: %s", str(collection_data)[:500])
            return web.json_response({"error": f"Invalid collection data: {str(e)}"}, status=400)
        
        # Save to file
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        logger.info("ScenePromptManager API: Attempting to save to: %s", prompt_json_path)
        logger.debug(
            "ScenePromptManager API: File exists before save: %s",
            os.path.exists(prompt_json_path),
        )
        
        try:
            # Convert to dict - scene_flags now preserved automatically
            collection_dict = collection.to_dict()
            
            logger.debug(
                "ScenePromptManager API: Collection dict keys: %s",
                list(collection_dict.keys()),
            )
            logger.debug(
                "ScenePromptManager API: Prompt keys in dict: %s",
                list(collection_dict.get('prompts', {}).keys()),
            )
            logger.debug(
                "ScenePromptManager API: Composition keys in dict: %s",
                list(collection_dict.get('compositions', {}).keys()),
            )
            
            with open(prompt_json_path, 'w', encoding='utf-8') as f:
                json.dump(collection_dict, f, indent=2, ensure_ascii=False)
            
            logger.debug(
                "ScenePromptManager API: File written successfully; exists=%s; size=%s",
                os.path.exists(prompt_json_path),
                os.path.getsize(prompt_json_path),
            )
            
            # Read back to verify
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            logger.debug(
                "ScenePromptManager API: Verification - read back %d prompts",
                len(saved_data.get('prompts', {})),
            )
            
            message = f"Saved {len(collection.prompts)} prompts and {len(collection.compositions)} compositions to {os.path.basename(scene_dir)}"
            logger.info("ScenePromptManager API: %s", message)
            return web.json_response({
                "success": True,
                "message": message
            })
        except Exception as e:
            logger.exception("ScenePromptManager API: Error saving to file")
            return web.json_response({"error": f"Failed to save prompts.json: {str(e)}"}, status=500)
    
    except Exception as e:
        logger.exception("Error saving scene prompts")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/scene/list")
async def scene_list(request):
    """
    Get list of available scene names.
    Returns: {"scenes": [str]}
    """
    try:
        scenes_dir = default_scenes_dir()
        available_scenes = get_subdirectories(scenes_dir)
        scene_names = sorted(available_scenes.keys()) if available_scenes else []
        
        return web.json_response({'scenes': scene_names})
    except Exception as e:
        logger.exception("fbTools API: Error listing scenes")
        return web.json_response({'error': str(e)}, status=500)


@routes.get("/fbtools/scene/thumbnail/{scene_name}")
async def get_scene_thumbnail(request):
    """
    Serve thumbnail image for a scene.
    Returns: thumbnail PNG image or 404 if not found
    """
    try:
        scene_name = request.match_info.get("scene_name")
        
        if not scene_name:
            return web.json_response({"error": "scene_name required"}, status=400)
        
        scenes_dir = default_scenes_dir()
        thumbnail_path = os.path.join(scenes_dir, scene_name, "thumbnail.png")
        
        if not os.path.exists(thumbnail_path):
            logger.warning("Thumbnail not found: %s", thumbnail_path)
            return web.json_response({"error": f"Thumbnail not found for scene '{scene_name}'"}, status=404)
        
        # Serve the image file
        return web.FileResponse(
            thumbnail_path,
            headers={
                'Content-Type': 'image/png',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        logger.exception("Error serving thumbnail")
        return web.json_response({"error": str(e)}, status=500)
