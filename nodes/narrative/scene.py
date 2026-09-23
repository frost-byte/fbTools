"""SceneInfo / mask-system core, extracted from extension.py (Plan 20).

The 10 Scene CRUD node classes (SceneSelect, SceneCreate, ...) and their 5
/fbtools/scene/* routes have NOT moved yet — they stay in extension.py and
import SceneInfo/MaskType/etc. back from here (see the Plan 20 re-export list
in extension.py's own import block). Story and lora-presets do the same.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict

from ..libber import LibberStateManager
from ..shared import default_scenes_dir
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
)
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
