"""
Testable utilities for scene image saving.

This module contains pure functions and classes that can be tested independently
of the ComfyUI node infrastructure.
"""

import os
from typing import Optional


class SceneImageSaveConfig:
    """Configuration for saving a scene image - testable pure data class"""
    def __init__(self, scene_name: str, scene_order: int, target_dir: str, image_format: str, quality: int = 95):
        self.scene_name = scene_name
        self.scene_order = scene_order
        self.target_dir = target_dir
        self.image_format = image_format
        self.quality = quality
    
    @staticmethod
    def from_descriptor(descriptor: dict, scene_index: int, image_format: str = "png", quality: int = 95) -> Optional["SceneImageSaveConfig"]:
        """Extract save configuration from scene descriptor - pure function, easily testable"""
        scene_name = descriptor.get("scene_name", "")
        scene_order = descriptor.get("scene_order", 0)
        job_input_dir = descriptor.get("job_input_dir", "")
        job_output_dir = descriptor.get("job_output_dir", "")
        target_dir = job_input_dir or job_output_dir
        
        if not scene_name or not target_dir:
            return None
        
        return SceneImageSaveConfig(scene_name, scene_order, target_dir, image_format, quality)
    
    def generate_filename(self) -> str:
        """Generate standardized filename - pure function, easily testable"""
        formatted_scene_name = self.scene_name.lower().replace(" ", "_").replace("-", "_")
        return f"{self.scene_order:03d}_{formatted_scene_name}.{self.image_format}"
    
    def generate_filepath(self) -> str:
        """Generate full file path - pure function, easily testable"""
        return os.path.join(self.target_dir, self.generate_filename())


class ImageSaver:
    """Abstraction for image saving - can be mocked in tests"""
    
    @staticmethod
    def tensor_to_pil(image_tensor):
        """Convert tensor to PIL Image - testable conversion logic"""
        import numpy as np
        from PIL import Image as PILImage
        
        img_tensor = image_tensor[0] if hasattr(image_tensor, 'shape') and len(image_tensor.shape) == 4 else image_tensor
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        return PILImage.fromarray(img_np)
    
    @staticmethod
    def save_pil_image(pil_image, filepath: str, image_format: str, quality: int = 95):
        """Save PIL image with format-specific options - testable I/O"""
        if image_format.lower() in ["jpg", "jpeg"]:
            pil_image.save(filepath, format="JPEG", quality=quality, optimize=True)
        elif image_format.lower() == "webp":
            pil_image.save(filepath, format="WEBP", quality=quality)
        else:
            pil_image.save(filepath, format="PNG", optimize=True)
    
    @staticmethod
    def ensure_directory(directory: str):
        """Ensure directory exists - testable I/O"""
        os.makedirs(directory, exist_ok=True)


def select_scene_descriptor(scene_batch: list, scene_index: int) -> Optional[dict]:
    """Select and return scene descriptor from batch - pure function, easily testable"""
    if not scene_batch:
        return None
    
    try:
        scenes_sorted = sorted(scene_batch, key=lambda d: d.get("scene_order", 0))
    except Exception:
        scenes_sorted = scene_batch
    
    safe_index = max(0, min(len(scenes_sorted) - 1, scene_index))
    return scenes_sorted[safe_index]


def generate_preview_text(config: SceneImageSaveConfig, filepath: str) -> str:
    """Generate preview text for UI - pure function, easily testable"""
    filename = config.generate_filename()
    preview_text = (
        f"Saved: {filename}\n"
        f"Scene: {config.scene_name} (order: {config.scene_order})\n"
        f"Path: {filepath}\n"
        f"Format: {config.image_format.upper()}"
    )
    if config.image_format.lower() in ["jpg", "jpeg", "webp"]:
        preview_text += f" (quality: {config.quality})"
    return preview_text
