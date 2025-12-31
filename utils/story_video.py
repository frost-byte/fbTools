"""
Utility functions for story video generation workflow.

This module provides testable helper functions for video generation from story scenes,
separated from ComfyUI dependencies for easier testing.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict


def list_job_ids(story_dir: str) -> List[str]:
    """
    List all job IDs for a story by scanning the jobs directory.
    
    Args:
        story_dir: Path to the story directory
        
    Returns:
        List of job ID strings, sorted by modification time (newest first)
    """
    jobs_dir = Path(story_dir) / "jobs"
    if not jobs_dir.exists():
        return []
    
    # Get all subdirectories (job IDs) with their modification times
    job_dirs = []
    for item in jobs_dir.iterdir():
        if item.is_dir():
            job_dirs.append((item.name, item.stat().st_mtime))
    
    # Sort by modification time, newest first
    job_dirs.sort(key=lambda x: x[1], reverse=True)
    
    return [job_id for job_id, _ in job_dirs]


def find_scene_image(job_input_dir: str, scene_order: int, scene_name: str, extensions: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the image file for a specific scene in the job input directory.
    
    Args:
        job_input_dir: Path to the job's input directory
        scene_order: Order number of the scene (0-based)
        scene_name: Name of the scene
        extensions: List of image extensions to search for (default: png, jpg, jpeg, webp)
        
    Returns:
        Full path to the image file, or None if not found
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    input_dir = Path(job_input_dir)
    if not input_dir.exists():
        return None
    
    # Format scene name for filename matching
    formatted_scene_name = scene_name.replace(" ", "_")
    expected_prefix = f"{scene_order:03d}_{formatted_scene_name}"
    
    # Search for files matching the pattern
    for ext in extensions:
        candidate = input_dir / f"{expected_prefix}{ext}"
        if candidate.exists():
            return str(candidate)
    
    return None


def pair_consecutive_scenes(scenes: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """
    Create pairs of consecutive scenes for video generation.
    
    Each pair represents a transition from one scene to the next.
    The last scene is paired with None (no next scene).
    
    Args:
        scenes: List of scene descriptor dictionaries (should be sorted by scene_order)
        
    Returns:
        List of tuples (current_scene, next_scene), where next_scene is None for the last scene
    """
    if not scenes:
        return []
    
    pairs = []
    for i in range(len(scenes)):
        current = scenes[i]
        next_scene = scenes[i + 1] if i + 1 < len(scenes) else None
        pairs.append((current, next_scene))
    
    return pairs


def generate_video_filename(from_scene_order: int, to_scene_order: Optional[int], 
                           from_scene_name: str, to_scene_name: Optional[str],
                           extension: str = "mp4") -> str:
    """
    Generate a filename for a video transition between two scenes.
    
    Args:
        from_scene_order: Order number of the starting scene
        to_scene_order: Order number of the ending scene (None for final scene)
        from_scene_name: Name of the starting scene
        to_scene_name: Name of the ending scene (None for final scene)
        extension: Video file extension (default: mp4)
        
    Returns:
        Filename string like "001_to_002_opening_to_battle.mp4" or "003_finale.mp4"
    """
    # Format scene names
    from_name_formatted = from_scene_name.replace(" ", "_")
    
    if to_scene_order is not None and to_scene_name is not None:
        to_name_formatted = to_scene_name.replace(" ", "_")
        return f"{from_scene_order:03d}_to_{to_scene_order:03d}_{from_name_formatted}_to_{to_name_formatted}.{extension}"
    else:
        # Final scene - no transition
        return f"{from_scene_order:03d}_{from_name_formatted}.{extension}"


def resolve_video_prompt(scene: dict, prompt_data: dict, next_scene: Optional[dict] = None) -> str:
    """
    Resolve the video prompt for a scene based on its video prompt configuration.
    
    Handles the video_prompt_source field:
    - "auto": Use the scene's image prompt
    - "prompt": Use video_prompt_key from prompt_data
    - "composition": Use video_prompt_key from compositions (requires composition processing)
    - "custom": Use video_custom_prompt directly
    
    Args:
        scene: Scene descriptor dictionary with video prompt fields
        prompt_data: Dictionary of available prompts
        next_scene: Optional next scene descriptor (for context-aware prompts)
        
    Returns:
        Resolved prompt string
    """
    video_prompt_source = scene.get("video_prompt_source", "auto")
    
    if video_prompt_source == "auto":
        # Use the image prompt
        return scene.get("positive_prompt", "")
    
    elif video_prompt_source == "custom":
        # Use custom video prompt
        return scene.get("video_custom_prompt", "")
    
    elif video_prompt_source == "prompt":
        # Look up video prompt key from prompt_data
        video_key = scene.get("video_prompt_key", "")
        if video_key:
            return prompt_data.get(video_key, "")
        return ""
    
    elif video_prompt_source == "composition":
        # Composition needs to be resolved elsewhere (requires LibberStateManager)
        # This function just returns the key; composition resolution happens at node level
        return scene.get("video_prompt_key", "")
    
    # Default fallback
    return scene.get("positive_prompt", "")


def build_video_descriptor(scene: dict, next_scene: Optional[dict],
                           job_input_dir: str, job_output_dir: str,
                           prompt_data: dict) -> Optional[dict]:
    """
    Build a complete video descriptor for a scene transition.
    
    Args:
        scene: Current scene descriptor
        next_scene: Next scene descriptor (None for final scene)
        job_input_dir: Path to job input directory (where images are)
        job_output_dir: Path to job output directory (where videos will be saved)
        prompt_data: Dictionary of available prompts
        
    Returns:
        Video descriptor dictionary or None if required images are missing
    """
    scene_order = scene.get("scene_order", 0)
    scene_name = scene.get("scene_name", "")
    
    # Find first frame image
    first_frame_path = find_scene_image(job_input_dir, scene_order, scene_name)
    if not first_frame_path:
        return None  # Cannot create video without first frame
    
    # Build descriptor
    descriptor = {
        "scene_order": scene_order,
        "scene_name": scene_name,
        "scene_id": scene.get("scene_id", ""),
        "first_frame_path": first_frame_path,
        "video_prompt": resolve_video_prompt(scene, prompt_data, next_scene),
        "lora_data": scene.get("lora_data"),
        "job_input_dir": job_input_dir,
        "job_output_dir": job_output_dir,
    }
    
    # Add next scene info if available
    if next_scene is not None:
        next_order = next_scene.get("scene_order", 0)
        next_name = next_scene.get("scene_name", "")
        last_frame_path = find_scene_image(job_input_dir, next_order, next_name)
        
        if last_frame_path:
            descriptor.update({
                "next_scene_order": next_order,
                "next_scene_name": next_name,
                "last_frame_path": last_frame_path,
                "has_transition": True,
            })
        else:
            descriptor["has_transition"] = False
    else:
        descriptor["has_transition"] = False
    
    # Generate output filename
    next_order = descriptor.get("next_scene_order")
    next_name = descriptor.get("next_scene_name")
    video_filename = generate_video_filename(scene_order, next_order, scene_name, next_name)
    descriptor["video_filename"] = video_filename
    descriptor["video_output_path"] = str(Path(job_output_dir) / video_filename)
    
    return descriptor
