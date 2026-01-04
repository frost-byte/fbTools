"""
Data models for story management system.

This module contains the SceneInStory and StoryInfo classes
extracted to a separate file for easier testing and reusability.
"""

import json
import os
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class SceneInStory(BaseModel):
    """Represents a scene within a story with its configuration
    
    Version 2 Schema:
    - prompt_source: "prompt" | "composition" | "custom"
    - prompt_key: key from scene's prompt_dict or composition_dict
    - custom_prompt: used when prompt_source="custom"
    
    Video Generation Fields:
    - video_prompt_source: "prompt" | "composition" | "custom" | "auto" (default: "auto")
    - video_prompt_key: key from scene's prompt_dict or composition_dict for video
    - video_custom_prompt: custom prompt for video generation
    """
    scene_id: str = ""  # Unique identifier for this scene instance
    scene_name: str
    scene_order: int
    mask_type: str = "combined"  # girl, male, combined, girl_no_bg, male_no_bg, combined_no_bg
    mask_background: bool = True
    
    # V2 fields - Image generation
    prompt_source: str = "prompt"  # "prompt", "composition", "custom"
    prompt_key: str = ""  # Key from prompt_dict or composition_dict
    custom_prompt: str = ""  # Used when prompt_source="custom"
    
    # Video generation fields
    video_prompt_source: str = "auto"  # "prompt", "composition", "custom", "auto" (uses image prompt)
    video_prompt_key: str = ""  # Key from prompt_dict or composition_dict for video
    video_custom_prompt: str = ""  # Custom prompt for video generation
    
    # Legacy V1 fields (for backwards compatibility during migration)
    prompt_type: str = ""  # DEPRECATED: girl_pos, male_pos, etc.
    
    depth_type: str = "depth"
    pose_type: str = "open"
    
    use_depth: bool = False
    use_mask: bool = False
    use_pose: bool = False
    use_canny: bool = False

    def __init__(self, **data):
        if 'scene_id' not in data or not data['scene_id']:
            import uuid
            data['scene_id'] = str(uuid.uuid4())
        
        # Migrate V1 to V2 if needed
        if 'prompt_type' in data and data.get('prompt_type') and not data.get('prompt_source'):
            prompt_type = data['prompt_type']
            if prompt_type == 'custom':
                data['prompt_source'] = 'custom'
                data['prompt_key'] = ''
            else:
                # Old prompt_type was a key in the old prompts.json (e.g., "girl_pos", "male_pos")
                # Map to new system: these are now keys in prompt_dict
                data['prompt_source'] = 'prompt'
                data['prompt_key'] = prompt_type
        
        super().__init__(**data)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StoryInfo(BaseModel):
    """Contains ordered list of scenes and story metadata"""
    version: int = 2  # Schema version (1=old prompt_type, 2=prompt_source/prompt_key)
    story_name: str
    story_dir: str
    scenes: List[SceneInStory] = []
    
    def get_scene_by_id(self, scene_id: str) -> Optional[SceneInStory]:
        """Get scene by unique ID"""
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None
    
    def get_scene_by_name(self, scene_name: str) -> Optional[SceneInStory]:
        """Get first scene matching the name (multiple scenes can share same name)"""
        for scene in self.scenes:
            if scene.scene_name == scene_name:
                # Ensure scene has UUID (for backward compatibility)
                if not scene.scene_id:
                    import uuid
                    scene.scene_id = str(uuid.uuid4())
                return scene
        return None
    
    def get_scenes_by_name(self, scene_name: str) -> List[SceneInStory]:
        """Get all scenes matching the name"""
        matching_scenes = [scene for scene in self.scenes if scene.scene_name == scene_name]
        # Ensure all scenes have UUID (for backward compatibility)
        for scene in matching_scenes:
            if not scene.scene_id:
                import uuid
                scene.scene_id = str(uuid.uuid4())
        return matching_scenes
    
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


# ============================================================================
# STORY FILE I/O FUNCTIONS
# ============================================================================

def save_story(story_info: StoryInfo, story_json_path: str):
    """Save story information to JSON file (V2 format)"""
    try:
        scenes_data = []
        for scene in story_info.scenes:
            scene_data = {
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key,
                "custom_prompt": scene.custom_prompt,
                "video_prompt_source": scene.video_prompt_source,
                "video_prompt_key": scene.video_prompt_key,
                "video_custom_prompt": scene.video_custom_prompt,
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
                "use_depth": scene.use_depth,
                "use_mask": scene.use_mask,
                "use_pose": scene.use_pose,
                "use_canny": scene.use_canny,
            }
            scenes_data.append(scene_data)
        
        story_data = {
            "version": 2,
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scenes": scenes_data
        }
        
        with open(story_json_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        raise Exception(f"Error saving story to '{story_json_path}': {str(e)}")


def load_story(story_json_path: str) -> Optional[StoryInfo]:
    """Load story information from JSON file
    
    Supports both V1 (prompt_type) and V2 (prompt_source/prompt_key) formats.
    V1 stories are automatically migrated to V2 on load.
    """
    if not os.path.isfile(story_json_path):
        return None
    
    try:
        with open(story_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle version (default to 1 for old stories)
        version = data.get("version", 1)
        
        scenes = []
        for scene_data in data.get("scenes", []):
            # V1 migration: convert prompt_type to prompt_source/prompt_key
            if version == 1 and 'prompt_type' in scene_data:
                prompt_type = scene_data.get('prompt_type', '')
                if prompt_type == 'custom':
                    scene_data['prompt_source'] = 'custom'
                    scene_data['prompt_key'] = ''
                elif prompt_type:
                    scene_data['prompt_source'] = 'prompt'
                    scene_data['prompt_key'] = prompt_type
            
            # Create scene with automatic V1->V2 migration via __init__
            scene = SceneInStory(**scene_data)
            scenes.append(scene)
        
        return StoryInfo(
            version=2,  # Always save as V2 going forward
            story_name=data.get("story_name", ""),
            story_dir=data.get("story_dir", ""),
            scenes=scenes
        )
        
    except Exception as e:
        raise Exception(f"Error loading story from '{story_json_path}': {str(e)}")
