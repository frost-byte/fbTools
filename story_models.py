"""
Data models for story management system.

This module contains the SceneInStory and StoryInfo classes
extracted to a separate file for easier testing and reusability.
"""

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
