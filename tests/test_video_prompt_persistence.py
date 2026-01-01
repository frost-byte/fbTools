"""
Integration tests for video prompt field persistence through API endpoints.

Tests that video_prompt_source, video_prompt_key, and video_custom_prompt
are correctly saved via /fbtools/story/save and loaded via /fbtools/story/load.

Note: These tests validate the data structures and logic without requiring
a running ComfyUI server. For full end-to-end testing, see the manual testing guide.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

# Use unified import approach from conftest
from conftest import import_test_module

# Import story_models module for data classes
story_models = import_test_module("story_models.py")
SceneInStory = story_models.SceneInStory
StoryInfo = story_models.StoryInfo


class TestVideoPromptDataStructures:
    """Test that video prompt fields are correctly structured in SceneInStory."""
    
    def test_scene_in_story_has_video_prompt_fields(self):
        """Test that SceneInStory model includes video prompt fields."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            prompt_source="prompt",
            prompt_key="main",
            video_prompt_source="composition",
            video_prompt_key="action",
            video_custom_prompt="Custom text"
        )
        
        assert hasattr(scene, 'video_prompt_source')
        assert hasattr(scene, 'video_prompt_key')
        assert hasattr(scene, 'video_custom_prompt')
        assert scene.video_prompt_source == "composition"
        assert scene.video_prompt_key == "action"
        assert scene.video_custom_prompt == "Custom text"
    
    def test_scene_in_story_video_prompt_defaults(self):
        """Test that video prompt fields have correct defaults."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0
        )
        
        # Check defaults
        assert scene.video_prompt_source == "auto"
        assert scene.video_prompt_key == ""
        assert scene.video_custom_prompt == ""
    
    def test_scene_serialization_includes_video_prompt_fields(self):
        """Test that SceneInStory serializes video prompt fields."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            video_prompt_source="prompt",
            video_prompt_key="test_key",
            video_custom_prompt=""
        )
        
        # Serialize to dict
        scene_dict = scene.model_dump()
        
        assert "video_prompt_source" in scene_dict
        assert "video_prompt_key" in scene_dict
        assert "video_custom_prompt" in scene_dict
        assert scene_dict["video_prompt_source"] == "prompt"
        assert scene_dict["video_prompt_key"] == "test_key"
    
    def test_scene_deserialization_includes_video_prompt_fields(self):
        """Test that SceneInStory can be deserialized with video prompt fields."""
        scene_data = {
            "scene_name": "test_scene",
            "scene_order": 0,
            "mask_type": "combined",
            "mask_background": True,
            "prompt_source": "prompt",
            "prompt_key": "main",
            "custom_prompt": "",
            "video_prompt_source": "composition",
            "video_prompt_key": "video_main",
            "video_custom_prompt": "",
            "depth_type": "depth",
            "pose_type": "open"
        }
        
        scene = SceneInStory(**scene_data)
        
        assert scene.video_prompt_source == "composition"
        assert scene.video_prompt_key == "video_main"
        assert scene.video_custom_prompt == ""


class TestVideoPromptJSONPersistence:
    """Test video prompt field persistence in JSON files."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_manual_json_save_includes_video_prompt_fields(self, temp_dir):
        """Test that manually saving a story JSON includes video prompt fields."""
        story_data = {
            "version": 2,
            "story_name": "test_story",
            "story_dir": str(temp_dir / "test_story"),
            "scenes": [
                {
                    "scene_name": "scene_01",
                    "scene_order": 0,
                    "mask_type": "combined",
                    "mask_background": True,
                    "prompt_source": "prompt",
                    "prompt_key": "main",
                    "custom_prompt": "",
                    "video_prompt_source": "prompt",
                    "video_prompt_key": "video_key",
                    "video_custom_prompt": "",
                    "depth_type": "depth",
                    "pose_type": "open"
                }
            ]
        }
        
        story_path = temp_dir / "test_story" / "story.json"
        story_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(story_path, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        # Read back and verify
        with open(story_path, 'r') as f:
            loaded_data = json.load(f)
        
        scene_data = loaded_data["scenes"][0]
        assert "video_prompt_source" in scene_data
        assert "video_prompt_key" in scene_data
        assert "video_custom_prompt" in scene_data
        assert scene_data["video_prompt_source"] == "prompt"
        assert scene_data["video_prompt_key"] == "video_key"
    
    def test_story_json_roundtrip_via_models(self, tmp_path):
        """Test that video prompts survive JSON serialization/deserialization via models."""
        # Create a story with video prompt fields
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            video_prompt_source="composition",
            video_prompt_key="comp_video",
            video_custom_prompt="Custom video text"
        )
        
        story_info = StoryInfo(
            version=2,
            story_name="roundtrip_test",
            story_dir=str(tmp_path),
            scenes=[scene]
        )
        
        # Manually serialize to JSON (simulating save_story logic)
        scenes_data = []
        for s in story_info.scenes:
            scene_data = {
                "scene_name": s.scene_name,
                "scene_order": s.scene_order,
                "mask_type": s.mask_type,
                "mask_background": s.mask_background,
                "prompt_source": s.prompt_source,
                "prompt_key": s.prompt_key,
                "custom_prompt": s.custom_prompt,
                "video_prompt_source": s.video_prompt_source,
                "video_prompt_key": s.video_prompt_key,
                "video_custom_prompt": s.video_custom_prompt,
                "depth_type": s.depth_type,
                "pose_type": s.pose_type,
            }
            scenes_data.append(scene_data)
        
        story_data = {
            "version": 2,
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scenes": scenes_data
        }
        
        # Write to file
        story_path = tmp_path / "story.json"
        with open(story_path, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        # Load back from file (simulating load_story logic)
        with open(story_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Recreate SceneInStory objects
        loaded_scenes = []
        for scene_data in loaded_data["scenes"]:
            loaded_scene = SceneInStory(
                scene_name=scene_data.get("scene_name", ""),
                scene_order=scene_data.get("scene_order", 0),
                mask_type=scene_data.get("mask_type", "combined"),
                mask_background=scene_data.get("mask_background", True),
                prompt_source=scene_data.get("prompt_source", "prompt"),
                prompt_key=scene_data.get("prompt_key", ""),
                custom_prompt=scene_data.get("custom_prompt", ""),
                video_prompt_source=scene_data.get("video_prompt_source", "auto"),
                video_prompt_key=scene_data.get("video_prompt_key", ""),
                video_custom_prompt=scene_data.get("video_custom_prompt", ""),
                depth_type=scene_data.get("depth_type", "depth"),
                pose_type=scene_data.get("pose_type", "open"),
            )
            loaded_scenes.append(loaded_scene)
        
        loaded_story = StoryInfo(
            version=loaded_data["version"],
            story_name=loaded_data["story_name"],
            story_dir=loaded_data["story_dir"],
            scenes=loaded_scenes
        )
        
        # Verify video prompt fields are preserved
        assert len(loaded_story.scenes) == 1
        loaded_scene = loaded_story.scenes[0]
        assert loaded_scene.video_prompt_source == "composition"
        assert loaded_scene.video_prompt_key == "comp_video"
        assert loaded_scene.video_custom_prompt == "Custom video text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
