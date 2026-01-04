"""
Integration tests for scene flags in story save/load cycle.

Tests verify that scene flags (use_depth, use_mask, use_pose, use_canny)
are properly preserved when saving and loading stories.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Use unified import approach from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import import_test_module

# Import modules to test
story_models = import_test_module("story_models.py")
SceneInStory = story_models.SceneInStory
StoryInfo = story_models.StoryInfo
save_story = story_models.save_story
load_story = story_models.load_story


class TestStorySceneFlagsSerialization:
    """Test that scene flags are properly serialized in stories."""
    
    def test_scene_in_story_has_flag_fields(self):
        """SceneInStory model should have all flag fields."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            use_depth=True,
            use_mask=False,
            use_pose=True,
            use_canny=False
        )
        
        assert scene.use_depth is True
        assert scene.use_mask is False
        assert scene.use_pose is True
        assert scene.use_canny is False
    
    def test_scene_flags_default_to_false(self):
        """Scene flags should default to False if not specified."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0
        )
        
        assert scene.use_depth is False
        assert scene.use_mask is False
        assert scene.use_pose is False
        assert scene.use_canny is False
    
    def test_scene_flags_serialize_to_dict(self):
        """Scene flags should be included when serializing to dict."""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            use_depth=True,
            use_mask=True,
            use_pose=False,
            use_canny=True
        )
        
        scene_dict = scene.model_dump()
        
        assert "use_depth" in scene_dict
        assert "use_mask" in scene_dict
        assert "use_pose" in scene_dict
        assert "use_canny" in scene_dict
        assert scene_dict["use_depth"] is True
        assert scene_dict["use_mask"] is True
        assert scene_dict["use_pose"] is False
        assert scene_dict["use_canny"] is True
    
    def test_scene_flags_deserialize_from_dict(self):
        """Scene flags should be loaded from dict."""
        scene_data = {
            "scene_name": "test_scene",
            "scene_order": 0,
            "use_depth": False,
            "use_mask": True,
            "use_pose": True,
            "use_canny": False
        }
        
        scene = SceneInStory(**scene_data)
        
        assert scene.use_depth is False
        assert scene.use_mask is True
        assert scene.use_pose is True
        assert scene.use_canny is False


class TestStoryFlagsPersistence:
    """Test that scene flags persist through file save/load using production functions."""
    
    def test_story_with_flags_saves_to_json(self):
        """Story with scene flags should save correctly to JSON using save_story()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create story with scenes that have flags
            story = StoryInfo(
                story_name="test_story",
                story_dir=tmpdir,
                scenes=[
                    SceneInStory(
                        scene_name="scene1",
                        scene_order=0,
                        use_depth=True,
                        use_mask=False,
                        use_pose=True,
                        use_canny=False
                    ),
                    SceneInStory(
                        scene_name="scene2",
                        scene_order=1,
                        use_depth=False,
                        use_mask=True,
                        use_pose=False,
                        use_canny=True
                    )
                ]
            )
            
            # Save to file using production save_story() function
            story_file = Path(tmpdir) / "story.json"
            save_story(story, str(story_file))
            
            # Verify file contents
            with open(story_file, 'r') as f:
                saved_data = json.load(f)
            
            # Check that flags are in the saved data
            assert len(saved_data["scenes"]) == 2
            
            scene1 = saved_data["scenes"][0]
            assert scene1["use_depth"] is True
            assert scene1["use_mask"] is False
            assert scene1["use_pose"] is True
            assert scene1["use_canny"] is False
            
            scene2 = saved_data["scenes"][1]
            assert scene2["use_depth"] is False
            assert scene2["use_mask"] is True
            assert scene2["use_pose"] is False
            assert scene2["use_canny"] is True
    
    def test_story_flags_roundtrip(self):
        """Flags should survive save/load roundtrip using production functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original story
            original_story = StoryInfo(
                story_name="test_story",
                story_dir=tmpdir,
                scenes=[
                    SceneInStory(
                        scene_name="scene1",
                        scene_order=0,
                        use_depth=True,
                        use_mask=True,
                        use_pose=False,
                        use_canny=True
                    )
                ]
            )
            
            # Save using production save_story() function
            story_file = Path(tmpdir) / "story.json"
            save_story(original_story, str(story_file))
            
            # Load using production load_story() function
            loaded_story = load_story(str(story_file))
            
            # Verify flags match
            assert len(loaded_story.scenes) == 1
            loaded_scene = loaded_story.scenes[0]
            assert loaded_scene.use_depth is True
            assert loaded_scene.use_mask is True
            assert loaded_scene.use_pose is False
            assert loaded_scene.use_canny is True


class TestStoryFlagsAPIFormat:
    """Test that flags work with the API format (what frontend sends)."""
    
    def test_api_format_scene_creation(self):
        """Test creating scenes from API format (frontend data)."""
        # Simulate data coming from StoryEdit frontend
        api_data = {
            "scene_id": "scene_123",
            "scene_name": "test_scene",
            "scene_order": 0,
            "mask_type": "combined",
            "mask_background": True,
            "prompt_source": "prompt",
            "prompt_key": "main_prompt",
            "custom_prompt": "",
            "video_prompt_source": "auto",
            "video_prompt_key": "",
            "video_custom_prompt": "",
            "depth_type": "depth",
            "pose_type": "open",
            "use_depth": True,
            "use_mask": False,
            "use_pose": True,
            "use_canny": False
        }
        
        scene = SceneInStory(**api_data)
        
        # Verify all fields are set correctly
        assert scene.scene_id == "scene_123"
        assert scene.scene_name == "test_scene"
        assert scene.use_depth is True
        assert scene.use_mask is False
        assert scene.use_pose is True
        assert scene.use_canny is False
    
    def test_story_save_endpoint_data_format(self):
        """Test the data format used by the /fbtools/story/save endpoint."""
        # This simulates what the backend receives from the frontend
        request_body = {
            "story_name": "test_story",
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "scene_name": "opening",
                    "scene_order": 0,
                    "use_depth": True,
                    "use_mask": False,
                    "use_pose": True,
                    "use_canny": False
                },
                {
                    "scene_id": "scene_2",
                    "scene_name": "middle",
                    "scene_order": 1,
                    "use_depth": False,
                    "use_mask": True,
                    "use_pose": False,
                    "use_canny": True
                }
            ]
        }
        
        # Process scenes as the backend would
        scenes = []
        for scene_data in request_body["scenes"]:
            scene = SceneInStory(**scene_data)
            scenes.append(scene)
        
        # Verify flags are preserved
        assert len(scenes) == 2
        
        assert scenes[0].use_depth is True
        assert scenes[0].use_mask is False
        assert scenes[0].use_pose is True
        assert scenes[0].use_canny is False
        
        assert scenes[1].use_depth is False
        assert scenes[1].use_mask is True
        assert scenes[1].use_pose is False
        assert scenes[1].use_canny is True


class TestStoryFlagsMigration:
    """Test backward compatibility with stories without flags."""
    
    def test_load_old_story_without_flags(self):
        """Loading old stories without flags should not fail."""
        old_story_data = {
            "version": 2,
            "story_name": "old_story",
            "story_dir": "/tmp/old_story",
            "scenes": [
                {
                    "scene_name": "scene1",
                    "scene_order": 0,
                    "mask_type": "combined",
                    "mask_background": True,
                    "prompt_source": "prompt",
                    "prompt_key": "girl_pos"
                    # No flag fields - old format
                }
            ]
        }
        
        # Should not raise error
        story = StoryInfo(**old_story_data)
        
        # Flags should default to False
        assert story.scenes[0].use_depth is False
        assert story.scenes[0].use_mask is False
        assert story.scenes[0].use_pose is False
        assert story.scenes[0].use_canny is False
    
    def test_mixed_scenes_with_and_without_flags(self):
        """Story can have mix of scenes with and without flags."""
        story_data = {
            "version": 2,
            "story_name": "mixed_story",
            "story_dir": "/tmp/mixed",
            "scenes": [
                {
                    "scene_name": "old_scene",
                    "scene_order": 0,
                    # No flags
                },
                {
                    "scene_name": "new_scene",
                    "scene_order": 1,
                    "use_depth": True,
                    "use_mask": True,
                    "use_pose": False,
                    "use_canny": True
                }
            ]
        }
        
        story = StoryInfo(**story_data)
        
        # Old scene has defaults
        assert story.scenes[0].use_depth is False
        assert story.scenes[0].use_mask is False
        
        # New scene has specified flags
        assert story.scenes[1].use_depth is True
        assert story.scenes[1].use_mask is True
        assert story.scenes[1].use_pose is False
        assert story.scenes[1].use_canny is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
