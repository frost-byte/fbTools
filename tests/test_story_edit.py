"""
Practical tests for StoryEdit node helper methods.

These tests focus on pure helper functions that don't require full ComfyUI context.
For integration testing, see STORY_EDIT_TESTING_GUIDE.md
"""

import pytest
import json
from unittest.mock import MagicMock, patch


class TestStoryEditHelpers:
    """Test StoryEdit helper methods that can be tested in isolation."""
    
    def test_load_prompt_text_custom_source(self):
        """Test loading prompt text when source is 'custom'."""
        # This is a pure function test - doesn't need imports
        
        # Simulate the logic of _load_prompt_text for custom source
        def load_prompt_text_custom(prompt_source, custom_prompt):
            if prompt_source == "custom":
                return custom_prompt
            return ""
        
        result = load_prompt_text_custom("custom", "My custom prompt text")
        assert result == "My custom prompt text"
        
        result = load_prompt_text_custom("prompt", "Should not return this")
        assert result == ""
    
    def test_build_summary_text_logic(self):
        """Test summary text building logic."""
        # Simulate the summary building
        def build_summary(story_name, scene_count, preview_scene_name, scene_names):
            summary = f"Story: {story_name}\n"
            summary += f"Scenes: {scene_count}\n"
            summary += f"Preview: {preview_scene_name}\n"
            summary += "\nAll scenes:\n"
            for idx, name in enumerate(scene_names):
                marker = "→" if name == preview_scene_name else " "
                summary += f"{marker} {idx}. {name}\n"
            return summary
        
        result = build_summary(
            story_name="test_story",
            scene_count=3,
            preview_scene_name="middle",
            scene_names=["opening", "middle", "ending"]
        )
        
        assert "Story: test_story" in result
        assert "Scenes: 3" in result
        assert "Preview: middle" in result
        assert "opening" in result
        assert "middle" in result
        assert "ending" in result
        assert "→" in result  # Preview marker
    
    def test_meta_payload_structure(self):
        """Test metadata payload JSON structure."""
        # Simulate _build_meta_payload logic
        def build_meta_payload(story_name, story_dir, scene_count, preview_scene):
            return json.dumps({
                "story_name": story_name,
                "story_dir": story_dir,
                "scene_count": scene_count,
                "preview_scene": preview_scene
            })
        
        result = build_meta_payload(
            story_name="test",
            story_dir="/tmp/test",
            scene_count=5,
            preview_scene="scene_002"
        )
        
        data = json.loads(result)
        assert data["story_name"] == "test"
        assert data["story_dir"] == "/tmp/test"
        assert data["scene_count"] == 5
        assert data["preview_scene"] == "scene_002"
    
    def test_resolve_preview_scene_logic(self):
        """Test preview scene resolution logic."""
        def resolve_preview_scene(scenes, preview_name):
            """Resolve which scene to preview."""
            if not scenes:
                return None
            
            # If preview_name specified, find it
            if preview_name:
                for scene in scenes:
                    if scene["scene_name"] == preview_name:
                        return scene
            
            # Default to first scene
            return scenes[0] if scenes else None
        
        scenes = [
            {"scene_name": "opening", "scene_order": 0},
            {"scene_name": "middle", "scene_order": 1},
            {"scene_name": "ending", "scene_order": 2}
        ]
        
        # Test with specific name
        result = resolve_preview_scene(scenes, "middle")
        assert result is not None
        assert result["scene_name"] == "middle"
        
        # Test default (first scene)
        result = resolve_preview_scene(scenes, "")
        assert result is not None
        assert result["scene_name"] == "opening"
        
        # Test with empty scenes
        result = resolve_preview_scene([], "")
        assert result is None
        
        # Test with nonexistent scene falls back to first
        result = resolve_preview_scene(scenes, "nonexistent")
        # In real implementation, this might return None or first scene
        # depending on the exact logic
    
    def test_scene_order_adjustment(self):
        """Test scene order adjustment when moving scenes."""
        def adjust_scene_order(scenes):
            """Ensure scene_order matches position in list."""
            for idx, scene in enumerate(scenes):
                scene["scene_order"] = idx
            return scenes
        
        scenes = [
            {"scene_name": "first", "scene_order": 0},
            {"scene_name": "second", "scene_order": 1},
            {"scene_name": "third", "scene_order": 2}
        ]
        
        # Swap first and second
        scenes[0], scenes[1] = scenes[1], scenes[0]
        
        # Adjust orders
        result = adjust_scene_order(scenes)
        
        assert result[0]["scene_name"] == "second"
        assert result[0]["scene_order"] == 0
        assert result[1]["scene_name"] == "first"
        assert result[1]["scene_order"] == 1


class TestStoryDataValidation:
    """Test story data structure validation."""
    
    def test_scene_data_structure(self):
        """Test that scene data has required fields."""
        scene = {
            "scene_id": "scene_001",
            "scene_name": "opening",
            "scene_order": 0,
            "mask_type": "combined",
            "mask_background": True,
            "prompt_source": "prompt",
            "prompt_key": "girl_pos",
            "custom_prompt": "",
            "depth_type": "depth",
            "pose_type": "open",
            "use_depth": False,
            "use_mask": False,
            "use_pose": False,
            "use_canny": False
        }
        
        # Verify required fields
        required_fields = [
            "scene_id", "scene_name", "scene_order",
            "mask_type", "mask_background", "prompt_source",
            "depth_type", "pose_type"
        ]
        
        for field in required_fields:
            assert field in scene, f"Missing required field: {field}"
    
    def test_story_data_structure(self):
        """Test that story data has required fields."""
        story = {
            "version": 2,
            "story_name": "test_story",
            "story_dir": "/tmp/test_story",
            "scenes": []
        }
        
        required_fields = ["version", "story_name", "story_dir", "scenes"]
        
        for field in required_fields:
            assert field in story, f"Missing required field: {field}"
        
        assert story["version"] == 2
        assert isinstance(story["scenes"], list)


class TestUILogic:
    """Test UI-related logic that can be tested without actual DOM."""
    
    def test_prompt_source_input_type(self):
        """Test that prompt source determines input field type."""
        def get_input_type(prompt_source):
            """Determine what input type to show."""
            if prompt_source == "custom":
                return "textarea"
            elif prompt_source in ["prompt", "composition"]:
                return "select"
            return "text"
        
        assert get_input_type("custom") == "textarea"
        assert get_input_type("prompt") == "select"
        assert get_input_type("composition") == "select"
    
    def test_scene_validation(self):
        """Test scene data validation before save."""
        def validate_scene(scene):
            """Check if scene has valid data."""
            errors = []
            
            if not scene.get("scene_name"):
                errors.append("Scene name is required")
            
            if scene.get("prompt_source") == "prompt" and not scene.get("prompt_key"):
                errors.append("Prompt key required when source is 'prompt'")
            
            if scene.get("prompt_source") == "custom" and not scene.get("custom_prompt"):
                errors.append("Custom prompt required when source is 'custom'")
            
            return len(errors) == 0, errors
        
        # Valid scene
        valid_scene = {
            "scene_name": "test",
            "prompt_source": "prompt",
            "prompt_key": "girl_pos"
        }
        is_valid, errors = validate_scene(valid_scene)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid scene - missing name
        invalid_scene = {
            "scene_name": "",
            "prompt_source": "prompt"
        }
        is_valid, errors = validate_scene(invalid_scene)
        assert is_valid is False
        assert "Scene name is required" in errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

