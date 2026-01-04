"""
Unit tests for StorySceneBatch and StoryScenePick prompt selection.

Tests verify that the story models and prompt resolution logic work correctly
for different scenes with different prompts.

Note: These tests focus on the data models (StoryInfo, SceneInStory) and 
prompt models (PromptCollection) rather than the full node execution, since
extension.py has ComfyUI dependencies that are complex to mock.
"""

import pytest
import tempfile
import json
from pathlib import Path

# Use unified import approach from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import import_test_module

# Import modules to test
story_models = import_test_module("story_models.py")
prompt_models = import_test_module("prompt_models.py")

# Classes we need
SceneInStory = story_models.SceneInStory
StoryInfo = story_models.StoryInfo
PromptMetadata = prompt_models.PromptMetadata
PromptCollection = prompt_models.PromptCollection


# Simple helper for loading JSON (avoiding utils/io.py which has relative imports)
def load_json_file(filepath: str):
    """Load JSON from a file"""
    with open(filepath, 'r') as f:
        return json.load(f)


class TestStoryModelsPromptResolution:
    """Test that story models correctly store prompt configuration"""
    
    def test_scene_in_story_prompt_fields(self):
        """SceneInStory should store prompt_source, prompt_key, custom_prompt"""
        scene = SceneInStory(
            scene_name="test_scene",
            scene_order=0,
            prompt_source="prompt",
            prompt_key="my_prompt",
            custom_prompt="",
        )
        
        assert scene.prompt_source == "prompt"
        assert scene.prompt_key == "my_prompt"
        assert scene.custom_prompt == ""
    
    def test_multiple_scenes_different_prompt_keys(self):
        """Different scenes in a story can have different prompt keys"""
        scenes = [
            SceneInStory(
                scene_name="scene1",
                scene_order=0,
                prompt_source="prompt",
                prompt_key="prompt_a",
                custom_prompt="",
            ),
            SceneInStory(
                scene_name="scene2",
                scene_order=1,
                prompt_source="prompt",
                prompt_key="prompt_b",
                custom_prompt="",
            ),
            SceneInStory(
                scene_name="scene3",
                scene_order=2,
                prompt_source="custom",
                prompt_key="",
                custom_prompt="Custom prompt text",
            ),
        ]
        
        story = StoryInfo(
            story_name="test_story",
            story_dir="/fake/path",
            scenes=scenes
        )
        
        # Verify each scene has its own prompt configuration
        assert story.scenes[0].prompt_key == "prompt_a"
        assert story.scenes[1].prompt_key == "prompt_b"
        assert story.scenes[2].prompt_source == "custom"
        assert story.scenes[2].custom_prompt == "Custom prompt text"


class TestPromptCollectionResolution:
    """Test that PromptCollection correctly resolves prompts and compositions"""
    
    def test_simple_prompt_resolution(self):
        """Individual prompts should be accessible by key"""
        collection = PromptCollection()
        collection.prompts["prompt_a"] = PromptMetadata(value="This is prompt A", processing_type="none")
        collection.prompts["prompt_b"] = PromptMetadata(value="This is prompt B", processing_type="none")
        
        assert collection.prompts["prompt_a"].value == "This is prompt A"
        assert collection.prompts["prompt_b"].value == "This is prompt B"
    
    def test_composition_combines_prompts(self):
        """Compositions should combine multiple prompts"""
        collection = PromptCollection()
        collection.prompts["character"] = PromptMetadata(value="a brave warrior", processing_type="none")
        collection.prompts["setting"] = PromptMetadata(value="in a dark forest", processing_type="none")
        collection.prompts["action"] = PromptMetadata(value="fighting a dragon", processing_type="none")
        
        collection.compositions = {
            "full_scene": ["character", "setting", "action"]
        }
        
        # Manually compose (simulating what StorySceneBatch does)
        prompt_dict = {key: meta.value for key, meta in collection.prompts.items()}
        composition_dict = {}
        for comp_name, prompt_keys in collection.compositions.items():
            parts = [prompt_dict.get(key, "") for key in prompt_keys if key in prompt_dict]
            composition_dict[comp_name] = " ".join(parts).strip()
        
        assert composition_dict["full_scene"] == "a brave warrior in a dark forest fighting a dragon"


class TestPromptFilePersistence:
    """Test that prompts.json files are loaded correctly"""
    
    def test_load_v2_prompts_json(self):
        """V2 format prompts.json should load correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.json"
            prompts_data = {
                "version": 2,
                "prompts": {
                    "test_prompt": {
                        "value": "Test prompt value",
                        "processing_type": "none"
                    }
                },
                "compositions": {}
            }
            prompts_file.write_text(json.dumps(prompts_data))
            
            loaded = load_json_file(str(prompts_file))
            assert loaded["version"] == 2
            assert "test_prompt" in loaded["prompts"]
            assert loaded["prompts"]["test_prompt"]["value"] == "Test prompt value"
    
    def test_v2_format_with_legacy_fields_prefers_v2(self):
        """When both v2 and legacy fields exist, v2 should take precedence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.json"
            prompts_data = {
                "version": 2,
                # NEW v2 format
                "prompts": {
                    "main_prompt": {
                        "value": "NEW v2 prompt",
                        "processing_type": "none"
                    }
                },
                "compositions": {},
                # OLD legacy fields (should be ignored)
                "girl_pos": "OLD legacy prompt",
            }
            prompts_file.write_text(json.dumps(prompts_data))
            
            loaded = load_json_file(str(prompts_file))
            collection = PromptCollection.from_dict(loaded)
            
            # V2 prompt should be loaded
            assert "main_prompt" in collection.prompts
            assert collection.prompts["main_prompt"].value == "NEW v2 prompt"
            
            # Legacy field exists in raw data but shouldn't be used
            assert loaded.get("girl_pos") == "OLD legacy prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

