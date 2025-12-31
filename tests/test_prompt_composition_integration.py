"""
Integration tests for prompt composition system
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Use unified import approach from conftest
from conftest import import_test_module

# Import from prompt_models using unified helper
prompt_models = import_test_module("prompt_models.py")
PromptMetadata = prompt_models.PromptMetadata
PromptCollection = prompt_models.PromptCollection


class MockLibber:
    """Mock Libber for testing substitution"""
    def __init__(self, libs):
        self.libs = libs
    
    def substitute(self, text):
        result = text
        for key, value in self.libs.items():
            result = result.replace(f"%{key}%", value)
        return result


class MockLibberManager:
    """Mock LibberStateManager for testing"""
    def __init__(self):
        self.libbers = {}
    
    def get_libber(self, name):
        return self.libbers.get(name)
    
    def add_libber(self, name, libs):
        self.libbers[name] = MockLibber(libs)


class TestPromptCollectionCompose:
    """Test PromptCollection.compose_prompts() method"""
    
    def test_compose_single_output(self):
        """Test composing a single output prompt"""
        collection = PromptCollection()
        collection.add_prompt("intro", "A beautiful scene")
        collection.add_prompt("subject", "with a warrior")
        collection.add_prompt("quality", "best quality")
        
        composition_map = {
            "main": ["intro", "subject", "quality"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert "main" in result
        assert result["main"] == "A beautiful scene with a warrior best quality"
    
    def test_compose_multiple_outputs(self):
        """Test composing multiple output prompts"""
        collection = PromptCollection()
        collection.add_prompt("char", "beautiful woman")
        collection.add_prompt("quality_high", "best quality")
        collection.add_prompt("quality_low", "normal quality")
        collection.add_prompt("style", "epic style")
        
        composition_map = {
            "image_prompt": ["char", "quality_high"],
            "video_high": ["char", "quality_high", "style"],
            "video_low": ["char", "quality_low", "style"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert len(result) == 3
        assert result["image_prompt"] == "beautiful woman best quality"
        assert result["video_high"] == "beautiful woman best quality epic style"
        assert result["video_low"] == "beautiful woman normal quality epic style"
    
    def test_compose_with_missing_keys(self):
        """Test that missing keys are skipped gracefully"""
        collection = PromptCollection()
        collection.add_prompt("existing", "value")
        
        composition_map = {
            "output": ["existing", "missing_key", "another_missing"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert result["output"] == "value"
    
    def test_compose_empty_output(self):
        """Test composing with no prompts"""
        collection = PromptCollection()
        
        composition_map = {
            "empty": []
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert result["empty"] == ""
    
    def test_compose_with_libber_substitution(self):
        """Test composition with libber processing"""
        collection = PromptCollection()
        collection.add_prompt(
            "templated",
            "A %quality% %type% character",
            processing_type="libber",
            libber_name="char_lib"
        )
        collection.add_prompt("suffix", "in epic pose")
        
        libber_manager = MockLibberManager()
        libber_manager.add_libber("char_lib", {
            "quality": "magnificent",
            "type": "warrior"
        })
        
        composition_map = {
            "main": ["templated", "suffix"]
        }
        
        result = collection.compose_prompts(composition_map, libber_manager)
        
        assert result["main"] == "A magnificent warrior character in epic pose"
    
    def test_compose_mixed_raw_and_libber(self):
        """Test composition with mix of raw and libber prompts"""
        collection = PromptCollection()
        collection.add_prompt("raw1", "beautiful scene", processing_type="raw")
        collection.add_prompt(
            "lib1",
            "%quality% lighting",
            processing_type="libber",
            libber_name="test_lib"
        )
        collection.add_prompt("raw2", "epic composition", processing_type="raw")
        
        libber_manager = MockLibberManager()
        libber_manager.add_libber("test_lib", {"quality": "dramatic"})
        
        composition_map = {
            "output": ["raw1", "lib1", "raw2"]
        }
        
        result = collection.compose_prompts(composition_map, libber_manager)
        
        assert result["output"] == "beautiful scene dramatic lighting epic composition"
    
    def test_compose_without_libber_manager(self):
        """Test that libber prompts pass through unchanged without manager"""
        collection = PromptCollection()
        collection.add_prompt(
            "templated",
            "A %quality% character",
            processing_type="libber",
            libber_name="missing_lib"
        )
        
        composition_map = {
            "output": ["templated"]
        }
        
        # No libber_manager provided
        result = collection.compose_prompts(composition_map)
        
        # Should return original value without substitution
        assert result["output"] == "A %quality% character"


class TestPromptCompositionSerialization:
    """Test that composition maps serialize correctly"""
    
    def test_composition_map_roundtrip(self):
        """Test composition map can be serialized and deserialized"""
        original_map = {
            "qwen_main": ["girl_pos", "male_pos", "quality"],
            "video_high": ["wan_prompt", "style"],
            "video_low": ["wan_low_prompt"]
        }
        
        # Serialize
        json_str = json.dumps(original_map)
        
        # Deserialize
        restored_map = json.loads(json_str)
        
        assert restored_map == original_map
    
    def test_composition_with_unicode(self):
        """Test composition with unicode prompt keys"""
        collection = PromptCollection()
        collection.add_prompt("日本語", "Japanese text")
        collection.add_prompt("emoji", "🎨 artistic")
        
        composition_map = {
            "output": ["日本語", "emoji"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert result["output"] == "Japanese text 🎨 artistic"


class TestLegacyPromptMigration:
    """Test migration of legacy prompt files"""
    
    def test_migrate_typical_legacy_file(self):
        """Test migrating a typical legacy prompts.json"""
        legacy_data = {
            "girl_pos": "beautiful woman, long hair",
            "male_pos": "handsome man, strong",
            "wan_prompt": "cinematic quality, 8k",
            "wan_low_prompt": "normal quality",
            "four_image_prompt": "four subjects in scene"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # All prompts should be migrated
        assert len(collection.prompts) == 5
        
        # Check values preserved
        assert collection.prompts["girl_pos"].value == "beautiful woman, long hair"
        assert collection.prompts["male_pos"].value == "handsome man, strong"
        
        # All should be raw type
        for prompt in collection.prompts.values():
            assert prompt.processing_type == "raw"
            assert prompt.libber_name is None
        
        # v1_backup should be preserved
        assert collection.v1_backup == legacy_data
    
    def test_compose_migrated_prompts(self):
        """Test that migrated prompts can be composed"""
        legacy_data = {
            "girl_pos": "woman",
            "male_pos": "man"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        composition_map = {
            "combined": ["girl_pos", "male_pos"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert result["combined"] == "woman man"
    
    def test_v2_format_detection(self):
        """Test that v2 format is correctly detected and loaded"""
        v2_data = {
            "version": 2,
            "prompts": {
                "test": {
                    "value": "test value",
                    "processing_type": "libber",
                    "libber_name": "test_lib"
                }
            }
        }
        
        collection = PromptCollection.from_dict(v2_data)
        
        assert collection.version == 2
        assert collection.prompts["test"].value == "test value"
        assert collection.prompts["test"].processing_type == "libber"
        assert collection.prompts["test"].libber_name == "test_lib"


class TestPromptCollectionFileOperations:
    """Test saving and loading PromptCollection"""
    
    def test_save_and_load_collection(self):
        """Test saving collection to JSON and loading it back"""
        collection = PromptCollection()
        collection.add_prompt(
            "test1",
            "test value",
            processing_type="libber",
            libber_name="my_lib",
            category="character"
        )
        collection.add_prompt("test2", "raw value", category="scene")
        
        # Convert to dict (as would be saved to JSON)
        data = collection.to_dict()
        
        # Restore from dict
        restored = PromptCollection.from_dict(data)
        
        assert len(restored.prompts) == 2
        assert restored.prompts["test1"].value == "test value"
        assert restored.prompts["test1"].processing_type == "libber"
        assert restored.prompts["test1"].libber_name == "my_lib"
        assert restored.prompts["test1"].category == "character"
        assert restored.prompts["test2"].value == "raw value"
        assert restored.prompts["test2"].processing_type == "raw"


class TestPromptCompositionWorkflows:
    """Test realistic workflow scenarios"""
    
    def test_image_generation_workflow(self):
        """Test typical image generation composition"""
        collection = PromptCollection()
        collection.add_prompt("char1", "beautiful woman")
        collection.add_prompt("char2", "handsome man")
        collection.add_prompt("setting", "in a garden")
        collection.add_prompt("quality", "masterpiece, best quality")
        collection.add_prompt("style", "oil painting style")
        
        # Image workflow: combine characters, setting, quality
        composition_map = {
            "qwen_main": ["char1", "char2", "setting", "quality", "style"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        expected = "beautiful woman handsome man in a garden masterpiece, best quality oil painting style"
        assert result["qwen_main"] == expected
    
    def test_video_generation_workflow(self):
        """Test video generation with high and low quality outputs"""
        collection = PromptCollection()
        collection.add_prompt("subject", "warrior in battle")
        collection.add_prompt("quality_high", "8k, cinematic")
        collection.add_prompt("quality_low", "720p")
        collection.add_prompt("motion", "dynamic camera movement")
        
        # Video workflow: different compositions for high/low
        composition_map = {
            "video_high": ["subject", "quality_high", "motion"],
            "video_low": ["subject", "quality_low"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert result["video_high"] == "warrior in battle 8k, cinematic dynamic camera movement"
        assert result["video_low"] == "warrior in battle 720p"
    
    def test_multi_image_workflow(self):
        """Test workflow with multiple specialized outputs"""
        collection = PromptCollection()
        collection.add_prompt("subject", "character")
        collection.add_prompt("close_up", "close up face")
        collection.add_prompt("wide", "wide angle shot")
        collection.add_prompt("quality", "best quality")
        
        composition_map = {
            "image_1": ["subject", "close_up", "quality"],
            "image_2": ["subject", "wide", "quality"],
            "image_3": ["subject", "quality"],
            "image_4": ["subject", "quality"]
        }
        
        result = collection.compose_prompts(composition_map)
        
        assert len(result) == 4
        assert "close up face" in result["image_1"]
        assert "wide angle shot" in result["image_2"]
    
    def test_libber_enhanced_workflow(self):
        """Test workflow using libber templates for consistency"""
        collection = PromptCollection()
        collection.add_prompt(
            "character",
            "A %appearance% %type%",
            processing_type="libber",
            libber_name="char_lib"
        )
        collection.add_prompt(
            "setting",
            "in a %environment%",
            processing_type="libber",
            libber_name="scene_lib"
        )
        collection.add_prompt("quality", "masterpiece")
        
        # Setup libbers
        libber_manager = MockLibberManager()
        libber_manager.add_libber("char_lib", {
            "appearance": "beautiful",
            "type": "warrior woman"
        })
        libber_manager.add_libber("scene_lib", {
            "environment": "mystical forest"
        })
        
        composition_map = {
            "main": ["character", "setting", "quality"]
        }
        
        result = collection.compose_prompts(composition_map, libber_manager)
        
        assert result["main"] == "A beautiful warrior woman in a mystical forest masterpiece"
