"""
Tests for prompt processing and migration features
"""

import pytest

# Use unified import approach from conftest
from conftest import import_test_module

# Import from prompt_models using unified helper
prompt_models = import_test_module("prompt_models.py")
PromptMetadata = prompt_models.PromptMetadata
PromptCollection = prompt_models.PromptCollection


class TestPromptMetadataSimplified:
    """Test simplified PromptMetadata with processing fields only"""
    
    def test_create_with_processing_type(self):
        """Test creating prompt with processing type"""
        prompt = PromptMetadata(
            value="A %quality% character",
            processing_type="libber",
            libber_name="my_libber"
        )
        assert prompt.value == "A %quality% character"
        assert prompt.processing_type == "libber"
        assert prompt.libber_name == "my_libber"
    
    def test_defaults(self):
        """Test default values"""
        prompt = PromptMetadata(value="test prompt")
        assert prompt.processing_type == "raw"
        assert prompt.libber_name is None
        assert prompt.category is None


class TestLegacyMigration:
    """Test migration from legacy prompt format"""
    
    def test_migrate_legacy_prompts(self):
        """Test migrating legacy prompt dictionary"""
        legacy = {
            "girl_pos": "beautiful woman",
            "male_pos": "handsome man",
            "wan_prompt": "high quality",
            "wan_low_prompt": "low quality",
            "four_image_prompt": "four images"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy)
        
        assert collection.version == 2
        assert len(collection.prompts) == 5
        assert collection.v1_backup == legacy
        
        # All prompts should be raw type
        for key in ["girl_pos", "male_pos", "wan_prompt", "wan_low_prompt", "four_image_prompt"]:
            assert collection.prompts[key].processing_type == "raw"
            assert collection.prompts[key].value == legacy[key]
    
    def test_migrate_empty_values(self):
        """Test migration skips empty values"""
        legacy = {
            "girl_pos": "beautiful woman",
            "male_pos": "",
            "wan_prompt": None,
        }
        
        collection = PromptCollection.from_legacy_dict(legacy)
        
        # Should only have girl_pos (non-empty string)
        assert len(collection.prompts) == 1
        assert "girl_pos" in collection.prompts
        assert "male_pos" not in collection.prompts
        assert "wan_prompt" not in collection.prompts


class TestSerializationWithProcessingFields:
    """Test serialization/deserialization with processing fields"""
    
    def test_to_dict_includes_processing_fields(self):
        """Test that to_dict includes processing fields"""
        collection = PromptCollection()
        collection.add_prompt(
            "test_prompt",
            "A %quality% character",
            processing_type="libber",
            libber_name="my_libber"
        )
        
        data = collection.to_dict()
        
        prompt_data = data["prompts"]["test_prompt"]
        assert prompt_data["value"] == "A %quality% character"
        assert prompt_data["processing_type"] == "libber"
        assert prompt_data["libber_name"] == "my_libber"
    
    def test_from_dict_loads_processing_fields(self):
        """Test that from_dict properly loads processing fields"""
        data = {
            "version": 2,
            "prompts": {
                "test": {
                    "value": "test value",
                    "processing_type": "libber",
                    "libber_name": "test_libber"
                }
            }
        }
        
        collection = PromptCollection.from_dict(data)
        
        prompt = collection.prompts["test"]
        assert prompt.value == "test value"
        assert prompt.processing_type == "libber"
        assert prompt.libber_name == "test_libber"
    
    def test_roundtrip_serialization(self):
        """Test that data survives roundtrip serialization"""
        original = PromptCollection()
        original.add_prompt(
            "p1",
            "test",
            processing_type="libber",
            libber_name="my_lib"
        )
        
        data = original.to_dict()
        restored = PromptCollection.from_dict(data)
        
        assert restored.prompts["p1"].processing_type == "libber"
        assert restored.prompts["p1"].libber_name == "my_lib"


class TestAddPromptWithProcessingFields:
    """Test add_prompt method with processing parameters"""
    
    def test_add_raw_prompt(self):
        """Test adding a raw prompt"""
        collection = PromptCollection()
        collection.add_prompt(
            "test",
            "test value",
            processing_type="raw"
        )
        
        assert collection.prompts["test"].processing_type == "raw"
        assert collection.prompts["test"].libber_name is None
    
    def test_add_libber_prompt(self):
        """Test adding a libber-processed prompt"""
        collection = PromptCollection()
        collection.add_prompt(
            "test",
            "A %quality% character",
            processing_type="libber",
            libber_name="char_libber"
        )
        
        assert collection.prompts["test"].processing_type == "libber"
        assert collection.prompts["test"].libber_name == "char_libber"


class TestPromptCollectionHelpers:
    """Test helper methods on PromptCollection"""
    
    def test_get_prompt_metadata(self):
        """Test getting full metadata for a prompt"""
        collection = PromptCollection()
        collection.add_prompt(
            "test",
            "test value",
            category="character",
            processing_type="libber",
            libber_name="test_lib"
        )
        
        metadata = collection.get_prompt_metadata("test")
        assert metadata is not None
        assert metadata.value == "test value"
        assert metadata.category == "character"
        assert metadata.processing_type == "libber"
        assert metadata.libber_name == "test_lib"
    
    def test_get_prompts_by_category(self):
        """Test filtering prompts by category"""
        collection = PromptCollection()
        collection.add_prompt("p1", "value1", category="character")
        collection.add_prompt("p2", "value2", category="scene")
        collection.add_prompt("p3", "value3", category="character")
        
        char_prompts = collection.get_prompts_by_category("character")
        assert len(char_prompts) == 2
        assert "p1" in char_prompts
        assert "p3" in char_prompts
        assert "p2" not in char_prompts
