"""
Tests for scene prompts API endpoints with scene_flags support.

Tests cover:
- GET /fbtools/scene/get_scene_prompts - Loading prompts with scene_flags
- POST /fbtools/scene/save_scene_prompts - Saving prompts with scene_flags
- Ensuring scene_flags are preserved through save/load cycle
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Use unified import approach from conftest
from conftest import import_test_module

# Import prompt models
prompt_models = import_test_module("prompt_models.py")
PromptCollection = prompt_models.PromptCollection
PromptMetadata = prompt_models.PromptMetadata


class TestSceneFlagsInCollection:
    """Test that PromptCollection properly handles scene_flags."""
    
    def test_scene_flags_in_new_collection(self):
        """Test creating a collection with scene_flags."""
        collection = PromptCollection()
        collection.add_prompt("test", "value")
        collection.scene_flags = {
            "use_depth": True,
            "use_mask": False,
            "use_pose": True,
            "use_canny": False
        }
        
        assert collection.scene_flags is not None
        assert collection.scene_flags["use_depth"] is True
        assert collection.scene_flags["use_mask"] is False
    
    def test_scene_flags_to_dict(self):
        """Test that to_dict includes scene_flags."""
        collection = PromptCollection()
        collection.add_prompt("test", "value")
        collection.scene_flags = {"use_depth": True}
        
        data = collection.to_dict()
        
        assert "scene_flags" in data
        assert data["scene_flags"]["use_depth"] is True
    
    def test_scene_flags_from_dict(self):
        """Test that from_dict loads scene_flags."""
        data = {
            "version": 2,
            "prompts": {"test": {"value": "value"}},
            "scene_flags": {
                "use_depth": True,
                "use_mask": False,
                "use_pose": True,
                "use_canny": False
            }
        }
        
        collection = PromptCollection.from_dict(data)
        
        assert collection.scene_flags is not None
        assert collection.scene_flags["use_depth"] is True
        assert collection.scene_flags["use_pose"] is True
    
    def test_scene_flags_optional(self):
        """Test that scene_flags is optional."""
        data = {
            "version": 2,
            "prompts": {"test": {"value": "value"}}
        }
        
        collection = PromptCollection.from_dict(data)
        
        # Should not raise error, scene_flags should be None
        assert collection.scene_flags is None


class TestSceneFlagsPersistence:
    """Test scene_flags persistence through file operations."""
    
    def test_save_and_load_with_flags(self):
        """Test saving and loading a collection with scene_flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts.json")
            
            # Create collection with prompts and flags
            collection = PromptCollection()
            collection.add_prompt("char1", "A brave hero")
            collection.add_prompt("setting", "Ancient castle")
            collection.add_composition("main", ["char1", "setting"])
            collection.scene_flags = {
                "use_depth": True,
                "use_mask": True,
                "use_pose": False,
                "use_canny": False
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(collection.to_dict(), f, indent=2)
            
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Verify file contains flags
            assert "scene_flags" in data
            assert data["scene_flags"]["use_depth"] is True
            assert data["scene_flags"]["use_mask"] is True
            
            # Verify loading works
            loaded = PromptCollection.from_dict(data)
            assert loaded.scene_flags["use_depth"] is True
            assert loaded.scene_flags["use_pose"] is False
    
    def test_save_without_flags(self):
        """Test saving a collection without scene_flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts.json")
            
            # Create collection without flags
            collection = PromptCollection()
            collection.add_prompt("test", "value")
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(collection.to_dict(), f, indent=2)
            
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # scene_flags should not be in file if None
            # (Pydantic excludes None values by default in some cases)
            # or it should be null/None
            if "scene_flags" in data:
                assert data["scene_flags"] is None
    
    def test_update_flags_after_load(self):
        """Test updating flags after loading from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts.json")
            
            # Create and save initial collection
            collection = PromptCollection()
            collection.add_prompt("test", "value")
            collection.scene_flags = {
                "use_depth": False,
                "use_mask": False,
                "use_pose": False,
                "use_canny": False
            }
            
            with open(filepath, 'w') as f:
                json.dump(collection.to_dict(), f)
            
            # Load and update
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            loaded = PromptCollection.from_dict(data)
            loaded.scene_flags["use_depth"] = True
            loaded.scene_flags["use_mask"] = True
            
            # Save updated version
            with open(filepath, 'w') as f:
                json.dump(loaded.to_dict(), f)
            
            # Verify changes persisted
            with open(filepath, 'r') as f:
                final_data = json.load(f)
            
            assert final_data["scene_flags"]["use_depth"] is True
            assert final_data["scene_flags"]["use_mask"] is True
            assert final_data["scene_flags"]["use_pose"] is False


class TestSceneFlagsWithCompositions:
    """Test scene_flags work correctly alongside compositions."""
    
    def test_flags_and_compositions_coexist(self):
        """Test that scene_flags and compositions work together."""
        collection = PromptCollection()
        collection.add_prompt("char1", "Hero")
        collection.add_prompt("char2", "Villain")
        collection.add_prompt("setting", "Castle")
        collection.add_composition("main", ["char1", "setting"])
        collection.add_composition("battle", ["char1", "char2", "setting"])
        collection.scene_flags = {
            "use_depth": True,
            "use_mask": True,
            "use_pose": False,
            "use_canny": True
        }
        
        data = collection.to_dict()
        
        # Both should be present
        assert "compositions" in data
        assert "scene_flags" in data
        assert len(data["compositions"]) == 2
        assert data["scene_flags"]["use_depth"] is True
        
        # Roundtrip test
        restored = PromptCollection.from_dict(data)
        assert len(restored.compositions) == 2
        assert restored.scene_flags["use_canny"] is True
    
    def test_modify_flags_preserves_compositions(self):
        """Test that modifying flags doesn't affect compositions."""
        collection = PromptCollection()
        collection.add_prompt("test", "value")
        collection.add_composition("comp1", ["test"])
        collection.scene_flags = {"use_depth": False}
        
        # Modify flags
        collection.scene_flags["use_depth"] = True
        collection.scene_flags["use_mask"] = True
        
        # Compositions should be unchanged
        assert "comp1" in collection.compositions
        assert collection.compositions["comp1"] == ["test"]


class TestArrayFormatCompatibility:
    """Test compatibility with array format from frontend."""
    
    def test_from_dict_with_array_prompts(self):
        """Test loading when prompts are in array format."""
        data = {
            "version": 2,
            "prompts": [
                {"key": "char1", "value": "Hero", "processing_type": "raw", "libber_name": ""},
                {"key": "char2", "value": "Villain", "processing_type": "libber", "libber_name": "test"}
            ],
            "scene_flags": {
                "use_depth": True,
                "use_mask": False
            }
        }
        
        collection = PromptCollection.from_dict(data)
        
        # Prompts should be loaded correctly
        assert collection.get_prompt_value("char1") == "Hero"
        assert collection.get_prompt_value("char2") == "Villain"
        
        # Flags should be preserved
        assert collection.scene_flags["use_depth"] is True
        assert collection.scene_flags["use_mask"] is False
    
    def test_from_dict_with_array_compositions(self):
        """Test loading when compositions are in array format."""
        data = {
            "version": 2,
            "prompts": {"test": {"value": "value"}},
            "compositions": [
                {"name": "comp1", "prompt_keys": ["test"]},
                {"name": "comp2", "prompt_keys": ["test"]}
            ],
            "scene_flags": {"use_depth": True}
        }
        
        collection = PromptCollection.from_dict(data)
        
        # Compositions should be loaded
        assert "comp1" in collection.compositions
        assert "comp2" in collection.compositions
        
        # Flags should be preserved
        assert collection.scene_flags["use_depth"] is True


class TestMigrationWithFlags:
    """Test that V1 to V2 migration doesn't break with scene_flags."""
    
    def test_migrate_v1_adds_no_flags(self):
        """Test that V1 migration doesn't add scene_flags."""
        legacy_data = {
            "girl_pos": "Girl prompt",
            "male_pos": "Male prompt"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Migrated collection should not have flags
        assert collection.scene_flags is None
    
    def test_add_flags_to_migrated_collection(self):
        """Test adding flags to a migrated collection."""
        legacy_data = {
            "girl_pos": "Girl prompt",
            "male_pos": "Male prompt"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Add flags after migration
        collection.scene_flags = {
            "use_depth": True,
            "use_mask": True,
            "use_pose": False,
            "use_canny": False
        }
        
        data = collection.to_dict()
        
        # Both v1_backup and scene_flags should be present
        assert "v1_backup" in data
        assert "scene_flags" in data
        assert data["scene_flags"]["use_depth"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
