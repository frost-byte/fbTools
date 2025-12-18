"""
Tests for PromptCollection and PromptMetadata classes.

Tests cover:
- V1 to V2 migration with v1_backup preservation
- CRUD operations (add, remove, get prompts)
- Serialization and deserialization
- Backward compatibility
- Edge cases and error handling
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Import from the standalone prompt_models module - no ComfyUI dependencies needed
from prompt_models import PromptCollection, PromptMetadata


class TestPromptMetadata:
    """Test PromptMetadata class functionality."""
    
    def test_create_basic_prompt_metadata(self):
        """Test creating a basic PromptMetadata with just value."""
        metadata = PromptMetadata(value="test prompt")
        assert metadata.value == "test prompt"
        assert metadata.category is None
        assert metadata.description is None
        assert metadata.tags is None
    
    def test_create_full_prompt_metadata(self):
        """Test creating PromptMetadata with all fields."""
        metadata = PromptMetadata(
            value="detailed prompt",
            category="character",
            description="Main character prompt",
            tags=["hero", "protagonist"]
        )
        assert metadata.value == "detailed prompt"
        assert metadata.category == "character"
        assert metadata.description == "Main character prompt"
        assert metadata.tags == ["hero", "protagonist"]


class TestPromptCollectionBasics:
    """Test basic PromptCollection functionality."""
    
    def test_create_empty_collection(self):
        """Test creating an empty PromptCollection."""
        collection = PromptCollection()
        assert collection.version == 2
        assert collection.v1_backup is None
        assert len(collection.prompts) == 0
    
    def test_add_prompt(self):
        """Test adding a prompt to collection."""
        collection = PromptCollection()
        collection.add_prompt("test_key", "test value")
        
        assert "test_key" in collection.prompts
        assert collection.get_prompt_value("test_key") == "test value"
    
    def test_add_prompt_with_metadata(self):
        """Test adding a prompt with full metadata."""
        collection = PromptCollection()
        collection.add_prompt(
            "character",
            "A brave warrior",
            category="main",
            description="Hero description",
            tags=["hero", "warrior"]
        )
        
        prompt = collection.prompts["character"]
        assert prompt.value == "A brave warrior"
        assert prompt.category == "main"
        assert prompt.description == "Hero description"
        assert prompt.tags == ["hero", "warrior"]
    
    def test_update_existing_prompt(self):
        """Test updating an existing prompt."""
        collection = PromptCollection()
        collection.add_prompt("key1", "value1")
        collection.add_prompt("key1", "value2")
        
        assert collection.get_prompt_value("key1") == "value2"
    
    def test_remove_prompt(self):
        """Test removing a prompt."""
        collection = PromptCollection()
        collection.add_prompt("key1", "value1")
        
        removed = collection.remove_prompt("key1")
        assert removed is True
        assert "key1" not in collection.prompts
    
    def test_remove_nonexistent_prompt(self):
        """Test removing a prompt that doesn't exist."""
        collection = PromptCollection()
        removed = collection.remove_prompt("nonexistent")
        assert removed is False
    
    def test_get_nonexistent_prompt(self):
        """Test getting a prompt that doesn't exist."""
        collection = PromptCollection()
        value = collection.get_prompt_value("nonexistent")
        assert value is None
    
    def test_list_prompt_names(self):
        """Test listing all prompt names."""
        collection = PromptCollection()
        collection.add_prompt("zebra", "value")
        collection.add_prompt("alpha", "value")
        collection.add_prompt("beta", "value")
        
        names = collection.list_prompt_names()
        assert names == ["alpha", "beta", "zebra"]  # Should be sorted


class TestPromptCollectionMigration:
    """Test V1 to V2 migration functionality."""
    
    def test_migrate_from_v1_basic(self):
        """Test basic V1 to V2 migration."""
        legacy_data = {
            "girl_pos": "A beautiful woman",
            "male_pos": "A handsome man",
            "wan_prompt": "High quality"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Check version and backup
        assert collection.version == 2
        assert collection.v1_backup == legacy_data
        
        # Check prompts migrated correctly
        assert collection.get_prompt_value("girl_pos") == "A beautiful woman"
        assert collection.get_prompt_value("male_pos") == "A handsome man"
        assert collection.get_prompt_value("wan_prompt") == "High quality"
    
    def test_migrate_preserves_all_v1_fields(self):
        """Test that migration preserves all V1 fields."""
        legacy_data = {
            "girl_pos": "test1",
            "male_pos": "test2",
            "wan_prompt": "test3",
            "wan_low_prompt": "test4",
            "four_image_prompt": "test5",
            "custom_field": "test6"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # All fields should be preserved
        for key, value in legacy_data.items():
            assert collection.get_prompt_value(key) == value
        
        # v1_backup should be identical
        assert collection.v1_backup == legacy_data
    
    def test_migrate_with_non_string_values(self):
        """Test migration handles non-string values gracefully."""
        legacy_data = {
            "girl_pos": "valid string",
            "invalid_number": 123,
            "invalid_dict": {"nested": "value"},
            "male_pos": "another valid string"
        }
        
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Only string values should be migrated to prompts
        assert collection.get_prompt_value("girl_pos") == "valid string"
        assert collection.get_prompt_value("male_pos") == "another valid string"
        assert collection.get_prompt_value("invalid_number") is None
        assert collection.get_prompt_value("invalid_dict") is None
        
        # But v1_backup should preserve everything
        assert collection.v1_backup == legacy_data


class TestPromptCollectionSerialization:
    """Test serialization and deserialization."""
    
    def test_to_dict_basic(self):
        """Test basic serialization to dict."""
        collection = PromptCollection()
        collection.add_prompt("key1", "value1")
        
        data = collection.to_dict()
        
        assert data["version"] == 2
        assert "prompts" in data
        assert "key1" in data["prompts"]
        assert data["prompts"]["key1"]["value"] == "value1"
    
    def test_to_dict_with_metadata(self):
        """Test serialization with full metadata."""
        collection = PromptCollection()
        collection.add_prompt(
            "key1",
            "value1",
            category="test",
            description="Test prompt",
            tags=["tag1", "tag2"]
        )
        
        data = collection.to_dict()
        prompt_data = data["prompts"]["key1"]
        
        assert prompt_data["value"] == "value1"
        assert prompt_data["category"] == "test"
        assert prompt_data["description"] == "Test prompt"
        assert prompt_data["tags"] == ["tag1", "tag2"]
    
    def test_to_dict_preserves_v1_backup(self):
        """Test that serialization preserves v1_backup."""
        legacy_data = {"girl_pos": "test", "male_pos": "test2"}
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        data = collection.to_dict()
        
        assert "v1_backup" in data
        assert data["v1_backup"] == legacy_data
    
    def test_from_dict_v2_format(self):
        """Test deserialization from V2 format."""
        data = {
            "version": 2,
            "prompts": {
                "key1": {"value": "value1"},
                "key2": {
                    "value": "value2",
                    "category": "test",
                    "description": "Test",
                    "tags": ["tag1"]
                }
            }
        }
        
        collection = PromptCollection.from_dict(data)
        
        assert collection.version == 2
        assert collection.get_prompt_value("key1") == "value1"
        assert collection.get_prompt_value("key2") == "value2"
        assert collection.prompts["key2"].category == "test"
    
    def test_from_dict_v1_format_auto_migrates(self):
        """Test that from_dict auto-migrates V1 format."""
        data = {
            "girl_pos": "test1",
            "male_pos": "test2"
        }
        
        collection = PromptCollection.from_dict(data)
        
        assert collection.version == 2
        assert collection.v1_backup == data
        assert collection.get_prompt_value("girl_pos") == "test1"
    
    def test_from_dict_v1_format_without_version(self):
        """Test auto-migration when version field missing."""
        data = {
            "wan_prompt": "test",
            "wan_low_prompt": "test2"
        }
        
        collection = PromptCollection.from_dict(data)
        
        assert collection.version == 2
        assert collection.v1_backup is not None
    
    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        original = PromptCollection()
        original.add_prompt("key1", "value1", category="cat1")
        original.add_prompt("key2", "value2", tags=["tag1"])
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = PromptCollection.from_dict(data)
        
        # Should be equivalent
        assert restored.version == original.version
        assert restored.get_prompt_value("key1") == "value1"
        assert restored.get_prompt_value("key2") == "value2"
        assert restored.prompts["key1"].category == "cat1"
        assert restored.prompts["key2"].tags == ["tag1"]


class TestPromptCollectionBackwardCompatibility:
    """Test backward compatibility scenarios."""
    
    def test_v1_backup_immutability_on_edits(self):
        """Test that v1_backup is not modified by subsequent edits."""
        legacy_data = {"girl_pos": "original", "male_pos": "original2"}
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Store original backup
        original_backup = collection.v1_backup.copy()
        
        # Make edits
        collection.add_prompt("girl_pos", "modified")
        collection.add_prompt("new_key", "new_value")
        collection.remove_prompt("male_pos")
        
        # v1_backup should be unchanged
        assert collection.v1_backup == original_backup
        
        # But current prompts should reflect changes
        assert collection.get_prompt_value("girl_pos") == "modified"
        assert collection.get_prompt_value("new_key") == "new_value"
        assert collection.get_prompt_value("male_pos") is None
    
    def test_access_legacy_fields_after_migration(self):
        """Test that legacy field access still works after migration."""
        legacy_data = {
            "girl_pos": "girl prompt",
            "male_pos": "male prompt",
            "wan_prompt": "wan prompt"
        }
        collection = PromptCollection.from_legacy_dict(legacy_data)
        
        # Should be able to access all legacy fields
        assert collection.get_prompt_value("girl_pos") == "girl prompt"
        assert collection.get_prompt_value("male_pos") == "male prompt"
        assert collection.get_prompt_value("wan_prompt") == "wan prompt"


class TestPromptCollectionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_key(self):
        """Test adding prompt with empty key."""
        collection = PromptCollection()
        collection.add_prompt("", "value")
        
        assert "" in collection.prompts
        assert collection.get_prompt_value("") == "value"
    
    def test_empty_value(self):
        """Test adding prompt with empty value."""
        collection = PromptCollection()
        collection.add_prompt("key", "")
        
        assert collection.get_prompt_value("key") == ""
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        collection = PromptCollection()
        collection.add_prompt("emoji_key", "🎨 Art prompt 日本語")
        
        assert collection.get_prompt_value("emoji_key") == "🎨 Art prompt 日本語"
    
    def test_large_prompt_value(self):
        """Test handling of very large prompt values."""
        large_value = "A" * 10000
        collection = PromptCollection()
        collection.add_prompt("large", large_value)
        
        assert collection.get_prompt_value("large") == large_value
    
    def test_many_prompts(self):
        """Test collection with many prompts."""
        collection = PromptCollection()
        
        # Add 1000 prompts
        for i in range(1000):
            collection.add_prompt(f"key_{i}", f"value_{i}")
        
        assert len(collection.prompts) == 1000
        assert collection.get_prompt_value("key_500") == "value_500"


class TestPromptCollectionFileOperations:
    """Test file-based operations (save/load via JSON)."""
    
    def test_save_and_load_v2_format(self):
        """Test saving and loading V2 format to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts.json")
            
            # Create and save
            collection = PromptCollection()
            collection.add_prompt("key1", "value1")
            collection.add_prompt("key2", "value2", category="test")
            
            with open(filepath, 'w') as f:
                json.dump(collection.to_dict(), f)
            
            # Load and verify
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            loaded = PromptCollection.from_dict(data)
            assert loaded.get_prompt_value("key1") == "value1"
            assert loaded.get_prompt_value("key2") == "value2"
            assert loaded.prompts["key2"].category == "test"
    
    def test_load_v1_file_and_migrate(self):
        """Test loading a V1 format file and auto-migrating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts_v1.json")
            
            # Create V1 format file
            v1_data = {
                "girl_pos": "v1 girl prompt",
                "male_pos": "v1 male prompt",
                "wan_prompt": "v1 wan prompt"
            }
            
            with open(filepath, 'w') as f:
                json.dump(v1_data, f)
            
            # Load and verify migration
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            collection = PromptCollection.from_dict(data)
            
            assert collection.version == 2
            assert collection.v1_backup == v1_data
            assert collection.get_prompt_value("girl_pos") == "v1 girl prompt"
    
    def test_save_migrated_collection_preserves_backup(self):
        """Test that saving a migrated collection preserves v1_backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prompts.json")
            
            # Create from legacy
            legacy_data = {"girl_pos": "original", "male_pos": "original2"}
            collection = PromptCollection.from_legacy_dict(legacy_data)
            
            # Make some changes
            collection.add_prompt("new_key", "new_value")
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(collection.to_dict(), f)
            
            # Load and verify backup preserved
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "v1_backup" in data
            assert data["v1_backup"] == legacy_data


class TestPromptCollectionIntegration:
    """Integration tests simulating real usage scenarios."""
    
    def test_typical_migration_workflow(self):
        """Test a typical user migration workflow."""
        # Step 1: Start with V1 data (loaded from old file)
        v1_data = {
            "girl_pos": "A beautiful character",
            "male_pos": "A strong character",
            "wan_prompt": "High quality, detailed",
            "wan_low_prompt": "Simple, basic",
            "four_image_prompt": "Four panel layout"
        }
        
        # Step 2: Auto-migrate on load
        collection = PromptCollection.from_dict(v1_data)
        
        # Step 3: Verify all old data accessible
        assert collection.get_prompt_value("girl_pos") == "A beautiful character"
        assert collection.get_prompt_value("four_image_prompt") == "Four panel layout"
        
        # Step 4: Add new custom prompts
        collection.add_prompt(
            "villain",
            "A menacing antagonist",
            category="character",
            tags=["villain", "antagonist"]
        )
        
        # Step 5: Modify existing prompt
        collection.add_prompt("girl_pos", "A beautiful and brave character")
        
        # Step 6: Save (would be to file in real usage)
        data = collection.to_dict()
        
        # Verify final state
        assert data["version"] == 2
        assert data["v1_backup"] == v1_data  # Original preserved
        assert "villain" in data["prompts"]
        assert data["prompts"]["girl_pos"]["value"] == "A beautiful and brave character"
    
    def test_collaborative_workflow(self):
        """Test workflow where prompts are shared and merged."""
        # User A creates prompts
        collection_a = PromptCollection()
        collection_a.add_prompt("hero", "The brave hero")
        collection_a.add_prompt("setting", "Ancient castle")
        
        # User B creates different prompts
        collection_b = PromptCollection()
        collection_b.add_prompt("villain", "The evil overlord")
        collection_b.add_prompt("setting", "Dark dungeon")  # Conflict!
        
        # Merge by loading A's data and adding B's unique prompts
        data_a = collection_a.to_dict()
        merged = PromptCollection.from_dict(data_a)
        
        # Add B's unique prompts (would need custom merge logic for conflicts)
        merged.add_prompt("villain", "The evil overlord")
        
        # Verify merged state
        assert merged.get_prompt_value("hero") == "The brave hero"
        assert merged.get_prompt_value("villain") == "The evil overlord"
        assert len(merged.prompts) == 3  # hero, setting, villain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
