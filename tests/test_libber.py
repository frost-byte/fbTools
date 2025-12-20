"""
Unit tests for Libber system
"""
import pytest
import json
import os
from pathlib import Path


# Copy the Libber class for testing (avoiding ComfyUI dependencies)
class Libber:
    """Test copy of Libber class"""
    
    def __init__(self, lib_dict=None, delimiter="%", max_depth=10):
        self.lib_dict = lib_dict or {}
        self.delimiter = delimiter
        self.max_depth = max_depth
    
    def add_lib(self, key: str, value: str):
        """Add or update a lib entry"""
        # Normalize key (lowercase, replace spaces/hyphens with underscores)
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        self.lib_dict[normalized_key] = value
    
    def remove_lib(self, key: str):
        """Remove a lib entry"""
        if key in self.lib_dict:
            del self.lib_dict[key]
    
    def get_lib(self, key: str):
        """Get a lib value"""
        return self.lib_dict.get(key)
    
    def list_libs(self):
        """Get sorted list of all keys"""
        return sorted(self.lib_dict.keys())
    
    def substitute(self, text: str, depth: int = 0):
        """Apply substitutions recursively"""
        if depth >= self.max_depth:
            return text
        
        if not text or not self.lib_dict:
            return text
        
        result = text
        changed = False
        
        for key, value in self.lib_dict.items():
            placeholder = f"{self.delimiter}{key}{self.delimiter}"
            if placeholder in result:
                result = result.replace(placeholder, value)
                changed = True
        
        # Recurse if we made changes
        if changed:
            return self.substitute(result, depth + 1)
        
        return result
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "delimiter": self.delimiter,
            "max_depth": self.max_depth,
            "lib_dict": self.lib_dict
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            lib_dict=data.get("lib_dict", {}),
            delimiter=data.get("delimiter", "%"),
            max_depth=data.get("max_depth", 10)
        )
    
    def save(self, filepath: str):
        """Save to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class TestLibberBasics:
    """Test basic Libber functionality"""
    
    def test_create_empty_libber(self):
        """Should create an empty Libber with defaults"""
        libber = Libber()
        assert libber.lib_dict == {}
        assert libber.delimiter == "%"
        assert libber.max_depth == 10
    
    def test_create_with_custom_delimiter(self):
        """Should create Libber with custom delimiter"""
        libber = Libber(delimiter="<<>>")
        assert libber.delimiter == "<<>>"
    
    def test_add_lib(self):
        """Should add a lib entry"""
        libber = Libber()
        libber.add_lib("warrior", "strong fighter")
        assert libber.get_lib("warrior") == "strong fighter"
    
    def test_add_lib_normalizes_key(self):
        """Should normalize key to lowercase with underscores"""
        libber = Libber()
        libber.add_lib("My Key", "value")
        assert libber.get_lib("my_key") == "value"
    
    def test_update_existing_lib(self):
        """Should update an existing lib entry"""
        libber = Libber()
        libber.add_lib("warrior", "fighter")
        libber.add_lib("warrior", "strong fighter")
        assert libber.get_lib("warrior") == "strong fighter"
    
    def test_remove_lib(self):
        """Should remove a lib entry"""
        libber = Libber()
        libber.add_lib("warrior", "fighter")
        libber.remove_lib("warrior")
        assert libber.get_lib("warrior") is None
    
    def test_list_libs(self):
        """Should return sorted list of keys"""
        libber = Libber()
        libber.add_lib("zebra", "value")
        libber.add_lib("alpha", "value")
        libber.add_lib("beta", "value")
        assert libber.list_libs() == ["alpha", "beta", "zebra"]


class TestLibberSubstitution:
    """Test substitution functionality"""
    
    def test_simple_substitution(self):
        """Should substitute a single placeholder"""
        libber = Libber({"quality": "best quality"})
        result = libber.substitute("A warrior, %quality%")
        assert result == "A warrior, best quality"
    
    def test_multiple_substitutions(self):
        """Should substitute multiple placeholders"""
        libber = Libber({
            "quality": "best quality",
            "style": "realistic"
        })
        result = libber.substitute("%quality%, %style% image")
        assert result == "best quality, realistic image"
    
    def test_recursive_substitution(self):
        """Should recursively substitute nested references"""
        libber = Libber({
            "quality": "best quality",
            "base": "%quality%, detailed",
            "final": "%base%, ultra realistic"
        })
        result = libber.substitute("%final%")
        assert result == "best quality, detailed, ultra realistic"
    
    def test_max_depth_prevents_infinite_loop(self):
        """Should stop at max_depth to prevent infinite loops"""
        libber = Libber({
            "a": "%b%",
            "b": "%a%"
        }, max_depth=5)
        # Should not hang, should stop after 5 iterations
        result = libber.substitute("%a%")
        assert "%a%" in result or "%b%" in result  # Should contain unresolved placeholder
    
    def test_no_substitution_without_placeholder(self):
        """Should not modify text without placeholders"""
        libber = Libber({"quality": "best"})
        result = libber.substitute("A simple text")
        assert result == "A simple text"
    
    def test_custom_delimiter(self):
        """Should use custom delimiter for substitution"""
        libber = Libber({"quality": "best quality"}, delimiter="<<")
        result = libber.substitute("Image with <<quality<<")
        assert result == "Image with best quality"
    
    def test_partial_substitution(self):
        """Should substitute only matching keys"""
        libber = Libber({"quality": "best"})
        result = libber.substitute("%quality% and %unknown%")
        assert result == "best and %unknown%"
    
    def test_empty_text(self):
        """Should handle empty text gracefully"""
        libber = Libber({"quality": "best"})
        assert libber.substitute("") == ""
        assert libber.substitute(None) == None
    
    def test_complex_nested_substitution(self):
        """Should handle complex nested substitutions"""
        libber = Libber({
            "chunky": "incredibly thick, and %yummy%",
            "yummy": "delicious",
            "character": "A %chunky% warrior"
        })
        result = libber.substitute("Look at this %character%!")
        assert result == "Look at this A incredibly thick, and delicious warrior!"


class TestLibberSerialization:
    """Test JSON serialization"""
    
    def test_to_dict(self):
        """Should convert to dictionary"""
        libber = Libber({
            "quality": "best quality",
            "style": "realistic"
        }, delimiter="<<>>", max_depth=20)
        
        data = libber.to_dict()
        assert data["delimiter"] == "<<>>"
        assert data["max_depth"] == 20
        assert data["lib_dict"]["quality"] == "best quality"
        assert data["lib_dict"]["style"] == "realistic"
    
    def test_from_dict(self):
        """Should create from dictionary"""
        data = {
            "delimiter": "<<>>",
            "max_depth": 20,
            "lib_dict": {
                "quality": "best quality",
                "style": "realistic"
            }
        }
        libber = Libber.from_dict(data)
        assert libber.delimiter == "<<>>"
        assert libber.max_depth == 20
        assert libber.get_lib("quality") == "best quality"
    
    def test_roundtrip_serialization(self):
        """Should maintain data through to_dict() -> from_dict()"""
        original = Libber({
            "quality": "best quality",
            "warrior": "strong fighter"
        }, delimiter="%%", max_depth=15)
        
        data = original.to_dict()
        restored = Libber.from_dict(data)
        
        assert restored.delimiter == original.delimiter
        assert restored.max_depth == original.max_depth
        assert restored.lib_dict == original.lib_dict


class TestLibberFileOperations:
    """Test file save/load"""
    
    def test_save_and_load(self, tmp_path):
        """Should save to and load from JSON file"""
        filepath = tmp_path / "test_libber.json"
        
        # Create and save
        original = Libber({
            "quality": "best quality",
            "warrior": "strong fighter"
        }, delimiter="<<>>", max_depth=20)
        original.save(str(filepath))
        
        # Load and verify
        loaded = Libber.load(str(filepath))
        assert loaded.delimiter == "<<>>"
        assert loaded.max_depth == 20
        assert loaded.get_lib("quality") == "best quality"
        assert loaded.get_lib("warrior") == "strong fighter"
    
    def test_save_creates_directory(self, tmp_path):
        """Should create directory if it doesn't exist"""
        filepath = tmp_path / "nested" / "dir" / "test.json"
        
        libber = Libber({"test": "value"})
        libber.save(str(filepath))
        
        assert filepath.exists()
        loaded = Libber.load(str(filepath))
        assert loaded.get_lib("test") == "value"


class TestLibberEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_key(self):
        """Should handle empty string as key"""
        libber = Libber()
        libber.add_lib("", "empty key value")
        assert libber.get_lib("") == "empty key value"
    
    def test_empty_value(self):
        """Should handle empty string as value"""
        libber = Libber()
        libber.add_lib("empty", "")
        assert libber.get_lib("empty") == ""
    
    def test_unicode_content(self):
        """Should support unicode characters"""
        libber = Libber()
        libber.add_lib("japanese", "日本語テキスト")
        libber.add_lib("emoji", "🎨✨🌟")
        
        assert libber.get_lib("japanese") == "日本語テキスト"
        result = libber.substitute("%emoji%")
        assert result == "🎨✨🌟"
    
    def test_large_value(self):
        """Should handle large text values"""
        large_text = "a" * 10000
        libber = Libber()
        libber.add_lib("large", large_text)
        result = libber.substitute("%large%")
        assert result == large_text
    
    def test_many_libs(self):
        """Should scale to many lib entries"""
        libber = Libber()
        for i in range(1000):
            libber.add_lib(f"key{i}", f"value{i}")
        
        assert len(libber.list_libs()) == 1000
        assert libber.get_lib("key500") == "value500"
    
    def test_special_characters_in_value(self):
        """Should handle special characters in values"""
        libber = Libber()
        libber.add_lib("special", "value with $pecial ch@rs & symbols!")
        result = libber.substitute("%special%")
        assert result == "value with $pecial ch@rs & symbols!"


class TestLibberIntegration:
    """Test realistic usage scenarios"""
    
    def test_character_presets_workflow(self):
        """Should support character preset workflow"""
        libber = Libber({
            "quality": "masterpiece, best quality, ultra detailed",
            "base_male": "handsome man, strong",
            "base_female": "beautiful woman, elegant",
            "warrior": "%base_male%, armor, sword",
            "mage": "%base_female%, robes, staff",
            "hero": "%warrior%, %quality%"
        })
        
        result = libber.substitute("Portrait of a %hero%")
        expected = "Portrait of a handsome man, strong, armor, sword, masterpiece, best quality, ultra detailed"
        assert result == expected
    
    def test_scene_components_workflow(self):
        """Should support scene building workflow"""
        libber = Libber({
            "lighting_day": "bright sunlight, clear sky",
            "lighting_night": "moonlight, stars",
            "location_castle": "medieval castle, stone walls",
            "scene": "%location_castle%, %lighting_day%"
        })
        
        result = libber.substitute("%scene%, knights training")
        assert result == "medieval castle, stone walls, bright sunlight, clear sky, knights training"
    
    def test_update_and_resubstitute(self):
        """Should support updating libs and reapplying"""
        libber = Libber({"style": "realistic"})
        
        result1 = libber.substitute("Image in %style% style")
        assert result1 == "Image in realistic style"
        
        # Update the lib
        libber.add_lib("style", "anime")
        result2 = libber.substitute("Image in %style% style")
        assert result2 == "Image in anime style"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
