#!/usr/bin/env python3
"""
Test script for the new generic mask system.

This script validates:
1. MaskDefinition class functionality
2. JSON serialization/deserialization
3. Mask loading and saving
4. Backward compatibility with legacy masks

Note: This test extracts and tests only the mask system components
without requiring the full ComfyUI environment.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Standalone implementations for testing
class MaskType(str, Enum):
    TRANSPARENT = "transparent"
    COLOR = "color"


@dataclass
class MaskDefinition:
    name: str
    type: MaskType
    has_background: bool
    color: Optional[Tuple[int, int, int]] = None
    
    def validate(self):
        if self.type == MaskType.TRANSPARENT and self.color is not None:
            raise ValueError(f"TRANSPARENT mask '{self.name}' should not have a color")
        if self.type == MaskType.COLOR and self.color is None:
            raise ValueError(f"COLOR mask '{self.name}' must have a color specified")
        if self.color is not None:
            for i, component in enumerate(self.color):
                if not (0 <= component <= 255):
                    raise ValueError(f"Color component {i} must be 0-255, got {component}")
    
    def get_filename(self) -> str:
        suffix = "_bkgd" if self.has_background else "_no_bkgd"
        return f"{self.name}_mask{suffix}.png"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "has_background": self.has_background,
            "color": list(self.color) if self.color else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MaskDefinition':
        color_data = data.get("color")
        color = tuple(color_data) if color_data else None
        return cls(
            name=data["name"],
            type=MaskType(data["type"]),
            has_background=data["has_background"],
            color=color
        )


def save_masks_json(scene_dir: str, masks: Dict[str, MaskDefinition]) -> bool:
    try:
        masks_path = Path(scene_dir) / "masks.json"
        masks_data = {
            "version": 1,
            "masks": [mask.to_dict() for mask in masks.values()]
        }
        with open(masks_path, 'w') as f:
            json.dump(masks_data, f, indent=2)
        return True
    except Exception:
        return False


def load_masks_json(scene_dir: str) -> Optional[Dict[str, MaskDefinition]]:
    try:
        masks_path = Path(scene_dir) / "masks.json"
        if not masks_path.exists():
            return None
        
        with open(masks_path, 'r') as f:
            data = json.load(f)
        
        masks = {}
        for mask_data in data.get("masks", []):
            mask = MaskDefinition.from_dict(mask_data)
            masks[mask.name] = mask
        
        return masks
    except Exception:
        return None


def test_mask_definition():
    """Test MaskDefinition creation and validation"""
    print("Testing MaskDefinition...")
    
    # Test transparent mask
    mask1 = MaskDefinition(
        name="character",
        type=MaskType.TRANSPARENT,
        has_background=True,
        color=None
    )
    mask1.validate()
    assert mask1.get_filename() == "character_mask_bkgd.png"
    print("✓ Transparent mask with background")
    
    # Test transparent mask without background
    mask2 = MaskDefinition(
        name="character",
        type=MaskType.TRANSPARENT,
        has_background=False,
        color=None
    )
    mask2.validate()
    assert mask2.get_filename() == "character_mask_no_bkgd.png"
    print("✓ Transparent mask without background")
    
    # Test color mask
    mask3 = MaskDefinition(
        name="region",
        type=MaskType.COLOR,
        has_background=True,
        color=(255, 0, 0)
    )
    mask3.validate()
    assert mask3.get_filename() == "region_mask_bkgd.png"
    print("✓ Color mask")
    
    # Test invalid: transparent with color
    try:
        mask4 = MaskDefinition(
            name="invalid",
            type=MaskType.TRANSPARENT,
            has_background=True,
            color=(255, 0, 0)  # Should be None
        )
        mask4.validate()
        print("✗ Should have raised error for transparent mask with color")
        return False
    except ValueError:
        print("✓ Correctly rejects transparent mask with color")
    
    # Test invalid: color without color
    try:
        mask5 = MaskDefinition(
            name="invalid",
            type=MaskType.COLOR,
            has_background=True,
            color=None  # Should have RGB
        )
        mask5.validate()
        print("✗ Should have raised error for color mask without color")
        return False
    except ValueError:
        print("✓ Correctly rejects color mask without color")
    
    return True


def test_json_serialization():
    """Test JSON serialization and deserialization"""
    print("\nTesting JSON serialization...")
    
    # Create mask definitions
    masks = {
        "hero": MaskDefinition("hero", MaskType.TRANSPARENT, True, None),
        "villain": MaskDefinition("villain", MaskType.TRANSPARENT, False, None),
        "environment": MaskDefinition("environment", MaskType.COLOR, True, (0, 255, 0))
    }
    
    # Convert to dict
    masks_dict = {name: mask.to_dict() for name, mask in masks.items()}
    
    # Convert to JSON and back
    json_str = json.dumps(masks_dict, indent=2)
    loaded_dict = json.loads(json_str)
    
    # Reconstruct from dict
    reconstructed = {name: MaskDefinition.from_dict(data) for name, data in loaded_dict.items()}
    
    # Verify
    for name in masks:
        original = masks[name]
        recon = reconstructed[name]
        assert original.name == recon.name
        assert original.type == recon.type
        assert original.has_background == recon.has_background
        assert original.color == recon.color
    
    print("✓ JSON serialization/deserialization works correctly")
    return True


def test_masks_json_io():
    """Test loading and saving masks.json files"""
    print("\nTesting masks.json I/O...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test masks
        test_masks = {
            "character1": MaskDefinition("character1", MaskType.TRANSPARENT, True, None),
            "character2": MaskDefinition("character2", MaskType.TRANSPARENT, False, None),
            "background": MaskDefinition("background", MaskType.COLOR, True, (100, 150, 200))
        }
        
        # Save to temp directory
        success = save_masks_json(tmpdir, test_masks)
        assert success, "Failed to save masks.json"
        print("✓ Successfully saved masks.json")
        
        # Verify file exists
        masks_json_path = Path(tmpdir) / "masks.json"
        assert masks_json_path.exists(), "masks.json file not created"
        print("✓ masks.json file created")
        
        # Load back
        loaded_masks = load_masks_json(tmpdir)
        assert loaded_masks is not None, "Failed to load masks.json"
        assert len(loaded_masks) == 3, f"Expected 3 masks, got {len(loaded_masks)}"
        print("✓ Successfully loaded masks.json")
        
        # Verify contents
        for name in test_masks:
            assert name in loaded_masks, f"Mask '{name}' not found in loaded data"
            original = test_masks[name]
            loaded = loaded_masks[name]
            assert original.name == loaded.name
            assert original.type == loaded.type
            assert original.has_background == loaded.has_background
            assert original.color == loaded.color
        
        print("✓ All mask definitions preserved correctly")
        
        # Test loading from non-existent directory
        nonexistent = load_masks_json("/nonexistent/path")
        assert nonexistent is None, "Should return None for nonexistent path"
        print("✓ Correctly handles nonexistent path")
    
    return True


def test_legacy_compatibility():
    """Test backward compatibility with legacy mask system"""
    print("\nTesting legacy compatibility...")
    
    # Test legacy mask name mappings
    legacy_names = ["girl", "male", "combined"]
    for name in legacy_names:
        mask = MaskDefinition(name, MaskType.TRANSPARENT, True, None)
        filename = mask.get_filename()
        expected = f"{name}_mask_bkgd.png"
        assert filename == expected, f"Expected {expected}, got {filename}"
    
    print("✓ Legacy mask names produce correct filenames")
    
    # Test no-background variants
    for name in legacy_names:
        mask = MaskDefinition(name, MaskType.TRANSPARENT, False, None)
        filename = mask.get_filename()
        expected = f"{name}_mask_no_bkgd.png"
        assert filename == expected, f"Expected {expected}, got {filename}"
    
    print("✓ Legacy no-background variants correct")
    
    return True


def test_validation_edge_cases():
    """Test edge cases in mask validation"""
    print("\nTesting validation edge cases...")
    
    # Test RGB bounds
    try:
        mask = MaskDefinition("test", MaskType.COLOR, True, (256, 0, 0))  # Out of range
        mask.validate()
        print("✗ Should reject RGB value > 255")
        return False
    except ValueError:
        print("✓ Rejects RGB value > 255")
    
    try:
        mask = MaskDefinition("test", MaskType.COLOR, True, (-1, 0, 0))  # Negative
        mask.validate()
        print("✗ Should reject negative RGB value")
        return False
    except ValueError:
        print("✓ Rejects negative RGB value")
    
    # Test valid edge values
    mask = MaskDefinition("test", MaskType.COLOR, True, (0, 0, 0))  # Black
    mask.validate()
    print("✓ Accepts RGB (0, 0, 0)")
    
    mask = MaskDefinition("test", MaskType.COLOR, True, (255, 255, 255))  # White
    mask.validate()
    print("✓ Accepts RGB (255, 255, 255)")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Generic Mask System - Test Suite")
    print("=" * 80)
    
    tests = [
        ("MaskDefinition", test_mask_definition),
        ("JSON Serialization", test_json_serialization),
        ("masks.json I/O", test_masks_json_io),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Validation Edge Cases", test_validation_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
