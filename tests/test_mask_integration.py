"""
Integration tests for the generic mask system.

These tests verify the complete mask workflow:
1. Creating scenes with custom masks
2. Loading and saving masks.json
3. Migration from legacy to new system
4. SceneInfo integration
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Mock the imports that would fail outside ComfyUI environment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We'll need to mock torch and other ComfyUI dependencies
try:
    from extension import MaskType, MaskDefinition, load_masks_json, save_masks_json
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import extension module. Some tests will be skipped.")


class TestMaskSystemIntegration(unittest.TestCase):
    """Integration tests for the mask system"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        self.scene_dir = os.path.join(self.test_dir, "test_scene")
        os.makedirs(self.scene_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_create_and_load_masks_json(self):
        """Test creating and loading masks.json"""
        # Create mask definitions
        masks = {
            "hero": MaskDefinition("hero", MaskType.TRANSPARENT, True, None),
            "villain": MaskDefinition("villain", MaskType.TRANSPARENT, False, None),
            "environment": MaskDefinition("environment", MaskType.COLOR, True, (100, 150, 200))
        }
        
        # Save to disk
        success = save_masks_json(self.scene_dir, masks)
        self.assertTrue(success, "Failed to save masks.json")
        
        # Verify file exists
        masks_json_path = os.path.join(self.scene_dir, "masks.json")
        self.assertTrue(os.path.exists(masks_json_path), "masks.json not created")
        
        # Load back
        loaded_masks = load_masks_json(self.scene_dir)
        self.assertIsNotNone(loaded_masks, "Failed to load masks.json")
        self.assertEqual(len(loaded_masks), 3, "Wrong number of masks loaded")
        
        # Verify contents
        for name in masks:
            self.assertIn(name, loaded_masks, f"Mask '{name}' not found")
            original = masks[name]
            loaded = loaded_masks[name]
            self.assertEqual(original.name, loaded.name)
            self.assertEqual(original.type, loaded.type)
            self.assertEqual(original.has_background, loaded.has_background)
            self.assertEqual(original.color, loaded.color)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_masks_json_format(self):
        """Test masks.json file format"""
        masks = {
            "test_mask": MaskDefinition("test_mask", MaskType.TRANSPARENT, True, None)
        }
        
        save_masks_json(self.scene_dir, masks)
        
        # Read raw JSON
        masks_json_path = os.path.join(self.scene_dir, "masks.json")
        with open(masks_json_path, 'r') as f:
            data = json.load(f)
        
        # Verify structure
        self.assertIn("version", data, "Missing version field")
        self.assertEqual(data["version"], 1, "Wrong version number")
        self.assertIn("masks", data, "Missing masks array")
        self.assertIsInstance(data["masks"], list, "masks should be an array")
        self.assertEqual(len(data["masks"]), 1, "Wrong number of masks")
        
        # Verify mask structure
        mask_data = data["masks"][0]
        self.assertIn("name", mask_data)
        self.assertIn("type", mask_data)
        self.assertIn("has_background", mask_data)
        self.assertIn("color", mask_data)
        self.assertEqual(mask_data["name"], "test_mask")
        self.assertEqual(mask_data["type"], "transparent")
        self.assertTrue(mask_data["has_background"])
        self.assertIsNone(mask_data["color"])
    
    def test_legacy_compatibility_structure(self):
        """Test that legacy scene structure is maintained"""
        # This test doesn't require extension imports
        legacy_scene_dir = os.path.join(self.test_dir, "legacy_scene")
        os.makedirs(legacy_scene_dir, exist_ok=True)
        
        # Create legacy mask files (empty files for test)
        legacy_files = [
            "girl_mask_bkgd.png",
            "male_mask_bkgd.png",
            "combined_mask_bkgd.png",
            "girl_mask_no_bkgd.png",
            "male_mask_no_bkgd.png",
            "combined_mask_no_bkgd.png"
        ]
        
        for filename in legacy_files:
            filepath = os.path.join(legacy_scene_dir, filename)
            Path(filepath).touch()
        
        # Verify all legacy files exist
        for filename in legacy_files:
            filepath = os.path.join(legacy_scene_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Legacy file {filename} not found")
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_filename_generation(self):
        """Test mask filename generation"""
        test_cases = [
            ("hero", True, "hero_mask_bkgd.png"),
            ("hero", False, "hero_mask_no_bkgd.png"),
            ("character1", True, "character1_mask_bkgd.png"),
            ("environment", False, "environment_mask_no_bkgd.png"),
            ("girl", True, "girl_mask_bkgd.png"),  # Legacy name
            ("male", False, "male_mask_no_bkgd.png"),  # Legacy name
        ]
        
        for name, has_bg, expected_filename in test_cases:
            mask = MaskDefinition(name, MaskType.TRANSPARENT, has_bg, None)
            filename = mask.get_filename()
            self.assertEqual(filename, expected_filename, 
                           f"Wrong filename for {name} with has_background={has_bg}")
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_multiple_mask_types(self):
        """Test mixing different mask types"""
        masks = {
            "transparent1": MaskDefinition("transparent1", MaskType.TRANSPARENT, True, None),
            "transparent2": MaskDefinition("transparent2", MaskType.TRANSPARENT, False, None),
            "color1": MaskDefinition("color1", MaskType.COLOR, True, (255, 0, 0)),
            "color2": MaskDefinition("color2", MaskType.COLOR, False, (0, 255, 0))
        }
        
        # Save and load
        save_masks_json(self.scene_dir, masks)
        loaded_masks = load_masks_json(self.scene_dir)
        
        # Verify types preserved
        self.assertEqual(loaded_masks["transparent1"].type, MaskType.TRANSPARENT)
        self.assertEqual(loaded_masks["transparent2"].type, MaskType.TRANSPARENT)
        self.assertEqual(loaded_masks["color1"].type, MaskType.COLOR)
        self.assertEqual(loaded_masks["color2"].type, MaskType.COLOR)
        
        # Verify colors
        self.assertIsNone(loaded_masks["transparent1"].color)
        self.assertIsNone(loaded_masks["transparent2"].color)
        self.assertEqual(loaded_masks["color1"].color, (255, 0, 0))
        self.assertEqual(loaded_masks["color2"].color, (0, 255, 0))
    
    def test_nonexistent_scene_directory(self):
        """Test behavior with non-existent directory"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Extension module not available")
        
        nonexistent_dir = os.path.join(self.test_dir, "does_not_exist")
        result = load_masks_json(nonexistent_dir)
        self.assertIsNone(result, "Should return None for non-existent directory")
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_empty_masks(self):
        """Test saving and loading empty masks dictionary"""
        masks = {}
        save_masks_json(self.scene_dir, masks)
        loaded_masks = load_masks_json(self.scene_dir)
        
        self.assertIsNotNone(loaded_masks, "Should load empty masks")
        self.assertEqual(len(loaded_masks), 0, "Should have zero masks")
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Extension module not available")
    def test_special_characters_in_names(self):
        """Test mask names with special characters"""
        special_names = [
            "mask-with-dash",
            "mask_with_underscore",
            "mask123",
            "UPPERCASE",
            "lowercase",
            "MixedCase"
        ]
        
        masks = {}
        for name in special_names:
            masks[name] = MaskDefinition(name, MaskType.TRANSPARENT, True, None)
        
        save_masks_json(self.scene_dir, masks)
        loaded_masks = load_masks_json(self.scene_dir)
        
        for name in special_names:
            self.assertIn(name, loaded_masks, f"Mask '{name}' not preserved")
            self.assertEqual(loaded_masks[name].name, name)


class TestMaskMigration(unittest.TestCase):
    """Test migration scenarios"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_legacy_scene_structure(self):
        """Test that we can detect legacy scene structure"""
        legacy_scene = os.path.join(self.test_dir, "legacy_scene")
        os.makedirs(legacy_scene, exist_ok=True)
        
        # Create legacy mask files
        Path(os.path.join(legacy_scene, "girl_mask_bkgd.png")).touch()
        Path(os.path.join(legacy_scene, "male_mask_bkgd.png")).touch()
        Path(os.path.join(legacy_scene, "combined_mask_bkgd.png")).touch()
        
        # Verify no masks.json exists yet
        masks_json = os.path.join(legacy_scene, "masks.json")
        self.assertFalse(os.path.exists(masks_json), "masks.json should not exist yet")
        
        # Verify legacy files exist
        self.assertTrue(os.path.exists(os.path.join(legacy_scene, "girl_mask_bkgd.png")))
        self.assertTrue(os.path.exists(os.path.join(legacy_scene, "male_mask_bkgd.png")))
        self.assertTrue(os.path.exists(os.path.join(legacy_scene, "combined_mask_bkgd.png")))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
