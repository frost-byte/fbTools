"""
Unit tests for StorySceneImageSave functionality

These tests demonstrate how the refactored code can be tested without 
needing actual file I/O or image tensors.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Use unified import approach from conftest
from conftest import import_test_module

# Import the testable components using the unified helper
scene_image_save = import_test_module("utils/scene_image_save.py")
SceneImageSaveConfig = scene_image_save.SceneImageSaveConfig
ImageSaver = scene_image_save.ImageSaver
select_scene_descriptor = scene_image_save.select_scene_descriptor
generate_preview_text = scene_image_save.generate_preview_text


class TestSceneImageSaveConfig:
    """Test the configuration data class"""
    
    def test_generate_filename_basic(self):
        """Test basic filename generation"""
        config = SceneImageSaveConfig(
            scene_name="test_scene",
            scene_order=1,
            target_dir="/tmp/test",
            image_format="png",
            quality=95
        )
        assert config.generate_filename() == "001_test_scene.png"
    
    def test_generate_filename_with_spaces(self):
        """Test filename generation with spaces in scene name"""
        config = SceneImageSaveConfig(
            scene_name="My Test Scene",
            scene_order=42,
            target_dir="/tmp/test",
            image_format="jpg",
            quality=85
        )
        assert config.generate_filename() == "042_my_test_scene.jpg"
    
    def test_generate_filename_with_hyphens(self):
        """Test filename generation with hyphens in scene name"""
        config = SceneImageSaveConfig(
            scene_name="scene-with-dashes",
            scene_order=5,
            target_dir="/tmp/test",
            image_format="webp",
            quality=90
        )
        assert config.generate_filename() == "005_scene_with_dashes.webp"
    
    def test_generate_filename_uppercase(self):
        """Test filename generation converts to lowercase"""
        config = SceneImageSaveConfig(
            scene_name="UPPERCASE_SCENE",
            scene_order=0,
            target_dir="/tmp/test",
            image_format="png",
            quality=95
        )
        assert config.generate_filename() == "000_uppercase_scene.png"
    
    def test_generate_filepath(self):
        """Test full filepath generation"""
        config = SceneImageSaveConfig(
            scene_name="scene_one",
            scene_order=3,
            target_dir="/output/test",
            image_format="png",
            quality=95
        )
        expected = os.path.join("/output/test", "003_scene_one.png")
        assert config.generate_filepath() == expected
    
    def test_from_descriptor_valid(self):
        """Test creating config from valid descriptor"""
        descriptor = {
            "scene_name": "my_scene",
            "scene_order": 10,
            "job_input_dir": "/path/to/input",
            "job_output_dir": "/path/to/output"
        }
        
        config = SceneImageSaveConfig.from_descriptor(
            descriptor, 
            scene_index=0, 
            image_format="jpg", 
            quality=80
        )
        
        assert config is not None
        assert config.scene_name == "my_scene"
        assert config.scene_order == 10
        assert config.target_dir == "/path/to/input"  # Prefers input_dir
        assert config.image_format == "jpg"
        assert config.quality == 80
    
    def test_from_descriptor_fallback_to_output(self):
        """Test fallback to output_dir when input_dir is missing"""
        descriptor = {
            "scene_name": "my_scene",
            "scene_order": 10,
            "job_input_dir": "",
            "job_output_dir": "/path/to/output"
        }
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, 0, "png", 95)
        
        assert config is not None
        assert config.target_dir == "/path/to/output"
    
    def test_from_descriptor_missing_scene_name(self):
        """Test returns None when scene_name is missing"""
        descriptor = {
            "scene_order": 10,
            "job_input_dir": "/path/to/input"
        }
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, 0, "png", 95)
        assert config is None
    
    def test_from_descriptor_missing_directories(self):
        """Test returns None when both directories are missing"""
        descriptor = {
            "scene_name": "my_scene",
            "scene_order": 10,
            "job_input_dir": "",
            "job_output_dir": ""
        }
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, 0, "png", 95)
        assert config is None


class TestSelectSceneDescriptor:
    """Test scene descriptor selection logic"""
    
    def test_select_from_single_scene(self):
        """Test selecting from batch with single scene"""
        batch = [{"scene_name": "scene1", "scene_order": 0}]
        result = select_scene_descriptor(batch, 0)
        assert result == {"scene_name": "scene1", "scene_order": 0}
    
    def test_select_with_sorting(self):
        """Test that scenes are sorted by scene_order"""
        batch = [
            {"scene_name": "scene3", "scene_order": 2},
            {"scene_name": "scene1", "scene_order": 0},
            {"scene_name": "scene2", "scene_order": 1}
        ]
        
        result = select_scene_descriptor(batch, 0)
        assert result["scene_name"] == "scene1"
        
        result = select_scene_descriptor(batch, 1)
        assert result["scene_name"] == "scene2"
        
        result = select_scene_descriptor(batch, 2)
        assert result["scene_name"] == "scene3"
    
    def test_select_clamps_negative_index(self):
        """Test that negative indices are clamped to 0"""
        batch = [
            {"scene_name": "scene1", "scene_order": 0},
            {"scene_name": "scene2", "scene_order": 1}
        ]
        
        result = select_scene_descriptor(batch, -5)
        assert result["scene_name"] == "scene1"
    
    def test_select_clamps_large_index(self):
        """Test that indices beyond length are clamped"""
        batch = [
            {"scene_name": "scene1", "scene_order": 0},
            {"scene_name": "scene2", "scene_order": 1}
        ]
        
        result = select_scene_descriptor(batch, 100)
        assert result["scene_name"] == "scene2"
    
    def test_select_empty_batch(self):
        """Test returns None for empty batch"""
        result = select_scene_descriptor([], 0)
        assert result is None
    
    def test_select_none_batch(self):
        """Test returns None for None batch"""
        result = select_scene_descriptor(None, 0)
        assert result is None
    
    def test_select_handles_sorting_errors(self):
        """Test gracefully handles scenes that can't be sorted"""
        batch = [
            {"scene_name": "scene1"},  # Missing scene_order
            {"scene_name": "scene2", "scene_order": 1}
        ]
        
        # Should not raise, should use original order
        result = select_scene_descriptor(batch, 0)
        assert result is not None


class TestGeneratePreviewText:
    """Test preview text generation"""
    
    def test_generate_preview_png(self):
        """Test preview text for PNG format"""
        config = SceneImageSaveConfig(
            scene_name="test_scene",
            scene_order=1,
            target_dir="/tmp/test",
            image_format="png",
            quality=95
        )
        filepath = "/tmp/test/001_test_scene.png"
        
        text = generate_preview_text(config, filepath)
        
        assert "Saved: 001_test_scene.png" in text
        assert "Scene: test_scene (order: 1)" in text
        assert "Path: /tmp/test/001_test_scene.png" in text
        assert "Format: PNG" in text
        assert "quality" not in text.lower()  # PNG doesn't show quality
    
    def test_generate_preview_jpeg_with_quality(self):
        """Test preview text for JPEG includes quality"""
        config = SceneImageSaveConfig(
            scene_name="test_scene",
            scene_order=5,
            target_dir="/tmp/test",
            image_format="jpg",
            quality=85
        )
        filepath = "/tmp/test/005_test_scene.jpg"
        
        text = generate_preview_text(config, filepath)
        
        assert "Format: JPG" in text
        assert "(quality: 85)" in text
    
    def test_generate_preview_webp_with_quality(self):
        """Test preview text for WebP includes quality"""
        config = SceneImageSaveConfig(
            scene_name="test_scene",
            scene_order=0,
            target_dir="/tmp/test",
            image_format="webp",
            quality=90
        )
        filepath = "/tmp/test/000_test_scene.webp"
        
        text = generate_preview_text(config, filepath)
        
        assert "Format: WEBP" in text
        assert "(quality: 90)" in text


class TestImageSaver:
    """Test ImageSaver class (these would use mocks for actual I/O)"""
    
    @patch('os.makedirs')
    def test_ensure_directory_creates_dir(self, mock_makedirs):
        """Test that ensure_directory calls os.makedirs"""
        ImageSaver.ensure_directory("/test/path")
        mock_makedirs.assert_called_once_with("/test/path", exist_ok=True)
    
    def test_tensor_to_pil_conversion(self):
        """Test tensor to PIL conversion logic"""
        # Create a mock tensor-like object
        import numpy as np
        
        # Simulate a torch tensor with shape (1, height, width, channels)
        mock_tensor = MagicMock()
        mock_tensor.shape = (1, 64, 64, 3)
        
        # Mock the indexing and cpu/numpy operations
        mock_inner = MagicMock()
        mock_tensor.__getitem__.return_value = mock_inner
        
        # Create actual numpy array for conversion
        test_array = np.random.rand(64, 64, 3)
        mock_inner.cpu.return_value.numpy.return_value = test_array
        
        with patch('PIL.Image.fromarray') as mock_fromarray:
            ImageSaver.tensor_to_pil(mock_tensor)
            
            # Verify fromarray was called
            mock_fromarray.assert_called_once()
            
            # Verify the array was scaled to 0-255
            called_array = mock_fromarray.call_args[0][0]
            assert called_array.dtype == np.uint8


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_config_to_filepath_flow(self):
        """Test complete flow from descriptor to filepath"""
        descriptor = {
            "scene_name": "Outdoor Scene",
            "scene_order": 42,
            "job_input_dir": "/output/story/jobs/abc123/input",
            "job_output_dir": "/output/story/jobs/abc123/output"
        }
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, 0, "webp", 92)
        assert config is not None
        
        filename = config.generate_filename()
        assert filename == "042_outdoor_scene.webp"
        
        filepath = config.generate_filepath()
        assert filepath == "/output/story/jobs/abc123/input/042_outdoor_scene.webp"
        
        preview = generate_preview_text(config, filepath)
        assert "042_outdoor_scene.webp" in preview
        assert "Outdoor Scene" in preview
        assert "quality: 92" in preview


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
