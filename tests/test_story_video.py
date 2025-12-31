"""
Unit tests for story video generation utilities.

Tests cover: job ID listing, scene image finding, scene pairing, 
video filename generation, prompt resolution, and video descriptor building.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# Use unified import approach from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import import_test_module

# Import the module to test
story_video = import_test_module("utils/story_video.py")


class TestListJobIds:
    """Tests for list_job_ids function"""
    
    def test_empty_jobs_directory(self):
        """Should return empty list when jobs directory doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = story_video.list_job_ids(tmpdir)
            assert result == []
    
    def test_single_job_id(self):
        """Should return single job ID"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "jobs"
            jobs_dir.mkdir()
            (jobs_dir / "abc123").mkdir()
            
            result = story_video.list_job_ids(tmpdir)
            assert result == ["abc123"]
    
    def test_multiple_job_ids_sorted_by_mtime(self):
        """Should return job IDs sorted by modification time, newest first"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "jobs"
            jobs_dir.mkdir()
            
            # Create jobs in sequence
            old_job = jobs_dir / "old_job"
            old_job.mkdir()
            
            import time
            time.sleep(0.01)  # Ensure different mtimes
            
            new_job = jobs_dir / "new_job"
            new_job.mkdir()
            
            result = story_video.list_job_ids(tmpdir)
            # Newest should be first
            assert result[0] == "new_job"
            assert result[1] == "old_job"
    
    def test_ignores_files_in_jobs_dir(self):
        """Should ignore non-directory files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "jobs"
            jobs_dir.mkdir()
            (jobs_dir / "job1").mkdir()
            (jobs_dir / "some_file.txt").write_text("ignore me")
            
            result = story_video.list_job_ids(tmpdir)
            assert result == ["job1"]


class TestFindSceneImage:
    """Tests for find_scene_image function"""
    
    def test_finds_png_image(self):
        """Should find PNG image with correct naming pattern"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            image_path = input_dir / "001_opening_scene.png"
            image_path.touch()
            
            result = story_video.find_scene_image(str(input_dir), 1, "opening scene")
            assert result == str(image_path)
    
    def test_finds_jpg_image(self):
        """Should find JPG image"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            image_path = input_dir / "002_battle.jpg"
            image_path.touch()
            
            result = story_video.find_scene_image(str(input_dir), 2, "battle")
            assert result == str(image_path)
    
    def test_prefers_first_matching_extension(self):
        """Should return first matching extension found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            png_path = input_dir / "003_finale.png"
            png_path.touch()
            jpg_path = input_dir / "003_finale.jpg"
            jpg_path.touch()
            
            result = story_video.find_scene_image(str(input_dir), 3, "finale")
            # Should prefer PNG as it comes first in default extensions list
            assert result == str(png_path)
    
    def test_returns_none_when_not_found(self):
        """Should return None when image doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            
            result = story_video.find_scene_image(str(input_dir), 5, "missing")
            assert result is None
    
    def test_returns_none_when_directory_missing(self):
        """Should return None when input directory doesn't exist"""
        result = story_video.find_scene_image("/nonexistent/path", 1, "scene")
        assert result is None
    
    def test_handles_scene_name_with_spaces(self):
        """Should correctly format scene names with spaces"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            image_path = input_dir / "010_my_cool_scene.png"
            image_path.touch()
            
            result = story_video.find_scene_image(str(input_dir), 10, "my cool scene")
            assert result == str(image_path)


class TestPairConsecutiveScenes:
    """Tests for pair_consecutive_scenes function"""
    
    def test_empty_scenes_list(self):
        """Should return empty list for empty input"""
        result = story_video.pair_consecutive_scenes([])
        assert result == []
    
    def test_single_scene(self):
        """Should pair single scene with None"""
        scenes = [{"scene_order": 0, "scene_name": "only"}]
        result = story_video.pair_consecutive_scenes(scenes)
        assert len(result) == 1
        assert result[0] == (scenes[0], None)
    
    def test_two_scenes(self):
        """Should create one transition pair plus final scene"""
        scenes = [
            {"scene_order": 0, "scene_name": "first"},
            {"scene_order": 1, "scene_name": "second"},
        ]
        result = story_video.pair_consecutive_scenes(scenes)
        assert len(result) == 2
        assert result[0] == (scenes[0], scenes[1])
        assert result[1] == (scenes[1], None)
    
    def test_multiple_scenes(self):
        """Should create pairs for all consecutive scenes"""
        scenes = [
            {"scene_order": 0, "scene_name": "a"},
            {"scene_order": 1, "scene_name": "b"},
            {"scene_order": 2, "scene_name": "c"},
        ]
        result = story_video.pair_consecutive_scenes(scenes)
        assert len(result) == 3
        assert result[0] == (scenes[0], scenes[1])
        assert result[1] == (scenes[1], scenes[2])
        assert result[2] == (scenes[2], None)


class TestGenerateVideoFilename:
    """Tests for generate_video_filename function"""
    
    def test_transition_between_scenes(self):
        """Should generate filename for scene transition"""
        result = story_video.generate_video_filename(1, 2, "opening", "battle")
        assert result == "001_to_002_opening_to_battle.mp4"
    
    def test_final_scene_no_transition(self):
        """Should generate filename for final scene without transition"""
        result = story_video.generate_video_filename(5, None, "finale", None)
        assert result == "005_finale.mp4"
    
    def test_handles_scene_names_with_spaces(self):
        """Should replace spaces with underscores"""
        result = story_video.generate_video_filename(0, 1, "my cool scene", "next scene")
        assert result == "000_to_001_my_cool_scene_to_next_scene.mp4"
    
    def test_custom_extension(self):
        """Should support custom video extension"""
        result = story_video.generate_video_filename(1, 2, "a", "b", extension="avi")
        assert result == "001_to_002_a_to_b.avi"
    
    def test_zero_padding(self):
        """Should zero-pad scene orders to 3 digits"""
        result = story_video.generate_video_filename(5, 10, "a", "b")
        assert result == "005_to_010_a_to_b.mp4"


class TestResolveVideoPrompt:
    """Tests for resolve_video_prompt function"""
    
    def test_auto_uses_image_prompt(self):
        """Should use image prompt when video_prompt_source is 'auto'"""
        scene = {
            "video_prompt_source": "auto",
            "positive_prompt": "a beautiful landscape",
        }
        result = story_video.resolve_video_prompt(scene, {})
        assert result == "a beautiful landscape"
    
    def test_custom_uses_custom_prompt(self):
        """Should use custom video prompt when source is 'custom'"""
        scene = {
            "video_prompt_source": "custom",
            "video_custom_prompt": "smooth camera pan over landscape",
            "positive_prompt": "ignored",
        }
        result = story_video.resolve_video_prompt(scene, {})
        assert result == "smooth camera pan over landscape"
    
    def test_prompt_key_lookup(self):
        """Should look up prompt from prompt_data when source is 'prompt'"""
        scene = {
            "video_prompt_source": "prompt",
            "video_prompt_key": "video_transition",
        }
        prompt_data = {
            "video_transition": "cinematic zoom transition",
        }
        result = story_video.resolve_video_prompt(scene, prompt_data)
        assert result == "cinematic zoom transition"
    
    def test_prompt_key_missing(self):
        """Should return empty string when prompt key not found"""
        scene = {
            "video_prompt_source": "prompt",
            "video_prompt_key": "missing_key",
        }
        result = story_video.resolve_video_prompt(scene, {})
        assert result == ""
    
    def test_composition_returns_key(self):
        """Should return composition key for external processing"""
        scene = {
            "video_prompt_source": "composition",
            "video_prompt_key": "my_composition",
        }
        result = story_video.resolve_video_prompt(scene, {})
        assert result == "my_composition"
    
    def test_default_fallback(self):
        """Should fall back to image prompt for unknown source"""
        scene = {
            "video_prompt_source": "unknown",
            "positive_prompt": "fallback prompt",
        }
        result = story_video.resolve_video_prompt(scene, {})
        assert result == "fallback prompt"


class TestBuildVideoDescriptor:
    """Tests for build_video_descriptor function"""
    
    def test_builds_descriptor_with_transition(self):
        """Should build complete descriptor for scene transition"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Create test images
            (input_dir / "001_opening.png").touch()
            (input_dir / "002_battle.png").touch()
            
            scene = {
                "scene_id": "id1",
                "scene_order": 1,
                "scene_name": "opening",
                "video_prompt_source": "auto",
                "positive_prompt": "epic opening",
                "lora_data": {"loras_high": []},
            }
            next_scene = {
                "scene_order": 2,
                "scene_name": "battle",
            }
            
            result = story_video.build_video_descriptor(
                scene, next_scene, str(input_dir), str(output_dir), {}
            )
            
            assert result is not None
            assert result["scene_order"] == 1
            assert result["scene_name"] == "opening"
            assert result["scene_id"] == "id1"
            assert result["has_transition"] is True
            assert result["next_scene_order"] == 2
            assert result["next_scene_name"] == "battle"
            assert "first_frame_path" in result
            assert "last_frame_path" in result
            assert result["video_prompt"] == "epic opening"
            assert result["video_filename"] == "001_to_002_opening_to_battle.mp4"
    
    def test_builds_descriptor_without_transition(self):
        """Should build descriptor for final scene with no transition"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            (input_dir / "003_finale.png").touch()
            
            scene = {
                "scene_id": "id3",
                "scene_order": 3,
                "scene_name": "finale",
                "video_prompt_source": "auto",
                "positive_prompt": "dramatic finale",
                "lora_data": None,
            }
            
            result = story_video.build_video_descriptor(
                scene, None, str(input_dir), str(output_dir), {}
            )
            
            assert result is not None
            assert result["has_transition"] is False
            assert "next_scene_order" not in result
            assert result["video_filename"] == "003_finale.mp4"
    
    def test_returns_none_when_first_frame_missing(self):
        """Should return None when first frame image doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            scene = {
                "scene_order": 1,
                "scene_name": "missing",
                "video_prompt_source": "auto",
                "positive_prompt": "",
                "lora_data": None,
            }
            
            result = story_video.build_video_descriptor(
                scene, None, str(input_dir), str(output_dir), {}
            )
            
            assert result is None
    
    def test_handles_missing_next_frame(self):
        """Should mark transition as unavailable if next frame missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            (input_dir / "001_scene_a.png").touch()
            # scene_b image doesn't exist
            
            scene = {
                "scene_order": 1,
                "scene_name": "scene_a",
                "video_prompt_source": "auto",
                "positive_prompt": "prompt a",
                "lora_data": None,
            }
            next_scene = {
                "scene_order": 2,
                "scene_name": "scene_b",
            }
            
            result = story_video.build_video_descriptor(
                scene, next_scene, str(input_dir), str(output_dir), {}
            )
            
            assert result is not None
            assert result["has_transition"] is False
            assert "last_frame_path" not in result
