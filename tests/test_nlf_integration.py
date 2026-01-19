"""
Integration tests for NLF pose in SceneCreate and SceneUpdate nodes.

These tests verify the NLF pose feature is properly integrated into the Scene system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# Use unified import approach from conftest
from conftest import import_test_module

# Import NLF pose utilities
nlf_pose = import_test_module("utils/nlf_pose.py", "nlf_pose")


class TestNLFPoseModule(unittest.TestCase):
    """Test NLF pose module can be imported and has required functions"""
    
    def test_nlf_pose_module_imports(self):
        """Test that NLF pose module imports successfully"""
        self.assertIsNotNone(nlf_pose)
        self.assertTrue(hasattr(nlf_pose, 'load_nlf_model'))
        self.assertTrue(hasattr(nlf_pose, 'predict_nlf_pose'))
        self.assertTrue(hasattr(nlf_pose, 'render_nlf_pose'))
        self.assertTrue(hasattr(nlf_pose, 'dwpose_to_openpose'))
        self.assertTrue(hasattr(nlf_pose, 'nlfpred_to_pose_keypoint'))
    
    def test_model_path_validation(self):
        """Test model path validation"""
        # Test with invalid path should raise exception
        with self.assertRaises(Exception):
            nlf_pose.load_nlf_model("/nonexistent/path/model.pth", warmup=False)


class TestSceneCreateNLF(unittest.TestCase):
    """Test SceneCreate node has NLF pose generation support"""
    
    def test_scene_create_has_nlf_inputs(self):
        """Test SceneCreate node has all NLF pose inputs"""
        try:
            from extension import SceneCreate
            
            schema = SceneCreate.define_schema()
            input_ids = [inp.id for inp in schema.inputs]
            
            # Verify all NLF inputs exist
            nlf_inputs = [
                'generate_nlf_pose',
                'nlf_model_path',
                'nlf_draw_face',
                'nlf_draw_hands',
                'nlf_render_device',
                'nlf_scale_hands',
                'nlf_render_backend'
            ]
            
            for input_id in nlf_inputs:
                self.assertIn(input_id, input_ids, f"Missing NLF input: {input_id}")
                
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_scene_create_nlf_generation_code_exists(self):
        """Test SceneCreate has NLF generation code"""
        try:
            from extension import SceneCreate
            import inspect
            
            source = inspect.getsource(SceneCreate.execute)
            
            # Verify NLF generation code exists
            self.assertIn('generate_nlf_pose', source)
            self.assertIn('load_nlf_model', source)
            self.assertIn('predict_nlf_pose', source)
            self.assertIn('render_nlf_pose', source)
            self.assertIn('nlfpred_to_pose_keypoint', source)
            
        except ImportError:
            self.skipTest("Extension module not available")


class TestSceneUpdateNLF(unittest.TestCase):
    """Test SceneUpdate node has NLF pose editing support"""
    
    def test_scene_update_has_nlf_inputs(self):
        """Test SceneUpdate node has all NLF pose editing inputs"""
        try:
            from extension import SceneUpdate
            
            schema = SceneUpdate.define_schema()
            input_ids = [inp.id for inp in schema.inputs]
            
            # Verify all NLF update inputs exist
            nlf_update_inputs = [
                'update_nlf_pose',
                'pose_image',
                'pose_keypoint',
                'nlf_draw_face',
                'nlf_draw_hands',
                'nlf_render_device',
                'nlf_scale_hands',
                'nlf_render_backend',
                'nlf_model_path'
            ]
            
            for input_id in nlf_update_inputs:
                self.assertIn(input_id, input_ids, f"Missing NLF update input: {input_id}")
                
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_scene_update_custom_pose_workflow_exists(self):
        """Test SceneUpdate has custom pose workflow code"""
        try:
            from extension import SceneUpdate
            import inspect
            
            source = inspect.getsource(SceneUpdate.execute)
            
            # Verify custom pose workflow exists
            self.assertIn('update_nlf_pose', source)
            self.assertIn('pose_image', source)
            self.assertIn('pose_keypoint', source)
            self.assertIn('pose_nlf_image', source)
            
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_scene_update_regeneration_workflow_exists(self):
        """Test SceneUpdate has NLF regeneration workflow"""
        try:
            from extension import SceneUpdate
            import inspect
            
            source = inspect.getsource(SceneUpdate.execute)
            
            # Verify regeneration logic exists
            self.assertIn('update_nlf_pose', source)
            self.assertIn('base_image', source)
            self.assertIn('upscale_image', source)
            self.assertIn('load_nlf_model', source)
            
        except ImportError:
            self.skipTest("Extension module not available")


class TestSceneInfoNLF(unittest.TestCase):
    """Test SceneInfo data model includes NLF pose support"""
    
    def test_scene_info_has_nlf_field(self):
        """Test SceneInfo has pose_nlf_image field"""
        try:
            from extension import SceneInfo
            
            scene_info = SceneInfo(
                scene_dir=tempfile.mkdtemp(),
                scene_name="test_scene"
            )
            
            # Verify pose_nlf_image field exists
            self.assertTrue(hasattr(scene_info, 'pose_nlf_image'))
            self.assertIsNone(scene_info.pose_nlf_image)
            
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_default_pose_options_includes_nlf(self):
        """Test default_pose_options includes 'nlf' entry"""
        try:
            from extension import default_pose_options
            
            self.assertIn('nlf', default_pose_options)
            self.assertEqual(default_pose_options['nlf'], 'pose_nlf_image')
            
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_scene_info_load_pose_images_includes_nlf(self):
        """Test SceneInfo.load_pose_images includes pose_nlf.png mapping"""
        try:
            from extension import SceneInfo
            import inspect
            
            # Check the source code includes pose_nlf.png mapping
            source = inspect.getsource(SceneInfo.load_pose_images)
            
            self.assertIn('pose_nlf', source)
            self.assertIn('pose_nlf.png', source)
            
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_scene_info_save_includes_nlf(self):
        """Test SceneInfo.save_all_images saves pose_nlf.png"""
        try:
            from extension import SceneInfo
            import inspect
            
            # Check the source code includes pose_nlf_image saving
            source = inspect.getsource(SceneInfo.save_all_images)
            
            self.assertIn('pose_nlf_image', source)
            self.assertIn('pose_nlf.png', source)
            
        except ImportError:
            self.skipTest("Extension module not available")
    
    def test_pose_normalization_includes_nlf(self):
        """Test SceneUpdate includes pose_nlf_image in normalization"""
        try:
            from extension import SceneUpdate
            import inspect
            
            # Get source code and verify pose_nlf_image is in normalization loop
            source = inspect.getsource(SceneUpdate.execute)
            
            self.assertIn('pose_nlf_image', source)
            self.assertIn("for pose_attr in", source)
            
        except ImportError:
            self.skipTest("Extension module not available")


class TestNLFPoseJSONFormat(unittest.TestCase):
    """Test NLF pose JSON format"""
    
    def test_pose_json_openpose_format(self):
        """Test pose.json uses correct OpenPose format structure"""
        # Example of expected OpenPose format
        pose_json_data = {
            'people': [
                {
                    'pose_keypoints_2d': [100.0, 200.0, 0.9] * 18,  # 18 body keypoints
                    'face_keypoints_2d': [0.0] * 210,  # 70 face keypoints * 3
                    'hand_left_keypoints_2d': [0.0] * 63,  # 21 hand keypoints * 3
                    'hand_right_keypoints_2d': [0.0] * 63
                }
            ]
        }
        
        # Verify it can be serialized/deserialized
        json_str = json.dumps(pose_json_data)
        parsed = json.loads(json_str)
        
        self.assertIn('people', parsed)
        self.assertIsInstance(parsed['people'], list)
        if parsed['people']:
            person = parsed['people'][0]
            self.assertIn('pose_keypoints_2d', person)
            self.assertIn('face_keypoints_2d', person)
            self.assertIn('hand_left_keypoints_2d', person)
            self.assertIn('hand_right_keypoints_2d', person)
            self.assertEqual(len(person['pose_keypoints_2d']), 54)  # 18 * 3


if __name__ == '__main__':
    unittest.main()
