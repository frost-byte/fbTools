#!/usr/bin/env python3
"""
Manual verification script for StoryEdit scene flags integration.

This script tests the complete flow:
1. Frontend data format → Backend parsing
2. Backend saving → File storage
3. File loading → Backend response
4. Backend response → Frontend data

Run this to verify the entire save/load cycle works correctly.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from story_models import SceneInStory, StoryInfo

def test_complete_story_flags_cycle():
    """Test the complete story flags save/load cycle."""
    
    print("\n" + "="*80)
    print("STORY FLAGS INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Step 1: Simulate frontend data (what StoryEdit sends to /fbtools/story/save)
    print("Step 1: Frontend sends scene data to API")
    print("-" * 80)
    
    frontend_request = {
        "story_name": "test_story",
        "scenes": [
            {
                "scene_id": "scene_001",
                "scene_name": "opening_scene",
                "scene_order": 0,
                "mask_type": "combined",
                "mask_background": True,
                "prompt_source": "prompt",
                "prompt_key": "main_prompt",
                "custom_prompt": "",
                "video_prompt_source": "auto",
                "video_prompt_key": "",
                "video_custom_prompt": "",
                "depth_type": "depth",
                "pose_type": "open",
                "use_depth": True,
                "use_mask": False,
                "use_pose": True,
                "use_canny": False
            },
            {
                "scene_id": "scene_002",
                "scene_name": "action_scene",
                "scene_order": 1,
                "mask_type": "combined",
                "mask_background": False,
                "prompt_source": "composition",
                "prompt_key": "action_comp",
                "custom_prompt": "",
                "video_prompt_source": "custom",
                "video_prompt_key": "",
                "video_custom_prompt": "Fast paced action sequence",
                "depth_type": "depth_any",
                "pose_type": "dw",
                "use_depth": False,
                "use_mask": True,
                "use_pose": False,
                "use_canny": True
            }
        ]
    }
    
    print("Frontend request body:")
    print(json.dumps(frontend_request, indent=2))
    
    # Step 2: Backend processes the request (simulate story_save endpoint)
    print("\n\nStep 2: Backend processes scenes")
    print("-" * 80)
    
    scenes = []
    for scene_data in frontend_request["scenes"]:
        scene = SceneInStory(**scene_data)
        scenes.append(scene)
        print(f"\n✓ Created scene: {scene.scene_name}")
        print(f"  Flags: depth={scene.use_depth}, mask={scene.use_mask}, pose={scene.use_pose}, canny={scene.use_canny}")
    
    # Step 3: Save to file (simulate save_story function)
    print("\n\nStep 3: Save story to JSON file")
    print("-" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        story_info = StoryInfo(
            story_name=frontend_request["story_name"],
            story_dir=tmpdir,
            scenes=scenes
        )
        
        story_file = Path(tmpdir) / "story.json"
        story_dict = story_info.model_dump()
        
        with open(story_file, 'w') as f:
            json.dump(story_dict, f, indent=2)
        
        print(f"✓ Saved to: {story_file}")
        print(f"\nFile contents (scenes only):")
        print(json.dumps(story_dict["scenes"], indent=2))
        
        # Step 4: Load from file (simulate load_story function)
        print("\n\nStep 4: Load story from JSON file")
        print("-" * 80)
        
        with open(story_file, 'r') as f:
            loaded_data = json.load(f)
        
        loaded_story = StoryInfo(**loaded_data)
        
        print(f"✓ Loaded story: {loaded_story.story_name}")
        print(f"  Scene count: {len(loaded_story.scenes)}")
        
        # Step 5: Verify all flags match
        print("\n\nStep 5: Verify flags are preserved")
        print("-" * 80)
        
        all_correct = True
        
        for i, (original_scene, loaded_scene) in enumerate(zip(scenes, loaded_story.scenes)):
            print(f"\nScene {i}: {loaded_scene.scene_name}")
            
            # Check each flag
            flags = ['use_depth', 'use_mask', 'use_pose', 'use_canny']
            for flag in flags:
                original_value = getattr(original_scene, flag)
                loaded_value = getattr(loaded_scene, flag)
                
                if original_value == loaded_value:
                    print(f"  ✓ {flag}: {loaded_value} (correct)")
                else:
                    print(f"  ✗ {flag}: {loaded_value} (expected {original_value})")
                    all_correct = False
        
        # Step 6: Simulate API response (what frontend receives)
        print("\n\nStep 6: API response to frontend")
        print("-" * 80)
        
        api_response = {
            "story_name": loaded_story.story_name,
            "story_dir": loaded_story.story_dir,
            "scenes": [scene.model_dump() for scene in loaded_story.scenes]
        }
        
        print("API response (scenes with flags):")
        for scene in api_response["scenes"]:
            print(f"\n  {scene['scene_name']}:")
            print(f"    use_depth: {scene['use_depth']}")
            print(f"    use_mask: {scene['use_mask']}")
            print(f"    use_pose: {scene['use_pose']}")
            print(f"    use_canny: {scene['use_canny']}")
        
        # Final result
        print("\n\n" + "="*80)
        if all_correct:
            print("✅ SUCCESS: All scene flags preserved through complete cycle!")
        else:
            print("❌ FAILURE: Some flags were not preserved correctly")
        print("="*80 + "\n")
        
        return all_correct


if __name__ == "__main__":
    success = test_complete_story_flags_cycle()
    sys.exit(0 if success else 1)
