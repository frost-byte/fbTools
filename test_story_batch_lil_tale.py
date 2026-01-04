#!/usr/bin/env python3
"""
Test script to run StorySceneBatch on lil_tale story and verify prompt assignments.
"""

import sys
import json
from pathlib import Path

# Add test directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))

# Import test utilities
from conftest import import_test_module

# Import required modules
story_models = import_test_module("story_models.py")
prompt_models = import_test_module("prompt_models.py")

StoryInfo = story_models.StoryInfo
SceneInStory = story_models.SceneInStory
PromptCollection = prompt_models.PromptCollection

def load_json_file(filepath: str):
    """Load JSON from a file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def test_lil_tale_story():
    """Test prompt resolution on lil_tale story - first 5 scenes"""
    
    print("\n" + "="*80)
    print("Testing Story Prompt Resolution on lil_tale story")
    print("="*80 + "\n")
    
    story_path = Path("/home/beerye/comfyui_env/ComfyUI-0.3.77/output/stories/lil_tale/story.json")
    
    if not story_path.exists():
        print(f"✗ Story not found at: {story_path}")
        return False
    
    # Load story
    story_data = load_json_file(str(story_path))
    print(f"✓ Loaded story: {story_data.get('story_name')}")
    print(f"  - Total scenes: {len(story_data.get('scenes', []))}")
    
    # Parse scenes
    scenes = []
    for scene_data in story_data.get("scenes", []):
        scene = SceneInStory(
            scene_id=scene_data.get("scene_id", ""),
            scene_name=scene_data["scene_name"],
            scene_order=scene_data["scene_order"],
            mask_type=scene_data.get("mask_type", "combined"),
            mask_background=scene_data.get("mask_background", "white"),
            prompt_source=scene_data.get("prompt_source", "prompt"),
            prompt_key=scene_data.get("prompt_key", ""),
            custom_prompt=scene_data.get("custom_prompt", ""),
            depth_type=scene_data.get("depth_type", "depth"),
            pose_type=scene_data.get("pose_type", "open"),
            use_depth=scene_data.get("use_depth", False),
            use_mask=scene_data.get("use_mask", False),
            use_pose=scene_data.get("use_pose", False),
            use_canny=scene_data.get("use_canny", False),
        )
        scenes.append(scene)
    
    # Sort by scene_order
    scenes.sort(key=lambda s: s.scene_order)
    
    print(f"\n{'='*80}")
    print(f"Checking first 5 scenes:")
    print(f"{'='*80}\n")
    
    scenes_dir = Path("/home/beerye/comfyui_env/ComfyUI-0.3.77/output/scenes")
    
    for i, scene in enumerate(scenes[:5]):
        print(f"\n--- Scene {i} (order={scene.scene_order}) ---")
        print(f"  Name: {scene.scene_name}")
        print(f"  Prompt source: {scene.prompt_source}")
        print(f"  Prompt key: {scene.prompt_key}")
        
        # Load scene's prompts.json
        scene_dir = scenes_dir / scene.scene_name
        prompt_path = scene_dir / "prompts.json"
        
        if not prompt_path.exists():
            print(f"  ⚠ No prompts.json found at: {prompt_path}")
            continue
        
        prompt_data_raw = load_json_file(str(prompt_path))
        
        # Load PromptCollection
        if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
            prompt_collection = PromptCollection.from_dict(prompt_data_raw)
        else:
            prompt_collection = PromptCollection.from_legacy_dict(prompt_data_raw)
        
        # Build prompt_dict
        prompt_dict = {key: metadata.value for key, metadata in prompt_collection.prompts.items()}
        
        # Build composition_dict
        composition_dict = {}
        if prompt_collection.compositions:
            for comp_name, prompt_keys in prompt_collection.compositions.items():
                parts = [prompt_dict.get(key, "") for key in prompt_keys if key in prompt_dict]
                composition_dict[comp_name] = " ".join(parts).strip()
        
        # Determine positive_prompt (same logic as StorySceneBatch)
        if scene.prompt_source == "custom":
            positive_prompt = scene.custom_prompt
        elif scene.prompt_source == "prompt" and scene.prompt_key:
            positive_prompt = prompt_dict.get(scene.prompt_key, "")
        elif scene.prompt_source == "composition" and scene.prompt_key:
            positive_prompt = composition_dict.get(scene.prompt_key, "")
        else:
            positive_prompt = ""
        
        print(f"  Prompt length: {len(positive_prompt)}")
        
        if positive_prompt:
            # Show the same snippet logic
            aurora_idx = positive_prompt.find('4ur0r4')
            if aurora_idx >= 0 and len(positive_prompt) > aurora_idx + 6:
                snippet = positive_prompt[aurora_idx + 6:aurora_idx + 126]
                print(f"  >>> 120 chars after '4ur0r4':")
                print(f"      '{snippet}'")
            elif len(positive_prompt) > 240:
                snippet = positive_prompt[120:240]
                print(f"  >>> Chars [120:240]:")
                print(f"      '{snippet}'")
            else:
                print(f"  >>> First 120 chars:")
                print(f"      '{positive_prompt[:120]}'")
        else:
            print(f"  ⚠ EMPTY PROMPT!")
    
    print(f"\n{'='*80}")
    print(f"✓ Test completed - check output above")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    success = test_lil_tale_story()
    sys.exit(0 if success else 1)

