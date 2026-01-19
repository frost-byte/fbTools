# Story Building Nodes for ComfyUI

This document describes the new Story building nodes added to the fbTools extension for ComfyUI.

## Overview

The Story building system allows users to create ordered sequences of scenes for storytelling purposes. Each scene in a story can have its own configuration for masks, prompts, depth images, and pose images.

## Data Models

### SceneInStory
Represents a single scene within a story with the following properties:
- `scene_name`: Name of the scene (references a pose directory)
- `scene_order`: Order position in the story (0-based)
- `mask_type`: Type of mask to use ("girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg")
- `mask_background`: Whether to include background in the mask (True/False)
- `prompt_type`: Type of prompt to use ("girl_pos", "male_pos", "four_image_prompt", "wan_prompt", "wan_low_prompt", "custom")
- `custom_prompt`: Custom prompt text (used when prompt_type is "custom")
- `depth_type`: Type of depth image ("depth", "depth_any", "midas", "zoe", "zoe_any")
- `pose_type`: Type of pose image ("dense", "dw", "edit", "face", "open")

### StoryInfo
Contains the complete story configuration:
- `story_name`: Name of the story
- `story_dir`: Directory where the story is saved
- `scenes`: List of SceneInStory objects

The StoryInfo class includes methods for:
- `get_scene_by_name(scene_name)`: Find a scene by its name
- `get_scene_by_order(order)`: Find a scene by its order position
- `add_scene(scene)`: Add a new scene to the story
- `remove_scene(scene_name)`: Remove a scene from the story
- `reorder_scene(scene_name, new_order)`: Change the order of a scene

## Nodes

### StoryCreate
**Category:** 🧊 frost-byte/Story

Creates a new story with an initial scene.

**Inputs:**
- `story_name`: Name for the new story (default: "my_story")
- `story_dir`: Directory to save the story (default: output/stories)
- `initial_scene`: First scene to add (dropdown of available poses)
- `mask_type`: Mask type for the initial scene
- `mask_background`: Include background in mask
- `prompt_type`: Type of prompt to use
- `custom_prompt`: Custom prompt text (if prompt_type is "custom")
- `depth_type`: Depth image type
- `pose_type`: Pose image type

**Outputs:**
- `story_info`: StoryInfo object containing the new story

### StoryEdit
**Category:** 🧊 frost-byte/Story

Edit an existing story by adding, removing, reordering, or modifying scenes.

**Inputs:**
- `story_info`: The story to edit (STORY_INFO type)
- `operation`: Operation to perform:
  - `add_scene`: Add a new scene to the story
  - `remove_scene`: Remove a scene from the story
  - `reorder_scene`: Change the order of a scene
  - `edit_scene`: Modify scene settings
  - `no_change`: Pass through without changes
- `scene_name`: Name of the scene to add/remove/edit
- `scene_order`: Order position for the scene
- `mask_type`: Mask type
- `mask_background`: Include background in mask
- `prompt_type`: Type of prompt
- `custom_prompt`: Custom prompt text
- `depth_type`: Depth image type
- `pose_type`: Pose image type

**Outputs:**
- `story_info`: Updated StoryInfo object

### StoryView
**Category:** 🧊 frost-byte/Story

View and select scenes from a story with comprehensive preview capabilities. This node allows you to:
- Select a specific scene from the story
- Preview the pose, mask, and depth images for that scene
- Edit or use file-based prompts with "use_file" or "use_edit" action
- Output a complete SceneInfo object for the selected scene

**Inputs:**
- `story_info`: The story to view (STORY_INFO type)
- `selected_scene`: Name of the scene to view (dropdown of available scenes)
- `prompt_in`: Editable prompt text field
- `prompt_action`: Choose "use_file" (use prompt from scene file) or "use_edit" (use edited prompt from input)

**Outputs:**
- `scene_info`: Complete SceneInfo object for the selected scene with updated prompt
- `story_name`: Name of the story
- `story_dir`: Directory of the story
- `scene_count`: Number of scenes in the story
- `scene_name`: Name of the selected scene
- `selected_prompt`: The prompt text (from file or edited, based on prompt_action)
- `pose_image`: Pose image for the selected scene (based on scene's pose_type setting)
- `mask_image`: Mask image for the selected scene (based on scene's mask_type and mask_background settings)
- `depth_image`: Depth image for the selected scene (based on scene's depth_type setting)

**UI Preview:** 
- **Images**: Shows pose, mask, and depth images side-by-side
- **Text**: Displays:
  - Story name and directory
  - Scene count and selected scene info
  - Prompt type and current prompt text
  - Complete list of all scenes with the selected scene marked

**How Prompt Selection Works:**
The node determines which prompt to load based on the selected scene's `prompt_type` setting:
- `girl_pos`: Loads girl_pos from prompts.json
- `male_pos`: Loads male_pos from prompts.json
- `four_image_prompt`: Loads four_image_prompt from prompts.json
- `wan_prompt`: Loads wan_prompt from prompts.json
- `wan_low_prompt`: Loads wan_low_prompt from prompts.json
- `custom`: Uses the scene's custom_prompt value

Then applies the `prompt_action`:
- `use_file`: Uses the prompt from the file
- `use_edit`: Uses the text from `prompt_in` input

The SceneInfo output includes the selected prompt in the appropriate field (e.g., if prompt_type is "girl_pos", the girl_pos field in SceneInfo will contain the selected prompt).

### StorySave
**Category:** 🧊 frost-byte/Story

Save the story configuration to a JSON file.

**Inputs:**
- `story_info`: The story to save (STORY_INFO type)
- `filename`: Name of the JSON file (default: "story.json")

**Outputs:**
- `save_path`: Full path where the story was saved

**UI Preview:** Displays save confirmation with path and scene count

### StoryLoad
**Category:** 🧊 frost-byte/Story

Load a story from a JSON file.

**Inputs:**
- `stories_dir`: Directory containing story folders
- `story_name`: Name of the story folder to load
- `filename`: Name of the JSON file (default: "story.json")

**Outputs:**
- `story_info`: Loaded StoryInfo object

## JSON File Format

Stories are saved in JSON format with the following structure:

```json
{
  "story_name": "my_story",
  "story_dir": "/path/to/output/stories/my_story",
  "scenes": [
    {
      "scene_name": "scene_001",
      "scene_order": 0,
      "mask_type": "combined",
      "mask_background": true,
      "prompt_type": "girl_pos",
      "custom_prompt": "",
      "depth_type": "depth",
      "pose_type": "open"
    },
    {
      "scene_name": "scene_002",
      "scene_order": 1,
      "mask_type": "girl",
      "mask_background": false,
      "prompt_type": "custom",
      "custom_prompt": "A beautiful sunset scene",
      "depth_type": "zoe",
      "pose_type": "dw"
    }
  ]
}
```

## Utility Functions

The following utility functions support the Story system:

### load_story(story_json_path: str) -> Optional[StoryInfo]
Loads a story from a JSON file and returns a StoryInfo object.

### save_story(story_info: StoryInfo, story_json_path: str)
Saves a StoryInfo object to a JSON file.

### default_stories_dir() -> str
Returns the default directory for storing stories (output/stories).

## Usage Example

### Creating a New Story
1. Add a **StoryCreate** node
2. Set the story name and initial scene
3. Configure the initial scene settings
4. Output connects to a **StoryEdit** node or **StorySave** node

### Building a Multi-Scene Story
1. Start with **StoryCreate**
2. Chain multiple **StoryEdit** nodes to add scenes:
   - First StoryEdit: operation="add_scene", scene_name="scene_002"
   - Second StoryEdit: operation="add_scene", scene_name="scene_003"
   - etc.
3. Use **StoryView** to preview the story structure
4. Connect to **StorySave** to save the configuration

### Loading and Modifying a Story
1. Use **StoryLoad** to load an existing story
2. Connect to **StoryView** to see current configuration
3. Connect to **StoryEdit** to make changes
4. Save with **StorySave**

### Removing or Reordering Scenes
1. Load or create a story
2. Connect to **StoryEdit** with operation="remove_scene" or "reorder_scene"
3. Specify the scene_name and new order (if reordering)
4. View changes with **StoryView**

### Using StoryView to Preview and Select Scenes
1. Load or create a story with multiple scenes
2. Connect to **StoryView**
3. In the `selected_scene` dropdown, choose a scene to preview
4. Choose `prompt_action`:
   - "use_file": Load the prompt from the scene's prompts.json file
   - "use_edit": Use the text from `prompt_in` input field
5. View the preview showing:
   - Pose, mask, and depth images side-by-side
   - Current prompt text
   - All scenes in the story with selected scene marked
6. Connect the `scene_info` output to other nodes for processing
7. Connect individual outputs (pose_image, mask_image, depth_image, selected_prompt) as needed

### Creating a Video Sequence from a Story
1. Use **StoryLoad** to load a story
2. For each scene in the story:
   - Connect to **StoryView** with that scene selected
   - Extract the `scene_info` output
   - Use SceneInfo with video generation nodes
3. Combine outputs into final sequence

## Integration with Scene System

The Story system works alongside the existing Scene system:
- Stories reference scenes by name (which correspond to pose directories)
- Each scene in a story can specify which mask, prompt, depth, and pose to use from that scene's data
- Scene nodes (SceneSelect, SceneCreate, etc.) manage individual scene data
- Story nodes manage the ordered sequence and configuration for using those scenes

## File Locations

- **Stories Directory:** `ComfyUI/output/stories/`
- **Story JSON Files:** `ComfyUI/output/stories/{story_name}/story.json`
- **Scene/Pose Data:** `ComfyUI/output/poses/{pose_name}/`

## Notes

- Scene order is automatically normalized when scenes are added or removed
- Duplicate scene names are allowed in a story (useful for using the same scene multiple times with different settings)
- The `STORY_INFO` type is a custom ComfyUI type that can be passed between nodes
- All Story nodes are in the "🧊 frost-byte/Story" category for easy discovery
