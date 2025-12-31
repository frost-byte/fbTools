# Story Video Generation Workflow

## Overview

The Story Video Generation system extends the Story building nodes to support video generation from story scene images. After generating images for each scene using `StorySceneBatch` and `StorySceneImageSave`, you can use `StoryVideoBatch` to create videos that transition between scenes.

## Architecture

The video generation workflow follows a similar pattern to image generation:

**Image Generation Flow:**
```
StoryLoad → StorySceneBatch → [Iterate] → Generate Image → StorySceneImageSave
```

**Video Generation Flow:**
```
StoryLoad → StoryVideoBatch → [Iterate] → Generate Video → Save Video
```

## Scene Configuration

Each scene in a story now supports video-specific prompts through three new fields:

### Video Prompt Fields

- **video_prompt_source**: Determines where the video prompt comes from
  - `"auto"` (default): Uses the scene's image generation prompt
  - `"prompt"`: Uses a specific prompt from the scene's prompt dictionary
  - `"composition"`: Uses a composition from the scene's compositions
  - `"custom"`: Uses a custom prompt string

- **video_prompt_key**: The key to look up in prompts or compositions (when source is "prompt" or "composition")

- **video_custom_prompt**: Custom prompt text used when source is "custom"

### Editing Video Prompts

You can set video prompts for scenes in two ways:

1. **Via StoryEdit node**: The StoryEdit UI allows you to configure video prompt settings for each scene

2. **Programmatically**: When creating scenes, include video prompt fields:
   ```python
   scene = SceneInStory(
       scene_name="opening",
       scene_order=0,
       # Image generation settings
       prompt_source="prompt",
       prompt_key="main_character",
       # Video generation settings
       video_prompt_source="custom",
       video_custom_prompt="smooth camera pan revealing character"
   )
   ```

## Node: StoryVideoBatch

### Purpose

Creates an ordered list of video descriptors for scene transitions. Each descriptor contains:
- First frame image path (current scene)
- Last frame image path (next scene) if available
- Video prompt (resolved from scene configuration)
- LoRa data from the scene
- Output filename and path

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| **story_info** | STORY_INFO | Story loaded from StoryLoad node |
| **job_id** | Combo | Job ID from available jobs (auto-populated, newest first) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| **video_count** | Int | Total number of video transitions |
| **video_batch** | VIDEO_BATCH | List of video descriptors for iteration |
| **job_id** | String | Selected job ID |
| **job_input_dir** | String | Path to job's input directory (where images are) |
| **job_output_dir** | String | Path to job's output directory (where videos will be saved) |

### Video Descriptor Format

Each video descriptor in the batch is a dictionary containing:

```python
{
    "scene_order": 0,                      # Order of current scene
    "scene_name": "opening",               # Name of current scene
    "scene_id": "uuid-string",             # Unique scene ID
    "first_frame_path": "/path/001_opening.png",  # Current scene image
    "last_frame_path": "/path/002_battle.png",    # Next scene image (if transition)
    "video_prompt": "camera pans across...",      # Resolved video prompt
    "lora_data": {                         # LoRa settings from scene
        "loras_high": [...],
        "loras_low": [...]
    },
    "has_transition": True,                # Whether there's a next scene
    "next_scene_order": 1,                 # Order of next scene (if transition)
    "next_scene_name": "battle",           # Name of next scene (if transition)
    "video_filename": "001_to_002_opening_to_battle.mp4",  # Generated filename
    "video_output_path": "/path/output/001_to_002_opening_to_battle.mp4",
    "job_input_dir": "/path/input/",
    "job_output_dir": "/path/output/"
}
```

For the final scene (no transition):
```python
{
    "scene_order": 5,
    "scene_name": "finale",
    "first_frame_path": "/path/005_finale.png",
    "video_prompt": "epic finale sequence",
    "has_transition": False,
    "video_filename": "005_finale.mp4",
    ...
}
```

## Directory Structure

Videos follow the same job-based organization as images:

```
output/
└── stories/
    └── {story_name}/
        └── jobs/
            └── {job_id}/
                ├── input/           # Scene images (from StorySceneImageSave)
                │   ├── 001_opening.png
                │   ├── 002_battle.png
                │   └── 003_finale.png
                └── output/          # Generated videos
                    ├── 001_to_002_opening_to_battle.mp4
                    ├── 002_to_003_battle_to_finale.mp4
                    └── 003_finale.mp4
```

## Example Workflows

### Basic Video Generation Workflow

```
1. StoryLoad
   ↓ (STORY_INFO)
2. StoryVideoBatch (select job_id)
   ↓ (VIDEO_BATCH)
3. Loop through video_batch:
   - Extract first_frame and last_frame paths
   - Load images
   - Apply video_prompt
   - Generate video (using your video generation node)
   - Save to video_output_path
```

### Integration with Video Generation Nodes

The VIDEO_BATCH output can be connected to iteration loops that:
1. Load the first and last frame images
2. Apply the video prompt to your video model
3. Use LoRa data for consistent character/style
4. Generate the transition video
5. Save to the specified output path

### Prompt Resolution Examples

**Auto (default):**
- Scene uses image generation prompt for video
- Good for consistent style between images and videos

**Custom:**
- Scene: `video_custom_prompt = "smooth camera zoom in"`
- Results in video with custom camera movement

**Prompt Key:**
- Scene: `video_prompt_source = "prompt"`, `video_prompt_key = "video_transition"`
- Looks up "video_transition" from scene's prompts.json
- Allows reusable video prompt definitions

**Composition:**
- Scene: `video_prompt_source = "composition"`, `video_prompt_key = "epic_transition"`
- Uses a composed prompt from multiple prompt keys
- Enables complex, layered video prompts

## Video Filename Convention

Transition videos are named:
```
{from_order:03d}_to_{to_order:03d}_{from_name}_to_{to_name}.mp4
```

Examples:
- `001_to_002_opening_to_battle.mp4`
- `002_to_003_battle_to_finale.mp4`

Final scene videos (no transition):
```
{order:03d}_{scene_name}.mp4
```

Example:
- `003_finale.mp4`

## Testing

Video generation utilities are fully tested with 29 unit tests covering:
- Job ID listing and sorting
- Scene image finding
- Scene pairing for transitions
- Video filename generation
- Prompt resolution
- Video descriptor building

Run tests with:
```bash
pytest tests/test_story_video.py -v
```

## Utility Functions

The `utils/story_video.py` module provides testable helper functions:

- `list_job_ids(story_dir)`: List all job IDs for a story
- `find_scene_image(job_input_dir, scene_order, scene_name)`: Find scene image file
- `pair_consecutive_scenes(scenes)`: Create scene transition pairs
- `generate_video_filename(from_order, to_order, from_name, to_name)`: Generate video filename
- `resolve_video_prompt(scene, prompt_data)`: Resolve video prompt from scene config
- `build_video_descriptor(scene, next_scene, job_input_dir, job_output_dir, prompt_data)`: Build complete video descriptor

These functions are separated from ComfyUI dependencies for easy testing and reuse.

## Future Enhancements

Potential additions to the video generation system:

1. **Video Style Presets**: Predefined transition styles (fade, zoom, pan, etc.)
2. **Video Duration Control**: Per-scene or per-transition duration settings
3. **Video Composition**: Combine multiple video clips into final story video
4. **Audio Integration**: Sync with audio tracks or dialogue
5. **Video Preview**: Preview transitions before full generation
6. **Batch Video Export**: Export all videos as a single concatenated file

## See Also

- [STORY_NODES_README.md](./STORY_NODES_README.md) - Story building nodes documentation
- [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Testing approach and patterns
- [utils/story_video.py](./utils/story_video.py) - Video generation utility functions
- [tests/test_story_video.py](./tests/test_story_video.py) - Video generation tests
