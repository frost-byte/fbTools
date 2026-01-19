# Video Prompt UX Implementation

## Overview
Enhanced the StoryEdit node's Advanced Flags tab to provide a dynamic, user-friendly interface for configuring video prompts per scene. The implementation includes:

1. **Dynamic input types** based on video_prompt_source selection
2. **Dropdown population** with available prompt/composition keys
3. **Live preview** showing the actual prompt text that will be used
4. **Custom prompt textarea** for freeform text entry

## Features

### Video Prompt Source Selection
- **auto**: Uses the image generation prompt (from prompt_key)
- **prompt**: Select from available prompts via dropdown
- **composition**: Select from available compositions via dropdown  
- **custom**: Enter custom text in a textarea

### Dynamic UI Behavior

#### When source is "prompt" or "composition":
- The video_prompt_key field becomes a dropdown
- Dropdown is populated with keys from the scene's `prompts.json`
- Preview shows the actual prompt value

#### When source is "auto":
- The video_prompt_key field is a text input (read-only usage)
- Preview shows "(Using image prompt - will be resolved at generation)"

#### When source is "custom":
- The video_prompt_key field is hidden
- A custom_prompt textarea is shown instead
- Preview shows the custom prompt text

### Preview Textarea
A readonly textarea below the video_prompt_key/custom_prompt field that displays:
- **auto**: The image prompt text (or placeholder message)
- **prompt**: The value of the selected prompt
- **composition**: The composed prompt (joined prompt values)
- **custom**: The custom prompt text

## Implementation Details

### Frontend (js/nodes/story.js)

#### renderFlagsTab()
- Conditionally renders input type based on video_prompt_source
- Shows dropdown for prompt/composition sources
- Shows custom textarea for custom source
- Always shows preview textarea

#### populateVideoPromptControls()
- Fetches prompt data for each scene using `SceneAPI.getScenePrompts(sceneDir)`
- Populates dropdown options with available keys
- Calls `updateVideoPromptPreview()` for each scene
- Caches prompt data in `scene._promptData` for later use

#### updateVideoPromptPreview()
- Resolves the appropriate preview text based on source and selection
- Updates the readonly preview textarea
- Handles different source types:
  - **auto**: Shows image prompt or placeholder
  - **prompt**: Shows prompt value from _promptData
  - **composition**: Resolves composition and joins prompt values
  - **custom**: Shows video_custom_prompt field

#### Event Handlers
- **video-prompt-source-select change**: Updates currentScenes, re-renders tab
- **video-prompt-key-input change**: Updates currentScenes, updates preview
- **video-prompt-key-select change**: Updates currentScenes, updates preview
- **video-custom-prompt-input input**: Updates currentScenes, updates preview

### Backend (extension.py)

#### SceneInStory Model
Already includes required fields:
```python
video_prompt_source: str = "auto"
video_prompt_key: str = ""
video_custom_prompt: str = ""
```

#### StoryEdit._build_meta_payload()
Updated to include all video prompt fields in JSON payload:
```python
"video_prompt_source": getattr(scene, "video_prompt_source", "auto"),
"video_prompt_key": getattr(scene, "video_prompt_key", ""),
"video_custom_prompt": getattr(scene, "video_custom_prompt", ""),
```

#### API Endpoint
The existing `/fbtools/story/save` endpoint properly handles the new fields through the SceneInStory Pydantic model.

## Data Flow

### Loading a Story
1. User selects story from dropdown
2. Backend loads story.json and builds metadata payload
3. Frontend receives scenes array with video prompt fields
4. renderTable() renders the Advanced Flags tab
5. populateVideoPromptControls() fetches prompt data for each scene
6. Dropdowns are populated and previews are generated

### Changing Video Prompt Source
1. User changes video_prompt_source dropdown
2. Event handler updates currentScenes[idx].video_prompt_source
3. renderTable() is called to re-render the tab
4. populateVideoPromptControls() runs again to set up new input type
5. Preview is updated based on new source

### Changing Video Prompt Key/Custom Prompt
1. User changes dropdown/text input/textarea
2. Event handler updates currentScenes[idx] field
3. updateVideoPromptPreview(idx) is called
4. Preview textarea is updated with resolved prompt text

### Saving Changes
1. User clicks "💾 Save Changes" button
2. saveStory() sends currentScenes array to backend
3. Backend validates data through SceneInStory model
4. story.json is updated with new video prompt fields
5. Success message is displayed

## Testing

All 150 tests pass, including:
- Story save/load functionality
- Video prompt resolution in StoryVideoBatch
- Scene metadata serialization
- Prompt collection and composition tests

## User Experience

The enhanced UX provides:
- **Discoverability**: Users can see all available prompts/compositions
- **Validation**: Dropdowns prevent typos in key names
- **Feedback**: Live preview shows exactly what text will be used
- **Flexibility**: Supports both structured (prompt/composition) and freeform (custom) input
- **Clarity**: Different sources have appropriate input types

## Future Enhancements

Potential improvements:
1. Add search/filter to dropdown when many prompts exist
2. Show prompt metadata (category, tags) in dropdown options
3. Add button to copy prompt text from preview
4. Highlight differences between image prompt and video prompt
5. Add validation warnings for missing prompts/compositions
