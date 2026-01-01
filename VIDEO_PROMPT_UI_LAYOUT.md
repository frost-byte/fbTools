# Video Prompt UI Layout Reference

## Advanced Flags Tab Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ 📋 Scenes    🏴 Advanced Flags                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ 💡 Advanced Flags: Control video prompts and which control      │
│    inputs are used per scene during generation.                  │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ scene_name_001                                               │ │
│ │                                                              │ │
│ │ 🎬 Video Prompt Settings:                                   │ │
│ │                                                              │ │
│ │ video_prompt_source: [auto ▼]                               │ │
│ │                                                              │ │
│ │ video_prompt_key:    [                      ] (text input)  │ │
│ │                                                              │ │
│ │ Preview:                                                     │ │
│ │ ┌──────────────────────────────────────────────────────────┐│ │
│ │ │ (Using image prompt - will be resolved at generation)    ││ │
│ │ │                                                           ││ │
│ │ └──────────────────────────────────────────────────────────┘│ │
│ │                                                              │ │
│ │ 🏴 Control Flags:                                           │ │
│ │ ☐ use_depth   ☑ use_mask   ☐ use_pose   ☐ use_canny       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ scene_name_002                                               │ │
│ │                                                              │ │
│ │ 🎬 Video Prompt Settings:                                   │ │
│ │                                                              │ │
│ │ video_prompt_source: [prompt ▼]                             │ │
│ │                                                              │ │
│ │ video_prompt_key:    [character_intro ▼] (dropdown)         │ │
│ │                                                              │ │
│ │ Preview:                                                     │ │
│ │ ┌──────────────────────────────────────────────────────────┐│ │
│ │ │ A young woman with flowing auburn hair, wearing a blue   ││ │
│ │ │ dress, standing in a sunlit garden                       ││ │
│ │ └──────────────────────────────────────────────────────────┘│ │
│ │                                                              │ │
│ │ 🏴 Control Flags:                                           │ │
│ │ ☑ use_depth   ☐ use_mask   ☑ use_pose   ☐ use_canny       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ scene_name_003                                               │ │
│ │                                                              │ │
│ │ 🎬 Video Prompt Settings:                                   │ │
│ │                                                              │ │
│ │ video_prompt_source: [composition ▼]                        │ │
│ │                                                              │ │
│ │ video_prompt_key:    [action_sequence ▼] (dropdown)         │ │
│ │                                                              │ │
│ │ Preview:                                                     │ │
│ │ ┌──────────────────────────────────────────────────────────┐│ │
│ │ │ running through the forest, leaping over fallen logs,    ││ │
│ │ │ dramatic action scene, dynamic camera angle              ││ │
│ │ └──────────────────────────────────────────────────────────┘│ │
│ │                                                              │ │
│ │ 🏴 Control Flags:                                           │ │
│ │ ☑ use_depth   ☑ use_mask   ☑ use_pose   ☑ use_canny       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ scene_name_004                                               │ │
│ │                                                              │ │
│ │ 🎬 Video Prompt Settings:                                   │ │
│ │                                                              │ │
│ │ video_prompt_source: [custom ▼]                             │ │
│ │                                                              │ │
│ │ custom_prompt:                                               │ │
│ │ ┌──────────────────────────────────────────────────────────┐│ │
│ │ │ slow motion shot of character looking back over shoulder ││ │
│ │ │ as the sun sets, golden hour lighting, cinematic        ││ │
│ │ │                                                           ││ │
│ │ │                                                           ││ │
│ │ └──────────────────────────────────────────────────────────┘│ │
│ │                                                              │ │
│ │ Preview:                                                     │ │
│ │ ┌──────────────────────────────────────────────────────────┐│ │
│ │ │ slow motion shot of character looking back over shoulder ││ │
│ │ │ as the sun sets, golden hour lighting, cinematic        ││ │
│ │ └──────────────────────────────────────────────────────────┘│ │
│ │                                                              │ │
│ │ 🏴 Control Flags:                                           │ │
│ │ ☐ use_depth   ☐ use_mask   ☐ use_pose   ☐ use_canny       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ 💾 Save Changes                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Input Type Matrix

| video_prompt_source | video_prompt_key field | custom_prompt field | Preview shows |
|---------------------|------------------------|---------------------|---------------|
| `auto`              | text input (empty)     | hidden              | Image prompt or placeholder |
| `prompt`            | dropdown (prompt keys) | hidden              | Selected prompt value |
| `composition`       | dropdown (comp names)  | hidden              | Composed prompt (joined) |
| `custom`            | hidden                 | textarea (editable) | Custom prompt text |

## Interaction Flow

### Selecting a Prompt Source
```
1. User clicks video_prompt_source dropdown
2. Selects "prompt" option
3. Tab re-renders with dropdown for video_prompt_key
4. Dropdown populates with available prompt keys from scene's prompts.json
5. Preview updates to show selected prompt value
```

### Changing Prompt Key
```
1. User clicks video_prompt_key dropdown
2. Selects "character_intro" from available options
3. updateVideoPromptPreview() is called
4. Preview textarea updates with prompt text:
   "A young woman with flowing auburn hair..."
5. Change is saved to currentScenes[idx].video_prompt_key
```

### Entering Custom Prompt
```
1. User selects "custom" in video_prompt_source
2. Tab re-renders with custom_prompt textarea
3. User types custom text
4. As user types, preview updates in real-time (on input event)
5. Change is saved to currentScenes[idx].video_custom_prompt
```

### Saving Changes
```
1. User clicks "💾 Save Changes"
2. currentScenes array is sent to /fbtools/story/save
3. Backend validates and saves to story.json
4. Success message appears: "✓ Story saved successfully"
5. Message fades after 2 seconds
```

## Field Descriptions

### video_prompt_source
Determines how the video generation prompt is obtained:
- **auto** (default): Use the scene's image generation prompt
- **prompt**: Use a specific prompt from prompts.json
- **composition**: Use a composition (multiple prompts joined)
- **custom**: Use custom freeform text

### video_prompt_key
When source is "prompt" or "composition":
- Key name to lookup in the scene's prompts.json
- Dropdown populated with available keys
- Selection updates preview immediately

### video_custom_prompt
When source is "custom":
- Freeform text for video generation
- Supports multiline input
- Updates preview as user types

### Preview
- Always visible, readonly textarea
- Shows the actual prompt text that will be used
- Updates dynamically based on source and selection
- Provides immediate feedback to user

## Styling

- Uses ComfyUI CSS variables for consistent theming
- Monospace font for preview textarea (easier to read prompts)
- Responsive grid layout for labels and inputs
- Emoji icons (🎬, 🏴) for visual distinction
- Proper spacing and padding for readability
