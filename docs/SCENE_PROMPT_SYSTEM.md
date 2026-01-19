# Scene Prompt Management System - Implementation Plan

## Overview
Node-level prompt composition system for flexible, workflow-specific prompt assembly.

## Architecture

### Data Layer (✅ COMPLETE)
**PromptMetadata** - Stores individual prompt information:
- `value`: The prompt text
- `processing_type`: "raw" or "libber"
- `libber_name`: Which libber to use (if processing_type="libber")
- `category`, `description`, `tags`: Organization/metadata

**PromptCollection** - Container for prompts:
- Stores prompts by key
- Legacy migration from v1 format
- Helper methods: `get_prompt_metadata()`, `get_prompts_by_category()`

### Node Layer (TODO)
**ScenePromptManager** - Create/edit prompts:
- **Inputs**: `scene_info` (optional), `prompt_collection_in` (JSON string)
- **UI**: Interactive table (similar to LibberManager)
  * Columns: Key | Value | Type | Libber Name | Actions
  * Add/Edit/Remove prompts
  * Visual indicators: 📝 (raw) vs 🔄 (libber)
- **Outputs**: Updated `scene_info`, `prompt_collection_json`

**ScenePromptComposer** - Assign prompts to outputs:
- **Inputs**: `scene_info` or `prompt_collection`, `composition_map_in` (JSON)
- **UI**: Interactive assignment table
  * List of available prompts (from collection)
  * Checkboxes to select prompts for each output
  * Drag-and-drop or up/down buttons to reorder
  * Four output sections: prompt_a, prompt_b, prompt_c, prompt_d
- **Outputs**: 
  * `prompt_a`, `prompt_b`, `prompt_c`, `prompt_d` (processed strings)
  * `composition_map_out` (JSON) - for saving/loading configurations

**SceneSelect** (UPDATE) - Enhanced to show prompts:
- Add read-only prompt display table
- Show processing type icons
- Keep existing functionality
- Optionally output composed prompts

## Composition Map Format

```json
{
  "prompt_a": [
    {"key": "girl_pos", "order": 0},
    {"key": "male_pos", "order": 1}
  ],
  "prompt_b": [
    {"key": "wan_prompt", "order": 0}
  ],
  "prompt_c": [
    {"key": "wan_low_prompt", "order": 0}
  ],
  "prompt_d": [
    {"key": "four_image_prompt", "order": 0}
  ]
}
```

## Use Cases

### Image Generation Workflow
1. ScenePromptManager: Create prompts (girl_pos, male_pos, quality, style)
2. ScenePromptComposer: Assign girl_pos + male_pos + quality → prompt_a
3. Connect prompt_a to image generation node

### Video Generation Workflow  
1. Same prompts from ScenePromptManager
2. ScenePromptComposer: Different assignment:
   - wan_prompt + quality + style → prompt_b (high model)
   - wan_low_prompt + style → prompt_c (low model)
3. Connect prompt_b/prompt_c to video nodes

### Libber-Enhanced Workflow
1. ScenePromptManager: Create prompt with placeholders
   - Key: "character", Value: "A %quality% %type% warrior", Type: libber, Libber: "char_lib"
2. LibberManager: Define libs (quality="epic", type="chunky")
3. ScenePromptComposer: Process and output
   - Result: "A epic chunky warrior"

## JavaScript UI Components

### PromptTable (for ScenePromptManager)
- Editable table with add/edit/remove
- Type dropdown (raw/libber)
- Libber name input (conditional on type)
- Similar to LibberManager table

### CompositionTable (for ScenePromptComposer)
- Left panel: Available prompts (checkboxes)
- Right panels: Four output slots with drag-and-drop
- Visual feedback for prompt assignments
- Up/down reorder buttons

## REST API Endpoints (if needed)
- `/fbtools/prompts/compose` - Process prompts with libbers
- Reuse existing prompt collection endpoints

## Migration Path
1. Load legacy scene JSON
2. PromptCollection.from_legacy_dict() converts to v2
3. User opens ScenePromptComposer
4. Suggested default composition based on legacy keys:
   - girl_pos, male_pos → prompt_a
   - wan_prompt → prompt_b  
   - wan_low_prompt → prompt_c
   - four_image_prompt → prompt_d
5. User can modify and save

## Next Steps
1. ✅ Simplify PromptMetadata (remove output_slot, order)
2. ✅ Update tests
3. ✅ Commit changes
4. 🔄 Create ScenePromptManager node
5. 🔄 Create ScenePromptComposer node
6. 🔄 Create JavaScript UI for both nodes
7. 🔄 Update SceneSelect to display prompts
8. 🔄 Test integration with LibberApply
9. 🔄 Documentation
10. 🔄 Examples/templates
