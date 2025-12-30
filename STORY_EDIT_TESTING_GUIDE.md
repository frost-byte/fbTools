# StoryEdit Node Testing Guide

This document describes testing approaches for the StoryEdit node, which integrates both Python backend (ComfyUI node) and JavaScript frontend (table-based UI).

## Test Strategy

Given the ComfyUI dependency complexity, we recommend a three-tier testing approach:

### 1. Manual Integration Testing (Primary Approach)

Since StoryEdit depends heavily on ComfyUI's runtime environment, manual testing provides the most reliable verification.

#### Test Scenario 1: Basic Story Loading
```python
# In ComfyUI workflow:
1. Add StoryEdit node to canvas
2. Select existing story from "story_select" dropdown
3. Verify:
   - Story scenes appear in table UI
   - Preview scene loads correctly
   - All fields populated (mask type, prompt source, etc.)
```

#### Test Scenario 2: Scene Editing
```python
# In StoryEdit UI table:
1. Click on "Mask Type" dropdown for a scene
2. Change from "combined" to "girl" or "man"
3. Click "Save Story"
4. Click "Refresh"
5. Verify:
   - Changes persist across refresh
   - story.json updated on disk
```

#### Test Scenario 3: Scene Reordering
```python
# In StoryEdit UI:
1. Load story with 3+ scenes
2. Click "Move Down" on first scene
3. Click "Save Story"
4. Verify:
   - Scene order updated in table
   - scene_order values adjusted correctly
   - Preview still works
```

#### Test Scenario 4: Adding New Scene
```python
# In StoryEdit UI:
1. Click "Add Scene" button
2. Fill in:
   - Scene name: "new_scene_001"
   - Mask type: "combined"
   - Prompt source: "custom"
   - Custom prompt: "Test scene"
3. Click "Save Story"
4. Verify:
   - New scene appears in table
   - Can select as preview scene
   - Saved to story.json
```

#### Test Scenario 5: Advanced Flags
```python
# In StoryEdit UI:
1. Switch to "Advanced Flags" tab
2. Toggle checkboxes for a scene:
   - use_depth: ON
   - use_mask: ON
   - use_pose: OFF
   - use_canny: OFF
3. Click "Save Story"
4. Return to "Scenes" tab
5. Verify:
   - Flags persist in story data
   - UI reflects correct state after refresh
```

#### Test Scenario 6: Prompt Source Switching
```python
# In StoryEdit UI:
1. Select scene with prompt_source="prompt"
2. Verify "Prompt Key" dropdown visible
3. Change prompt_source to "custom"
4. Verify:
   - Dropdown replaced with textarea
   - Can enter custom text
   - Save persists custom prompt
```

### 2. Unit Testing (Helper Methods Only)

For stateless helper methods that don't require ComfyUI context:

```python
# tests/test_story_edit_helpers.py

def test_build_summary_text():
    """Test summary text generation"""
    from extension import StoryEdit
    
    # Mock minimal data
    story_data = MagicMock()
    story_data.story_name = "test"
    story_data.scenes = [MagicMock(), MagicMock(), MagicMock()]
    story_data.scenes[0].scene_name = "opening"
    story_data.scenes[1].scene_name = "middle"
    story_data.scenes[2].scene_name = "ending"
    
    preview_scene = story_data.scenes[1]
    preview_scene.scene_name = "middle"
    
    result = StoryEdit._build_summary_text(story_data, preview_scene)
    
    assert "Story: test" in result
    assert "Scenes: 3" in result
    assert "Preview: middle" in result
```

### 3. Frontend UI Testing (Jest)

Test UI rendering and event handling in isolation:

```javascript
// js-tests/story_edit.test.js (simplified)

describe("StoryEdit UI Rendering", () => {
    test("renders empty state when no story loaded", () => {
        const container = document.createElement("div");
        // Render empty UI
        expect(container.innerHTML).toContain("No story loaded");
    });
    
    test("renders scenes table with data", () => {
        const mockData = {
            scenes: [
                { scene_name: "scene_1", mask_type: "combined" },
                { scene_name: "scene_2", mask_type: "girl" }
            ]
        };
        // Render with data
        // Assert table contains scene names
    });
});
```

## Test Data Setup

### Sample Story Structure
```json
{
    "version": 2,
    "story_name": "test_story",
    "story_dir": "/path/to/test_story",
    "scenes": [
        {
            "scene_id": "scene_001",
            "scene_name": "opening",
            "scene_order": 0,
            "mask_type": "combined",
            "mask_background": true,
            "prompt_source": "prompt",
            "prompt_key": "girl_pos",
            "custom_prompt": "",
            "depth_type": "depth",
            "pose_type": "open",
            "use_depth": false,
            "use_mask": false,
            "use_pose": false,
            "use_canny": false
        }
    ]
}
```

## Running Tests

### Python Tests
```bash
cd /path/to/comfyui-fbTools
python3 -m pytest tests/test_story_edit_helpers.py -v
```

### JavaScript Tests
```bash
cd /path/to/comfyui-fbTools
npm test -- js-tests/story_edit.test.js
```

### Manual Testing Checklist

- [ ] Load existing story - UI shows all scenes
- [ ] Edit scene mask type - change persists
- [ ] Edit scene prompt source - UI updates correctly
- [ ] Add new scene - appears in table
- [ ] Delete scene - removed from table (with confirmation)
- [ ] Move scene up/down - order updates
- [ ] Toggle advanced flags - persists on save
- [ ] Save story - writes to disk
- [ ] Refresh - reloads from disk
- [ ] Preview scene selection - updates preview outputs
- [ ] Execute node in workflow - outputs correct images
- [ ] Metadata JSON - valid structure

## Known Testing Limitations

1. **ComfyUI Dependencies**: Extension.py heavily depends on ComfyUI modules (comfy, folder_paths, nodes) that aren't easily mockable
2. **File System**: Tests require actual file system operations for loading/saving stories
3. **Image Loading**: Preview assets require actual image files
4. **UI Integration**: JavaScript UI testing requires DOM environment

## Recommended Approach

For production testing:
1. Use **manual integration tests** for primary verification
2. Add **helper method unit tests** for pure functions
3. Consider **snapshot testing** for UI rendering
4. Implement **end-to-end tests** with actual ComfyUI instance running

## Test Coverage Goals

- ✅ Schema validation (inputs/outputs defined correctly)
- ✅ Story loading from disk
- ✅ Preview scene resolution
- ✅ Scene asset loading
- ✅ Summary text generation
- ✅ Metadata payload construction
- ✅ UI rendering (table, tabs, buttons)
- ✅ Event handlers (save/refresh/add/delete/move)
- ✅ State management (scene updates, flags)
- ✅ REST API calls (when implemented)

## Future Enhancements

1. Add Docker-based ComfyUI test environment
2. Implement Playwright for browser automation testing
3. Create test fixtures for common story structures
4. Add performance benchmarks for large stories (50+ scenes)
5. Implement visual regression testing for UI
