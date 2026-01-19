# ComfyUI fbTools Development Guide

## Critical Checklist: Adding New Nodes

When creating a new ComfyUI node, **always** complete these steps:

1. Create node class definition in `extension.py`
2. Define schema with `@classmethod def define_schema(cls)`
   - **⚠️ CRITICAL**: Input and output names must be unique across the entire node
   - Example: Cannot have both an input `mask` and an output `mask` - use `input_mask` and `output_mask`
   - Violation causes: `"Ids must be unique between inputs and outputs"` error
3. Implement `@classmethod def execute(cls, ...)`
   - Parameter names must match input names from schema
   - **⚠️ CRITICAL**: Use relative imports for our utility modules: `from .utils.images import ...`
   - Wrong: `from utils.images import ...` (causes `ModuleNotFoundError`)
   - **⚠️ CRITICAL**: Return `io.NodeOutput(arg1, arg2, ...)` with positional args, NOT a tuple or dict
   - Wrong: `return (val1, val2)` or `return {"key": val}` (causes `'dict' object has no attribute 'shape'` errors)
   - Correct: `return io.NodeOutput(val1, val2)` (args must match OUTPUT_TYPES order)
4. **⚠️ REGISTER in `get_node_list()` at end of extension.py** (line ~4961)
5. Restart ComfyUI service: `sudo systemctl restart comfyui_377`
6. Test node in browser

## Tensor Format Conventions

### Masks
- **Format**: `[B, H, W]` (3D tensor)
- **Values**: `0.0 - 1.0` (float32)
- **Device**: Preserve original device when processing
- **Wrong**: `[B, H, W, 1]` or `[1, 1, H, W]` (causes pipeline breaks)

### Images
- **Format**: `[B, H, W, C]` (4D tensor, NHWC)
- **Channels**: 3 (RGB) or 4 (RGBA)
- **Values**: `0.0 - 1.0` (float32)
- **Device**: Preserve original device

### General Tensor Rules
- Always use `.to(device)` when creating new tensors
- Use `.detach()` before converting to numpy/PIL
- Preserve dtype unless explicitly converting

## Architecture Patterns

### Scene Image Workflow
- **base_image**: The original input image (saved as `base.png` in scene directory)
  - The true source of all scene content
  - User can update via `update_base=True` in SceneUpdate
  - Updating base_image triggers regeneration of everything
- **upscale_image**: Scaled version of base_image (saved as `upscale.png`)
  - Created from base_image using upscale_factor and upscale_method
  - Acts as the source for all derived images (pose, depth, canny, etc.)
  - In **SceneCreate**: Created by upscaling the input base_image
  - In **SceneUpdate**: 
    - If `update_base=True` + new base_image: Regenerate from new base
    - If `update_upscale=True`: Apply scale factor to existing upscale_image
    - If neither: Use existing upscale_image from scene
    - All other regeneration (pose, depth, canny) uses whichever upscale_image we have
- **Derived images**: pose_*, depth_*, canny_image
  - Generated FROM the upscale_image
  - Can be regenerated individually using `update_*` flags
  - All normalized to match upscale_image dimensions
  
**Image Hierarchy**: base_image → upscale_image → derived images (pose, depth, canny)

### Singleton Pattern
- **LibberStateManager**: Use `LibberStateManager.instance()` not `LibberStateManager()`
- Single source of truth for libber data across all nodes

### Data Models
- **PromptCollection v2**: Current format with `prompts` dict and `compositions` dict
- **SceneInfo**: Contains scene metadata, prompts, compositions, images
- **PromptMetadata**: `key`, `value`, `category`, `processing_type`

### State Management
- Scene data persists in `.json` files
- Libber data loaded from `libber.yaml` files
- Priority: `collection_json` → `scene_info.prompts` → new collection

## File Organization

### Backend (Python)
- **extension.py**: Main node definitions (4800+ lines)
- **utils/images.py**: Image/mask processing helpers
- **utils/io.py**: File I/O utilities
- **utils/pose.py**: Pose-related helpers
- **utils/util.py**: General utilities

### Frontend (JavaScript)
- **js/nodes/scene.js**: Scene node UI setup
- **js/nodes/libber.js**: Libber node UI
- **js/nodes/story.js**: Story node UI
- **js/api/**: API client wrappers
- **js/ui/**: Reusable UI components

### API Endpoints
- Pattern: `/fbtools/{category}/{action}`
- Example: `/fbtools/scene/get_scene_prompts` (GET)
- Example: `/fbtools/scene/process_compositions` (POST)

## Common Operations

### Adding Helper Functions
- **Image processing**: `utils/images.py`
- **Mask operations**: `utils/images.py` (with cv2 availability check)
- **File operations**: `utils/io.py`

### API Integration
1. Add endpoint in `extension.py` under `@server.PromptServer.instance.routes`
2. Create API client in `js/api/{category}.js`
3. Call from node setup in `js/nodes/{category}.js`

### UI Components
- Tables: Use `createTable()` pattern
- Tabs: Track with state variable (e.g., `activeDisplayTab`)
- Dropdowns: Populate in `onConfigure` hook
- Drag-and-drop: HTML5 API with visual feedback

## Testing Workflow

### After Code Changes
1. Save changes
2. Restart service: `sudo systemctl restart comfyui_377`
3. Refresh browser (hard refresh: Ctrl+Shift+R)
4. Test in ComfyUI workflow
5. Check browser console for errors (F12)
6. Check backend logs: `sudo journalctl -u comfyui_377 -f`

### Common Issues
- **Node not appearing**: Check `get_node_list()` registration
- **"Ids must be unique" error**: Input and output names conflict (e.g., both named "mask")
- **Tensor shape errors**: Verify 3D vs 4D format
- **Device errors**: Ensure `.to(device)` on new tensors
- **API not responding**: Check endpoint definition and routing
- **UI not updating**: Clear browser cache, check onConfigure hook

## Dependencies

### Python
- `torch`: Tensor operations
- `cv2` (opencv): Image processing (check `_HAS_CV2` flag)
- `kornia`: Advanced image operations (check `_HAS_KORNIA` flag)
- `skimage`: Histogram matching (check `_HAS_SKIMAGE` flag)
- `PIL/Pillow`: Image loading/saving
- `numpy`: Array operations

### JavaScript
- ES6 modules
- LiteGraph (ComfyUI framework)
- Fetch API for requests

## Naming Conventions

### Nodes
- PascalCase: `SceneSelect`, `LibberManager`, `MaskProcessor`
- Display names: Space-separated with proper caps

### Categories
- Format: `"🧊 frost-byte/{Category}"`
- Examples: `"🧊 frost-byte/Scene"`, `"🧊 frost-byte/Image Processing"`

### Variables
- Python: `snake_case`
- JavaScript: `camelCase`
- Constants: `UPPER_SNAKE_CASE`

## Code Quality

### Error Handling
- Always check for cv2/kornia availability before using
- Graceful fallbacks when dependencies missing
- Clear error messages with context

### Logging
- Use `print()` for important operations
- Include context: `f"NodeName: operation - details"`
- Debug flags for verbose output

### Documentation
- Docstrings for all classes and complex functions
- Inline comments for non-obvious logic
- Update relevant .md files when adding features

## Git Workflow

## Workflow Optimization

Use the following preferences for:

1. **Git commits**: 
   - Suggest commits when functionality is complete for a feature or bug fix
   - Can create series of local commits during development, then squash/cherry-pick into unified commit to keep history clean
   - Detailed commit messages using concise language; detailed descriptions go in CHANGELOG.md
   - Ask whether to push after suggesting commit

2. **Service restarts**:
   - **Always explicitly remind** to run `sudo systemctl restart comfyui_377` after backend/Python changes
   - User may defer restart if suggesting additional changes before testing

3. **Browser refresh**:
   - Remind to refresh browser after service restart and frontend code updates
   - Remind to refresh/reload nodes in ComfyUI if inputs/outputs have changed in a node

4. **Testing reminders**:
   - Draft test code for new features and changes (both frontend and backend)
   - Provide overview of each test to aid understanding
   - Follow existing methodology from `tests/` and `js-tests/` directories
   - Suggest testing steps after implementations
   - Test scenarios should verify behavior matches design goals

5. **Documentation**:
   - Update all relevant .md files when features are added or changed
   - Perform documentation pass at end of session before pushing
   - Review: README, SCENE_NODES_README, STORY_NODES_README, LIBBER_NODES_README, etc.
   - Update CHANGELOG.md with detailed changes

6. **Code organization**:
   - Suggest refactoring when files get large or there's redundancy/repetition
   - Recommend module partitioning to reduce structural complexity
   - Create new utility files or extend existing ones when there's overlap in usage

7. **Error checking**:
   - Run `get_errors` tool **after all edits** for a request are complete
   - Check Python and JavaScript files in our repository
   - Errors may resolve themselves after all changes are implemented

8. **Session management**:
   - Provide summary when user says "That is all for now" or "Thanks, that's good enough for now"
   - Can offer summary after significant milestones (ask first)
   - Track TODO items across sessions
   - Provide reminder/overview of previous session when new session starts

---

*Last updated: December 21, 2025*
