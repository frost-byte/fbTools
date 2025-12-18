# Plan: Flexible Multi-Prompt System with Libber Integration + Libber Bug Fixes

Refactor the hardcoded prompt system to support unlimited named prompts while maintaining backwards compatibility, integrating with Libber, AND fixing current Libber UI synchronization issues by implementing server-side state management via REST API.

## Steps

### 1. Create PromptCollection data model with REST API support

Define PromptMetadata(BaseModel) with fields: value, category, description, tags. Define PromptCollection(BaseModel) with fields:
- version=2
- v1_backup (Optional dict)
- prompts (dict[str, PromptMetadata])

Add from_legacy_dict() classmethod that preserves original data in v1_backup. Implement to_dict() for serialization. Register REST endpoints:
- POST /fbtools/prompts/create
- POST /fbtools/prompts/add
- POST /fbtools/prompts/remove
- GET /fbtools/prompts/list_names
- POST /fbtools/prompts/apply_libber (takes libber_session_id + prompt_collection_json, returns substituted collection)

### 2. Update SceneInfo with PromptCollection and backward compat properties

Replace individual prompt fields in SceneInfo with `prompts: Optional[PromptCollection] = None`. Add @property methods: `girl_pos`, `male_pos`, `wan_prompt`, `wan_low_prompt`, `four_image_prompt` that return `prompts.get_prompt_value("girl_pos")` if prompts exists, else return legacy field. Update save_prompts() to save v2 format with v1_backup section. Update load_prompt_json() to detect version, auto-migrate v1→v2 while preserving v1_backup.

### 3. Create prompt name discovery and dynamic selectors

Create node schema with inputs:
- prompt_collection_json_in
- operation (add/remove/update/rename/list)
- prompt_name
- prompt_value
- prompt_category
- prompt_description

Implement execute() to use REST API calls via LibberStateManager pattern for state management. Return UI text array: [0]=updated_prompt_collection_json, [1]=prompt_names_json (for dropdown). Add JS handler similar to StoryEdit to update prompt_name selector dropdown from text[1]. Add button widget in JS to trigger execute like LibberEdit button.

Add get_available_prompt_names(pose_dir: str) utility that reads prompts.json and returns union of v1 keys (girl_pos, male_pos, wan_prompt, wan_low_prompt, four_image_prompt) and v2 prompt names. Update SceneSelect, SceneCreate, SceneUpdate prompt_type Combo.Input to dynamically populate from available names. Add REST endpoint GET /fbtools/prompts/discover/{scene_name} that returns available prompt names. Add JS handler to fetch and update prompt selector dropdown on scene change.

### 4. Implement Scene REST API for lightweight metadata operations

Create SceneStateManager service class similar to LibberStateManager for server-side Scene instance management. Register REST endpoints for operations that don't require tensor manipulation:
- POST /fbtools/scene/update_prompts (update prompt text fields)
- POST /fbtools/scene/update_metadata (update pose_name, mask_type, depth_type, resolution, loras)
- POST /fbtools/scene/save_metadata (persist metadata to files: prompts.json, loras.json, pose.json)
- GET /fbtools/scene/list_scenes/{story_dir}
- POST /fbtools/scene/update_loras (add/remove lora entries)

Keep tensor operations (image loading, depth map generation, pose processing) in normal execute() flow for performance. REST API handles only text/metadata for responsive UI updates. Add JS handlers to SceneEdit/SceneUpdate nodes to use REST endpoints for prompt/lora/metadata changes, similar to LibberEdit pattern.

### 8. Implement execution-based output organization for two-stage Story workflows

**Us5. Add PromptCollectionEdit node with REST backend

Create node schema with inputs:
- prompt_collection_json_in
- operation (add/remove/update/rename/list)
- prompt_name
- prompt_value
- prompt_category
- prompt_description

Implement execute() to use REST API calls via LibberStateManager pattern for state management. Return UI text array: [0]=updated_prompt_collection_json, [1]=prompt_names_json (for dropdown). Add JS handler similar to StoryEdit to update prompt_name selector dropdown from text[1]. Add button widget in JS to trigger execute like LibberEdit button.

### 6 Case**: Two-stage pipeline:
1. **Image Generation**: User provides input image → loops through story scenes → generates 1+ images per scene using prompts/masks/sampling
2. **Video Generation**: Takes images from Stage 1 → uses first-to-last frame transitions between consecutive scenes → creates videos → stitches together

**Requirements**:
- All images from one execution must be in single directory (not scattered across scene subdirectories)
- Filenames sort by scene order (000_scene1, 001_scene2, etc.)
- Easy to identify images from specific execution for Stage 2 video workflow
- Support multiple variants per scene (if sampling generates multiple images)
- Video workflow needs sequential access to scene images for transitions

**Directory Structure**:
```
stories/{story_name}/jobs/{job_id}/
  executions/{exec_id}/
    input/
      source_{input_hash}.png
    images/
      {scene_order:03d}_{scene_name}_{variant:02d}.png
      000_opening_00.png
      000_opening_01.png
      001_action_00.png
      001_action_01.png
      002_finale_00.png
    videos/
      000_opening_to_001_action.mp4
      001_action_to_002_finale.mp4
    metadata.json  # execution metadata: input_hash, timestamp, scene_count
```

**Implementation Steps**:

1. **Add StoryExecutionInit node**:
   - Inputs: IMAGE (input image), story_info, job_id (from StorySceneBatch), optional exec_id
   - Generates exec_id from timestamp+random if not provided (format: `YYYYMMDD_HHMMSS_{rand6}`)
   - Computes input_hash from image tensor (SHA256 of normalized tensor, first 12 chars)
   - Creates execution directory structure: `{story_dir}/jobs/{job_id}/executions/{exec_id}/`
   - Creates subdirs: `input/`, `images/`, `videos/`
   - Saves input image to `input/source_{input_hash}.png`
   - Saves metadata.json with: exec_id, input_hash, timestamp, story_name, scene_count
   - Outputs: exec_id, exec_root_dir, images_dir, videos_dir, input_hash, IMAGE (passthrough)

2. **Update StorySceneBatch to support execution context**:
   - Add optional input: exec_id (STRING)
   - When exec_id provided, add to each scene descriptor: `exec_id`, `exec_images_dir`
   - Output additional field: exec_images_dir (for connecting to SaveImage nodes)

3. **Update StoryScenePick to expose execution paths**:
   - Extract exec_id and exec_images_dir from descriptor if present
   - Add outputs: exec_images_dir (STRING), exec_videos_dir (STRING)
   - Allows connecting directly to SaveImage or video nodes

4. **Add StoryImageNamer utility node**:
   - Inputs: scene_order (INT), scene_name (STRING), variant_index (INT, default=0)
   - Generates standardized filename: `{scene_order:03d}_{scene_name}_{variant_index:02d}`
   - Output: filename_prefix (STRING) - can connect to SaveImage's filename_prefix
   - Pure naming logic - no path resolution

5. **Add StoryPathResolver utility node**:
   - Inputs: exec_images_dir (STRING), filename_prefix (STRING), extension (STRING, default=".png")
   - Computes multiple path formats for different SaveImage node requirements
   - Outputs:
     - `abs_directory` (STRING) - absolute path to directory: `/path/to/ComfyUI/output/stories/my_story/jobs/job1/executions/exec1/images`
     - `abs_filepath` (STRING) - absolute full path with extension: `/path/to/.../images/000_scene_00.png`
     - `abs_filepath_no_ext` (STRING) - absolute path without extension: `/path/to/.../images/000_scene_00`
     - `rel_directory` (STRING) - relative to ComfyUI output: `stories/my_story/jobs/job1/executions/exec1/images`
     - `rel_filepath` (STRING) - relative full path with extension: `stories/.../images/000_scene_00.png`
     - `rel_filepath_no_ext` (STRING) - relative path without extension: `stories/.../images/000_scene_00`
     - `basename` (STRING) - filename only: `000_scene_00`
     - `basename_with_ext` (STRING) - filename with extension: `000_scene_00.png`
   - Handles different SaveImage node conventions automatically
   - User picks appropriate output based on their SaveImage node's requirements

6. **Add StoryImageCollector utility node**:
   - Inputs: exec_images_dir (STRING), scene_order_filter (INT, optional), scene_name_filter (STRING, optional)
   - Scans exec_images_dir for images matching filters
   - Parses filenames to extract scene_order, scene_name, variant
   - Outputs: 
     - image_paths (LIST[STRING]) - sorted by scene order, then variant
     - image_count (INT)
     - scene_groups (DICT) - grouped by scene: `{scene_order: [path1, path2, ...]}`
   - Used in Stage 2 to collect images for video generation

6. **Add StoryVideoNamer utility node**:
   - Inputs: scene_order (INT), scene_name (STRING), next_scene_name (STRING)
   - Generates filename: `{scene_order:03d}_{scene_name}_to_{next_scene_name}`
   - Output: filename_prefix (STRING) - for video SaveImage/SaveVideo nodes

7. **Add StoryVideoPathResolver utility node**:
   - Inputs: exec_videos_dir (STRING), filename_prefix (STRING), extension (STRING, default=".mp4")
   - Same multi-format outputs as StoryPathResolver but for videos directory
   - Supports different video SaveVideo/SaveImage node conventions
   - Outputs: abs_directory, abs_filepath, rel_directory, rel_filepath, basename, basename_with_ext, etc.

8. **Backward compatibility**:
   - When exec_id not provided, nodes function as before
   - StorySceneBatch without exec_id uses original `jobs/{job_id}/{scene_order:03d}_{scene_name}/` structure
   - Existing workflows unaffected

**Usage Example - Stage 1 (Image Generation)**:
``StoryPathResolver(exec_images_dir, filename_prefix, ".png") → path outputs
  [Image Generation Pipeline: CLIPEncode, KSampler, etc.]
  
  # Different SaveImage node examples:
  # Option A: Node wants directory + basename separately
  SaveImage(directory=abs_directory, filename=basename_with_ext)
  
  # Option B: Node wants filename_prefix (no extension)
  SaveImage(filename_prefix=basename, output_path=abs_directory)
  
  # Option C: Node wants relative path to ComfyUI output
  SaveImage(subfolder=rel_directory, filename_prefix=basename)
  
String(exec_videos_dir) → from StoryExecutionInit
Loop over scene_groups:
  LoadImage(scene_N_image) → first_frame
  LoadImage(scene_N+1_image) → last_frame
  StoryVideoNamer(scene_order, scene_name, next_scene_name) → video_filename
  StoryVideoPathResolver(exec_videos_dir, video_filename, ".mp4") → path outputs
  [Video Generation Pipeline: frame interpolation, etc.]
  
  # Video save node examples:
  # Option A: Separate directory + filename
  SaveVideo(directory=abs_directory, filename=basename_with_ext)
  
  # Option B: Relative path convention
  SaveVideo(subfolder=rel_directory, filename_prefix=basename)
  [Image Generation Pipeline: CLIPEncode, KSampler, etc.]
  SaveImage(filename_prefix, exec_images_dir) → saves to executions/{exec_id}/images/
```

**Usage Example - Stage 2 (Video Generation)**:
```
String(exec_id) → StoryImageCollector(exec_images_dir) → image_paths, scene_groups
Loop over scene_groups:
  LoadImage(scene_N_image) → first_frame
  LoadImage(scene_N+1_image) → last_frame
  StoryVideoNamer(scene_order, scene_name, next_scene_name) → video_filename
  [Video Generation Pipeline: frame interpolation, etc.]
  SaveVideo(video_filename, exec_videos_dir) → saves to executions/{exec_id}/videos/
```

**# 7. Research and implement ComfyUI v3 REST API for Libber

Investigate comfy_api.latest extension API for registering custom endpoints (check ComfyExtension base class methods, PromptServer integration). Create LibberStateManager service class that maintains in-memory Libber instances keyed by session/workflow ID. Register REST endpoints:
- POST /fbtools/libber/create
- POST /fbtools/libber/add_lib
- POST /fbtools/libber/remove_lib
- GET /fbtools/libber/keys
- GET /fbtools/libber/get_lib/{key}

Reference global_seed_manager.py pattern but adapt for FBToolsExtension.register_routes() method or similar v3 API pattern.

### 8. Refactor LibberEdit to use REST API instead of complex client state

Remove onWidgetChanged and callback manipulation from fb_tools.js LibberEdit handler. Replace with fetch() calls to REST endpoints when operation button clicked. Update key_selector dropdown by fetching /fbtools/libber/keys after each operation. Store libber_session_id widget to track which server-side Libber instance is being edited. Simplify JS to just: trigger REST call on button click → wait for response → update dropdowns from server response.

##Benefits**:
- **Single directory per execution**: All images in `executions/{exec_id}/images/` - easy to find
- **Sortable filenames**: Scene order prefix ensures correct ordering for video generation
- **Stage separation**: Images and videos in separate subdirs, clean organization
- **Multiple variants supported**: Can generate N images per scene, all properly named
- **Metadata tracking**: metadata.json records execution details for reference
- **Video workflow ready**: StoryImageCollector and scene_groups enable easy sequential processing
- **No filename collisions**: exec_id ensures unique directory per workflow run
- **Human readable**: Filenames encode scene order and name, easy to understand

## Further Considerations

### 1. Session management for server-side state

How to handle multiple users/workflows editing Libbers simultaneously?
- **Option A**: Use workflow_id from ComfyUI execution context
- **Option B**: Generate unique session IDs on LibberCreate/Load
- **Option C**: Store in temporary Redis/dict keyed by node_id + execution_id
- **Recommended: Option B with 30min TTL** - simpler, works for both API and UI usage

### 2. Error handling and state recovery

What happens if server restarts while user editing Libber?
- **Option A**: Auto-save to temp files every operation
- **Option B**: Only persist on explicit LibberSave
- **Option C**: Save state in browser localStorage + server
- **Recommended: Option B + localStorage** - explicit save prevents data loss, localStorage allows client-side recovery

### 3. Testing strategy

How to test REST endpoints and state management? Should we add test_libber_api.py using pytest + aiohttp test client? Should JS changes have integration tests?
- **Recommended**: Add pytest tests for REST endpoints, manual testing for JS (ComfyUI doesn't have standard JS test framework)

### 4. Migration rollout

Should v2 prompt system be opt-in or automatic?
- **Option A**: Add "use_v2_prompts" flag to SceneInfo
- **Option B**: Automatic migration with warning log
- **Option C**: Separate V2 nodes (SceneSelectV2, etc.)
- **Recommended: Option B** - seamless upgrade path, less user confusion, v1_backup provides safety net

## Migration Details

### Non-destructive Migration Strategy

All migration operations must be non-destructive:

1. **File-level backup**: When migrating prompts.json:
   - Create prompts.json.v1.backup before any modifications
   - Store original data in v1_backup field within new structure
   - Structure: `{"version": 2, "v1_backup": {original data}, "prompts": {migrated data}}`

2. **v1_backup immutability**: 
   - Once created, v1_backup field is frozen
   - All edits apply only to v2 prompts section
   - v1_backup provides rollback option if needed

3. **Backward compatibility**: 
   - @property methods on SceneInfo delegate to PromptCollection
   - Old code accessing scene.girl_pos continues to work
   - New code can use scene.prompts.get("girl_pos") for explicit access

### Prompt Naming Conventions

Instead of hardcoded standards, use file-based discovery:

1. **Dynamic dropdown population**: 
   - Read prompts.json from selected scene directory
   - Extract all v1 keys (girl_pos, male_pos, etc.) if v1_backup exists
   - Extract all v2 prompt names from prompts dict
   - Combine and deduplicate for dropdown options

2. **User-defined prompt names**:
   - No validation on prompt naming - fully flexible
   - Categories are optional metadata, not enforced
   - Users can create any prompt structure their workflow needs

3. **Discovery across scenes**:
   - get_available_prompt_names(pose_dir) returns names from single scene
   - get_all_prompt_names_in_story(story_info) aggregates across story
   - Useful for consistency checking without constraining creativity
