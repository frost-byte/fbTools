# Scene Building Nodes Documentation

## Overview

The Scene Building system provides a comprehensive workflow for creating, managing, and visualizing scene data for ComfyUI. Each scene contains pose information, depth maps, masks, and prompts that can be used in image generation workflows.

A **Scene** consists of:
- **Pose data**: Multiple pose detection formats (DWPose, OpenPose, DensePose)
- **Depth maps**: Various depth estimation methods (Depth Anything v2, MiDaS, Zoe)
- **Masks**: Character segmentation masks (with/without background)
- **Prompts**: Multiple prompt types for different generation methods
- **LoRA configurations**: High and low quality LoRA presets

## Scene Data Structure

### SceneInfo

The core data structure containing all scene information:

```python
SceneInfo(
    # Metadata
    pose_dir: str              # Directory where scene files are stored
    pose_name: str             # Unique name for the scene
    resolution: int            # Target resolution for images
    
    # Prompts (5 types)
    girl_pos: str              # Positive prompt for female character
    male_pos: str              # Positive prompt for male character(s)
    four_image_prompt: str     # Four-image generation prompt
    wan_prompt: str            # WanVideo high-quality prompt
    wan_low_prompt: str        # WanVideo low-quality prompt
    
    # Pose Data
    pose_json: str             # OpenPose keypoint data (JSON)
    pose_dwpose_json: str      # DWPose specific data (optional)
    
    # Depth Images (5 variants)
    depth_image: Tensor        # Depth Anything v2
    depth_any_image: Tensor    # Depth Anything v1
    depth_midas_image: Tensor  # MiDaS depth
    depth_zoe_image: Tensor    # Zoe depth
    depth_zoe_any_image: Tensor # Zoe Anything depth
    
    # Pose Images (7 variants)
    pose_dense_image: Tensor   # DensePose visualization
    pose_dw_image: Tensor      # DWPose output
    pose_edit_image: Tensor    # Editable pose
    pose_face_image: Tensor    # Face-focused pose
    pose_open_image: Tensor    # OpenPose output
    canny_image: Tensor        # Canny edge detection
    upscale_image: Tensor      # Upscaled base image
    
    # Mask Images (6 variants)
    girl_mask_bkgd_image: Tensor      # Female mask with background
    male_mask_bkgd_image: Tensor      # Male mask with background
    combined_mask_bkgd_image: Tensor  # Combined mask with background
    girl_mask_no_bkgd_image: Tensor   # Female mask without background
    male_mask_no_bkgd_image: Tensor   # Male mask without background
    combined_mask_no_bkgd_image: Tensor # Combined mask without background
    
    # LoRA Configurations
    loras_high: list           # High-quality LoRA list
    loras_low: list            # Low-quality LoRA list
)
```

### Helper Methods

The `SceneInfo` class provides several helper methods for common operations:

#### Loading Images
```python
# Load all depth images from a directory
depth_images = SceneInfo.load_depth_images(pose_dir)

# Load all pose images
pose_images = SceneInfo.load_pose_images(pose_dir)

# Load all mask images
mask_images = SceneInfo.load_mask_images(pose_dir)

# Load everything at once
all_images = SceneInfo.load_all_images(pose_dir)
```

#### Saving Data
```python
# Save all images
scene_info.save_all_images(pose_dir)

# Save prompts to prompts.json
scene_info.save_prompts(pose_dir)

# Save pose keypoints to pose.json
scene_info.save_pose_json(pose_dir)

# Save LoRA configurations
scene_info.save_loras(pose_dir)

# Create necessary directories
scene_info.ensure_directories(pose_dir)

# Save everything (images, prompts, pose_json, loras)
scene_info.save_all(pose_dir)
```

#### Factory Method
```python
# Create SceneInfo by loading from a directory
scene_info = SceneInfo.from_pose_directory(
    pose_dir="/path/to/scene",
    pose_name="my_scene",
    prompt_data=None,  # Will load from prompts.json if None
    pose_json="",
    loras_high=None,
    loras_low=None
)
```

## Scene Nodes

### 1. SceneCreate

Creates a new scene from a base image by generating all pose, depth, and edge detection data.

**Inputs:**
- `base_image` (IMAGE, required): Source image for the scene
- `poses_dir` (STRING): Root directory for storing scenes (defaults to `ComfyUI/poses/`)
- `pose_name` (STRING): Name for this scene (default: "default_pose")
- `resolution` (INT): Target resolution (default: 512)
- `upscale_method` (COMBO): Upscaling method
  - Options: lanczos, nearest-exact, bilinear, area, bicubic
  - Default: nearest-exact
- `upscale_factor` (FLOAT): Upscaling factor (0.1-10.0, default: 1.0)

**Pose Detection Settings:**
- `densepose_model` (COMBO): DensePose model selection
  - densepose_r50_fpn_dl.torchscript
  - densepose_r101_fpn_dl.torchscript
- `densepose_cmap` (COMBO): DensePose colormap (viridis, parula)

**Depth Estimation Settings:**
- `depth_any_ckpt` (COMBO): Depth Anything v1 checkpoint
  - depth_anything_vitl14.pth
  - depth_anything_vitb14.pth
  - depth_anything_vits14.pth
- `depth_any_v2_ckpt` (COMBO): Depth Anything v2 checkpoint
  - depth_anything_v2_vitg.pth
  - depth_anything_v2_vitl.pth
  - depth_anything_v2_vitb.pth
  - depth_anything_v2_vits.pth
- `midas_a` (FLOAT): MiDaS scaling parameter A (default: 2π)
- `midas_bg_thresh` (FLOAT): MiDaS background threshold (default: 0.1)
- `zoe_environment` (COMBO): Zoe environment (indoor, outdoor)

**Edge Detection Settings:**
- `canny_low_threshold` (INT): Canny low threshold (0-255, default: 100)
- `canny_high_threshold` (INT): Canny high threshold (0-255, default: 200)

**Prompts:**
- `girl_pos` (STRING, multiline): Female character positive prompt
- `male_pos` (STRING, multiline): Male character(s) positive prompt
- `four_image_prompt` (STRING, multiline): Four-image generation prompt
- `wan_prompt` (STRING, multiline): WanVideo high-quality prompt
- `wan_low_prompt` (STRING, multiline): WanVideo low-quality prompt

**LoRAs:**
- `loras_high` (WANVIDLORA): High-quality LoRA list
- `loras_low` (WANVIDLORA): Low-quality LoRA list

**Outputs:**
- `scene_info` (SCENE_INFO): Complete scene data structure

**Behavior:**
1. Creates scene directory structure: `{poses_dir}/{pose_name}/`
2. Upscales base image according to settings
3. Generates all pose detections (DensePose, DWPose, OpenPose)
4. Generates all depth maps (Depth Anything v1/v2, MiDaS, Zoe, Zoe Any)
5. Generates Canny edge detection
6. Automatically saves all generated data to disk
7. Returns complete `SceneInfo` object

**Note:** Currently, masks must be generated externally and placed in the scene directory. Future updates will integrate SAM2, SAM Ultra, and other segmentation tools for automatic mask generation.

### 2. SceneUpdate

Updates specific components of an existing scene without regenerating everything.

**Inputs:**
- `scene_info_in` (SCENE_INFO, required): Existing scene to update
- `girl_pos` (STRING, multiline): Updated female character prompt
- `male_pos` (STRING, multiline): Updated male character(s) prompt
- `pose_json` (STRING): Updated pose JSON data

**Update Flags (all BOOLEAN, default: False):**
- `update_prompts` (default: True): Update text prompts
- `update_zoe`: Regenerate Zoe depth images
- `update_depth`: Regenerate Depth Anything images
- `update_densepose`: Regenerate DensePose
- `update_openpose`: Regenerate OpenPose
- `update_midas`: Regenerate MiDaS depth
- `update_canny`: Regenerate Canny edges
- `update_upscale`: Re-upscale base image
- `update_pose_json`: Update pose keypoint data
- `update_facepose`: Regenerate face pose
- `update_editpose`: Regenerate edit pose
- `update_dwpose`: Regenerate DWPose
- `update_high_loras`: Update high-quality LoRAs
- `update_low_loras`: Update low-quality LoRAs

**Settings:** (Same as SceneCreate for relevant updates)
- Resolution, upscaling, model selection parameters

**LoRA Inputs:**
- `high_loras` (WANVIDLORA, optional): New high-quality LoRA list
- `low_loras` (WANVIDLORA, optional): New low-quality LoRA list

**Outputs:**
- `scene_info_out` (SCENE_INFO): Updated scene data

**Behavior:**
1. Takes existing `SceneInfo` as input
2. Only regenerates components with update flags set to True
3. Automatically saves updated LoRAs to disk when modified
4. Returns updated `SceneInfo` object
5. Preserves all unchanged data

**Use Cases:**
- Update prompts without regenerating poses/depth
- Regenerate a single depth method (e.g., just Zoe)
- Update LoRA configurations
- Fine-tune specific components after initial creation

### 3. SceneView

Visualizes a scene with selectable depth and pose types, displaying prompts.

**Inputs:**
- `scene_info` (SCENE_INFO, required): Scene to visualize
- `depth_type` (COMBO): Which depth map to display
  - Options: depth, depth_any, depth_midas, depth_zoe, depth_zoe_any
- `pose_type` (COMBO): Which pose to display
  - Options: dense, dw, edit, face, open

**Outputs:**
- `depth_image` (IMAGE): Selected depth visualization
- `pose_image` (IMAGE): Selected pose visualization
- `pose_name` (STRING): Name of the scene
- `pose_dir` (STRING): Directory path
- `girl_pos` (STRING): Female character prompt
- `male_pos` (STRING): Male character prompt

**UI Preview:**
- Shows depth and pose images side-by-side
- Displays both prompts in text preview

**Behavior:**
1. Loads requested depth and pose types from `SceneInfo`
2. Normalizes images for display
3. Combines images in batch for preview
4. Shows prompts in text UI
5. Outputs selected images and metadata

**Use Cases:**
- Preview different depth/pose combinations
- Compare depth estimation methods
- Verify scene data before use
- Extract specific visualizations for downstream nodes

### 4. SceneSave

Saves all scene data to disk in a structured format.

**Inputs:**
- `scene_info` (SCENE_INFO, required): Scene data to save
- `pose_dir` (STRING, optional): Override save directory

**Outputs:**
- UI preview showing save confirmation and prompts

**File Structure Created:**
```
{pose_dir}/
├── input/              # Input images directory (created)
├── output/             # Output images directory (created)
├── depth.png           # Depth Anything v2
├── depth_any.png       # Depth Anything v1
├── depth_midas.png     # MiDaS depth
├── depth_zoe.png       # Zoe depth
├── depth_zoe_any.png   # Zoe Any depth
├── pose_dense.png      # DensePose
├── pose_dw.png         # DWPose
├── pose_edit.png       # Edit pose
├── pose_face.png       # Face pose
├── pose_open.png       # OpenPose
├── canny.png           # Canny edges
├── upscale.png         # Upscaled base image
├── girl_mask_bkgd.png  # Female mask with background
├── male_mask_bkgd.png  # Male mask with background
├── combined_mask_bkgd.png # Combined mask with background
├── girl_mask_no_bkgd.png  # Female mask no background
├── male_mask_no_bkgd.png  # Male mask no background
├── combined_mask_no_bkgd.png # Combined mask no background
├── prompts.json        # All 5 prompt types
├── pose.json           # Pose keypoint data
└── loras.json          # LoRA configurations
```

**Behavior:**
1. Creates directory structure if needed
2. Uses `scene_info.save_all()` helper method
3. Saves all images in PNG format
4. Saves metadata in JSON format
5. Skips None/empty fields
6. Shows confirmation message with save path

**Use Cases:**
- Persist scene for later use
- Export scene for external processing
- Backup scene data
- Share scenes between workflows

### 5. SceneSelect

Loads and selects a specific scene from available poses, with mask and prompt configuration.

**Inputs:**
- `poses_dir` (STRING): Directory containing scene folders
- `selected_scene` (STRING): Name of scene to load
- `mask_type` (COMBO): Character mask selection
  - Options: girl, male, combined
- `mask_background` (BOOLEAN): Include background in mask (default: True)
- `depth_type` (COMBO): Depth map type to use
  - Options: depth, depth_any, depth_midas, depth_zoe, depth_zoe_any
- `pose_type` (COMBO): Pose type to use
  - Options: dense, dw, edit, face, open
- `prompt_type` (COMBO): Which prompt to use
  - Options: girl_pos, male_pos, four_image_prompt, wan_prompt, wan_low_prompt
- `prompt_action` (COMBO): Prompt source selection
  - "use_file": Use prompt from prompts.json file
  - "use_edit": Use manually edited prompt from widget
- `prompt_in` (STRING, multiline): Editable prompt text

**Outputs:**
- `scene_info` (SCENE_INFO): Complete scene data
- `depth_image` (IMAGE): Selected depth map
- `pose_image` (IMAGE): Selected pose image
- `mask_image` (IMAGE): Selected character mask
- `prompt_out` (STRING): Selected/edited prompt

**UI Preview:**
- Shows depth, pose, and mask images
- Displays selected prompt

**Behavior:**
1. Loads scene from `{poses_dir}/{selected_scene}/`
2. Uses `SceneInfo.load_all_images()` to load all assets
3. Loads prompts from `prompts.json`
4. Loads LoRAs from `loras.json` if present
5. Selects requested depth, pose, and mask variants
6. Returns prompt based on `prompt_action`:
   - "use_file": Returns prompt from JSON file
   - "use_edit": Returns manually edited text from widget
7. Normalizes all images for output
8. Creates complete `SceneInfo` with all data

**Use Cases:**
- Load existing scenes into workflow
- Switch between different mask/depth/pose combinations
- Edit prompts while preserving scene data
- Select specific prompt types for different generation methods

### 6. SceneInput

Provides a dropdown UI for selecting from available scenes in a directory.

**Inputs:**
- `poses_dir` (STRING): Directory to scan for scenes
- `selected_scene` (COMBO): Dropdown of available scene names

**Outputs:**
- `selected_scene_name` (STRING): Name of selected scene

**Behavior:**
1. Scans `poses_dir` for subdirectories
2. Populates dropdown with scene names
3. Returns selected scene name as string
4. Can be connected to SceneSelect's `selected_scene` input

**Use Cases:**
- Provide user-friendly scene selection
- Browse available scenes
- Chain to SceneSelect for loading

## Mask Generation (Future Integration)

Currently, masks must be generated externally using tools like Photoshop, GIMP, or other segmentation software. The six mask variants should be saved in the scene directory with these filenames:

**Current Mask Requirements:**
- `girl_mask_bkgd.png` - Female character with background
- `male_mask_bkgd.png` - Male character(s) with background
- `combined_mask_bkgd.png` - All characters with background
- `girl_mask_no_bkgd.png` - Female character, no background
- `male_mask_no_bkgd.png` - Male character(s), no background
- `combined_mask_no_bkgd.png` - All characters, no background

**Planned Integration:**

Future updates will integrate automatic mask generation directly into **SceneCreate** and **SceneUpdate** nodes:

### Planned Mask Generation Tools:
1. **SAM2 (Segment Anything Model 2)**
   - Interactive point/box prompting
   - Automatic instance segmentation
   - High-quality character separation

2. **SAM Ultra**
   - Enhanced version of SAM
   - Better edge detection
   - Improved small object handling

3. **Additional Tools** (Under consideration)
   - MediaPipe Segmentation
   - Background Matting v2
   - MODNet (Mobile Objective Decomposition Network)
   - Custom trained segmentation models

### Planned Workflow:
```
SceneCreate
├── Generate poses/depth (current)
├── Run segmentation model
├── Detect characters (male/female classification)
├── Generate base masks
├── Create with/without background variants
└── Save all 6 mask types automatically
```

### Configuration Options (Planned):
- Mask generation method selection
- Character detection confidence threshold
- Edge refinement settings
- Background removal quality
- Interactive correction interface

## Workflow Examples

### Example 1: Basic Scene Creation
```
LoadImage → SceneCreate → SceneSave
                ↓
          [Generates all pose/depth data]
          [Saves to disk automatically]
```

### Example 2: Scene with Custom Prompts
```
LoadImage → SceneCreate → SceneUpdate → SceneSave
              ↓              ↓
         [Initial]     [Update prompts]
         [generation]  [Keep poses/depth]
```

### Example 3: Scene Selection and Use
```
SceneInput → SceneSelect → [Your Generation Nodes]
    ↓            ↓
[Browse]   [Load specific]
[scenes]   [depth/pose/mask]
```

### Example 4: Scene Visualization
```
SceneLoad → SceneView
              ↓
        [Preview depth/pose]
        [Compare methods]
```

### Example 5: Scene Update Workflow
```
SceneLoad → SceneUpdate → SceneSave
              ↓
        [update_depth: True]
        [update_prompts: True]
        [Keep everything else]
```

## Directory Structure

**Default poses directory:** `ComfyUI/poses/`

**Per-scene structure:**
```
poses/
├── scene_name_1/
│   ├── input/              # For input images
│   ├── output/             # For generated outputs
│   ├── [depth images]      # 5 depth variants
│   ├── [pose images]       # 7 pose variants
│   ├── [mask images]       # 6 mask variants (manually created)
│   ├── prompts.json        # 5 prompt types
│   ├── pose.json           # Keypoint data
│   └── loras.json          # LoRA configurations
├── scene_name_2/
│   └── ...
└── scene_name_3/
    └── ...
```

## Prompt Management

### Prompt Types

1. **girl_pos**: Positive prompt for female character
   - Focus: Appearance, clothing, style
   - Used in: Character-specific generation

2. **male_pos**: Positive prompt for male character(s)
   - Focus: Male character details
   - Used in: Character-specific generation

3. **four_image_prompt**: Four-image generation method prompt
   - Focus: Complete scene description
   - Used in: Multi-image generation workflows

4. **wan_prompt**: WanVideo high-quality prompt
   - Focus: Detailed, high-quality output
   - Used in: High-fidelity video generation

5. **wan_low_prompt**: WanVideo low-quality prompt
   - Focus: Efficient, lower-resource generation
   - Used in: Fast preview or low-memory workflows

### Prompt Storage

Prompts are stored in `prompts.json`:
```json
{
  "girl_pos": "beautiful woman, detailed face, elegant dress...",
  "male_pos": "handsome man, casual clothing, confident pose...",
  "four_image_prompt": "scene description with both characters...",
  "wan_prompt": "high quality, detailed environment, cinematic...",
  "wan_low_prompt": "simple scene, clear composition..."
}
```

### Prompt Editing

Two methods for editing prompts:

1. **Via SceneUpdate node:**
   - Set `update_prompts: True`
   - Enter new text in `girl_pos` and `male_pos` fields
   - Saves automatically when node executes

2. **Via SceneSelect node:**
   - Set `prompt_action: "use_edit"`
   - Edit text in `prompt_in` widget
   - Widget updates dynamically
   - Select `prompt_type` to choose which prompt to edit

## LoRA Configuration

LoRAs (Low-Rank Adaptations) are stored in `loras.json`:

```json
{
  "high": [
    {
      "lora_name": "detail_enhancer.safetensors",
      "strength": 0.8,
      "blocks": {},
      "layer_filter": "",
      "low_mem_load": false,
      "merge_loras": false
    }
  ],
  "low": [
    {
      "lora_name": "fast_style.safetensors",
      "strength": 0.5,
      "blocks": {},
      "layer_filter": "",
      "low_mem_load": true,
      "merge_loras": false
    }
  ]
}
```

## Best Practices

### Scene Creation
1. **Start with high-quality base images** (1024x1024 or larger)
2. **Use appropriate upscale factors** (1.0-2.0 for most cases)
3. **Set resolution to match your generation needs** (512, 768, 1024)
4. **Create descriptive scene names** for easy identification
5. **Generate masks externally** (for now) before using scenes

### Scene Organization
1. **Use consistent naming conventions** (e.g., `character_action_variant`)
2. **Group related scenes** in subdirectories when appropriate
3. **Document special requirements** in scene names or separate readme
4. **Keep backup copies** of important scenes

### Prompt Management
1. **Be specific and detailed** in character descriptions
2. **Include important details** (clothing, hair, expression)
3. **Use consistent terminology** across related scenes
4. **Test prompts** with SceneView before full generation
5. **Update prompts separately** from poses/depth when iterating

### Performance Optimization
1. **Use SceneUpdate** instead of SceneCreate for small changes
2. **Reuse depth/pose data** when only changing prompts
3. **Select appropriate model sizes** (smaller for iteration, larger for final)
4. **Cache frequently used scenes** by loading once and reusing

### Workflow Design
1. **Separate scene creation from generation** workflows
2. **Use SceneSelect** for production workflows
3. **Use SceneView** for validation and testing
4. **Save scenes explicitly** with SceneSave after creation/updates

## Troubleshooting

### Common Issues

**Issue:** "Scene not found" error
- **Solution:** Check `poses_dir` path and `pose_name` spelling
- **Solution:** Ensure scene directory exists with required files

**Issue:** Missing mask images in output
- **Solution:** Masks must be created manually (for now)
- **Solution:** Ensure masks are saved with correct filenames

**Issue:** Prompts not updating
- **Solution:** Set `update_prompts: True` in SceneUpdate
- **Solution:** Check `prompt_action` setting in SceneSelect

**Issue:** LoRAs not loading
- **Solution:** Verify `loras.json` exists in scene directory
- **Solution:** Check LoRA file paths are correct
- **Solution:** Ensure LoRA files exist in ComfyUI loras folder

**Issue:** Depth/pose images look incorrect
- **Solution:** Try different model selections (vitl vs vitb)
- **Solution:** Adjust resolution settings
- **Solution:** Use SceneUpdate to regenerate specific components

**Issue:** Out of memory errors
- **Solution:** Reduce resolution
- **Solution:** Use smaller model variants (vits instead of vitl)
- **Solution:** Close other applications
- **Solution:** Update one component at a time instead of all

### File Permission Issues
- Ensure ComfyUI has write permissions to `poses_dir`
- Check that scene directories are not read-only
- Verify disk space is available

### Validation
Use SceneView to verify:
- All required images are present
- Prompts are correctly stored
- Depth/pose visualizations look correct
- Masks are properly formatted

## API Reference

### SceneInfo Methods

```python
# Class methods (static)
SceneInfo.load_depth_images(pose_dir: str) -> dict
SceneInfo.load_pose_images(pose_dir: str) -> dict
SceneInfo.load_mask_images(pose_dir: str) -> dict
SceneInfo.load_all_images(pose_dir: str) -> dict
SceneInfo.from_pose_directory(
    pose_dir: str,
    pose_name: str,
    prompt_data: Optional[dict] = None,
    pose_json: str = "",
    loras_high: Optional[list] = None,
    loras_low: Optional[list] = None
) -> SceneInfo

# Instance methods
scene_info.save_all_images(pose_dir: Optional[str] = None)
scene_info.save_prompts(pose_dir: Optional[str] = None)
scene_info.save_pose_json(pose_dir: Optional[str] = None)
scene_info.save_loras(pose_dir: Optional[str] = None)
scene_info.ensure_directories(pose_dir: Optional[str] = None)
scene_info.save_all(pose_dir: Optional[str] = None)

# Utility methods
scene_info.three_image_prompt() -> str
scene_info.input_img_glob() -> str
scene_info.input_img_dir() -> str
scene_info.output_dir() -> str
```

### Default Mappings

```python
# Depth type mappings
default_depth_options = {
    "depth": "depth_image",
    "depth_any": "depth_any_image",
    "depth_midas": "depth_midas_image",
    "depth_zoe": "depth_zoe_image",
    "depth_zoe_any": "depth_zoe_any_image"
}

# Pose type mappings
default_pose_options = {
    "dense": "pose_dense_image",
    "dw": "pose_dw_image",
    "edit": "pose_edit_image",
    "face": "pose_face_image",
    "open": "pose_open_image"
}

# Mask type mappings
default_mask_options = {
    "girl": "girl_mask_bkgd",
    "male": "male_mask_bkgd",
    "combined": "combined_mask_bkgd",
    "girl_no_bg": "girl_mask_no_bkgd",
    "male_no_bg": "male_mask_no_bkgd",
    "combined_no_bg": "combined_mask_no_bkgd"
}
```

## Future Enhancements

### Planned Features
1. **Automatic mask generation** with SAM2/SAM Ultra integration
2. **Batch scene creation** from multiple images
3. **Scene animation** support (pose interpolation)
4. **Advanced pose editing** interface
5. **Scene comparison** tools
6. **Automatic character detection** and classification
7. **Scene templates** for common scenarios
8. **Cloud storage** integration
9. **Scene versioning** and history
10. **Multi-person scene** improvements

### Under Consideration
- Real-time preview during creation
- GPU acceleration options
- Mobile device support
- Collaborative scene editing
- Scene marketplace integration
- AI-assisted prompt generation
- Depth map quality improvements
- Alternative pose detection models

## Support and Resources

### Related Documentation
- [Story Building Nodes](STORY_NODES_README.md) - Scene sequencing
- ComfyUI Documentation - Base system info
- ControlNet Aux - Pose/depth models

### Community
- Report issues on GitHub
- Share scenes and workflows
- Request features
- Contribute improvements

---

**Version:** 1.0  
**Last Updated:** December 2025  
**Compatibility:** ComfyUI with comfyui_controlnet_aux
