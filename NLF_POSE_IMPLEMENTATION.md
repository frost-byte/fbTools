# NLF Pose Integration Implementation Guide

## Overview
This document outlines the integration of NLF (Neural Lifting Framework) pose generation into fbTools Scene nodes.

## ✅ Completed Work

### 1. NLF Pose Utility Module (`utils/nlf_pose.py`)
Created comprehensive utility functions for NLF pose operations:

- **`load_nlf_model()`**: Load/download NLF model with optional warmup
- **`predict_nlf_pose()`**: Generate NLF pose predictions from images
- **`load_vitpose_model()`**: Load VitPose ONNX model for detection
- **`dwpose_to_openpose()`**: Convert DWPose format to OpenPose POSE_KEYPOINT format
- **`nlfpred_to_pose_keypoint()`**: Convert NLFPRED to POSE_KEYPOINT for OpenposeEditorNode compatibility
- **`render_nlf_pose()`**: Render NLF poses to images with configurable options

### 2. SceneInfo Data Model Updates
- Added `pose_nlf_image` field to store NLF pose renders
- Updated `default_pose_options` to include "nlf" option

### 3. Backend API Enhancement
- Updated `/fbtools/scene/get_scene_prompts` endpoint to return mask data for dynamic combo updates
- SceneSelect node now uses dynamic combo for `mask_name` instead of string input

### 4. Frontend Dynamic Updates
- SceneSelect's `mask_name` combo now updates automatically when scene selection changes
- Integrated with existing scene API to fetch available masks

## 🚧 Remaining Work

### 1. SceneCreate Node Enhancement
**Location**: `extension.py`, class `SceneCreate`

**Required Changes**:
```python
# Add inputs for NLF pose generation:
- nlf_model (COMBO): NLF model selection (auto-downloads if needed)
  - Options: ["nlf_l_multi_0.3.2.torchscript", "nlf_l_multi_0.2.2.torchscript"]
  - Default: "nlf_l_multi_0.3.2.torchscript"
- generate_nlf_pose (BOOLEAN): Toggle NLF pose generation
- nlf_draw_face (BOOLEAN): Draw face keypoints (default: True)
- nlf_draw_hands (BOOLEAN): Draw hand keypoints (default: True)
- nlf_render_device (COMBO): ["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"]
- nlf_scale_hands (BOOLEAN): Scale hands (default: True)
- nlf_render_backend (COMBO): ["taichi", "torch"] (default: "torch")
```

**Implementation**:
```python
# In execute method, after other pose generation:
if generate_nlf_pose:
    from .utils.nlf_pose import load_nlf_model, predict_nlf_pose, render_nlf_pose
    
    # Load model
    nlf_model = load_nlf_model(nlf_model_path)
    
    # Predict poses
    nlf_pred, bboxes = predict_nlf_pose(nlf_model, upscale_image)
    
    # Render poses
    nlf_image, nlf_mask = render_nlf_pose(
        nlf_pred, W, H,
        draw_face=nlf_draw_face,
        draw_hands=nlf_draw_hands,
        render_device=nlf_render_device,
        scale_hands=nlf_scale_hands,
        render_backend=nlf_render_backend
    )
    
    # Convert to POSE_KEYPOINT format and save as pose.json
    from .utils.nlf_pose import nlfpred_to_pose_keypoint
    pose_keypoints = nlfpred_to_pose_keypoint(nlf_pred, W, H)
    pose_json = json.dumps(pose_keypoints)
    
    # Add to SceneInfo
    scene_info.pose_nlf_image = nlf_image
    scene_info.pose_json = pose_json  # Save for editing
```

### 2. SceneUpdate Node Enhancement
**Location**: `extension.py`, class `SceneUpdate`

**Required Changes**:
```python
# Add new inputs:
- update_nlf_pose (BOOLEAN): Toggle NLF pose update
- pose_image (IMAGE, optional): Custom pose image to use
- pose_keypoint (POSE_KEYPOINT, optional): Custom pose keypoints

# Same NLF configuration inputs as SceneCreate:
- nlf_model_path, nlf_draw_face, nlf_draw_hands, etc.
```

**Implementation Logic**:
```python
if update_nlf_pose:
    if pose_image is not None and pose_keypoint is not None:
        # User provided custom pose - use it directly
        scene_info.pose_nlf_image = pose_image
        scene_info.pose_json = json.dumps(pose_keypoint)
    else:
        # Generate from base_image (same as SceneCreate logic)
        # Load model, predict, render, convert format
        pass
```

### 3. Scene Loading Updates
**Location**: `extension.py`, `SceneInfo` class methods

**Required Changes**:
- `load_pose_images()`: Add loading of `pose_nlf.png`
- `save_all_images()`: Add saving of `pose_nlf_image`
- `load_preview_assets()`: Include NLF pose in preview batch when `pose_attr="pose_nlf_image"`

### 4. Documentation Updates
**Files to update**:
- `docs/SCENE_NODES_README.md`: Document NLF pose feature
- Add section explaining:
  - NLF pose generation workflow
  - Required models (NLF model, VitPose model)
  - Model downloads and installation
  - Editing workflow using OpenposeEditorNode
  - Configuration options

### 5. Model Management
**Considerations**:
- NLF models should auto-download on first use (already implemented)
- VitPose models need to be in `ComfyUI/models/detection/` folder
- Document where users can get these models
- Consider adding model validation/download helpers

## 📋 Implementation Checklist

- [x] Create `utils/nlf_pose.py` with all utility functions
- [x] Add `pose_nlf_image` field to SceneInfo
- [x] Update `default_pose_options` dict
- [x] Add format conversion functions
- [ ] Update SceneCreate node inputs and logic
- [ ] Update SceneUpdate node inputs and logic
- [ ] Update SceneInfo load/save methods for NLF pose
- [ ] Add SceneSelect support for nlf pose type
- [ ] Document NLF pose feature
- [ ] Test full workflow end-to-end

## 🎯 Usage Workflow (Planned)

### Creating a Scene with NLF Pose:
1. User loads base image into SceneCreate
2. User enables `generate_nlf_pose` toggle
3. User selects NLF model and VitPose model (or uses defaults)
4. User configures rendering options (face, hands, backend)
5. Node generates NLF pose and saves as `pose_nlf.png` + `pose.json`
6. User can select "nlf" as pose type in SceneSelect

### Editing NLF Pose:
1. User loads scene with NLF pose
2. User connects pose.json to OpenposeEditorNode
3. User edits pose in editor
4. User connects edited pose + rendered image to SceneUpdate
5. SceneUpdate with `update_nlf_pose=True` saves new pose

### Using NLF Pose in Story:
1. StoryEdit allows selecting "nlf" as pose_type for each scene
2. StoryView renders preview with NLF pose
3. Story generation workflows use NLF pose images

## 🔧 Technical Notes

### Dependencies:
- **Required**: 
  - torch
  - numpy
  - comfyui_controlnet_aux (for VitPose)
- **Optional**:
  - taichi (for faster rendering)
  - onnxruntime (for GPU-accelerated pose detection)

### Model Files:
- **NLF Model**: `~/.comfyui/models/nlf/nlf_l_multi_0.3.2.torchscript` (auto-download)
- **VitPose Model**: `~/.comfyui/models/detection/vitpose-h-wholebody.onnx` (manual download)

### Format Specifications:

**NLFPRED Format**:
```python
{
    'joints3d_nonparam': [
        [torch.Tensor(B, 24, 3), ...]  # List of 3D joint tensors per person
    ]
}
```

**POSE_KEYPOINT Format** (OpenPose):
```python
{
    'canvas_width': int,
    'canvas_height': int,
    'people': [
        {
            'pose_keypoints_2d': [x1, y1, c1, ...],  # 18 body points * 3 = 54 values
            'face_keypoints_2d': [x1, y1, c1, ...],  # 70 face points * 3 = 210 values
            'hand_left_keypoints_2d': [x1, y1, c1, ...],  # 21 hand points * 3 = 63 values
            'hand_right_keypoints_2d': [x1, y1, c1, ...]
        }
    ]
}
```

## 🚀 Next Steps

1. Complete SceneCreate node integration (highest priority)
2. Complete SceneUpdate node integration
3. Update load/save methods in SceneInfo
4. Add comprehensive documentation
5. Create example workflows
6. Test with various models and configurations

## 📝 Notes

- The implementation uses a "torch" render backend by default since taichi may not be installed
- Format conversions are approximate - proper camera projection would improve accuracy
- Consider adding validation for model files before attempting to load
- May need to handle cases where NLF pose detection fails (no person detected)
