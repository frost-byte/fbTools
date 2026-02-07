# fbTools Dependencies

This document details all external dependencies for comfyui-fbTools and their purpose.

## Required Dependencies

### ComfyUI Core
- **ComfyUI** >= 0.3.0
- Required for: All functionality

### Custom Nodes

#### comfyui_controlnet_aux
- **Repository**: https://github.com/Fannovel16/comfyui_controlnet_aux
- **Installation**: Via ComfyUI-Manager (search "controlnet aux")
- **Required for**: 
  - Pose detection (DWPose, OpenPose, DensePose)
  - Depth estimation (Depth Anything v2, MiDaS, Zoe)
  - Canny edge detection
  - Face detection
- **Used by**: All Scene nodes (SceneCreate, SceneUpdate, SceneView, etc.)
- **Status**: **REQUIRED** - Most scene functionality will not work without this

### Python Packages
Listed in `requirements.txt`:
- `kornia~=0.8` - Image processing and transformations
- `opencv_python~=4.11` - Computer vision operations
- `scikit-image~=0.25` - Image processing utilities

## Optional Dependencies

### ComfyUI-SCAIL-Pose
- **Repository**: https://github.com/kijai/ComfyUI-SCAIL-Pose
- **Installation**: Via ComfyUI-Manager (search "SCAIL-Pose" or "ComfyUI-SCAIL-Pose")
- **Required for**: NLF (Neural Lifting Framework) 3D pose generation
- **Features provided**:
  - Advanced 3D pose estimation from 2D images
  - High-quality pose rendering with torch/taichi backends
  - Support for multi-person 3D pose tracking
- **Used by**: 
  - SceneCreate node (`generate_nlf_pose` parameter)
  - SceneUpdate node (`update_nlf_pose` parameter)
- **Graceful degradation**: 
  - ✅ Basic DWPose functionality still works
  - ✅ Other scene features unaffected
  - ⚠️ NLF-specific features return black placeholder images
  - 📝 Clear error messages in logs and UI status widget
- **Additional optional**: `taichi` Python package for GPU-accelerated rendering (faster than torch backend)

### ComfyUI-WanVideoWrapper
- **Repository**: https://github.com/kijai/ComfyUI-WanVideoWrapper/
- **Installation**: Via ComfyUI-Manager (search "WanVideoWrapper")
- **Required for**: LoRA configuration features
- **Features provided**:
  - WANVIDLORA type for video LoRA configurations
  - High/low quality LoRA settings
- **Used by**: SceneWanVideoLoraMultiSave node
- **Graceful degradation**:
  - ✅ All other nodes work normally
  - ⚠️ SceneWanVideoLoraMultiSave node unavailable

## Dependency Status Checking

### At Startup
The extension checks for dependencies at load time and logs warnings for missing optional dependencies:

```
[fbTools] ComfyUI-SCAIL-Pose detected - NLF pose rendering will be available
```

or

```
[fbTools] ComfyUI-SCAIL-Pose not found. NLF pose features will be disabled.
          Install from ComfyUI-Manager or https://github.com/smthemex/ComfyUI-SCAIL-Pose
```

### At Runtime
When NLF features are used without ComfyUI-SCAIL-Pose installed:
- **Logs**: Clear warning messages indicating missing dependency and installation URL
- **UI**: Status widget shows error message with dependency name
- **Behavior**: Returns black placeholder images, continues processing other features
- **No crashes**: Extension remains functional for non-NLF workflows

## Installation Guide

### Quick Install (Recommended)
1. Open ComfyUI-Manager in ComfyUI
2. Click "Install Custom Nodes"
3. Search for and install:
   - ✅ **Required**: "controlnet aux" (comfyui_controlnet_aux)
   - ⚠️ **Optional**: "SCAIL-Pose" (ComfyUI-SCAIL-Pose) - for NLF features
   - ⚠️ **Optional**: "WanVideoWrapper" (ComfyUI-WanVideoWrapper) - for LoRA features
4. Restart ComfyUI

### Manual Install
```bash
cd ComfyUI/custom_nodes

# Required
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# Optional - for NLF 3D pose features
git clone https://github.com/smthemex/ComfyUI-SCAIL-Pose.git

# Optional - for LoRA features  
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Install Python dependencies for each
cd comfyui_controlnet_aux && pip install -r requirements.txt
cd ../ComfyUI-SCAIL-Pose && pip install -r requirements.txt  # if installing
cd ../ComfyUI-WanVideoWrapper && pip install -r requirements.txt  # if installing
```

## Troubleshooting

### "NLF pose rendering unavailable"
**Cause**: ComfyUI-SCAIL-Pose not installed
**Solution**: Install via ComfyUI-Manager or manually from https://github.com/smthemex/ComfyUI-SCAIL-Pose
**Workaround**: Use DWPose instead (works without SCAIL-Pose)

### "Pose detection failed"
**Cause**: comfyui_controlnet_aux not installed
**Solution**: Install via ComfyUI-Manager (required dependency)
**Note**: This is a required dependency - scene nodes will not function without it

### Slow NLF rendering
**Cause**: Using torch backend without GPU acceleration
**Solution**: 
1. Install taichi: `pip install taichi`
2. Set `render_backend="taichi"` in SceneUpdate node
3. Ensure CUDA/GPU is available

## Version Compatibility

| fbTools Version | ComfyUI Min | controlnet_aux | SCAIL-Pose | WanVideoWrapper |
|----------------|-------------|----------------|------------|-----------------|
| 1.0.x          | 0.3.0       | Latest         | Latest     | Latest          |

## License Notes

Each dependency has its own license:
- **comfyui_controlnet_aux**: Apache 2.0
- **ComfyUI-SCAIL-Pose**: Check repository for license
- **ComfyUI-WanVideoWrapper**: Check repository for license

See individual repositories for full license information.
