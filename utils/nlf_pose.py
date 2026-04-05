"""
NLF Pose Generation Utilities
Integrates NLF pose detection and rendering from WanVideoWrapper and ComfyUI-SCAIL-Pose
"""
import os
import numpy as np
import torch
import copy
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths
try:
    from .util import import_virtual_package, add_custom_node_to_syspath
except ImportError:
    # Test harness imports this module as top-level; allow absolute fallback.
    from utils.util import import_virtual_package, add_custom_node_to_syspath
# Import logging utils - handle both relative and absolute imports for testing
try:
    from .logging_utils import get_logger
except ImportError:
    from utils.logging_utils import get_logger

logger = get_logger(__name__)

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
candidates=["ComfyUI-SCAIL-Pose", "comfyui-scail-pose"]
scail_pose_path = add_custom_node_to_syspath(candidates)

if scail_pose_path is not None:
    import_virtual_package("scail_pose", scail_pose_path)

# Check if ComfyUI-SCAIL-Pose is available at module load time
_SCAIL_POSE_AVAILABLE = scail_pose_path is not None

if _SCAIL_POSE_AVAILABLE:
    logger.info("ComfyUI-SCAIL-Pose detected - NLF pose rendering will be available")
    try:
        from scail_pose.NLFPoseExtract.nlf_render import ( # type: ignore
            render_nlf_as_images,
            render_multi_nlf_as_images,
            intrinsic_matrix_from_field_of_view
        )
        _HAS_NLF_RENDER = True
    except ImportError as e:
        logger.warning(f"Failed to import from NLFPoseExtract.nlf_render: {e}")
else:
    logger.warning(
        "ComfyUI-SCAIL-Pose not found. NLF pose features will be disabled. "
        "Install from ComfyUI-Manager or https://github.com/kijai/ComfyUI-SCAIL-Pose"
    )

# Add model folder paths
folder_paths.add_model_folder_path("nlf", os.path.join(folder_paths.models_dir, "nlf"))
folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

# Default model URLs
NLF_MODEL_URLS = [
    "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript",
    "https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_l_multi_0.2.2.torchscript",
]

# Available NLF model names (basename of URLs)
NLF_MODEL_NAMES = [
    "nlf_l_multi_0.3.2.torchscript",
    "nlf_l_multi_0.2.2.torchscript",
]

def check_jit_script_function():
    """Check if torch.jit.script has been modified by another custom node"""
    if torch.jit.script.__name__ != "script":
        module = torch.jit.script.__module__
        qualname = getattr(torch.jit.script, '__qualname__', 'unknown')
        try:
            code_file = torch.jit.script.__code__.co_filename
            code_line = torch.jit.script.__code__.co_firstlineno
            logger.warning(
                f"torch.jit.script has been modified by another custom node.\n"
                f"  Function name: {torch.jit.script.__name__}\n"
                f"  Module: {module}\n"
                f"  Qualified name: {qualname}\n"
                f"  Defined in: {code_file}:{code_line}\n"
                f"This may cause issues with the NLF model."
            )
        except:
            logger.warning(
                f"torch.jit.script function is: {torch.jit.script.__name__} from module {module}, "
                f"this has been modified by another custom node. This may cause issues with the NLF model."
            )


def load_nlf_model(model_name: Optional[str] = None, warmup: bool = True) -> torch.jit.ScriptModule:
    """
    Load NLF model from file or download if needed
    
    Args:
        model_name: Model filename (e.g., 'nlf_l_multi_0.3.2.torchscript'), or None to use default
        warmup: Whether to warmup the model after loading
    
    Returns:
        Loaded NLF model
    """
    check_jit_script_function()
    
    # Handle empty string as None
    if model_name is not None and not model_name.strip():
        model_name = None
    
    if model_name is None:
        # Use default model name
        model_name = NLF_MODEL_NAMES[0]
    
    # Construct full path in models/nlf directory
    model_path = os.path.join(folder_paths.models_dir, "nlf", model_name)
    
    # Download if not exists
    if not os.path.exists(model_path):
        logger.info(f"Downloading NLF model '{model_name}' to: {model_path}")
        
        # Find corresponding URL for this model name
        model_url = None
        for url in NLF_MODEL_URLS:
            if url.endswith(model_name):
                model_url = url
                break
        
        if model_url is None:
            raise ValueError(f"Unknown NLF model name: {model_name}. Available: {NLF_MODEL_NAMES}")
        
        import requests
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded NLF model: {model_name}")
        else:
            raise RuntimeError(f"Failed to download NLF model from {model_url}: {response.status_code}")
    
    model = torch.jit.load(model_path).eval()
    
    if warmup:
        logger.info("Warming up NLF model...")
        dummy_input = torch.zeros(1, 3, 256, 256, device=device)
        jit_profiling_prev_state = torch._C._jit_set_profiling_executor(True)
        try:
            for _ in range(2):
                _ = model.detect_smpl_batched(dummy_input)
        finally:
            torch._C._jit_set_profiling_executor(jit_profiling_prev_state)
        logger.info("NLF model warmed up")
    
    model = model.to(offload_device)
    return model


def predict_nlf_pose(model: torch.jit.ScriptModule, images: torch.Tensor, per_batch: int = -1) -> Tuple[Dict, List]:
    """
    Predict NLF poses from images
    
    Args:
        model: NLF model
        images: Input images tensor [B, H, W, C]
        per_batch: Batch size for processing (-1 for all at once)
    
    Returns:
        Tuple of (pose_results dict, bboxes list)
    """
    check_jit_script_function()
    model = model.to(device)
    
    num_images = images.shape[0]
    
    # Determine batch size
    if per_batch == -1:
        batch_size = num_images
    else:
        batch_size = per_batch
    
    # Initialize result containers
    all_boxes = []
    all_joints3d_nonparam = []
    
    # Process in batches
    for i in range(0, num_images, batch_size):
        end_idx = min(i + batch_size, num_images)
        batch_images = images[i:end_idx]
        
        jit_profiling_prev_state = torch._C._jit_set_profiling_executor(True)
        try:
            pred = model.detect_smpl_batched(batch_images.permute(0, 3, 1, 2).to(device))
        finally:
            torch._C._jit_set_profiling_executor(jit_profiling_prev_state)
        
        # Collect boxes and joints from this batch
        if 'boxes' in pred:
            all_boxes.extend(pred['boxes'])
        if 'joints3d_nonparam' in pred:
            all_joints3d_nonparam.extend(pred['joints3d_nonparam'])
    
    model = model.to(offload_device)
    
    # Move collected results to offload device
    all_boxes = [box.to(offload_device) for box in all_boxes]
    all_joints3d_nonparam = [joints.to(offload_device) for joints in all_joints3d_nonparam]
    
    # Maintain the original nested format
    pose_results = {
        'joints3d_nonparam': [all_joints3d_nonparam],
    }
    
    # Convert bboxes to list format: [x_min, y_min, x_max, y_max] for each detection
    formatted_boxes = []
    for box in all_boxes:
        if box.numel() == 0 or box.shape[0] == 0:
            formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            bbox_values = box[0, :4].cpu().tolist()
            formatted_boxes.append(bbox_values)
    
    return (pose_results, formatted_boxes)


def load_vitpose_model(model_name: str = "vitpose-h-wholebody.onnx"):
    """
    Load VitPose ONNX model
    
    Args:
        model_name: Name of the VitPose model file
    
    Returns:
        Dictionary with detector and pose model
    """
    try:
        from comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import DwposeDetector
        
        # Load detector (YOLO)
        yolo_repo = "yzd-v/DWPose"
        bbox_detector = "yolox_l.onnx"
        
        # Load VitPose model
        model_path = folder_paths.get_full_path("detection", model_name)
        if model_path is None:
            raise FileNotFoundError(f"VitPose model {model_name} not found in detection folder")
        
        # Create detector
        detector = DwposeDetector.from_pretrained(
            "yzd-v/DWPose",
            yolo_repo,
            det_filename=bbox_detector,
            pose_filename=None,  # We'll use VitPose separately
            torchscript_device=str(device)
        )
        
        return {
            "yolo": detector,
            "vitpose_path": model_path
        }
    except Exception as e:
        logger.error(f"Failed to load VitPose model: {e}")
        raise


def dwpose_to_openpose(dwpose_dict: Dict) -> Dict:
    """
    Convert DWPose format to OpenPose POSE_KEYPOINT format
    
    DWPose format has:
    - bodies: {'candidate': array of [N, 18, 2], 'subset': array of indices}
    - hands: array of hand keypoints
    - faces: array of face keypoints
    
    OpenPose format needs:
    - canvas_width, canvas_height
    - people: list of dicts with pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d
    
    Args:
        dwpose_dict: DWPose format dictionary
    
    Returns:
        OpenPose format dictionary
    """
    # Default canvas size (will be overridden with actual dimensions when available)
    canvas_width = 512
    canvas_height = 768
    
    people = []
    
    bodies = dwpose_dict.get('bodies', {})
    candidates = bodies.get('candidate', np.array([]))
    faces = dwpose_dict.get('faces', np.array([]))
    hands = dwpose_dict.get('hands', np.array([]))
    
    # Number of people detected
    num_people = candidates.shape[0] if len(candidates.shape) > 1 else 0
    
    for i in range(num_people):
        person = {}
        
        # Body keypoints (18 points * 3 values = 54 values)
        if len(candidates.shape) > 1 and i < candidates.shape[0]:
            body_kp = candidates[i]  # Shape: [18, 2]
            # Convert to [x, y, confidence] format (assuming confidence = 1.0)
            pose_keypoints_2d = []
            for kp in body_kp:
                pose_keypoints_2d.extend([
                    float(kp[0]) * canvas_width,
                    float(kp[1]) * canvas_height,
                    1.0  # confidence
                ])
            person['pose_keypoints_2d'] = pose_keypoints_2d
        else:
            person['pose_keypoints_2d'] = [0.0] * 54
        
        # Face keypoints (70 points * 3 values = 210 values)
        if len(faces.shape) > 1 and i < faces.shape[0]:
            face_kp = faces[i]  # Expecting [70, 2] but might be [68, 2]
            face_keypoints_2d = []
            for kp in face_kp:
                face_keypoints_2d.extend([
                    float(kp[0]) * canvas_width,
                    float(kp[1]) * canvas_height,
                    1.0
                ])
            # Pad to 70 points if needed
            while len(face_keypoints_2d) < 210:
                face_keypoints_2d.extend([0.0, 0.0, 0.0])
            person['face_keypoints_2d'] = face_keypoints_2d[:210]
        else:
            person['face_keypoints_2d'] = [0.0] * 210
        
        # Hand keypoints (21 points * 3 values = 63 values each)
        # Hands are stored as left, right pairs
        hand_idx = i * 2
        if len(hands.shape) > 1:
            # Left hand
            if hand_idx < hands.shape[0]:
                hand_left_kp = hands[hand_idx]
                hand_left_keypoints_2d = []
                for kp in hand_left_kp:
                    hand_left_keypoints_2d.extend([
                        float(kp[0]) * canvas_width,
                        float(kp[1]) * canvas_height,
                        1.0
                    ])
                person['hand_left_keypoints_2d'] = hand_left_keypoints_2d[:63]
            else:
                person['hand_left_keypoints_2d'] = [0.0] * 63
            
            # Right hand
            if hand_idx + 1 < hands.shape[0]:
                hand_right_kp = hands[hand_idx + 1]
                hand_right_keypoints_2d = []
                for kp in hand_right_kp:
                    hand_right_keypoints_2d.extend([
                        float(kp[0]) * canvas_width,
                        float(kp[1]) * canvas_height,
                        1.0
                    ])
                person['hand_right_keypoints_2d'] = hand_right_keypoints_2d[:63]
            else:
                person['hand_right_keypoints_2d'] = [0.0] * 63
        else:
            person['hand_left_keypoints_2d'] = [0.0] * 63
            person['hand_right_keypoints_2d'] = [0.0] * 63
        
        people.append(person)
    
    return {
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'people': people
    }


def nlfpred_to_pose_keypoint(nlf_pred: Dict, width: int = 512, height: int = 768) -> List[Dict]:
    """
    Convert NLFPRED format to POSE_KEYPOINT format (list of OpenPose dicts)
    
    This requires rendering the NLF pose first to get 2D keypoints
    For now, we'll create a placeholder that can be used with OpenposeEditorNode
    
    Args:
        nlf_pred: NLF prediction dictionary with joints3d_nonparam
        width: Canvas width
        height: Canvas height
    
    Returns:
        List of OpenPose format dictionaries
    """
    logger.debug("nlfpred_to_pose_keypoint: Input nlf_pred keys: %s", list(nlf_pred.keys()) if nlf_pred else 'None')
    
    # Extract 3D joints with better error handling
    joints3d_nested = nlf_pred.get('joints3d_nonparam', [[]])
    logger.debug("nlfpred_to_pose_keypoint: joints3d_nested type: %s, length: %s", 
                type(joints3d_nested), len(joints3d_nested) if hasattr(joints3d_nested, '__len__') else 'N/A')
    
    if not joints3d_nested or len(joints3d_nested) == 0:
        logger.warning("nlfpred_to_pose_keypoint: No joints3d_nonparam data, returning empty list")
        return [{'canvas_width': width, 'canvas_height': height, 'people': []}]
    
    joints3d = joints3d_nested[0]
    logger.debug("nlfpred_to_pose_keypoint: joints3d type: %s, length: %s", 
                type(joints3d), len(joints3d) if hasattr(joints3d, '__len__') else 'N/A')
    
    if not joints3d or len(joints3d) == 0:
        logger.warning("nlfpred_to_pose_keypoint: joints3d is empty, returning empty list")
        return [{'canvas_width': width, 'canvas_height': height, 'people': []}]
    
    pose_keypoints = []
    
    for idx, joints in enumerate(joints3d):
        logger.debug("nlfpred_to_pose_keypoint: Processing person %d, joints type: %s", idx, type(joints))
        
        # Check for empty tensor/array
        if isinstance(joints, torch.Tensor) and joints.numel() == 0:
            logger.warning("nlfpred_to_pose_keypoint: Person %d has empty tensor, skipping", idx)
            continue
        elif hasattr(joints, '__len__') and len(joints) == 0:
            logger.warning("nlfpred_to_pose_keypoint: Person %d has empty data, skipping", idx)
            continue
        
        # Simple orthographic projection (just use x, y coordinates)
        # This is a simplified conversion - ideally we'd use proper camera projection
        if isinstance(joints, torch.Tensor):
            joints_np = joints[0].cpu().numpy() if joints.ndim > 2 else joints.cpu().numpy()
        else:
            joints_np = joints
        
        # Create person dict
        person = {
            'canvas_width': width,
            'canvas_height': height,
            'people': []
        }
        
        if joints_np.shape[0] > 0:
            # Convert 3D to 2D by orthographic projection
            # Normalize to canvas size
            pose_2d = joints_np[:, :2]  # Take x, y coordinates
            
            # Simple normalization (this should be improved with proper camera projection)
            pose_2d_norm = (pose_2d - pose_2d.min(axis=0)) / (pose_2d.max(axis=0) - pose_2d.min(axis=0) + 1e-6)
            pose_2d_canvas = pose_2d_norm * np.array([width, height])
            
            # Create OpenPose format (18 body keypoints minimum)
            pose_keypoints_2d = []
            for i in range(min(18, pose_2d_canvas.shape[0])):
                pose_keypoints_2d.extend([
                    float(pose_2d_canvas[i, 0]),
                    float(pose_2d_canvas[i, 1]),
                    1.0  # confidence
                ])
            
            # Pad to 18 keypoints if needed
            while len(pose_keypoints_2d) < 54:
                pose_keypoints_2d.extend([0.0, 0.0, 0.0])
            
            person['people'].append({
                'pose_keypoints_2d': pose_keypoints_2d,
                'face_keypoints_2d': [0.0] * 210,
                'hand_left_keypoints_2d': [0.0] * 63,
                'hand_right_keypoints_2d': [0.0] * 63
            })
        
        pose_keypoints.append(person)
    
    return pose_keypoints if pose_keypoints else [{'canvas_width': width, 'canvas_height': height, 'people': []}]


def test_taichi_init(render_device: str = "gpu") -> Tuple[bool, Optional[str]]:
    """
    Test taichi initialization with the same parameters used by NLF rendering.
    Returns (success, error_message)
    """
    try:
        import taichi as ti
        device_map = {
            "cpu": ti.cpu,
            "gpu": ti.gpu,
            "opengl": ti.opengl,
            "cuda": ti.cuda,
            "vulkan": ti.vulkan,
            "metal": ti.metal,
        }
        arch = device_map.get(render_device.lower())
        logger.info(f"Testing taichi initialization with arch={arch} (device={render_device})")
        ti.init(arch=arch)
        logger.info(f"Taichi initialized successfully on {arch}")
        return True, None
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def render_nlf_pose(
    nlf_pred: Dict,
    width: int,
    height: int,
    dw_poses: Optional[Dict] = None,
    ref_dw_pose: Optional[Dict] = None,
    draw_face: bool = True,
    draw_hands: bool = True,
    render_device: str = "gpu",
    scale_hands: bool = True,
    render_backend: str = "torch"  # Default to torch since taichi may not be installed
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render NLF poses to images
    
    Args:
        nlf_pred: NLF prediction dictionary
        width: Output width
        height: Output height
        dw_poses: Optional DW poses for 2D overlay
        ref_dw_pose: Optional reference DW pose for alignment
        draw_face: Whether to draw face keypoints
        draw_hands: Whether to draw hand keypoints
        render_device: Rendering device (gpu/cpu for taichi backend)
        scale_hands: Whether to scale hands
        render_backend: "taichi" or "torch"
    
    Returns:
        Tuple of (rendered images tensor, mask tensor)
    """

    
    if not _HAS_NLF_RENDER:
        logger.info(
            "Returning black placeholder for NLF pose. "
            "To enable NLF functionality, install ComfyUI-SCAIL-Pose via ComfyUI-Manager."
        )
        # Return placeholder image
        placeholder = torch.zeros(1, height, width, 3)
        mask = torch.zeros(1, height, width)
        return placeholder, mask
    
    # Extract pose input
    if isinstance(nlf_pred, dict):
        pose_input = nlf_pred['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_pred else nlf_pred
    else:
        pose_input = nlf_pred
    
    dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
    
    # Get camera intrinsics
    intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
    
    # Test taichi if requested
    if render_backend == "taichi":
        success, error = test_taichi_init(render_device)
        if not success:
            logger.warning(f"Taichi initialization test failed: {error}")
            logger.warning("This is the specific error that causes fallback to torch backend")
    
    # Render poses with detailed error logging
    try:
        logger.debug(f"Calling render with backend={render_backend}, device={render_device}")
        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(
                pose_input, dw_pose_input, height, width, len(pose_input),
                intrinsic_matrix=intrinsic_matrix,
                draw_face=draw_face,
                draw_hands=draw_hands,
                render_backend=render_backend
            )
        else:
            frames_np = render_nlf_as_images(
                pose_input, dw_pose_input, height, width, len(pose_input),
                intrinsic_matrix=intrinsic_matrix,
                draw_face=draw_face,
                draw_hands=draw_hands,
                render_backend=render_backend
            )
        logger.debug(f"Render completed successfully with {render_backend} backend")
    except Exception as e:
        logger.error(f"Error during NLF rendering with {render_backend} backend: {type(e).__name__}: {e}", exc_info=True)
        # If taichi failed, try falling back to torch
        if render_backend == "taichi":
            logger.warning("Falling back to torch backend after taichi error")
            render_backend = "torch"
            if pose_input[0].shape[0] > 1:
                frames_np = render_multi_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input),
                    intrinsic_matrix=intrinsic_matrix,
                    draw_face=draw_face,
                    draw_hands=draw_hands,
                    render_backend=render_backend
                )
            else:
                frames_np = render_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input),
                    intrinsic_matrix=intrinsic_matrix,
                    draw_face=draw_face,
                    draw_hands=draw_hands,
                    render_backend=render_backend
                )
        else:
            raise
    
    frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
    frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
    
    return (frames_tensor.cpu().float(), mask.cpu().float())
