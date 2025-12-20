import node_helpers
import math
from comfy.utils import common_upscale
import folder_paths
import comfy.model_management as model_management

from typing_extensions import override
from folder_paths import get_output_directory
from nodes import ImageScaleBy
from .utils.util import (
    draw_pose_json,
    draw_pose,
    extend_scalelist,
    pose_normalized,
    select_text_by_action,
    update_ui_widget,
    get_workflow_all_nodes,
    listify_nodes_data,
    listify_node_inputs,
    node_input_details,
    find_node_by_id,
    get_node_inputs
)
from .utils.io import save_json_file, load_prompt_json, load_json_file
from .utils.images import image_resize_ess, find_nearest_qwen_aspect_ratio
from .utils.pose import estimate_dwpose, dense_pose, depth_anything, depth_anything_v2, zoe, zoe_any, openpose, midas, canny

from .utils.images import (
    make_empty_image,
    _compute_ref_stats,
    _pick_ref_image,
    proc_deflicker_luma,
    proc_deflicker_clahe,
    proc_color_histmatch,
    proc_color_meanstd,
    proc_bilateral_cv2,
    proc_unsharp,
    _stack_if_same_shape
)

from .utils.images import _HAS_KORNIA, _HAS_SKIMAGE, _HAS_CV2, load_image_comfyui, save_image_comfyui, make_placeholder_tensor, normalize_image_tensor
from comfy_api.latest import ComfyExtension, io, ui
from inspect import cleandoc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import os
from pathlib import Path
import json
import uuid
import re
import copy
from pydantic import BaseModel, ConfigDict

try:
    from westNeighbor_comfyui_ultimate_openpose_editor.openpose_editor_nodes import OpenposeEditorNode  # type: ignore
except Exception:
    OpenposeEditorNode = None


OpenposeJSON = dict

# Extension-wide node prefix to keep node_id globally unique across ComfyUI
EXTENSION_PREFIX = "fbt"

def prefixed_node_id(display_name: str) -> str:
    """Construct a globally-unique node_id using the shared extension prefix."""
    return f"{EXTENSION_PREFIX}_{display_name}"


# ============================================================================
# LIBBER - String Template/Substitution System
# ============================================================================

class Libber:
    """
    Libber: A string templating system for ComfyUI prompts.
    
    Allows defining reusable text snippets (libs) that can be referenced
    in other strings using a delimiter (default: %). Supports recursive
    substitution with depth limiting to prevent infinite loops.
    
    Example:
        libs = {
            "chunky": "incredibly thick, and %yummy%",
            "yummy": "delicious",
            "character": "A %chunky% warrior"
        }
        libber = Libber(libs)
        libber.substitute("Look at this %character%!")
        # Result: "Look at this A incredibly thick, and delicious warrior!"
    """
    
    def __init__(self, lib_dict=None, delimiter="%", max_depth=10):
        """
        Initialize a Libber instance.
        
        Args:
            lib_dict: Dictionary of lib_key -> value mappings
            delimiter: Character(s) used to mark lib references (default: "%")
            max_depth: Maximum recursion depth for nested lib substitution
        """
        self.libs = lib_dict or {}
        self.delimiter = delimiter
        self.max_depth = max_depth
    
    def add_lib(self, key: str, value: str):
        """Add or update a lib entry."""
        # Normalize key to lowercase with underscores
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        self.libs[normalized_key] = value
    
    def remove_lib(self, key: str):
        """Remove a lib entry."""
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        if normalized_key in self.libs:
            del self.libs[normalized_key]
            return True
        return False
    
    def get_lib(self, key: str) -> Optional[str]:
        """Get a lib value by key."""
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        return self.libs.get(normalized_key)
    
    def list_libs(self) -> List[str]:
        """Return a list of all lib keys."""
        return sorted(self.libs.keys())
    
    def substitute(self, text: str, depth: int = 0) -> str:
        """
        Recursively substitute lib references in text.
        
        Args:
            text: Input string containing lib references like %lib_name%
            depth: Current recursion depth (used internally)
            
        Returns:
            String with all lib references substituted
        """
        if depth >= self.max_depth:
            return text
        
        # Pattern: delimiter + lowercase/underscore words + delimiter
        # e.g., %chunky%, %my_lib%, %test_123%
        pattern = re.escape(self.delimiter) + r'([a-z0-9_]+)' + re.escape(self.delimiter)
        
        def replacer(match):
            lib_key = match.group(1)
            if lib_key in self.libs:
                # Get the value and recursively substitute
                value = self.libs[lib_key]
                return self.substitute(value, depth + 1)
            # Return unchanged if not found
            return match.group(0)
        
        return re.sub(pattern, replacer, text)
    
    def to_dict(self) -> dict:
        """Convert Libber instance to a dictionary for serialization."""
        return {
            "libs": self.libs,
            "delimiter": self.delimiter,
            "max_depth": self.max_depth
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Libber":
        """Create a Libber instance from a dictionary."""
        return cls(
            lib_dict=data.get("libs", {}),
            delimiter=data.get("delimiter", "%"),
            max_depth=data.get("max_depth", 10)
        )
    
    def save(self, filepath: str):
        """Save Libber configuration to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "Libber":
        """Load Libber configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self):
        return f"Libber(libs={len(self.libs)}, delimiter='{self.delimiter}', max_depth={self.max_depth})"


def default_libber_dir():
    """Get default directory for storing libber files."""
    output_dir = get_output_directory()
    default_dir = os.path.join(output_dir, "libbers")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
    return default_dir



def load_pose(
    show_body=True,
    show_face=True,
    show_hands=True,
    resolution_x=-1,
    pose_marker_size=4,
    face_marker_size=3,
    hand_marker_size=2,
    hands_scale=1.0,
    body_scale=1.0,
    head_scale=1.0,
    overall_scale=1.0,
    scalelist_behavior="poses",
    match_scalelist_method="loop extend",
    only_scale_pose_index=99,
    POSE_KEYPOINT=None
):
    if POSE_KEYPOINT is not None:
        POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
        hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
            scalelist_behavior, POSE_JSON, hands_scale, body_scale, head_scale, overall_scale,
            match_scalelist_method, only_scale_pose_index)
        normalized_pose_json = pose_normalized(POSE_JSON)
        pose_imgs, POSE_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
        if pose_imgs:
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return {
                "ui": {"POSE_JSON": [json.dumps(POSE_SCALED, indent=4)]},
                "result": (torch.from_numpy(pose_imgs_np), POSE_SCALED, json.dumps(POSE_SCALED, indent=4))
            }

    # otherwise output blank images
    W=512
    H=768
    pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
    pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
    people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

    W_scaled = resolution_x
    if resolution_x < 64:
        W_scaled = W
    H_scaled = int(H*(W_scaled*1.0/W))
    pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
    pose_img_np = np.array(pose_img).astype(np.float32) / 255

    return {
        "ui": {"POSE_JSON": people},
        "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
    }

@io.comfytype(io_type="DICT")
class DictType:
    Type = dict
    
    class Output(io.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class Input(io.Input):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class SAMPreprocessNHWC(io.ComfyNode):
    """
    Prepare IMAGE for SAM predictor inside other nodes:
      - Ensure RGB (drop alpha)
      - Resize so long side == 1024 (keeps aspect)
      - Scale to 0..1 float32
      - Return NHWC back (ComfyUI IMAGE), which the next node will convert as needed
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SAMPreprocessNHWC"),
            display_name="SAMPreprocessNHWC",
            category="🧊 frost-byte/Preprocessing",
            inputs=[
                io.Image.Input("input_image", tooltip="Input IMAGE to preprocess for SAM" ),
            ],
            outputs=[
                io.Image.Output("output_image", tooltip="Preprocessed IMAGE in NHWC format"),
                io.String.Output("info", tooltip="Information about the preprocessing"),
            ],
        )
    @classmethod
    def execute(cls, input_image):
        if input_image.ndim != 4:
            raise RuntimeError("IMAGE must be [B,H,W,C]")

        print(f"SAMPreprocessNHWC: image in shape={input_image.shape}")        
        b, h, w, c = input_image.shape
        img = input_image

        # drop alpha if present
        if c == 4:
            img = img[..., :3]
            c = 3
        if c != 3:
            raise RuntimeError(f"SAM expects RGB 3ch, got {c}")

        # convert to float32, scale to 0..255 (SAM torch path often expects that)
        img = img.to(torch.float32).clamp(0, 1)

        # resize so max(H,W)=1024 with aspect
        long_side = max(h, w)
        if long_side != 1024:
            scale = 1024.0 / long_side
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img = F.interpolate(
                img.permute(0, 3, 1, 2),  # NHWC -> NCHW for interpolate
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1).contiguous()  # back to NHWC
            #.contiguous()  # we do not want to go back to NHWC, output needs to be NCHW for SAM predictor
        # AssertionError: set_torch_image input must be BCHW with long side 1024
        # /home/beerye/comfyui_env/.venv/lib/python3.12/site-packages/segment_anything/predictor.py", line 80, in set_torch_image
        info = f"[fbTools: SAMPreprocessNHWC] out={tuple(img.shape)} range=[{img.min().item():.1f},{img.max().item():.1f}]"
        print(info)
        return io.NodeOutput({
            "output_image": img,
            "info": info
        })

class TailEnhancePro(io.ComfyNode):
    """
    TailEnhancePro:
      - Split last K frames of a LIST[IMAGE], run selected processing chain on them, recombine.
      - Processing toggles + parameters:
          * Deflicker: luma-scale OR CLAHE
          * Color match: histogram OR mean/std affine (with blend amount)
          * Sharpen: unsharp mask (kornia)
          * Denoise: bilateral (opencv)
      - Reference window: how many HEAD frames to compute stats / pick histogram reference from.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("TailEnhancePro"),
            display_name="TailEnhancePro",
            category="🧊 frost-byte/Video",
            inputs=[
                io.Image.Input("input_frames", tooltip="Input IMAGE frames to process"),
                io.Int.Input("tail_count", default=6, min=1, max=999),
                io.Int.Input("ref_window", default=24, min=1, max=999),
                io.Boolean.Input("ref_from_head", default=True, tooltip="if False, uses tail as fallback source"),
                # Deflicker
                io.Boolean.Input("enable_deflicker", default=True),
                io.Combo.Input("deflicker_mode", options=["luma_scale", "clahe"], default="luma_scale", tooltip="Deflicker method"),
                io.Float.Input("deflicker_strength", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="Deflicker strength (0=off, 1=full)"),
                io.Float.Input("clahe_clip_limit", default=2.0, min=0.1, max=10.0, step=0.1, tooltip="CLAHE clip limit (0.1 to 10.0, only if CLAHE mode)"),
                io.Int.Input("clahe_grid_w", default=8, min=2, max=64, tooltip="CLAHE grid width (only if CLAHE mode)"),
                io.Int.Input("clahe_grid_h", default=8, min=2, max=64, tooltip="CLAHE grid height (only if CLAHE mode)"),
                # Color
                io.Boolean.Input("enable_color_match", default=True),
                io.Combo.Input("color_mode", options=["histogram", "meanstd"], default="histogram"),
                io.Float.Input("color_amount", default=0.6, min=0.0, max=1.0, step=0.05, tooltip="Color match amount (0=off, 1=full)"),
                # Sharpen
                io.Boolean.Input("enable_unsharp", default=True),
                io.Float.Input("unsharp_radius", default=1.5, min=0.1, max=10.0, step=0.1),
                io.Float.Input("unsharp_amount", default=0.5, min=0.0, max=3.0, step=0.05),
                # Denoise
                io.Boolean.Input("enable_bilateral", default=False),
                io.Int.Input("bilateral_d", default=5, min=1, max=25),
                io.Float.Input("bilateral_sigma_color", default=25.0, min=1.0, max=250.0, step=1.0),
                io.Float.Input("bilateral_sigma_space", default=7.0, min=1.0, max=100.0, step=1.0),
            ],
            outputs=[
                io.Image.Output("output_frames", tooltip="Processed IMAGE frames"),
                io.Image.Output("batched", tooltip="Batched output if all frames same shape"),
                io.String.Output("info", tooltip="Info / debug messages"),
            ],
        )

    @classmethod
    def execute(
        cls,
        input_frames,
        tail_count,
        ref_window,
        ref_from_head,
        enable_deflicker,
        deflicker_mode,
        deflicker_strength,
        clahe_clip_limit,
        clahe_grid_w,
        clahe_grid_h,
        enable_color_match,
        color_mode,
        color_amount,
        enable_unsharp,
        unsharp_radius,
        unsharp_amount,
        enable_bilateral,
        bilateral_d,
        bilateral_sigma_color,
        bilateral_sigma_space
    ):

        info_msgs = []
        if not input_frames or len(input_frames) == 0:
            return ([], None, "[TailEnhancePro] empty input")

        n = len(input_frames)
        k = max(1, min(int(tail_count), n))
        head = input_frames[: n - k]
        tail = input_frames[n - k :]

        # Reference set
        ref_src = head if (ref_from_head and len(head) > 0) else (tail if len(tail) > 0 else input_frames)
        mean_c, std_c, mean_luma = _compute_ref_stats(ref_src, ref_window)
        ref_img_for_hist = _pick_ref_image(ref_src, ref_window)

        if enable_deflicker and deflicker_mode == "clahe" and not _HAS_KORNIA:
            info_msgs.append("CLAHE requested but kornia not installed -> skipped")
        if enable_color_match and color_mode == "histogram" and not _HAS_SKIMAGE:
            info_msgs.append("Histogram match requested but scikit-image not installed -> skipped")
        if enable_bilateral and not _HAS_CV2:
            info_msgs.append("Bilateral requested but opencv-python not installed -> skipped")

        out_tail = []
        for img in tail:
            x = img
            if enable_deflicker:
                if deflicker_mode == "luma_scale":
                    x = proc_deflicker_luma(x, mean_luma, deflicker_strength)
                else:
                    x = proc_deflicker_clahe(x, clahe_clip_limit, clahe_grid_w, clahe_grid_h)

            if enable_color_match:
                if color_mode == "histogram" and ref_img_for_hist is not None and _HAS_SKIMAGE:
                    x = proc_color_histmatch(x, ref_img_for_hist, color_amount)
                else:
                    x = proc_color_meanstd(x, mean_c.to(x), std_c.to(x), color_amount)

            if enable_bilateral:
                x = proc_bilateral_cv2(x, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

            if enable_unsharp:
                x = proc_unsharp(x, unsharp_radius, unsharp_amount)

            out_tail.append(x.clamp(0,1))

        out_frames = list(head) + out_tail
        batched = _stack_if_same_shape(out_frames)

        msg = f"[TailEnhancePro] n={n} tail={k} ref_window={ref_window} " \
            f"ops(deflicker={enable_deflicker}:{deflicker_mode}, color={enable_color_match}:{color_mode}, " \
            f"bilateral={enable_bilateral}, unsharp={enable_unsharp})"
        if info_msgs:
            msg += " | " + " ; ".join(info_msgs)

        return io.NodeOutput({
            "output_frames": out_frames,
            "batched": batched,
            "info": msg
        })

class TailSplit(io.ComfyNode):
    """
    Splits the input image batch into two parts: the main part and a tail part.
    The tail part is defined as the last `tail_size` images in the batch.
    - IMAGE is expected as [B, H, W, C],
    - Returns:
        - main_image: [B - tail_size, H, W, C]
        - tail_image: [tail_size, H, W, C]
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("TailSplit"),
            display_name="TailSplit",
            category="🧊 frost-byte/Video",
            inputs=[
                io.Image.Input("image", tooltip="Input image batch"),
                io.Int.Input("tail_size", default=5, min=1, max=100, tooltip="Number of images to include in the tail part"),
                io.Boolean.Input("debug", default=False, tooltip="If true, will print debug info to console"),
            ],
            outputs=[
                io.Image.Output("main_image", tooltip="Main image batch without tail"),
                io.Image.Output("tail_image", tooltip="Tail image batch"),
                io.String.Output("debug_info", tooltip="Debug information"),
            ],
        )

    @classmethod
    def execute(cls, image, tail_size=1, debug=False):
        # image: torch.FloatTensor [B, H, W, C]
        if not torch.is_tensor(image):
            raise ValueError("fbTools -> TailSplit: Input 'image' must be a torch tensor")

        print(f"fbTools -> TailSplit: image in shape={image.shape}, tail_size={tail_size}, dtype={image.dtype}, device={image.device}") if debug else None        
        b, h, w, c = image.shape
        print(f"fbTools -> TailSplit: b={b}, h={h}, w={w}, c={c}") if debug else None
        
        if tail_size >= b:
            raise ValueError("tail_size must be less than the batch size")
        
        main_image = image[:-tail_size]  # [B - tail_size, H, W, C]
        tail_image = image[-tail_size:]   # [tail_size, H, W, C]
        
        try:
            mn = image.detach().min().item()
            mx = image.detach().max().item()
            alpha_summary = f" range=[{mn:.6f},{mx:.6f}]"
        except Exception:
            alpha_summary = ""
            
        msg = (
            f"fbTools -> TailSplit: image in shape={image.shape}, tail_size={tail_size}, dtype={image.dtype}, device={image.device}, "
            f"-> main_image shape={main_image.shape}, tail_image shape={tail_image.shape}{alpha_summary}"
        )
        
        if debug:
            print(msg)

        return io.NodeOutput({
            "main_image": main_image,
            "tail_image": tail_image,
            "debug_info": msg
        })

class OpaqueAlpha(io.ComfyNode):
    """
    Creates an opaque mask (all 1.0) matching the input image's spatial size and applies it
    as an alpha channel to the input image. Handles RGB or RGBA input images and batches.
    - IMAGE is expected as [B, C, H, W], float 0..1
    - Returns:
        - image_rgba: [B, 4, H, W]
        - mask: [B, 1, H, W] (float 0..1)
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("OpaqueAlpha"),
            display_name="OpaqueAlpha",
            category="🧊 frost-byte/Image Processing",
            inputs=[
                io.Image.Input("image", tooltip="Input image, RGB or RGBA"),
                io.Float.Input("alpha_value", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Alpha value to set in the mask"),
                io.Boolean.Input("force_replace_alpha", default=True, tooltip="If true, will replace existing alpha channel if input image is RGBA"),
                io.Boolean.Input("debug", default=False, tooltip="If true, will print debug info to console"),
            ],
            outputs=[
                io.Image.Output("image_rgba", tooltip="Output image with RGBA channels"),
                io.Image.Output("mask", tooltip="Opaque alpha mask"),
                io.String.Output("debug_info", tooltip="Debug information"),
            ],
        )

    @classmethod
    def execute(cls, image, alpha_value=1.0, force_replace_alpha=True, debug=False):
        # image: torch.FloatTensor [B, H, W, C], C=3 or 4, float 0..1
        if not torch.is_tensor(image):
            raise ValueError("Input 'image' must be a torch tensor")

        print(f"OpaqueAlpha: image in shape={image.shape}, alpha_value={alpha_value}, force_replace_alpha={force_replace_alpha},dtype={image.dtype}, device={image.device}") if debug else None        
        b, h, w, c = image.shape
        print(f"OpaqueAlpha: b={b}, h={h}, w={w}, c={c}") if debug else None
        device = image.device
        dtype = image.dtype
        
        # Build an opaque mask [B, H, W, 1]
        mask = torch.full((b, h, w, 1), fill_value=alpha_value, device=device, dtype=dtype)
        
        if c == 4:
            if force_replace_alpha:
                # Replace existing alpha channel
                image_rgba = image.clone()
                image_rgba[:, :, :, 3:4] = mask
            else:
                # Keep existing alpha channel
                image_rgba = image
        elif c == 3:
            # Add alpha channel
            image_rgba = torch.cat([image, mask], dim=3)  # [B, H, W, 4]
        else:
            raise ValueError("Input 'image' must have 3 (RGB) or 4 (RGBA) channels")
        
        try:
            mn = image.detach().min().item()
            mx = image.detach().max().item()
            alpha_summary = f" alpha_range=[{mn:.6f},{mx:.6f}]"
        except Exception:
            alpha_summary = ""
            
        msg = (
            f"OpaqueAlpha: image in shape={image.shape}, alpha_value={alpha_value}, force_replace_alpha={force_replace_alpha},dtype={image.dtype}, device={image.device}, "
            f"range=[{mn:.6f},{mx:.6f}] -> image out shape={image_rgba.shape}, mask shape={mask.shape}{alpha_summary}"
        )
        
        if debug:
            print(msg)

        return io.NodeOutput({
            "image_rgba": image_rgba,
            "mask": mask,
            "debug_info": msg
        })

def get_subdirectories(directory_path: str) -> dict:
    """Return a dictionary mapping subdirectory names to their full paths."""
    subdir_dict = {}

    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' does not exist or is not a directory.")
        return subdir_dict

    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_dir():
                # print(f"Found subdirectory: {entry.name} at {entry.path}")
                subdir_dict[entry.name] = entry.path

    return subdir_dict

class SubdirLister(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SubdirLister"),
            display_name="SubdirLister",
            category="🧊 frost-byte/File",
            inputs=[
                io.String.Input("directory_path", default="", tooltip="Path to the directory"),
            ],
            outputs=[
                io.Custom("DICT").Output("dir_dict", tooltip="Dictionary of subdirectory names to paths"),
                io.String.Output("dir_names", tooltip="List of subdirectory names"),
            ],
        )
    
    @classmethod
    def execute(cls, directory_path):

        subdir_dict = get_subdirectories(directory_path)

        return io.NodeOutput({
            "dir_dict": subdir_dict,
            "dir_names": list(subdir_dict.keys()) if subdir_dict else []
        })

def default_poses_dir():
    output_dir = get_output_directory()
    default_dir = os.path.join(output_dir, "poses")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
        os.makedirs(os.path.join(default_dir, "default_pose"), exist_ok=True)
    return default_dir

class QwenAspectRatio(io.ComfyNode):
    """
    QwenAspectRatio:
      - Computes aspect ratio string for Qwen input from IMAGE dimensions.
      - Outputs recommended width and height based upon standard aspect ratios.
      - Outputs the layout type based upon the aspect ratio, e.g., "portrait", "landscape", "square".
      - Outputs string like "16:9" or "4:3".
      - Outputs the float value of the aspect ratio (width / height).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            description=cleandoc("""
            Computes recommended width and height based upon the aspect ratio of the input IMAGE.
            Also provides the layout type (e.g., 'portrait', 'landscape', 'square'), aspect ratio string (e.g., '16:9'),
            and the float value of the aspect ratio (width / height).
            """),
            node_id=prefixed_node_id("QwenAspectRatio"),
            display_name="QwenAspectRatio",
            category="🧊 frost-byte/Image Processing",
            inputs=[
                io.Image.Input("input_image", tooltip="Input IMAGE to compute aspect ratio from"),
            ],
            outputs=[
                io.Int.Output(id="width", display_name="width", tooltip="Recommended width for Qwen based on aspect ratio"),
                io.Int.Output(id="height", display_name="height", tooltip="Recommended height for Qwen based on aspect ratio"),
                io.String.Output(id="layout", display_name="layout", tooltip="Layout type based on aspect ratio (e.g., 'portrait', 'landscape', 'square')"),
                io.String.Output(id="aspect_ratio", display_name="aspect_ratio", tooltip="Aspect ratio string for Qwen (e.g., '16:9')"),
                io.Float.Output(id="aspect_ratio_float", display_name="aspect_ratio_float", tooltip="Float value of the aspect ratio (width / height)"),
            ],
        )

    @classmethod
    def execute(
        cls,
        input_image,
    ):
        if input_image is None:
            w, h = 512, 512
        elif input_image.ndim == 3:
            b, h, w = input_image.shape
        else:
            b, h, w, c = input_image.shape

        print(f"QwenAspectRatio: input image shape={input_image.shape} -> w={w}, h={h}")
        recommended_w, recommended_h, layout, aspect_ratio_str, aspect_ratio_float = find_nearest_qwen_aspect_ratio(w, h)
        print(f"QwenAspectRatio: recommended_w={recommended_w}, recommended_h={recommended_h}, layout={layout}, aspect_ratio_str={aspect_ratio_str}, aspect_ratio_float={aspect_ratio_float}")

        return io.NodeOutput(
            recommended_w,
            recommended_h,
            layout,
            aspect_ratio_str,
            aspect_ratio_float
        )


# ============================================================================
# PROMPT COLLECTION - Flexible Multi-Prompt System
# ============================================================================

# Import the data models from separate module for better testability
from .prompt_models import PromptMetadata, PromptCollection


class SceneInfo(BaseModel):
    #metadata
    pose_dir: str
    pose_name: str
    
    # Legacy individual prompt fields - maintained for backward compatibility
    girl_pos: str = ""
    male_pos: str = ""
    wan_prompt: str = ""
    wan_low_prompt: str = ""
    four_image_prompt: str = ""
    
    # V2 PromptCollection - new flexible prompt system
    prompts: Optional[PromptCollection] = None
    
    pose_json: str
    resolution: int

    # Image Tensors (ComfyUI uses torch.Tensor with shape [B,H,W,C] for IMAGE)
    depth_image: Optional[torch.Tensor] = None
    depth_any_image: Optional[torch.Tensor] = None
    depth_midas_image: Optional[torch.Tensor] = None
    depth_zoe_image: Optional[torch.Tensor] = None
    depth_zoe_any_image: Optional[torch.Tensor] = None
    pose_dense_image: Optional[torch.Tensor] = None
    pose_dw_image: Optional[torch.Tensor] = None
    pose_dwpose_json: Optional[str] = None
    pose_edit_image: Optional[torch.Tensor] = None
    pose_face_image: Optional[torch.Tensor] = None
    pose_open_image: Optional[torch.Tensor] = None
    canny_image: Optional[torch.Tensor] = None
    upscale_image: Optional[torch.Tensor] = None
    loras_high: Optional[list] = None
    loras_low: Optional[list] = None

    # masks
    girl_mask_bkgd_image: Optional[torch.Tensor] = None
    male_mask_bkgd_image: Optional[torch.Tensor] = None
    combined_mask_bkgd_image: Optional[torch.Tensor] = None
    girl_mask_no_bkgd_image: Optional[torch.Tensor] = None
    male_mask_no_bkgd_image: Optional[torch.Tensor] = None
    combined_mask_no_bkgd_image: Optional[torch.Tensor] = None
    
    # Backward compatibility properties - delegate to PromptCollection if present
    def get_prompt_field(self, field_name: str, legacy_value: str) -> str:
        """Get prompt from PromptCollection if available, else return legacy field."""
        if self.prompts:
            value = self.prompts.get_prompt_value(field_name)
            return value if value is not None else legacy_value
        return legacy_value

    def three_image_prompt(self) -> str:
        return f"{self.girl_pos} {self.male_pos}"

    def input_img_glob(self) -> str:
        return os.path.join(self.pose_dir, "input") + "/*.png"

    def input_img_dir(self) -> str:
        return f"poses/{self.pose_name}/input/img"

    def output_dir(self) -> str:
        return f"poses/{self.pose_name}/output"

    @classmethod
    def load_depth_images(cls, pose_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load depth images from a pose directory, optionally filtering by keys."""
        mapping = {
            'depth_image': "depth.png",
            'depth_any_image': "depth_any.png",
            'depth_midas_image': "depth_midas.png",
            'depth_zoe_image': "depth_zoe.png",
            'depth_zoe_any_image': "depth_zoe_any.png",
        }

        def _img(path: str):
            img, _ = load_image_comfyui(path, include_mask=False)
            return img

        selected_keys = list(mapping.keys()) if keys is None else list(keys)
        images = {}
        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            images[key] = _img(os.path.join(pose_dir, filename))
        return images

    @classmethod
    def load_pose_images(cls, pose_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load pose images from a pose directory, optionally filtering by keys."""
        mapping = {
            'pose_dense_image': "pose_dense.png",
            'pose_dw_image': "pose_dw.png",
            'pose_edit_image': "pose_edit.png",
            'pose_face_image': "pose_face.png",
            'pose_open_image': "pose_open.png",
            'canny_image': "canny.png",
            'upscale_image': "upscale.png",
        }

        def _img(path: str):
            img, _ = load_image_comfyui(path, include_mask=False)
            return img

        selected_keys = list(mapping.keys()) if keys is None else list(keys)
        images = {}
        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            images[key] = _img(os.path.join(pose_dir, filename))
        return images

    @classmethod
    def load_mask_images(cls, pose_dir: str, keys: Optional[list[str]] = None) -> tuple[dict, dict]:
        """Load mask images and their masks from a pose directory.

        Returns a tuple `(images, masks)` where images maps mask keys to IMAGE tensors,
        and masks maps the same keys to [B,H,W] float masks (1.0 means masked-out).
        """
        mapping = {
            "girl": "girl_mask_bkgd.png",
            "male": "male_mask_bkgd.png",
            "combined": "combined_mask_bkgd.png",
            "girl_no_bg": "girl_mask_no_bkgd.png",
            "male_no_bg": "male_mask_no_bkgd.png",
            "combined_no_bg": "combined_mask_no_bkgd.png",
        }

        images = {}
        masks = {}
        selected_keys = list(mapping.keys()) if keys is None else list(keys)

        for key in selected_keys:
            filename = mapping.get(key)
            if not filename:
                continue
            image, mask = load_image_comfyui(os.path.join(pose_dir, filename), include_mask=True)
            images[key] = image
            masks[key] = mask

        return images, masks

    @classmethod
    def load_all_images(cls, pose_dir: str) -> dict:
        """Load all images (depth, pose, mask) from a pose directory"""
        all_images = {}
        all_images.update(cls.load_depth_images(pose_dir))
        all_images.update(cls.load_pose_images(pose_dir))
        mask_images, _ = cls.load_mask_images(pose_dir)
        all_images.update(mask_images)
        return all_images

    @classmethod
    def load_preview_assets(
            cls,
            pose_dir: str,
            depth_attr: str,
            pose_attr: str,
            mask_type: str,
            mask_background: bool = True,
            include_upscale: bool = False,
            include_canny: bool = False,
    ) -> dict:
        """Load a minimal, normalized bundle for preview/output (depth, pose, mask, optional upscale/canny).

        Returns dict keys:
            depth_image, pose_image, mask_image, mask (B,H,W,1), mask_preview (B,H,W,3),
            base_image, canny_image, preview_batch (list of tensors), H, W, resolution,
            plus raw dictionaries depth_images/pose_images/mask_images for downstream SceneInfo population.
        """
        mask_key = resolve_mask_key(mask_type, mask_background)

        depth_keys = {depth_attr, "depth_image"}
        pose_keys = {pose_attr, "pose_open_image"}
        if include_upscale:
            pose_keys.add("upscale_image")
        if include_canny:
            pose_keys.add("canny_image")
        mask_keys = {mask_key, "combined"}

        depth_images = cls.load_depth_images(pose_dir, keys=list(depth_keys))
        pose_images = cls.load_pose_images(pose_dir, keys=list(pose_keys))
        mask_images, mask_tensors = cls.load_mask_images(pose_dir, keys=list(mask_keys))

        # Determine spatial size from available images
        empty_image = make_empty_image(1, 512, 512)
        base_image = pose_images.get("upscale_image") if include_upscale else None
        depth_image_raw = depth_images.get("depth_image")
        pose_image_raw = pose_images.get(pose_attr, pose_images.get("pose_open_image", empty_image))
        mask_image_raw = mask_images.get(mask_key, mask_images.get("combined", empty_image))

        if base_image is not None:
            H, W = base_image.shape[1], base_image.shape[2]
        elif depth_image_raw is not None:
            H, W = depth_image_raw.shape[1], depth_image_raw.shape[2]
        elif pose_image_raw is not None:
            H, W = pose_image_raw.shape[1], pose_image_raw.shape[2]
        elif mask_image_raw is not None:
            H, W = mask_image_raw.shape[1], mask_image_raw.shape[2]
        else:
            H, W = 512, 512

        # Normalize images to a consistent size
        depth_image = normalize_image_tensor(depth_images.get(depth_attr, depth_images.get("depth_image", empty_image)), H, W)
        pose_image = normalize_image_tensor(pose_image_raw, H, W)
        base_image = normalize_image_tensor(base_image, H, W) if include_upscale else None
        mask_image = normalize_image_tensor(mask_image_raw, H, W)

        # Build mask output (single-channel) and preview (3-channel)
        mask = None
        mask_tensor = mask_tensors.get(mask_key)
        unsqueeze_me = False
        if mask_tensor is not None:
            print(f"SceneInfo.load_preview_assets: using mask tensor for key '{mask_key}'")
            mask = mask_tensor
            unsqueeze_me = True
        elif mask_image is not None:
            print(f"SceneInfo.load_preview_assets: building empty mask matching mask_image shape")
            b, hh, ww, _ = mask_image.shape
            mask = torch.zeros((b, hh, ww, 1), device=mask_image.device, dtype=torch.float32)
        else:
            print(f"SceneInfo.load_preview_assets: building empty mask of size (1,{H},{W},1)")
            mask = torch.zeros((1, H, W, 1), dtype=torch.float32)

        if mask is not None and mask.dtype != torch.float32:
            print(f"SceneInfo.load_preview_assets: converting mask to float32")
            mask = mask.float()

        mask_preview = None
        if mask is not None:
            preview_mask = mask
            if unsqueeze_me:
                preview_mask = mask.unsqueeze(-1)
            if preview_mask.shape[-1] == 1:
                preview_mask = preview_mask.repeat(1, 1, 1, 3)
            mask_preview = normalize_image_tensor(preview_mask, H, W)

        canny_image = None
        if include_canny:
            canny_image = normalize_image_tensor(pose_images.get("canny_image"), H, W)

        preview_batch = []
        if base_image is not None:
            preview_batch.append(base_image)
        if mask_image is not None:
            preview_batch.append(mask_image)
        if pose_image is not None:
            preview_batch.append(pose_image)
        if depth_image is not None:
            preview_batch.append(depth_image)
        if mask_preview is not None:
            preview_batch.append(mask_preview)

        resolution = max(H, W)

        return {
            "depth_image": depth_image,
            "pose_image": pose_image,
            "mask_image": mask_image,
            "mask": mask,
            "mask_preview": mask_preview,
            "base_image": base_image,
            "canny_image": canny_image,
            "preview_batch": preview_batch,
            "H": H,
            "W": W,
            "resolution": resolution,
            "depth_images": depth_images,
            "pose_images": pose_images,
            "mask_images": mask_images,
        }


    @classmethod
    def from_story_scene(
            cls,
            scene: "SceneInStory",
            poses_dir: Optional[str] = None,
            prompt_in: str = "",
            prompt_action: str = "use_file",
            include_upscale: bool = False,
            include_canny: bool = False,
            prompt_override: Optional[str] = None,
            pose_dir_override: Optional[str] = None,
    ) -> tuple["SceneInfo", dict, str, dict, Optional[str]]:
        """Build SceneInfo + assets from a SceneInStory configuration.

        Returns (scene_info, assets, selected_prompt, prompt_data, prompt_widget_text).
        """

        poses_dir = poses_dir or default_poses_dir()
        pose_dir = pose_dir_override if pose_dir_override else os.path.join(poses_dir, scene.scene_name)

        if not os.path.isdir(pose_dir):
            raise ValueError(f"from_story_scene: pose_dir '{pose_dir}' is invalid")

        prompt_json_path = os.path.join(pose_dir, "prompts.json")
        prompt_data = load_prompt_json(prompt_json_path) or {}

        prompt_file_text = build_positive_prompt(scene.prompt_type, prompt_data, scene.custom_prompt)
        class_name = f"{cls.__name__}.from_story_scene"
        selected_prompt, prompt_widget_text = select_text_by_action(
            prompt_in,
            prompt_file_text,
            prompt_action,
            class_name,
        )
        if prompt_override:
            selected_prompt = prompt_override

        pose_json_path = os.path.join(pose_dir, "pose.json")
        pose_json_obj = load_json_file(pose_json_path)
        pose_json = json.dumps(pose_json_obj) if pose_json_obj else "[]"

        loras_path = os.path.join(pose_dir, "loras.json")
        loras_high, loras_low = load_loras(loras_path) if os.path.isfile(loras_path) else (None, None)

        depth_attr = default_depth_options.get(scene.depth_type, "depth_image")
        pose_attr = default_pose_options.get(scene.pose_type, "pose_open_image")

        assets = cls.load_preview_assets(
            pose_dir,
            depth_attr=depth_attr,
            pose_attr=pose_attr,
            mask_type=scene.mask_type,
            mask_background=scene.mask_background,
            include_upscale=include_upscale,
            include_canny=include_canny,
        )

        depth_images = assets.get("depth_images", {})
        pose_images = assets.get("pose_images", {})
        mask_images = assets.get("mask_images", {})

        girl_pos_val = selected_prompt if scene.prompt_type == "girl_pos" else prompt_data.get("girl_pos", "")
        male_pos_val = selected_prompt if scene.prompt_type == "male_pos" else prompt_data.get("male_pos", "")
        four_image_prompt_val = selected_prompt if scene.prompt_type == "four_image_prompt" else prompt_data.get("four_image_prompt", "")
        wan_prompt_val = selected_prompt if scene.prompt_type == "wan_prompt" else prompt_data.get("wan_prompt", "")
        wan_low_prompt_val = selected_prompt if scene.prompt_type == "wan_low_prompt" else prompt_data.get("wan_low_prompt", "")

        scene_info = cls(
            pose_dir=pose_dir,
            pose_name=scene.scene_name,
            girl_pos=girl_pos_val,
            male_pos=male_pos_val,
            four_image_prompt=four_image_prompt_val,
            wan_prompt=wan_prompt_val,
            wan_low_prompt=wan_low_prompt_val,
            pose_json=pose_json,
            resolution=assets.get("resolution", 0),
            loras_high=loras_high,
            loras_low=loras_low,
            **depth_images,
            **pose_images,
            **mask_images,
        )

        return scene_info, assets, selected_prompt or "", prompt_data, prompt_widget_text

    @classmethod
    def from_pose_directory(cls, pose_dir: str, pose_name: str, prompt_data: Optional[dict] = None, 
                           pose_json: str = "", loras_high: Optional[list] = None, loras_low: Optional[list] = None):
        """Create a SceneInfo instance by loading all data from a pose directory"""
        if prompt_data is None:
            prompt_json_path = os.path.join(pose_dir, "prompts.json")
            prompt_data = load_prompt_json(prompt_json_path)
        
        # Migrate legacy prompts to PromptCollection
        prompt_collection = None
        if prompt_data:
            # Check if it's v2 format (has "version" field)
            if "version" in prompt_data and prompt_data.get("version") == 2:
                prompt_collection = PromptCollection.from_dict(prompt_data)
            else:
                # Legacy format - migrate
                prompt_collection = PromptCollection.from_legacy_dict(prompt_data)
                print(f"SceneInfo.from_pose_directory: Migrated {len(prompt_collection.prompts)} legacy prompts")
        else:
            # No prompts file - create empty collection
            prompt_collection = PromptCollection()
        
        # Load all images
        all_images = cls.load_all_images(pose_dir)
        
        # Determine resolution from depth_image
        depth_image = all_images.get('depth_image')
        if depth_image is not None:
            H, W = depth_image.shape[1], depth_image.shape[2]
            resolution = max(H, W)
        else:
            resolution = 512
        
        return cls(
            pose_dir=pose_dir,
            pose_name=pose_name,
            prompts=prompt_collection,
            pose_json=pose_json,
            resolution=resolution,
            loras_high=loras_high,
            loras_low=loras_low,
            **all_images
        )

    def save_all_images(self, pose_dir: Optional[str] = None):
        """Save all images to the pose directory"""
        from pathlib import Path
        
        pose_path = Path(pose_dir) if pose_dir else Path(self.pose_dir)
        
        # Save depth images
        if self.depth_image is not None:
            save_image_comfyui(self.depth_image, pose_path / "depth.png")
        if self.depth_any_image is not None:
            save_image_comfyui(self.depth_any_image, pose_path / "depth_any.png")
        if self.depth_midas_image is not None:
            save_image_comfyui(self.depth_midas_image, pose_path / "depth_midas.png")
        if self.depth_zoe_image is not None:
            save_image_comfyui(self.depth_zoe_image, pose_path / "depth_zoe.png")
        if self.depth_zoe_any_image is not None:
            save_image_comfyui(self.depth_zoe_any_image, pose_path / "depth_zoe_any.png")
        
        # Save pose images
        if self.pose_dense_image is not None:
            save_image_comfyui(self.pose_dense_image, pose_path / "pose_dense.png")
        if self.pose_dw_image is not None:
            save_image_comfyui(self.pose_dw_image, pose_path / "pose_dw.png")
        if self.pose_edit_image is not None:
            save_image_comfyui(self.pose_edit_image, pose_path / "pose_edit.png")
        if self.pose_face_image is not None:
            save_image_comfyui(self.pose_face_image, pose_path / "pose_face.png")
        if self.pose_open_image is not None:
            save_image_comfyui(self.pose_open_image, pose_path / "pose_open.png")
        if self.canny_image is not None:
            save_image_comfyui(self.canny_image, pose_path / "canny.png")
        if self.upscale_image is not None:
            save_image_comfyui(self.upscale_image, pose_path / "upscale.png")
        
        # Save mask images
        if self.girl_mask_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_bkgd_image, pose_path / "girl_mask_bkgd.png")
        if self.male_mask_bkgd_image is not None:
            save_image_comfyui(self.male_mask_bkgd_image, pose_path / "male_mask_bkgd.png")
        if self.combined_mask_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_bkgd_image, pose_path / "combined_mask_bkgd.png")
        if self.girl_mask_no_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_no_bkgd_image, pose_path / "girl_mask_no_bkgd.png")
        if self.male_mask_no_bkgd_image is not None:
            save_image_comfyui(self.male_mask_no_bkgd_image, pose_path / "male_mask_no_bkgd.png")
        if self.combined_mask_no_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_no_bkgd_image, pose_path / "combined_mask_no_bkgd.png")

    def save_prompts(self, pose_dir: Optional[str] = None):
        """Save prompts to prompts.json in v2 format with v1_backup"""
        from pathlib import Path
        
        pose_path = Path(pose_dir) if pose_dir else Path(self.pose_dir)
        prompts_path = pose_path / "prompts.json"
        
        # If using PromptCollection, save v2 format
        if self.prompts:
            save_json_file(prompts_path, self.prompts.to_dict())
        else:
            # Legacy mode: save v1 format but wrap in v2 structure for migration
            legacy_data = {
                "girl_pos": self.girl_pos if self.girl_pos else "",
                "male_pos": self.male_pos if self.male_pos else "",
                "wan_prompt": self.wan_prompt if self.wan_prompt else "",
                "wan_low_prompt": self.wan_low_prompt if self.wan_low_prompt else "",
                "four_image_prompt": self.four_image_prompt if self.four_image_prompt else "",
            }
            # Auto-migrate to v2 format on save
            prompt_collection = PromptCollection.from_legacy_dict(legacy_data)
            save_json_file(prompts_path, prompt_collection.to_dict())

    def save_pose_json(self, pose_dir: Optional[str] = None):
        """Save pose_json to pose.json in the pose directory"""
        from pathlib import Path
        import json
        
        if not self.pose_json:
            return
        
        pose_path = Path(pose_dir) if pose_dir else Path(self.pose_dir)
        pose_json_path = pose_path / "pose.json"
        save_json_file(pose_json_path, json.loads(self.pose_json))

    def save_loras(self, pose_dir: Optional[str] = None):
        """Save LoRAs to loras.json in the pose directory"""
        from pathlib import Path
        
        if self.loras_high is None and self.loras_low is None:
            return
        
        pose_path = Path(pose_dir) if pose_dir else Path(self.pose_dir)
        loras_path = pose_path / "loras.json"
        # save_loras function handles None values, but we need to provide defaults
        save_loras(self.loras_high or [], self.loras_low or [], str(loras_path))

    def ensure_directories(self, pose_dir: Optional[str] = None):
        """Ensure pose directory and input/output subdirectories exist"""
        import os
        
        pose_path = pose_dir if pose_dir else self.pose_dir
        
        if not os.path.exists(pose_path):
            os.makedirs(pose_path, exist_ok=True)
            print(f"SceneInfo: Created pose_dir='{pose_path}'")
        
        input_dir = os.path.join(pose_path, "input")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            print(f"SceneInfo: Created input_dir='{input_dir}'")
        
        output_dir = os.path.join(pose_path, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"SceneInfo: Created output_dir='{output_dir}'")

    def save_all(self, pose_dir: Optional[str] = None):
        """Save all scene data (images, prompts, pose_json, loras) to the pose directory"""
        target_dir = pose_dir if pose_dir else self.pose_dir
        self.ensure_directories(target_dir)
        self.save_all_images(target_dir)
        self.save_prompts(target_dir)
        self.save_pose_json(target_dir)
        self.save_loras(target_dir)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

class SceneInStory(BaseModel):
    """Represents a scene within a story with its configuration"""
    scene_id: str = ""  # Unique identifier for this scene instance
    scene_name: str
    scene_order: int
    mask_type: str = "combined"  # girl, male, combined, girl_no_bg, male_no_bg, combined_no_bg
    mask_background: bool = True
    prompt_type: str = "girl_pos"  # girl_pos, male_pos, four_image_prompt, wan_prompt, wan_low_prompt, custom
    custom_prompt: str = ""
    depth_type: str = "depth"
    pose_type: str = "open"
    
    def __init__(self, **data):
        if 'scene_id' not in data or not data['scene_id']:
            import uuid
            data['scene_id'] = str(uuid.uuid4())
        super().__init__(**data)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class StoryInfo(BaseModel):
    """Contains ordered list of scenes and story metadata"""
    story_name: str
    story_dir: str
    scenes: List[SceneInStory] = []
    
    def get_scene_by_id(self, scene_id: str) -> Optional[SceneInStory]:
        """Get scene by unique ID"""
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None
    
    def get_scene_by_name(self, scene_name: str) -> Optional[SceneInStory]:
        """Get first scene matching the name (multiple scenes can share same name)"""
        for scene in self.scenes:
            if scene.scene_name == scene_name:
                # Ensure scene has UUID (for backward compatibility)
                if not scene.scene_id:
                    import uuid
                    scene.scene_id = str(uuid.uuid4())
                return scene
        return None
    
    def get_scenes_by_name(self, scene_name: str) -> List[SceneInStory]:
        """Get all scenes matching the name"""
        matching_scenes = [scene for scene in self.scenes if scene.scene_name == scene_name]
        # Ensure all scenes have UUID (for backward compatibility)
        for scene in matching_scenes:
            if not scene.scene_id:
                import uuid
                scene.scene_id = str(uuid.uuid4())
        return matching_scenes
    
    def get_scene_by_order(self, order: int) -> Optional[SceneInStory]:
        for scene in self.scenes:
            if scene.scene_order == order:
                # Ensure scene has UUID (for backward compatibility)
                if not scene.scene_id:
                    import uuid
                    scene.scene_id = str(uuid.uuid4())
                return scene
        return None
    
    def add_scene(self, scene: SceneInStory):
        """Add a scene to the story. Each scene gets a unique ID, allowing duplicates of same scene_name."""
        # Ensure scene has a unique ID
        if not scene.scene_id:
            import uuid
            scene.scene_id = str(uuid.uuid4())
        
        # Ensure unique order
        existing_orders = [s.scene_order for s in self.scenes]
        if scene.scene_order in existing_orders:
            # Find next available order
            scene.scene_order = max(existing_orders) + 1 if existing_orders else 0
        
        self.scenes.append(scene)
        self.scenes.sort(key=lambda s: s.scene_order)
    
    def remove_scene(self, scene_identifier: str):
        """Remove scene by ID or by name (removes first match if name)"""
        # Try to find by ID first
        scene_to_remove = self.get_scene_by_id(scene_identifier)
        if not scene_to_remove:
            # Fall back to name
            scene_to_remove = self.get_scene_by_name(scene_identifier)
        
        if scene_to_remove:
            self.scenes = [s for s in self.scenes if s.scene_id != scene_to_remove.scene_id]
            # Reorder remaining scenes
            for idx, scene in enumerate(sorted(self.scenes, key=lambda s: s.scene_order)):
                scene.scene_order = idx
    
    def reorder_scene(self, scene_identifier: str, new_order: int):
        """Reorder scene by ID or by name (reorders first match if name)"""
        # Try to find by ID first
        scene = self.get_scene_by_id(scene_identifier)
        if not scene:
            # Fall back to name
            scene = self.get_scene_by_name(scene_identifier)
        
        if scene:
            scene.scene_order = new_order
            self.scenes.sort(key=lambda s: s.scene_order)
            # Normalize orders
            for idx, s in enumerate(self.scenes):
                s.scene_order = idx
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

def load_loras(loras_json_path: str) -> tuple[list, list] | tuple[None, None]:
    if os.path.isfile(loras_json_path):
        data = load_json_file(loras_json_path)
        print(f"load_loras: loaded data from {loras_json_path}: {data}")
        loras_high = []
        loras_low = []
        if not data or not isinstance(data, dict):
            return (loras_high, loras_low)

        for lora_type in ["high", "low"]:
            preset = data.get(lora_type, "")
            if not preset:
                continue
            for item in preset:
                lora_name = item["lora_name"]
                strength = item["strength"]
                low_mem_load = item.get("low_mem_load", False)
                merge_loras = item.get("merge_loras", False)

                if not lora_name or lora_name == "none" or strength == 0.0:
                    continue
                try:
                    full_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                except Exception as e:
                    print(f"Could not resolve path for LoRA '{lora_name}': {e}")
                    continue

                # Use saved blocks/layer_filter if present
                saved_blocks = item.get("blocks", {})
                saved_layer_filter = item.get("layer_filter", "")

                if lora_type == "low":
                    target_list = loras_low
                else:
                    target_list = loras_high

                target_list.append({
                    "path": full_path,
                    "strength": strength,
                    "name": os.path.splitext(lora_name)[0],
                    "blocks": saved_blocks,
                    "layer_filter": saved_layer_filter,
                    "low_mem_load": low_mem_load,
                    "merge_loras": merge_loras,
                })

        return (loras_high, loras_low)
    
    return (None,None)        

def save_loras(loras_high: list, loras_low: list, loras_json_path: str):
    high = []
    low = []

    for lora in loras_high:
        lora_name = os.path.basename(lora["path"])
        strength = lora["strength"]
        blocks = lora.get("blocks", {})
        layer_filter = lora.get("layer_filter", "")
        low_mem_load = lora.get("low_mem_load", False)
        merge_loras = lora.get("merge_loras", False)

        high.append({
            "lora_name": lora_name,
            "strength": strength,
            "blocks": blocks,
            "layer_filter": layer_filter,
            "low_mem_load": low_mem_load,
            "merge_loras": merge_loras,
        })
    for lora in loras_low:
        lora_name = os.path.basename(lora["path"])
        strength = lora["strength"]
        blocks = lora.get("blocks", {})
        layer_filter = lora.get("layer_filter", "")
        low_mem_load = lora.get("low_mem_load", False)
        merge_loras = lora.get("merge_loras", False)

        low.append({
            "lora_name": lora_name,
            "strength": strength,
            "blocks": blocks,
            "layer_filter": layer_filter,
            "low_mem_load": low_mem_load,
            "merge_loras": merge_loras,
        })

    data =  {
        "high": high,
        "low": low
    }
    save_json_file(loras_json_path, data)

def load_story(story_json_path: str) -> Optional[StoryInfo]:
    """Load story information from JSON file"""
    if not os.path.isfile(story_json_path):
        print(f"fbTools: story_json_path '{story_json_path}' is not a valid file")
        return None
    
    try:
        with open(story_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenes = []
        for scene_data in data.get("scenes", []):
            scene = SceneInStory(
                scene_name=scene_data.get("scene_name", ""),
                scene_order=scene_data.get("scene_order", 0),
                mask_type=scene_data.get("mask_type", "combined"),
                mask_background=scene_data.get("mask_background", True),
                prompt_type=scene_data.get("prompt_type", "girl_pos"),
                custom_prompt=scene_data.get("custom_prompt", ""),
                depth_type=scene_data.get("depth_type", "depth"),
                pose_type=scene_data.get("pose_type", "open"),
            )
            scenes.append(scene)
        
        story_info = StoryInfo(
            story_name=data.get("story_name", ""),
            story_dir=data.get("story_dir", ""),
            scenes=scenes
        )
        
        print(f"load_story: loaded story from {story_json_path} with {len(scenes)} scenes")
        return story_info
    except Exception as e:
        print(f"fbTools: Error loading story JSON from '{story_json_path}': {e}")
        return None

def save_story(story_info: StoryInfo, story_json_path: str):
    """Save story information to JSON file"""
    try:
        scenes_data = []
        for scene in story_info.scenes:
            scene_data = {
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "prompt_type": scene.prompt_type,
                "custom_prompt": scene.custom_prompt,
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
            }
            scenes_data.append(scene_data)
        
        story_data = {
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scenes": scenes_data
        }
        
        save_json_file(story_json_path, story_data)
        print(f"save_story: saved story to {story_json_path} with {len(scenes_data)} scenes")
    except Exception as e:
        print(f"fbTools: Error saving story to '{story_json_path}': {e}")

def default_stories_dir():
    output_dir = get_output_directory()
    default_dir = os.path.join(output_dir, "stories")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
    return default_dir

class NodeInputSelect(io.ComfyNode):
    """
    NodeInputSelect:
      - The user is presented with a dropdown list of available nodes - a string containing the node id and type, separated using an _.
      - The user is presented with a dropdown list of available names for the inputs in the selected node.
      - A node that allows selection of a input from a list of available inputs.
      - Outputs the selected input name
      - Outputs the selected input id as a string.
      - Outputs the selected input value as a string.
    """

    @classmethod
    def define_schema(cls):
        node_data = None
        input_name = "unknown_input"        
        default_inputs = ["unknown_input"]
        # All nodes for the workflow
        nodes_data = get_workflow_all_nodes(cls.__name__)
        
        # List of node names for the dropdown
        nodes = listify_nodes_data(nodes_data)
        nodes = nodes if nodes is not None else []
        # The selected node, default to the first node if available
        first_node_key = list(nodes_data.keys())[0] if nodes_data and isinstance(nodes_data, dict) and len(nodes_data) > 0 else None

        if isinstance(nodes_data, dict) and first_node_key:
            node_data = nodes_data.get(first_node_key, None)

        default_node_name = nodes[0] if nodes and len(nodes) > 0 else "1_Unknown_Node"
        #print(f"{cls.__name__}: available nodes={nodes}; default_node_name='{default_node_name}'; first_node_key='{first_node_key}'")
        node_inputs = node_input_details(cls.__name__, node_data) if node_data else []
        
        if isinstance(node_inputs, dict):
            default_inputs = listify_node_inputs(node_inputs)
        
        if not default_inputs:
            default_inputs = ["unknown_input"]

        if node_inputs and isinstance(node_inputs, dict) and len(node_inputs) > 0:
            input_name = list(node_inputs.keys())[0]
    
        return io.Schema(
            node_id=prefixed_node_id("NodeInputSelect"),
            display_name="NodeInputSelect",
            category="🧊 frost-byte/Nodes",
            inputs=[
                io.Combo.Input(
                    id="node_name",
                    display_name="node_name",
                    options=nodes,
                    default=default_node_name,
                    tooltip="Select a node from the available nodes"
                ),
                io.Combo.Input(
                    id="input_name_in",
                    display_name="input_name",
                    options=default_inputs,
                    default=input_name,
                    tooltip="Select a widget from the available options"
                ),
            ],
            outputs=[
                io.String.Output(id="input_name_out", display_name="input_name", tooltip="Name of the selected input"),
                io.String.Output(id="input_value", display_name="input_value", tooltip="Value of the selected input"),
            ],
        )

    @classmethod
    def execute(
        cls,
        node_name: str = "1_Unknown_Node",
        input_name_in: str = "unknown_input",
    ):
        class_name = cls.__name__
        input_name_out= "No Inputs"
        input_value = ""

        print(f"{class_name}: node='{node_name}'; input_name_in='{input_name_in}'")

        # All nodes for the workflow
        nodes_data = get_workflow_all_nodes(cls.__name__)

        if nodes_data is None or not isinstance(nodes_data, dict) or len(nodes_data) == 0:
            print(f"{class_name}: No nodes available.")
            return io.NodeOutput(
                input_name_in,
                ""
            )

        print(f"{class_name}: nodes_data keys={list(nodes_data.keys()) if nodes_data else 'None'}")

        # List of node names for the dropdown
        nodes = listify_nodes_data(nodes_data)
        print(f"{class_name}: available nodes={nodes}")

        # The default is the first node, if available
        node_id = list(nodes_data.keys())[0] if nodes_data and len(nodes_data) > 0 else None

        # If a node name is provided, extract the node id
        if node_name != "1_Unknown_Node":
            node_id = node_name.split("_", 1)[0] if "_" in node_name else None

        if node_id is None:
            print(f"{class_name}: Could not determine node_id from node_name='{node_name}'")
            return io.NodeOutput(
                input_name_in,
                ""
            )

        print(f"{class_name}: selected node_id={node_id}")

        if isinstance(nodes_data, dict):
            node_data = nodes_data.get(str(node_id), None)
            
            if node_data is None:
                print(f"{class_name}: No data found for node_id={node_id}")
                return io.NodeOutput(
                    input_name_in,
                    ""
                )

            node_inputs = node_input_details(cls.__name__, node_data)

            if node_inputs and isinstance(node_inputs, dict):
                print(f"{class_name}: node_inputs keys={list(node_inputs.keys())}")
                input_name_out = input_name_in if input_name_in and input_name_in in node_inputs.keys() else None
                
                # If the specified input name is not found, default to the first input
                if input_name_out == "No Inputs" or input_name_out is None:
                    input_name_out = list(node_inputs.keys())[0]

                input_value = node_inputs.get(input_name_out, "")

        print(f"{class_name}: selected input_name='{input_name_out}'; input_value='{input_value}'")

        return io.NodeOutput(
            input_name_out,
            input_value,
        )

class SceneSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_dir = os.path.join(output_dir, "poses")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            os.makedirs(os.path.join(default_dir, "default_pose"), exist_ok=True)

        subdir_dict = get_subdirectories(default_dir)
        default_options = sorted(subdir_dict.keys()) if subdir_dict else ["default_pose"]
        default_pose = default_options[0]
        pose_options = list(default_pose_options.keys())
        depth_options = list(default_depth_options.keys())

        # print(f"SceneSelect: default poses_dir='{default_dir}'; default_pose='{default_pose}'; default_options={default_options}")

        return io.Schema(
            node_id=prefixed_node_id("SceneSelect"),
            display_name="SceneSelect",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("poses_dir", default=default_dir, tooltip="Directory containing pose subdirectories"),
                io.Combo.Input('selected_pose', options=default_options, default=default_pose, tooltip="Select a pose name"),
                io.String.Input(id="girl_pos_in", display_name="girl_pos", multiline=True, default="", tooltip="The positive prompt for the girl"),
                io.Combo.Input(id="girl_action", display_name="action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the girl prompt"),
                io.String.Input(id="male_pos_in", display_name="male_pos", multiline=True, default="", tooltip="The positive prompt for the male"),
                io.Combo.Input(id="male_action", display_name="action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the male prompt"),
                io.String.Input(id="four_image_prompt_in", display_name="four_image_prompt", multiline=True, default="", tooltip="The four image prompt for the scene"),
                io.Combo.Input(id="four_image_prompt_action", display_name="four_image_prompt_action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the four image prompt"),
                io.String.Input(id="loras_high_in", display_name="loras_high", multiline=True, default="", tooltip="LoRAs list in JSON format to override pose defaults"),
                io.String.Input(id="loras_low_in", display_name="loras_low", multiline=True, default="", tooltip="Low-memory LoRAs list in JSON format to override pose defaults"),
                io.String.Input(id="wan_prompt_in", display_name="wan_prompt", placeholder="Provide the Wan high positive prompt for the scene", tooltip="WanVideoWrapper high prompt for the scene", multiline=True),
                io.String.Input(id="wan_low_prompt_in", display_name="wan_low_prompt", placeholder="Provide the Wan low positive prompt for the scene", tooltip="WanVideoWrapper low prompt for the scene", multiline=True),
                io.Combo.Input(id="wan_prompt_action", display_name="wan_prompt_action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the Wan prompt"),
                io.Combo.Input(id="depth_image_type", display_name="depth_image_type", options=depth_options, default="depth", tooltip="Type of depth image to generate from the pose"),
                io.Combo.Input(id="pose_image_type", display_name="pose_image_type", options=pose_options, default="open", tooltip="Type of pose image to generate from the pose"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Whether to mask the background in the scene"),
                io.Combo.Input(id="mask_type", display_name="mask_type", options=["girl", "male", "combined"], default="combined", tooltip="Subject mask to apply"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images"),
                io.String.Output(id="pose_name", display_name="pose_name", tooltip="Name of the selected pose"),
                io.String.Output(id="pose_dir", display_name="pose_dir", tooltip="Directory of the selected pose"),
                io.String.Output(id="input_img_glob", display_name="input_img_glob", tooltip="Input image glob pattern for the pose"),
                io.String.Output(id="output_image_prefix", display_name="output_image_prefix", tooltip="Output image prefix for the scene"),
                io.String.Output(id="output_video_prefix", display_name="output_video_prefix", tooltip="Output video prefix for the scene"),
                io.String.Output(id="girl_pos", display_name="girl_pos", tooltip="Girl's positive prompt"),
                io.String.Output(id="male_pos", display_name="male_pos", tooltip="Male's positive prompt"),
                io.String.Output(id="four_image_prompt", display_name="four_image_prompt", tooltip="Four image prompt for the scene"),
                io.String.Output(id="wan_prompt_out", display_name="wan_prompt", tooltip="WAN prompt, (high positive) from the pose"),
                io.String.Output(id="wan_low_prompt_out", display_name="wan_low_prompt", tooltip="WAN low prompt, (low positive) from the pose"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth IMAGE from the pose"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask IMAGE from the pose"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from the selected mask image"),
                io.Image.Output(id='canny_image', display_name='canny_image', tooltip='Canny IMAGE from the pose'),
                io.Image.Output(id='pose_image', display_name='pose_image', tooltip='Pose IMAGE from the pose'),
                io.Custom("WANVIDLORA").Output(id="loras_high_out", display_name="loras_high", tooltip="WanVideoWrapper Multi-Lora list" ),
                io.Custom("WANVIDLORA").Output(id="loras_low_out", display_name="loras_low", tooltip="WanVideoWrapper Multi-Lora list" ),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo 
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        poses_dir="",
        selected_pose="default_pose",
        girl_pos_in="",
        girl_action="use_file",
        male_pos_in="",
        male_action="use_file",
        four_image_prompt_in="",
        four_image_prompt_action="use_file",
        loras_high_in="",
        loras_low_in="",
        wan_prompt_in="",
        wan_low_prompt_in="",
        wan_prompt_action="use_file",
        depth_image_type="depth",
        pose_image_type="open",
        mask_background=True,
        mask_type="combined",
    ) -> io.NodeOutput:
        className = cls.__name__
        input_types = cls.INPUT_TYPES()
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        print(f"{className}: unique_id ='{unique_id}'; extra_pnginfo='{extra_pnginfo}'")

        if (type(input_types) is dict):
            inputs = input_types.get('required', {})
        elif (type(input_types) is tuple):
            inputs = input_types[0] if input_types else {}

        if not poses_dir:
            poses_dir = default_poses_dir()

        if not poses_dir or not selected_pose:
            print(f"{className}: poses_dir or selected_pose is empty")
            return io.NodeOutput(None)
        
        pose_dir = os.path.join(poses_dir, selected_pose)

        if not os.path.isdir(pose_dir):
            print(f"{className}: pose_dir '{pose_dir}' is not a valid directory")
            return io.NodeOutput(None)
        
        prompt_json_path = os.path.join(pose_dir, "prompts.json")
        prompt_data = load_prompt_json(prompt_json_path)
        pose_json_path = os.path.join(pose_dir, "pose.json")
        pose_json = load_json_file(pose_json_path)
        
        if not pose_json:
            pose_json = "[]"
        else:
            pose_json = json.dumps(pose_json)

        loras_path = os.path.join(pose_dir, "loras.json")
        
        loras = None
        if not os.path.isfile(loras_path):
            print(f"{className}: loras.json not found at '{loras_path}'")
        else:
            loras_high, loras_low = load_loras(os.path.join(pose_dir, "loras.json"))

        girl_file_text = prompt_data.get("girl_pos", "")
        girl_pos, girl_widget_text = select_text_by_action(
            girl_pos_in, 
            girl_file_text, 
            girl_action, 
            className
        )

        if girl_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, girl_widget_text,"girl_pos_in", inputs)

        girl_pos_ui_text = girl_widget_text if girl_widget_text is not None else girl_file_text

        male_file_text = prompt_data.get("male_pos", "")
        male_pos, male_widget_text = select_text_by_action(
            male_pos_in, 
            male_file_text, 
            male_action, 
            className
        )

        if male_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, male_widget_text,"male_pos_in", inputs)

        male_pos_ui_text = male_widget_text if male_widget_text is not None else male_file_text

        # four_image_prompt
        four_image_file_text = prompt_data.get("four_image_prompt", "")
        four_image_prompt, four_image_widget_text = select_text_by_action(
            four_image_prompt_in, 
            four_image_file_text, 
            four_image_prompt_action, 
            className
        )
        
        if four_image_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, four_image_widget_text,"four_image_prompt_in", inputs)
        four_image_prompt_ui_text = four_image_widget_text if four_image_widget_text is not None else four_image_file_text

        # wan_prompt - high positive prompt
        wan_file_text = prompt_data.get("wan_prompt", "")
        wan_prompt, wan_widget_text = select_text_by_action(
            wan_prompt_in, 
            wan_file_text, 
            wan_prompt_action, 
            className
        )

        if wan_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, wan_widget_text, "wan_prompt_in", inputs)
            
        wan_prompt_ui_text = wan_widget_text if wan_widget_text is not None else wan_file_text

        # wan_low_prompt - low positive prompt
        wan_low_file_text = prompt_data.get("wan_low_prompt", "")
        wan_low_prompt, wan_low_widget_text = select_text_by_action(
            wan_low_prompt_in, 
            wan_low_file_text,
            wan_prompt_action, 
            className
        )

        if wan_low_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, wan_low_widget_text, "wan_low_prompt_in", inputs)

        wan_low_prompt_ui_text = wan_low_widget_text if wan_low_widget_text is not None else wan_low_file_text

        loras_low_file_text = json.dumps(loras_low, indent=2) if loras_low else "[]"
        loras_low_text, loras_low_widget_text = select_text_by_action(
            loras_low_in, 
            loras_low_file_text,
            "use_file",
            className
        )
        if loras_low_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, loras_low_widget_text, "loras_low_in", inputs)

        loras_low_ui_text = loras_low_widget_text if loras_low_widget_text is not None else loras_low_file_text
        
        loras_high_file_text = json.dumps(loras_high, indent=2) if loras_high else "[]"
        loras_high_text, loras_high_widget_text = select_text_by_action(
            loras_high_in, 
            loras_high_file_text,
            "use_file",
            className
        )

        if loras_high_widget_text is not None:
            update_ui_widget(className, unique_id, extra_pnginfo, loras_high_widget_text, "loras_high_in", inputs)
        
        loras_high_ui_text = loras_high_widget_text if loras_high_widget_text is not None else loras_high_file_text

        # Load selected/normalized assets (and mask preview/output separation)
        selected_depth_attr = default_depth_options.get(depth_image_type, "depth_image")
        selected_pose_attr = default_pose_options.get(pose_image_type, "pose_open_image")
        mask_key = resolve_mask_key(mask_type, mask_background)
        print(f"{className}: Loading assets from pose_dir='{pose_dir}'; mask_key='{mask_key}'")
        assets = SceneInfo.load_preview_assets(
            pose_dir,
            depth_attr=selected_depth_attr,
            pose_attr=selected_pose_attr,
            mask_type=mask_type,
            mask_background=mask_background,
            include_upscale=True,
            include_canny=True,
        )

        # Also load full images for SceneInfo completeness (depth variants, masks, canny)
        depth_images_full = SceneInfo.load_depth_images(pose_dir)
        pose_images_full = SceneInfo.load_pose_images(pose_dir)
        mask_images_full, mask_tensors_full = SceneInfo.load_mask_images(pose_dir)
        # Ensure canny present even if missing on disk
        canny_image = pose_images_full.get("canny_image")

        base_image = assets["base_image"]
        selected_depth_image = assets["depth_image"]
        pose_image = assets["pose_image"]
        mask_image = assets["mask_image"]
        mask = assets["mask"]
        preview_mask = assets["mask_preview"]
        H, W = assets["H"], assets["W"]
        resolution = assets["resolution"]

        # Normalize canny to match preview size
        canny_image = normalize_image_tensor(canny_image, H, W)

        print(f"{className}: depth_image shape: {selected_depth_image.shape if selected_depth_image is not None else 'None'}")
        print(f"{className}: upscale_image shape: {base_image.shape if base_image is not None else 'None'}")

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None

        ui_data = {
            "images": preview_image.as_dict().get("images", []) if preview_image else None,
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
            "text": [
                str(girl_pos_ui_text), 
                str(male_pos_ui_text),
                str(loras_high_ui_text),
                str(loras_low_ui_text),
                str(wan_prompt_ui_text),
                str(wan_low_prompt_ui_text),
                str(four_image_prompt_ui_text),
            ],
        }

        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=selected_pose,
            girl_pos=girl_pos,
            male_pos=male_pos,
            four_image_prompt=four_image_prompt,
            wan_prompt=wan_prompt,
            wan_low_prompt=wan_low_prompt,
            pose_json=pose_json,
            resolution=resolution,
            depth_image=depth_images_full.get("depth_image"),
            depth_any_image=depth_images_full.get("depth_any_image"),
            depth_midas_image=depth_images_full.get("depth_midas_image"),
            depth_zoe_image=depth_images_full.get("depth_zoe_image"),
            depth_zoe_any_image=depth_images_full.get("depth_zoe_any_image"),
            pose_dense_image=pose_images_full.get("pose_dense_image"),
            pose_dw_image=pose_images_full.get("pose_dw_image"),
            pose_edit_image=pose_images_full.get("pose_edit_image"),
            pose_face_image=pose_images_full.get("pose_face_image"),
            pose_open_image=pose_images_full.get("pose_open_image"),
            canny_image=canny_image,
            upscale_image=pose_images_full.get("upscale_image"),
            girl_mask_bkgd_image=mask_images_full.get('girl'),
            male_mask_bkgd_image=mask_images_full.get('male'),
            combined_mask_bkgd_image=mask_images_full.get('combined'),
            girl_mask_no_bkgd_image=mask_images_full.get('girl_no_bg'),
            male_mask_no_bkgd_image=mask_images_full.get('male_no_bg'),
            combined_mask_no_bkgd_image=mask_images_full.get('combined_no_bg'),
            loras_high=loras_high,
            loras_low=loras_low,
        )

        return io.NodeOutput(
            scene_info,
            selected_pose,
            pose_dir,
            scene_info.input_img_glob(),
            scene_info.input_img_dir(),
            os.path.join(scene_info.output_dir(), "vid_"),
            girl_pos,
            male_pos,
            four_image_prompt,
            wan_prompt,
            wan_low_prompt,
            selected_depth_image,
            mask_image,
            mask,
            canny_image,
            pose_image,
            loras_high,
            loras_low,
            ui=ui_data
        )

default_depth_options = {
    "depth": "depth_image",
    "depth_any": "depth_any_image",
    "midas": "depth_midas_image",
    "zoe": "depth_zoe_image",
    "zoe_any": "depth_zoe_any_image",
}

default_pose_options = {
    "dense": "pose_dense_image",
    "dw": "pose_dw_image",
    "edit": "pose_edit_image",
    "face": "pose_face_image",
    "open": "pose_open_image",
}

default_mask_options = {
    "girl": "girl_mask_bkgd",
    "male": "male_mask_bkgd",
    "combined": "combined_mask_bkgd",
    "girl_no_bg": "girl_mask_no_bkgd",
    "male_no_bg": "male_mask_no_bkgd",
    "combined_no_bg": "combined_mask_no_bkgd",
}

def resolve_mask_key(mask_type: str, mask_background: bool) -> str:
    """Return the mask key to use given mask type and background preference."""
    key = mask_type or "combined"
    if not mask_background and not key.endswith("_no_bg"):
        key = f"{key}_no_bg"
    return key

def build_positive_prompt(prompt_type: str, prompt_data: dict, custom_prompt: str = "") -> str:
    """Select the prompt text for a scene based on prompt_type and available prompt data."""
    prompt_data = prompt_data or {}
    girl = prompt_data.get("girl_pos", "") or ""
    male = prompt_data.get("male_pos", "") or ""
    four = prompt_data.get("four_image_prompt", "") or ""
    wan_hi = prompt_data.get("wan_prompt", "") or ""
    wan_low = prompt_data.get("wan_low_prompt", "") or ""

    if prompt_type == "custom":
        return custom_prompt or ""
    if prompt_type == "combined":
        return " ".join([p for p in [girl, male] if p]).strip()
    if prompt_type == "girl_pos":
        return girl
    if prompt_type == "male_pos":
        return male
    if prompt_type == "four_image_prompt":
        return four
    if prompt_type == "wan_prompt":
        return wan_hi
    if prompt_type == "wan_low_prompt":
        return wan_low
    return girl or male or four or wan_hi or wan_low

class SceneWanVideoLoraMultiSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneWanVideoLoraMultiSave"),
            display_name="SceneWanVideoLoraMultiSave",
            category="🧊 frost-byte/Scene",
            description=cleandoc("""
                                 Saves the name, weights, layer filter and blocks of multiple LoRA models using the
                                 output from WanVideoWrapper's WanVideoLoraSelectMulti node to the directory for the given scene.
            """),
            inputs=[
                io.Custom("SCENE_INFO").Input(id="info_in", display_name="scene_info", tooltip="SceneInfo, which provides the path to save the LoRA information to"),
                io.Custom("WANVIDLORA").Input(id="loras_high", display_name="lora", tooltip="WanVideoSelectMulti output with multiple High LoRA entries"),
                io.Custom("WANVIDLORA").Input(id="loras_low", display_name="lora", tooltip="WanVideoSelectMulti output with multiple Low LoRA entries"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="info_out", display_name="scene_info", tooltip="Save operation information"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        info_in,
        loras_high=None,
        loras_low=None,
    ) -> io.NodeOutput:
        className = cls.__name__

        if info_in is None or loras_high is None or loras_low is None:
            return io.NodeOutput(None)

        pose_dir = info_in.pose_dir
        if not pose_dir or not os.path.isdir(pose_dir):
            print(f"{className}: Invalid pose_dir '{pose_dir}' in SceneInfo")
            return io.NodeOutput(None)

        if not loras_high is None:
            print(f"{className}: Saving {len(loras_high)} High LoRA entries to pose_dir '{pose_dir}'")
            loras_high_path = os.path.join(pose_dir, "loras_high.json")
        else:
            loras_high = []
        if not loras_low is None:
            print(f"{className}: Saving {len(loras_low)} Low LoRA entries to pose_dir '{pose_dir}'")
            loras_low_path = os.path.join(pose_dir, "loras_low.json")
        else:
            loras_low = []

        loras_path = os.path.join(pose_dir, "loras.json")
        save_loras(loras_high, loras_low, loras_path)
        print(f"Saved LoRA preset to: {loras_path}")

        return io.NodeOutput(info_in)

class SceneCreate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneCreate"),
            display_name="SceneCreate",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="poses_dir", display_name="poses_dir", tooltip="Root Directory where all scene subdirectories are saved"),
                io.String.Input(id="pose_name", display_name="pose_name", tooltip="Name of the pose"),
                io.Int.Input(id="resolution", display_name="resolution", tooltip="Resolution for the pose, depth and other images", default=512),
                io.Combo.Input(
                    id="upscale_method",
                    display_name="upscale_method",
                    options=["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],
                    default="nearest-exact",
                    tooltip="Method to use for upscaling the base image"
                ),
                io.Float.Input(
                    id="upscale_factor", 
                    display_name="upscale_factor", 
                    tooltip="Factor to upscale the base image by", 
                    default=1.0, min=0.1, max=10.0, step=0.1
                ),
                io.Combo.Input(
                    id="densepose_model", 
                    display_name="densepose_model",
                    options=["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], 
                    default="densepose_r50_fpn_dl.torchscript", 
                    tooltip="DensePose model to use"
                ),
                io.Combo.Input(
                    id="densepose_cmap",
                    display_name="densepose_cmap",
                    options=["viridis", "parula"],
                    default="viridis",
                    tooltip="Color map to use for DensePose visualization"
                ),
                io.Combo.Input(
                    id="depth_any_ckpt",
                    display_name="depth_any_ckpt",
                    options=["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"],
                    default="depth_anything_vitl14.pth",
                    tooltip="Checkpoint for Depth Any model"
                ),
                io.Combo.Input(
                    id="depth_any_v2_ckpt",
                    display_name="depth_any_v2_ckpt",
                    options=["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                    default="depth_anything_v2_vitl.pth",
                    tooltip="Checkpoint for Depth Any v2 model"
                ),
                io.Float.Input(
                    id="midas_a",
                    display_name="midas_a",
                    tooltip="MiDas parameter A for depth scaling",
                    default=np.pi * 2.0, min=0.0, max=np.pi * 5.0, step=0.1
                ),
                io.Float.Input(
                    id="midas_bg_thresh",
                    display_name="midas_bg_thresh",
                    tooltip="MiDas parameter Bg threshold for depth scaling",
                    default=0.1, min=0.1, max=np.pi * 5.0, step=0.1
                ),
                io.Combo.Input(
                    id="zoe_environment",
                    display_name="zoe_environment",
                    options=["indoor", "outdoor"],
                    default="indoor",
                    tooltip="Environment setting for Zoe Any model"
                ),
                io.Int.Input(
                    id="canny_low_threshold",
                    display_name="canny_low_threshold",
                    tooltip="Canny edge detector low threshold",
                    default=100, min=0, max=255, step=1
                ),
                io.Int.Input(
                    id="canny_high_threshold",
                    display_name="canny_high_threshold",
                    tooltip="Canny edge detector high threshold",
                    default=200, min=0, max=255, step=1
                ),
                io.Image.Input(id="base_image", display_name="base_image", tooltip="Base image for the scene"),
                io.Custom("WANVIDLORA").Input(id="loras_high", display_name="loras_high", tooltip="WanVideoWrapper High Multi-Lora list", optional=True),
                io.Custom("WANVIDLORA").Input(id="loras_low", display_name="loras_low", tooltip="WanVideoWrapper Low Multi-Lora list", optional=True),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene Information"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        poses_dir="",
        pose_name="default_pose",
        resolution=512,
        upscale_method="nearest-exact",
        upscale_factor=1.0,
        densepose_model="densepose_r50_fpn_dl.torchscript",
        densepose_cmap="viridis",
        depth_any_ckpt="depth_anything_vitl14.pth",
        depth_any_v2_ckpt="depth_anything_v2_vitl.pth",
        midas_a=np.pi * 2.0,
        midas_bg_thresh=0.1,
        zoe_environment="indoor",
        canny_low_threshold=100,
        canny_high_threshold=200,
        base_image=None,
        loras_high=None,
        loras_low=None,
    ) -> io.NodeOutput:
        if base_image is None:
            print("SceneCreate: base_image is None")
            return io.NodeOutput(None)
        
        if not poses_dir:
            poses_dir = default_poses_dir()
        
        if not pose_name:
            pose_name = "default_pose"

        pose_dir = os.path.join(poses_dir, pose_name)

        upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
        print(f"SceneCreate: upscale_image is of type: {type(upscale_image)} with shape {upscale_image.shape if torch.is_tensor(upscale_image) else 'N/A'}")

        # DensePose
        dense_pose_image = dense_pose(upscale_image, densepose_model, densepose_cmap, resolution)

        # Depth Anything
        depth_any_image = depth_anything(upscale_image, ckpt=depth_any_ckpt, resolution=resolution)
        
        # Depth Anything V2
        depth_image = depth_anything_v2(upscale_image, ckpt=depth_any_v2_ckpt, resolution=resolution)

        # MiDas
        midas_depth_image = midas(upscale_image, a=midas_a, bg_thresh=midas_bg_thresh)

        # Zoe
        depth_zoe_image = zoe(upscale_image, resolution=resolution)
        
        # Zoe Any
        depth_zoe_any_image = zoe_any(upscale_image, environment=zoe_environment, resolution=resolution)

        if type(depth_any_image) is not torch.Tensor:
            H = 512
            W = 512
        elif not depth_any_image is None and type(depth_any_image) is torch.Tensor:
            H, W = depth_any_image.shape[1], depth_any_image.shape[2]
        pose_dw_image, pose_json = estimate_dwpose(upscale_image, resolution=resolution)
        normalized_upscale_image = image_resize_ess(upscale_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)

        pose_open_image = openpose(normalized_upscale_image, resolution=resolution)
        canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"
        pose_dwpose_json = json.dumps(pose_json)

        # Create empty PromptCollection for new scenes
        # Users will add prompts via ScenePromptManager
        prompt_collection = PromptCollection()

        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=pose_name,
            resolution=resolution,
            prompts=prompt_collection,
            upscale_image=upscale_image,
            depth_image=depth_image,
            depth_any_image=depth_any_image,
            depth_midas_image=midas_depth_image,
            depth_zoe_image=depth_zoe_image,
            depth_zoe_any_image=depth_zoe_any_image,
            pose_dense_image=dense_pose_image,
            pose_dw_image=pose_dw_image,
            pose_edit_image=pose_dw_image,
            pose_dwpose_json=pose_dwpose_json,
            pose_open_image=pose_open_image,
            pose_face_image=pose_dw_image,
            pose_json=pose_json,
            canny_image=canny_image,
            loras_high=loras_high,
            loras_low=loras_low,
        )
        
        # Save all scene data using the helper method
        scene_info.save_all(pose_dir)
        print(f"SceneCreate: Saved all scene data to '{pose_dir}'")
        
        return io.NodeOutput(
            scene_info,
        )

class SceneUpdate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneUpdate"),
            display_name="SceneUpdate",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info_in", display_name="scene_info", tooltip="Scene Information" ),
                io.String.Input(id="girl_pos", multiline=True, display_name="girl_pos", tooltip="The positive prompt for the girl in the scene"),
                io.String.Input(id="male_pos", multiline=True, display_name="male_pos", tooltip="The positive prompt for the male(s) in the scene"),
                io.Boolean.Input(id="update_prompts", display_name="update_prompts", tooltip="If true, will update the prompts in the scene_info", default=True),
                io.Boolean.Input(id="update_zoe", display_name="update_zoe", tooltip="If true, will update the Zoe depth images in the scene_info", default=False),
                io.Boolean.Input(id="update_depth", display_name="update_depth", tooltip="If true, will update the Depth Anything images in the scene_info", default=False),
                io.Boolean.Input(id="update_densepose", display_name="update_densepose", tooltip="If true, will update the DensePose image in the scene_info", default=False),
                io.Boolean.Input(id="update_openpose", display_name="update_openpose", tooltip="If true, will update the OpenPose image in the scene_info", default=False),
                io.Boolean.Input(id="update_midas", display_name="update_midas", tooltip="If true, will update the MiDas depth image in the scene_info", default=False),
                io.Boolean.Input(id="update_canny", display_name="update_canny", tooltip="If true, will update the Canny edge image in the scene_info", default=False),
                io.Boolean.Input(id="update_upscale", display_name="update_upscale", tooltip="If true, will update the Upscale image in the scene_info", default=False),
                io.Boolean.Input(id="update_pose_json", display_name="update_pose_json", tooltip="If true, will update the pose_json in the scene_info", default=False),
                io.Boolean.Input(id="update_facepose", display_name="update_facepose", tooltip="If true, will update the Face Pose image in the scene_info", default=False),
                io.Boolean.Input(id="update_editpose", display_name="update_editpose", tooltip="If true, will update the Edit Pose image in the scene_info", default=False),
                io.Boolean.Input(id="update_dwpose", display_name="update_dwpose", tooltip="If true, will update the DensePose image in the scene_info", default=False),
                io.Boolean.Input(id="update_high_loras", display_name="update_high_loras", tooltip="If true, will update the High LoRAs list in the scene_info", default=False),
                io.Boolean.Input(id="update_low_loras", display_name="update_low_loras", tooltip="If true, will update the Low LoRAs list in the scene_info", default=False),
                io.String.Input(id="pose_json", display_name="pose_json", tooltip="JSON string for the pose keypoints"),
                io.Int.Input(id="resolution", display_name="resolution", tooltip="Resolution for the pose, depth and other images", default=512),
                io.Combo.Input(
                    id="upscale_method",
                    display_name="upscale_method",
                    options=["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],
                    default="nearest-exact",
                    tooltip="Method to use for upscaling the base image"
                ),
                io.Float.Input(
                    id="upscale_factor", 
                    display_name="upscale_factor", 
                    tooltip="Factor to upscale the base image by", 
                    default=1.0, min=0.1, max=10.0, step=0.1
                ),
                io.Combo.Input(
                    id="densepose_model", 
                    display_name="densepose_model",
                    options=["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], 
                    default="densepose_r50_fpn_dl.torchscript", 
                    tooltip="DensePose model to use"
                ),
                io.Combo.Input(
                    id="densepose_cmap",
                    display_name="densepose_cmap",
                    options=["viridis", "parula"],
                    default="viridis",
                    tooltip="Color map to use for DensePose visualization"
                ),
                io.Combo.Input(
                    id="depth_any_ckpt",
                    display_name="depth_any_ckpt",
                    options=["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"],
                    default="depth_anything_vitl14.pth",
                    tooltip="Checkpoint for Depth Any model"
                ),
                io.Combo.Input(
                    id="depth_any_v2_ckpt",
                    display_name="depth_any_v2_ckpt",
                    options=["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                    default="depth_anything_v2_vitl.pth",
                    tooltip="Checkpoint for Depth Any v2 model"
                ),
                io.Float.Input(
                    id="midas_a",
                    display_name="midas_a",
                    tooltip="MiDas parameter A for depth scaling",
                    default=np.pi * 2.0, min=0.0, max=np.pi * 5.0, step=0.1
                ),
                io.Float.Input(
                    id="midas_bg_thresh",
                    display_name="midas_bg_thresh",
                    tooltip="MiDas parameter Bg threshold for depth scaling",
                    default=0.1, min=0.1, max=np.pi * 5.0, step=0.1
                ),
                io.Combo.Input(
                    id="zoe_environment",
                    display_name="zoe_environment",
                    options=["indoor", "outdoor"],
                    default="indoor",
                    tooltip="Environment setting for Zoe Any model"
                ),
                io.Int.Input(
                    id="canny_low_threshold",
                    display_name="canny_low_threshold",
                    tooltip="Canny edge detector low threshold",
                    default=100, min=0, max=255, step=1
                ),
                io.Int.Input(
                    id="canny_high_threshold",
                    display_name="canny_high_threshold",
                    tooltip="Canny edge detector high threshold",
                    default=200, min=0, max=255, step=1
                ),
                io.Custom("WANVIDLORA").Input(id="high_loras", display_name="high_loras", tooltip="WanVideoWrapper Multi-Lora list", optional=True ),
                io.Custom("WANVIDLORA").Input(id="low_loras", display_name="low_loras", tooltip="WanVideoWrapper Multi-Lora list", optional=True ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info_out", display_name="scene_info", tooltip="Updated Scene Information"),
            ],
        )
        
    @classmethod
    async def execute(
        cls,
        scene_info_in=None,
        girl_pos=None,
        male_pos=None,
        update_prompts=True,
        update_zoe=False,
        update_depth=False,
        update_densepose=False,
        update_openpose=False,
        update_midas=False,
        update_canny=False,
        update_upscale=False,
        update_pose_json=False,
        update_facepose=False,
        update_editpose=False,
        update_dwpose=False,
        update_high_loras=False,
        update_low_loras=False,
        pose_json="[]",
        resolution=512,
        upscale_method="nearest-exact",
        upscale_factor=1.0,
        densepose_model="densepose_r50_fpn_dl.torchscript",
        densepose_cmap="viridis",
        depth_any_ckpt="depth_anything_vitl14.pth",
        depth_any_v2_ckpt="depth_anything_v2_vitl.pth",
        midas_a=np.pi * 2.0,
        midas_bg_thresh=0.1,
        zoe_environment="indoor",
        canny_low_threshold=100,
        canny_high_threshold=200,
        high_loras=None,
        low_loras=None,
    ):
        if scene_info_in is None:
            print("SceneUpdate: scene_info is None")
            return io.NodeOutput(None)

        scene_info_out = scene_info_in
        if update_prompts:
            if girl_pos is not None:
                scene_info_out.girl_pos = girl_pos
            if male_pos is not None:
                scene_info_out.male_pos = male_pos

        print(f"SceneUpdate: Wan Low Prompt -> {scene_info_in.wan_low_prompt[:32]}...")                

        upscale_image = scene_info_in.upscale_image

        if upscale_image is None:
            print("SceneUpdate: base upscale_image is None in scene_info")
            return io.NodeOutput(scene_info_out)

        if update_upscale:
            upscale_image, = ImageScaleBy().upscale(upscale_image, upscale_method=upscale_method, scale_by=upscale_factor)
            scene_info_out.upscale_image = upscale_image

        if upscale_image is None:
            print("SceneUpdate: upscale_image is None after upscaling")
            return io.NodeOutput(scene_info_out)

        #if update_facepose:
            #pose_face_image, pose_json = face(upscale_image, resolution=resolution)
            #scene_info_out.pose_face_image = pose_face_image
            #scene_info_out.pose_json = pose_json
        if update_densepose:
            scene_info_out.pose_dense_image = dense_pose(upscale_image, densepose_model, densepose_cmap, resolution)

        if update_depth:
            # Depth Anything
            scene_info_out.depth_any_image = depth_anything(upscale_image, ckpt=depth_any_ckpt, resolution=resolution)
            scene_info_out.depth_image = depth_anything_v2(upscale_image, ckpt=depth_any_v2_ckpt, resolution=resolution)

        # MiDas
        if update_midas:
            scene_info_out.depth_midas_image = midas(upscale_image, a=midas_a, bg_thresh=midas_bg_thresh)

        # Zoe
        if update_zoe:
            depth_zoe_image = zoe(upscale_image, resolution=resolution)
            scene_info_out.depth_zoe_image = depth_zoe_image
            depth_zoe_any_image = zoe_any(upscale_image, environment=zoe_environment, resolution=resolution)
            scene_info_out.depth_zoe_any_image = depth_zoe_any_image

        # Pose Json
        if update_pose_json:
            scene_info_out.pose_json = pose_json
        
        if update_canny:
            canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)
            scene_info_out.canny_image = canny_image

        if update_dwpose:
            pose_dw_image, pose_json = estimate_dwpose(upscale_image, resolution=resolution)
            scene_info_out.pose_dw_image = pose_dw_image
            #scene_info_out.pose_json = pose_json

        depth_any_image = scene_info_out.depth_any_image
        
        if type(depth_any_image) is not torch.Tensor or depth_any_image is None:
            H = 512
            W = 512
        elif type(depth_any_image) is torch.Tensor and not depth_any_image is None:
            H, W = depth_any_image.shape[1], depth_any_image.shape[2]

        normalized_upscale_image = image_resize_ess(upscale_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)

        if update_openpose or update_editpose:
            pose_open_image = openpose(normalized_upscale_image, resolution=resolution)
            scene_info_out.pose_open_image = pose_open_image

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"

        # Update LoRAs
        if update_high_loras and high_loras is not None:
            scene_info_out.loras_high = high_loras
        
        if update_low_loras and low_loras is not None:
            scene_info_out.loras_low = low_loras
        
        # Save LoRAs if updated
        if update_high_loras or update_low_loras:
            scene_info_out.save_loras()
            print(f"SceneUpdate: Saved LoRA presets to: {scene_info_in.pose_dir}/loras.json")

        return io.NodeOutput(
            scene_info_out,
        )

class SceneView(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneView"),
            display_name="SceneView",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information" ),
                io.Combo.Input(
                    id="depth_type", options=list(default_depth_options.keys())
                ),
                io.Combo.Input(
                    id="pose_type", options=list(default_pose_options.keys())
                ),
                io.Combo.Input(id="mask_type", display_name="mask_type", options=["girl", "male", "combined"], default="combined", tooltip="Mask to preview"),
                io.Boolean.Input(id="include_mask_bg", display_name="include_mask_bg", default=True, tooltip="Include background in mask selection"),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected Depth Image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected Pose Image"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Selected Mask Image"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from selected mask image"),
                io.String.Output(id="pose_name", display_name="pose_name", tooltip="Name of the selected pose"),
                io.String.Output(id="pose_dir", display_name="pose_dir", tooltip="Directory of the selected pose"),
                io.String.Output(id="girl_pos", display_name="girl_pos", tooltip="The positive prompt for the girl in the scene"),
                io.String.Output(id="male_pos", display_name="male_pos", tooltip="The positive prompt for the male(s) in the scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    async def execute(
        cls,
        scene_info=Optional[SceneInfo],
        depth_type="depth",
        pose_type="dense",
        mask_type="combined",
        include_mask_bg=True,
    ) -> io.NodeOutput:
        if scene_info is None:
            print("SceneView: scene_info is None")
            return io.NodeOutput(None, None, None, None, None, None, None, None)
        
        if not isinstance(scene_info, SceneInfo):
            print("SceneView: scene_info is not of type SceneInfo")
            return io.NodeOutput(None, None, None, None, None, None, None, None)

        assets = scene_info.load_preview_assets(
            scene_info.pose_dir,
            depth_attr=depth_type,
            pose_attr=pose_type,
            mask_type=mask_type,
            mask_background=include_mask_bg,
            include_canny=True,
        )

        mask_image = assets["mask_image"]
        mask = assets["mask"]
        depth_image = assets["depth_image"]
        pose_image = assets["pose_image"]
        girl_pos = getattr(scene_info, "girl_pos", "")
        male_pos = getattr(scene_info, "male_pos", "")
        pose_name = getattr(scene_info, "pose_name", "")
        pose_dir = getattr(scene_info, "pose_dir", "")

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        combined_prompt = f"Girl Positive Prompt: {girl_pos}\nMale Positive Prompt: {male_pos}"
        text_ui = ui.PreviewText(value=combined_prompt)
 
        ui_data = {
            "text": text_ui.as_dict().get("text", ''),
            "images": preview_image.as_dict().get("images", []) if preview_image else [],
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
        }

        return io.NodeOutput(
            depth_image,
            pose_image,
            mask_image,
            mask,
            pose_name,
            pose_dir,
            girl_pos,
            male_pos,
            ui=ui_data
        )
 
class SceneOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneOutput"),
            display_name="SceneOutput",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information"),
            ],
            outputs=[
                io.String.Output(id="pose_dir", display_name="pose_dir", tooltip="Directory where the scene is saved"),
                io.String.Output(id="pose_name", display_name="pose_name", tooltip="Name of the pose"),
                io.String.Output(id="girl_pos", display_name="girl_pos", tooltip="Girl Positive Prompt"),
                io.String.Output(id="male_pos", display_name="male_pos", tooltip="Male Positive Prompt"),
                io.String.Output(id="four_image_prompt", display_name="four_image_prompt", tooltip="Four Image Prompt"),
                io.String.Output(id="wan_prompt", display_name="wan_prompt", tooltip="Wan High Positive Prompt"),
                io.String.Output(id="wan_low_prompt", display_name="wan_low_prompt", tooltip="Wan Low Positive Prompt"),
                io.String.Output(id="pose_json", display_name="pose_json", tooltip="Pose JSON data"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth Image"),
                io.Image.Output(id="depth_any_image", display_name="depth_any_image", tooltip="Depth Any Image"),
                io.Image.Output(id="depth_midas_image", display_name="depth_midas_image", tooltip="Depth Midas Image"),
                io.Image.Output(id="depth_zoe_image", display_name="depth_zoe_image", tooltip="Depth Zoe Image"),
                io.Image.Output(id="depth_zoe_any_image", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image"),
                io.Image.Output(id="pose_dense_image", display_name="pose_dense_image", tooltip="Pose Dense Image"),
                io.Image.Output(id="pose_dw_image", display_name="pose_dw_image", tooltip="Pose DW Image"),
                io.Image.Output(id="pose_edit_image", display_name="pose_edit_image", tooltip="Pose Edit Image"),
                io.Image.Output(id="pose_face_image", display_name="pose_face_image", tooltip="Pose Face Image"),
                io.Image.Output(id="pose_open_image", display_name="pose_open_image", tooltip="Pose Open Image"),
                io.Image.Output(id="canny_image", display_name="canny_image", tooltip="Canny Image"),
                io.Image.Output(id="upscale_image", display_name="upscale_image", tooltip="Upscale Image"),
                io.Image.Output(id="girl_mask_image", display_name="girl_mask_image", tooltip="Girl Mask Image, with background"),
                io.Image.Output(id="male_mask_image", display_name="male_mask_image", tooltip="Male Mask Image, with background"),
                io.Image.Output(id="combined_mask_image", display_name="combined_mask_image", tooltip="Combined Mask Image, with background"),
                io.Image.Output(id="girl_mask_nobg_image", display_name="girl_mask_nobg_image", tooltip="Girl Mask Image, no background"),
                io.Image.Output(id="male_mask_nobg_image", display_name="male_mask_nobg_image", tooltip="Male Mask Image, no background"),
                io.Image.Output(id="combined_mask_nobg_image", display_name="combined_mask_nobg_image", tooltip="Combined Mask Image, no background"),
                io.Custom("WANVIDLORA").Output(id="high_loras", display_name="high_loras", tooltip="WanVideoWrapper High Multi-Lora list"),
                io.Custom("WANVIDLORA").Output(id="low_loras", display_name="low_loras", tooltip="WanVideoWrapper Low Multi-Lora list"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info=None,
    ) -> io.NodeOutput:
        if scene_info is None:
            print("SceneOutput: scene_info is None")
            return io.NodeOutput((
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ))
        
        print(
            f"SceneOutput: scene_info.pose_dir='{scene_info.pose_dir}', "
            f"pose_name='{scene_info.pose_name}', "
            f"girl_pos='{scene_info.girl_pos[:32]}', "
            f"male_pos='{scene_info.male_pos[:32]}', "
            f"wan_prompt='{scene_info.wan_prompt[:32]}', "
            f"wan_low_prompt='{scene_info.wan_low_prompt[:32]}', "
            f"depth_image shape: {scene_info.depth_image.shape if scene_info.depth_image is not None else 'None'}"
        )
        return io.NodeOutput(
            scene_info.pose_dir,
            scene_info.pose_name,
            scene_info.girl_pos,
            scene_info.male_pos,
            scene_info.four_image_prompt,
            scene_info.wan_prompt,
            scene_info.wan_low_prompt,
            scene_info.pose_json,
            scene_info.depth_image,
            scene_info.depth_any_image,
            scene_info.depth_midas_image,
            scene_info.depth_zoe_image,
            scene_info.depth_zoe_any_image,
            scene_info.pose_dense_image,
            scene_info.pose_dw_image,
            scene_info.pose_edit_image,
            scene_info.pose_face_image,
            scene_info.pose_open_image,
            scene_info.canny_image,
            scene_info.upscale_image,
            scene_info.girl_mask_bkgd_image,
            scene_info.male_mask_bkgd_image,
            scene_info.combined_mask_bkgd_image,
            scene_info.girl_mask_no_bkgd_image,
            scene_info.male_mask_no_bkgd_image,
            scene_info.combined_mask_no_bkgd_image,
            scene_info.high_loras,
            scene_info.low_loras,
        )

class SceneSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneSave"),
            display_name="SceneSave",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Info Input"),
                io.String.Input(id="pose_dir", display_name="pose_dir", optional=True, tooltip="The Pose directory for the scene, overrides the scene_info", multiline=False, default=""),
            ],
            outputs=[],
            is_output_node=True,
        )        

    @classmethod
    def execute(
        cls,
        scene_info=None,
        pose_dir="",
    ) -> io.NodeOutput:
        if scene_info is None or not scene_info.pose_name:
            print("SaveScene: scene_info is None or pose_name is empty")
            return io.NodeOutput(None)

        # Use provided pose_dir or fall back to scene_info's pose_dir
        target_dir = pose_dir if pose_dir else scene_info.pose_dir
        if not target_dir:
            target_dir = str(Path(default_poses_dir()) / scene_info.pose_name)

        print(f"SaveScene: pose_name='{scene_info.pose_name}'; dest_dir='{target_dir}'")
        
        # Use the unified save_all method
        scene_info.save_all(target_dir)

        return io.NodeOutput(
            ui=ui.PreviewText(f"Scene saved to '{target_dir}' with prompt='The girl {scene_info.girl_pos}, The male {scene_info.male_pos}'"),
        )

class SceneInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneInput"),
            display_name="SceneInput",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="pose_dir", display_name="pose_dir", tooltip="Directory where the scene is saved", multiline=False, default=""),
                io.String.Input(id="pose_name", display_name="pose_name", tooltip="Name of the pose", multiline=False, default=""),
                io.String.Input(id="girl_pos", display_name="girl_pos", tooltip="The prompt for the girl in the scene", multiline=True, default=""),
                io.String.Input(id="male_pos", display_name="male_pos", tooltip="The prompt for the male(s) in the scene", multiline=True, default=""),
                io.String.Input(id="four_image_prompt", display_name="four_image_prompt", tooltip="The Four Image prompt for the scene", multiline=True, default=""),
                io.String.Input(id="wan_prompt", display_name="wan_prompt", tooltip="The Wan High positive prompt for the scene", multiline=True, default=""),
                io.String.Input(id="wan_low_prompt", display_name="wan_low_prompt", tooltip="The Wan Low positive prompt for the scene", multiline=True, default=""),
                io.String.Input(id="pose_json", display_name="pose_json", tooltip="Pose JSON data", multiline=True, default=""),
                io.Image.Input(id="depth_image", display_name="depth_image", tooltip="Depth Image", optional=True),
                io.Image.Input(id="depth_any_image", display_name="depth_any_image", tooltip="Depth Any Image", optional=True),
                io.Image.Input(id="depth_midas_image", display_name="depth_midas_image", tooltip="Depth Midas Image", optional=True),
                io.Image.Input(id="depth_zoe_image", display_name="depth_zoe_image", tooltip="Depth Zoe Image", optional=True),
                io.Image.Input(id="depth_zoe_any_image", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image", optional=True),
                io.Image.Input(id="pose_dense_image", display_name="pose_dense_image", tooltip="Pose Dense Image", optional=True),
                io.Image.Input(id="pose_dw_image", display_name="pose_dw_image", tooltip="Pose DW Image", optional=True),
                io.Image.Input(id="pose_edit_image", display_name="pose_edit_image", tooltip="Pose Edit Image", optional=True),
                io.Image.Input(id="pose_face_image", display_name="pose_face_image", tooltip="Pose Face Image", optional=True),
                io.Image.Input(id="pose_open_image", display_name="pose_open_image", tooltip="Pose Open Image", optional=True),
                io.Image.Input(id="canny_image", display_name="canny_image", tooltip="Canny Image", optional=True),
                io.Image.Input(id="upscale_image", display_name="upscale_image", tooltip="Upscale Image", optional=True),
                io.Image.Input(id="girl_mask_image", display_name="girl_mask_image", tooltip="Girl Mask Image, with background", optional=True),
                io.Image.Input(id="male_mask_image", display_name="male_mask_image", tooltip="Male Mask Image, with background", optional=True),
                io.Image.Input(id="combined_mask_image", display_name="combined_mask_image", tooltip="Combined Mask Image, with background", optional=True),
                io.Image.Input(id="girl_mask_nobg_image", display_name="girl_mask_nobg_image", tooltip="Girl Mask Image, no background", optional=True),
                io.Image.Input(id="male_mask_nobg_image", display_name="male_mask_nobg_image", tooltip="Male Mask Image, no background", optional=True),
                io.Image.Input(id="combined_mask_nobg_image", display_name="combined_mask_nobg_image", tooltip="Combined Mask Image, no background", optional=True),
                io.Custom("WANVIDLORA").Input(id="high_loras", display_name="high_loras", tooltip="WanVideoWrapper High Multi-Lora list", optional=True ),
                io.Custom("WANVIDLORA").Input(id="low_loras", display_name="low_loras", tooltip="WanVideoWrapper Low Multi-Lora list", optional=True),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pose_dir="",
        pose_name="",
        girl_pos="",
        male_pos="",
        four_image_prompt="",
        wan_prompt="",
        wan_low_prompt="",
        pose_json="",
        depth_image=None,
        depth_any_image=None,
        depth_midas_image=None,
        depth_zoe_image=None,
        depth_zoe_any_image=None,
        pose_dense_image=None,
        pose_dw_image=None,
        pose_edit_image=None,
        pose_face_image=None,
        pose_open_image=None,
        canny_image=None,
        upscale_image=None,
        girl_mask_image=None,
        male_mask_image=None,
        combined_mask_image=None,
        girl_mask_no_bkgd_image=None,
        male_mask_no_bkgd_image=None,
        combined_mask_no_bkgd_image=None,
        high_loras=None,
        low_loras=None,
    ) -> io.NodeOutput:
        if not pose_dir or not os.path.isdir(pose_dir):
            print(f"SceneInput: pose_dir '{pose_dir}' is invalid")
            return io.NodeOutput(None)

        print(f"SceneInput: pose_dir='{pose_dir}'; pose_name='{pose_name}'")
        resolution = min(depth_image.shape[1], depth_image.shape[2]) if depth_image is not None else 512

        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=pose_name,
            girl_pos=girl_pos,
            male_pos=male_pos,
            four_image_prompt=four_image_prompt,
            wan_prompt=wan_prompt,
            wan_low_prompt=wan_low_prompt,
            pose_json=pose_json,
            depth_image=depth_image,
            depth_any_image=depth_any_image,
            depth_midas_image=depth_midas_image,
            depth_zoe_image=depth_zoe_image,
            depth_zoe_any_image=depth_zoe_any_image,
            pose_dense_image=pose_dense_image,
            pose_dw_image=pose_dw_image,
            pose_edit_image=pose_edit_image,
            pose_face_image=pose_face_image,
            pose_open_image=pose_open_image,
            girl_mask_bkgd_image=girl_mask_image,
            male_mask_bkgd_image=male_mask_image,
            combined_mask_bkgd_image=combined_mask_image,
            girl_mask_no_bkgd_image=girl_mask_no_bkgd_image,
            male_mask_no_bkgd_image=male_mask_no_bkgd_image,
            combined_mask_no_bkgd_image=combined_mask_no_bkgd_image,
            canny_image=canny_image,
            upscale_image=upscale_image,
            loras_high=high_loras,
            loras_low=low_loras,
            resolution=resolution,
        )

        return io.NodeOutput(
            scene_info
        )

class StoryCreate(io.ComfyNode):
    """Create a new story with an initial scene"""
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_stories_dir_path = default_stories_dir()
        default_poses_dir_path = default_poses_dir()
        
        # Get available poses
        poses_subdir_dict = get_subdirectories(default_poses_dir_path)
        available_poses = sorted(poses_subdir_dict.keys()) if poses_subdir_dict else ["default_pose"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryCreate"),
            display_name="StoryCreate",
            category="🧊 frost-byte/Story",
            inputs=[
                io.String.Input(id="story_name", display_name="story_name", default="my_story", tooltip="Name of the story"),
                io.String.Input(id="story_dir", display_name="story_dir", default=default_stories_dir_path, tooltip="Directory to save the story"),
                io.Combo.Input(id="initial_scene", display_name="initial_scene", options=available_poses, default=available_poses[0], tooltip="First scene to add to the story"),
                io.Combo.Input(id="mask_type", display_name="mask_type", options=["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"], default="combined", tooltip="Mask type for the scene"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Include background in mask"),
                io.Combo.Input(id="prompt_type", display_name="prompt_type", options=["girl_pos", "male_pos", "combined", "four_image_prompt", "wan_prompt", "wan_low_prompt", "custom"], default="girl_pos", tooltip="Type of prompt to use"),
                io.String.Input(id="custom_prompt", display_name="custom_prompt", default="", multiline=True, tooltip="Custom prompt (only used if prompt_type is 'custom')"),
                io.Combo.Input(id="depth_type", display_name="depth_type", options=list(default_depth_options.keys()), default="depth", tooltip="Depth image type"),
                io.Combo.Input(id="pose_type", display_name="pose_type", options=list(default_pose_options.keys()), default="open", tooltip="Pose image type"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info", display_name="story_info", tooltip="Story information"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        story_name="my_story",
        story_dir="",
        initial_scene="default_pose",
        mask_type="combined",
        mask_background=True,
        prompt_type="girl_pos",
        custom_prompt="",
        depth_type="depth",
        pose_type="open",
    ) -> io.NodeOutput:
        if not story_dir:
            story_dir = default_stories_dir()
        
        # Create story directory if it doesn't exist
        story_path = Path(story_dir) / story_name
        os.makedirs(story_path, exist_ok=True)
        
        # Create initial scene
        initial_scene_obj = SceneInStory(
            scene_name=initial_scene,
            scene_order=0,
            mask_type=mask_type,
            mask_background=mask_background,
            prompt_type=prompt_type,
            custom_prompt=custom_prompt,
            depth_type=depth_type,
            pose_type=pose_type,
        )
        
        story_info = StoryInfo(
            story_name=story_name,
            story_dir=str(story_path),
            scenes=[initial_scene_obj]
        )
        
        print(f"StoryCreate: Created story '{story_name}' with initial scene '{initial_scene}'")
        
        return io.NodeOutput(story_info)

class StoryEdit(io.ComfyNode):
    """Edit a story by adding, removing, or reordering scenes"""
    @classmethod
    def define_schema(cls):
        default_poses_dir_path = default_poses_dir()
        poses_subdir_dict = get_subdirectories(default_poses_dir_path)
        available_poses = sorted(poses_subdir_dict.keys()) if poses_subdir_dict else ["default_pose"]
        # story_scene_selector should reflect the current story, so seed with a neutral placeholder.
        story_scene_options = ["0: (no story scenes)"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryEdit"),
            display_name="StoryEdit",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to edit"),
                io.Combo.Input(id="story_action", display_name="story_action", options=["use_file", "use_edit"], default="use_file", tooltip="Use story_info from file or from edited JSON state"),
                io.String.Input(id="story_json_in", display_name="story_json_in", default="", multiline=True, tooltip="Serialized story state (auto-updated after each operation)"),
                io.Combo.Input(id="operation", display_name="operation", options=["add_scene", "remove_scene", "reorder_scene", "edit_scene", "no_change"], default="no_change", tooltip="Operation to perform"),
                io.Combo.Input(id="scene_name", display_name="scene_name", options=available_poses, default=available_poses[0], tooltip="Scene to add/remove/edit (available poses)"),
                io.Combo.Input(id="story_scene_selector", display_name="story_scene_selector", options=story_scene_options, default=story_scene_options[0], tooltip="Select a scene in the current story by index and name"),
                io.Int.Input(id="scene_order", display_name="scene_order", default=0, tooltip="Order position for scene (for add/reorder)"),
                io.Combo.Input(id="mask_type", display_name="mask_type", options=["girl", "male", "combined",], default="combined", tooltip="Mask type"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Include background in mask"),
                io.Combo.Input(id="prompt_type", display_name="prompt_type", options=["girl_pos", "male_pos", "combined", "four_image_prompt", "wan_prompt", "wan_low_prompt", "custom"], default="girl_pos", tooltip="Prompt type"),
                io.String.Input(id="custom_prompt", display_name="custom_prompt", default="", multiline=True, tooltip="Shows current prompt; edit when prompt_type='custom'"),
                io.Combo.Input(id="depth_type", display_name="depth_type", options=list(default_depth_options.keys()), default="depth", tooltip="Depth image type"),
                io.Combo.Input(id="pose_type", display_name="pose_type", options=list(default_pose_options.keys()), default="open", tooltip="Pose image type"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info_out", display_name="story_info", tooltip="Updated story information"),
                io.Image.Output(id="base_image", display_name="base_image", tooltip="Base/upscale image for selected scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask image for selected scene"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from selected mask image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Pose image for selected scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth image for selected scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        story_info=None,
        story_action="use_file",
        story_json_in="",
        operation="no_change",
        scene_name="default_pose",
        story_scene_selector="0: (no story scenes)",
        scene_order=0,
        mask_type="combined",
        mask_background=True,
        prompt_type="girl_pos",
        custom_prompt="",
        depth_type="depth",
        pose_type="open",
    ) -> io.NodeOutput:
        if story_info is None:
            print("StoryEdit: story_info is None")
            return io.NodeOutput(None, None, None, None, None, None)
        
        # Create a copy to avoid modifying the original
        import copy
        
        # Determine which story state to use based on story_action
        if story_action == "use_edit" and story_json_in:
            # Deserialize from JSON state
            try:
                story_data = json.loads(story_json_in)
                scenes = []
                for scene_data in story_data.get("scenes", []):
                    scenes.append(SceneInStory(
                        scene_id=scene_data.get("scene_id", ""),
                        scene_name=scene_data.get("scene_name", ""),
                        scene_order=scene_data.get("scene_order", 0),
                        mask_type=scene_data.get("mask_type", "combined"),
                        mask_background=scene_data.get("mask_background", True),
                        prompt_type=scene_data.get("prompt_type", "girl_pos"),
                        custom_prompt=scene_data.get("custom_prompt", ""),
                        depth_type=scene_data.get("depth_type", "depth"),
                        pose_type=scene_data.get("pose_type", "open"),
                    ))
                story_copy = StoryInfo(
                    story_name=story_data.get("story_name", story_info.story_name),
                    story_dir=story_data.get("story_dir", story_info.story_dir),
                    scenes=scenes
                )
                print(f"StoryEdit: Loaded story from JSON state with {len(scenes)} scenes")
            except Exception as e:
                print(f"StoryEdit: Error deserializing story_json_in: {e}. Falling back to story_info.")
                story_copy = copy.deepcopy(story_info)
        else:
            # Use story_info from file/input
            story_copy = copy.deepcopy(story_info)

        # Ensure scenes from the incoming story_info are present (merge if JSON lacked them)
        if story_info and getattr(story_info, "scenes", None):
            existing_ids = {s.scene_id for s in story_copy.scenes if getattr(s, "scene_id", None)}
            existing_names = {(s.scene_name, s.scene_order) for s in story_copy.scenes}
            for src_scene in story_info.scenes:
                key = getattr(src_scene, "scene_id", None)
                name_key = (src_scene.scene_name, src_scene.scene_order)
                if (key and key in existing_ids) or name_key in existing_names:
                    continue
                story_copy.scenes.append(copy.deepcopy(src_scene))
            # Re-sort after merge so downstream indexing is stable
            story_copy.scenes = sorted(story_copy.scenes, key=lambda s: s.scene_order)
        
        # Helper to pick a scene in story by index string "idx: name"
        def get_scene_by_selector(story_obj, selector: str) -> Optional[SceneInStory]:
            try:
                idx_str, _ = selector.split(":", 1)
                idx_val = int(idx_str.strip())
            except Exception:
                return None
            scenes_sorted = sorted(story_obj.scenes, key=lambda s: s.scene_order)
            if 0 <= idx_val < len(scenes_sorted):
                return scenes_sorted[idx_val]
            return None

        target_scene = get_scene_by_selector(story_copy, story_scene_selector)

        if operation == "add_scene":
            new_scene = SceneInStory(
                scene_name=scene_name,
                scene_order=scene_order,
                mask_type=mask_type,
                mask_background=mask_background,
                prompt_type=prompt_type,
                custom_prompt=custom_prompt,
                depth_type=depth_type,
                pose_type=pose_type,
            )
            story_copy.add_scene(new_scene)
            print(f"StoryEdit: Added scene '{scene_name}' at order {scene_order}")
            
        elif operation == "remove_scene":
            identifier = target_scene.scene_id if target_scene else scene_name
            story_copy.remove_scene(identifier)
            print(f"StoryEdit: Removed scene '{identifier}'")
            
        elif operation == "reorder_scene":
            identifier = target_scene.scene_id if target_scene else scene_name
            story_copy.reorder_scene(identifier, scene_order)
            print(f"StoryEdit: Reordered scene '{identifier}' to position {scene_order}")
            
        elif operation == "edit_scene":
            existing_scene = target_scene if target_scene else story_copy.get_scene_by_name(scene_name)
            if existing_scene:
                existing_scene.mask_type = mask_type
                existing_scene.mask_background = mask_background
                existing_scene.prompt_type = prompt_type
                existing_scene.custom_prompt = custom_prompt
                existing_scene.depth_type = depth_type
                existing_scene.pose_type = pose_type
                print(f"StoryEdit: Edited scene '{existing_scene.scene_name}' (id={existing_scene.scene_id[:8]})")
            else:
                print(f"StoryEdit: Scene '{scene_name}' not found for editing")
        
        elif operation == "no_change":
            print("StoryEdit: No operation performed")
        
        # Build selector options from the current story state (after any modifications)
        scenes_sorted = sorted(story_copy.scenes, key=lambda s: s.scene_order)
        story_scene_options = [f"{idx}: {scene.scene_name}" for idx, scene in enumerate(scenes_sorted)]
        if not story_scene_options:
            story_scene_options = ["0: (no story scenes)"]
        if story_scene_selector not in story_scene_options:
            story_scene_selector = story_scene_options[0]

        # Load preview images for the selected scene
        poses_dir = default_poses_dir()
        # For preview, prefer the scene selected from the story (by index) if available
        preview_scene = get_scene_by_selector(story_copy, story_scene_selector)
        if preview_scene is None and target_scene is not None:
            preview_scene = target_scene
        if preview_scene is None and scenes_sorted:
            preview_scene = scenes_sorted[0]
        if preview_scene is None:
            preview_scene = story_copy.get_scene_by_name(scene_name)
        preview_scene_name = preview_scene.scene_name if preview_scene else scene_name
        preview_mask_type = preview_scene.mask_type if preview_scene else mask_type
        preview_mask_background = preview_scene.mask_background if preview_scene else mask_background
        preview_prompt_type = preview_scene.prompt_type if preview_scene else prompt_type
        preview_custom_prompt = preview_scene.custom_prompt if preview_scene else custom_prompt
        preview_depth_type = preview_scene.depth_type if preview_scene else depth_type
        preview_pose_type = preview_scene.pose_type if preview_scene else pose_type

        pose_dir = os.path.join(poses_dir, preview_scene_name)
        
        base_image = None
        mask_image = None
        mask = None
        preview_mask = None
        pose_image = None
        depth_image = None
        selected_prompt_text = ""
        
        if os.path.isdir(pose_dir):
            selected_depth_attr = default_depth_options.get(preview_depth_type, "depth_image")
            selected_pose_attr = default_pose_options.get(preview_pose_type, "pose_open_image")
            mask_key = resolve_mask_key(preview_mask_type, preview_mask_background)

            assets = SceneInfo.load_preview_assets(
                pose_dir,
                depth_attr=selected_depth_attr,
                pose_attr=selected_pose_attr,
                mask_type=preview_mask_type,
                mask_background=preview_mask_background,
                include_upscale=True,
                include_canny=False,
            )

            base_image = assets["base_image"]
            depth_image = assets["depth_image"]
            pose_image = assets["pose_image"]
            mask_image = assets["mask_image"]
            mask = assets["mask"]
            preview_mask = assets["mask_preview"]
            H, W = assets["H"], assets["W"]

            print(f"StoryEdit: Loading mask with key '{mask_key}' (bg={preview_mask_background})")

            # Load prompt text
            prompt_json_path = os.path.join(pose_dir, "prompts.json")
            prompt_data = load_prompt_json(prompt_json_path)
            
            # Determine the prompt text based on prompt_type
            # Use custom_prompt only when prompt_type is 'custom', otherwise load from file
            if preview_prompt_type == "custom":
                selected_prompt_text = preview_custom_prompt
            else:
                selected_prompt_text = prompt_data.get(preview_prompt_type, "")
        
        # Create preview UI
        preview_batch = assets.get("preview_batch", [])
        preview_image_ui = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        # Create scene list text with scene IDs
        preview_scene_id = preview_scene.scene_id if preview_scene else ""
        scene_list_lines = []
        selected_scene_options = []
        scene_index = 0
        for scene in sorted(story_copy.scenes, key=lambda s: s.scene_order):
            marker = "▶ " if preview_scene_id and scene.scene_id == preview_scene_id else "  "
            mask_suffix = "" if scene.mask_background else " (no bg)"
            selected_scene_options.append(f"{scene_index}: {scene.scene_name}")
            scene_index += 1
            scene_line = (
                f"{marker}{scene.scene_order}: {scene.scene_name} [{scene.scene_id[:8]}] | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={scene.prompt_type} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
            if scene.prompt_type == "custom" and scene.custom_prompt:
                scene_line += f" | custom='{scene.custom_prompt[:30]}...'"
            scene_list_lines.append(scene_line)
        
        scene_list_text = "\n".join(scene_list_lines) if scene_list_lines else "No scenes"
        
        # Serialize story to JSON for persistence
        scenes_data = []
        for scene in story_copy.scenes:
            scenes_data.append({
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "prompt_type": scene.prompt_type,
                "custom_prompt": scene.custom_prompt,
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
            })
        
        story_json = json.dumps({
            "story_name": story_copy.story_name,
            "story_dir": story_copy.story_dir,
            "scenes": scenes_data
        }, indent=2)
        
        scene_options_json = json.dumps(selected_scene_options) if selected_scene_options else "['0: (no story scenes)']"
        # Combine UI elements - text array will contain:
        # [0]=scene_list, [1]=selected_prompt, [2]=story_json, [3]=selector options JSON, [4]=scene options JSON
        combined_ui = {
            "text": [scene_list_text, selected_prompt_text, story_json, json.dumps(story_scene_options), scene_options_json],
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        return io.NodeOutput(
            story_copy,
            base_image,
            mask_image,
            mask,
            pose_image,
            depth_image,
            ui=combined_ui
        )

class StoryView(io.ComfyNode):
    """View and select scenes from a story with preview capabilities"""
    @classmethod
    def define_schema(cls):
        # Get default scene options for when no story is loaded
        default_poses_dir_path = default_poses_dir()
        poses_subdir_dict = get_subdirectories(default_poses_dir_path)
        default_scene_options = sorted(poses_subdir_dict.keys()) if poses_subdir_dict else ["default_pose"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryView"),
            display_name="StoryView",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to view"),
                io.Combo.Input(id="selected_scene", display_name="selected_scene", options=default_scene_options, default=default_scene_options[0], tooltip="Select a scene from the story"),
                io.String.Input(id="prompt_in", display_name="prompt_in", multiline=True, default="", tooltip="Editable prompt text"),
                io.Combo.Input(id="prompt_action", display_name="prompt_action", options=["use_file", "use_edit"], default="use_file", tooltip="Use file prompt or edited prompt"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info_out", display_name="story_info", tooltip="Story information (pass-through for chaining to StorySave)"),
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information for selected scene"),
                io.String.Output(id="story_name", display_name="story_name", tooltip="Name of the story"),
                io.String.Output(id="story_dir", display_name="story_dir", tooltip="Directory of the story"),
                io.Int.Output(id="scene_count", display_name="scene_count", tooltip="Number of scenes in the story"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="selected_prompt", display_name="selected_prompt", tooltip="The selected prompt text"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Pose image for selected scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask image for selected scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth image for selected scene"),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo 
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        story_info=None,
        selected_scene="default_pose",
        prompt_in="",
        prompt_action="use_file",
    ) -> io.NodeOutput:
        className = cls.__name__
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        
        if story_info is None:
            print("StoryView: story_info is None")
            return io.NodeOutput(None, None, "", "", 0, "", "", None, None, None)
        
        # Find the selected scene configuration in the story
        scene_config = None
        for scene in story_info.scenes:
            if scene.scene_name == selected_scene:
                scene_config = scene
                break
        
        if scene_config is None and story_info.scenes:
            print(f"StoryView: Scene '{selected_scene}' not found in story, defaulting to first scene '{story_info.scenes[0].scene_name}'")
            scene_config = story_info.scenes[0]
            selected_scene = scene_config.scene_name

        # If scene not found in story, create a default configuration
        if scene_config is None:
            print(f"StoryView: Scene '{selected_scene}' not found in story, using defaults")
            scene_config = SceneInStory(
                scene_name=selected_scene,
                scene_order=0,
                mask_type="combined",
                mask_background=True,
                prompt_type="girl_pos",
                custom_prompt="",
                depth_type="depth",
                pose_type="open",
            )
        
        # Load scene data from pose directory
        poses_dir = default_poses_dir()
        pose_dir = os.path.join(poses_dir, selected_scene)
        
        if not os.path.isdir(pose_dir):
            print(f"StoryView: pose_dir '{pose_dir}' is not a valid directory")
            return io.NodeOutput(story_info, None, story_info.story_name, story_info.story_dir, len(story_info.scenes), selected_scene, "", None, None, None)
        
        try:
            scene_info, assets, selected_prompt, prompt_data, prompt_widget_text = SceneInfo.from_story_scene(
                scene_config,
                poses_dir=poses_dir,
                prompt_in=prompt_in,
                prompt_action=prompt_action,
                include_upscale=False,
                include_canny=False,
            )
        except Exception as e:
            print(f"StoryView: failed to build SceneInfo for '{selected_scene}': {e}")
            return io.NodeOutput(story_info, None, story_info.story_name, story_info.story_dir, len(story_info.scenes), selected_scene, "", None, None, None)

        if prompt_widget_text is not None:
            input_types = cls.INPUT_TYPES()
            inputs = input_types.get('required', {}) if isinstance(input_types, dict) else {}
            update_ui_widget(className, unique_id, extra_pnginfo, prompt_widget_text, "prompt_in", inputs)

        selected_depth_image = assets.get("depth_image")
        selected_pose_image = assets.get("pose_image")
        selected_mask_image = assets.get("mask_image")
        mask = assets.get("mask")
        
        # Create preview UI combining pose, mask, and depth
        preview_batch = assets.get("preview_batch", [])
        preview_image_ui = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        # Create text preview with scene IDs
        scene_list_lines = []
        for scene in sorted(story_info.scenes, key=lambda s: s.scene_order):
            marker = "▶ " if scene.scene_name == selected_scene else "  "
            mask_suffix = "" if scene.mask_background else " (no bg)"
            scene_line = (
                f"{marker}{scene.scene_order}: {scene.scene_name} [{scene.scene_id[:8]}] | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={scene.prompt_type} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
            if scene.prompt_type == "custom" and scene.custom_prompt:
                scene_line += f" | custom='{scene.custom_prompt[:30]}...'"
            scene_list_lines.append(scene_line)
        
        scene_list_text = "\n".join(scene_list_lines) if scene_list_lines else "No scenes"
        
        preview_text = (
            f"Story: {story_info.story_name}\n"
            f"Dir: {story_info.story_dir}\n"
            f"Scenes: {len(story_info.scenes)}\n"
            f"Selected: {selected_scene} (order {scene_config.scene_order})\n"
            f"Prompt Type: {scene_config.prompt_type}\n"
            f"Prompt: {selected_prompt}\n\n"
            f"All Scenes:\n{scene_list_text}"
        )
        text_ui = ui.PreviewText(value=preview_text)
        
        # Combine UI elements
        combined_ui = {
            "text": text_ui.as_dict().get("text", []),
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        print(f"StoryView: Story '{story_info.story_name}' - Selected scene '{selected_scene}' with prompt_type '{scene_config.prompt_type}'")
        
        return io.NodeOutput(
            story_info,
            scene_info,
            story_info.story_name,
            story_info.story_dir,
            len(story_info.scenes),
            selected_scene,
            selected_prompt,
            selected_pose_image,
            selected_mask_image,
            selected_depth_image,
            ui=combined_ui
        )

class StorySceneBatch(io.ComfyNode):
    """Create an ordered list of scene descriptors for iteration."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StorySceneBatch"),
            display_name="StorySceneBatch",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to expand into an ordered scene list"),
                io.String.Input(id="job_id", display_name="job_id", default="", tooltip="Optional job id; auto-generated when empty"),
                io.String.Input(id="job_root_dir", display_name="job_root_dir", default="", tooltip="Base directory for per-job inputs/outputs; defaults to story_dir/jobs/<job_id>"),
            ],
            outputs=[
                io.Int.Output(id="scene_count", display_name="scene_count", tooltip="Total number of scenes"),
                io.Custom("SCENE_BATCH").Output(id="scene_batch", display_name="scene_batch", tooltip="Ordered list of scene dictionaries"),
                io.String.Output(id="job_id_out", display_name="job_id", tooltip="Job id used for this batch"),
                io.String.Output(id="job_root_dir_out", display_name="job_root_dir", tooltip="Resolved job root directory"),
            ],
        )

    @classmethod
    def execute(
        cls,
        story_info=None,
        job_id: str = "",
        job_root_dir: str = "",
    ) -> io.NodeOutput:
        if story_info is None or not getattr(story_info, "scenes", None):
            print("StorySceneBatch: story_info is empty")
            return io.NodeOutput(0, [], job_id or "", job_root_dir or "")

        resolved_job_id = job_id.strip() or uuid.uuid4().hex[:12]
        default_root = Path(story_info.story_dir) / "jobs" / resolved_job_id
        job_root = Path(job_root_dir) if job_root_dir else default_root
        job_root.mkdir(parents=True, exist_ok=True)

        poses_dir = default_poses_dir()
        batch: list[dict] = []

        scenes_sorted = sorted(story_info.scenes, key=lambda s: s.scene_order)
        for scene in scenes_sorted:
            pose_dir = os.path.join(poses_dir, scene.scene_name)
            prompt_path = os.path.join(pose_dir, "prompts.json")
            prompt_data = load_prompt_json(prompt_path) or {}

            mask_key = resolve_mask_key(scene.mask_type, scene.mask_background)
            depth_key = default_depth_options.get(scene.depth_type, "depth_image")
            pose_key = default_pose_options.get(scene.pose_type, "pose_open_image")
            positive_prompt = build_positive_prompt(scene.prompt_type, prompt_data, scene.custom_prompt)

            job_scene_dir = job_root / f"{scene.scene_order:03d}_{scene.scene_name}"
            job_input_dir = job_scene_dir / "input"
            job_output_dir = job_scene_dir / "output"
            job_input_dir.mkdir(parents=True, exist_ok=True)
            job_output_dir.mkdir(parents=True, exist_ok=True)

            source_input_dir = Path(pose_dir) / "input"
            first_input_image = None
            for ext in ["png", "jpg", "jpeg", "webp"]:
                matches = sorted(source_input_dir.glob(f"*.{ext}"))
                if matches:
                    first_input_image = str(matches[0])
                    break

            descriptor = {
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "mask_key": mask_key,
                "prompt_type": scene.prompt_type,
                "custom_prompt": scene.custom_prompt,
                "depth_type": scene.depth_type,
                "depth_key": depth_key,
                "pose_type": scene.pose_type,
                "pose_key": pose_key,
                "pose_dir": pose_dir,
                "story_dir": story_info.story_dir,
                "job_id": resolved_job_id,
                "job_root": str(job_root),
                "job_scene_dir": str(job_scene_dir),
                "job_input_dir": str(job_input_dir),
                "job_output_dir": str(job_output_dir),
                "source_input_dir": str(source_input_dir),
                "source_output_dir": str(Path(pose_dir) / "output"),
                "positive_prompt": positive_prompt,
                "wan_prompt": prompt_data.get("wan_prompt", ""),
                "wan_low_prompt": prompt_data.get("wan_low_prompt", ""),
                "four_image_prompt": prompt_data.get("four_image_prompt", ""),
                "girl_pos": prompt_data.get("girl_pos", ""),
                "male_pos": prompt_data.get("male_pos", ""),
                "input_image_path": first_input_image,
                "prompt_data": prompt_data,
            }

            batch.append(descriptor)

        print(f"StorySceneBatch: Prepared {len(batch)} scenes with job_id={resolved_job_id} at {job_root}")

        return io.NodeOutput(
            len(batch),
            batch,
            resolved_job_id,
            str(job_root),
        )


class StoryScenePick(io.ComfyNode):
    """Select one scene descriptor by index and load the assets for generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StoryScenePick"),
            display_name="StoryScenePick",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("SCENE_BATCH").Input(id="scene_batch", display_name="scene_batch", tooltip="Scene descriptor list from StorySceneBatch"),
                io.Int.Input(id="scene_index", display_name="scene_index", default=0, tooltip="Index into scene_batch (0-based)"),
            ],
            outputs=[
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Selected mask image"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Single-channel mask"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected depth image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected pose image"),
                io.Image.Output(id="canny_image", display_name="canny_image", tooltip="Canny edge image"),
                io.String.Output(id="positive_prompt", display_name="positive_prompt", tooltip="Chosen positive prompt"),
                io.String.Output(id="wan_prompt", display_name="wan_prompt", tooltip="Wan high positive prompt"),
                io.String.Output(id="wan_low_prompt", display_name="wan_low_prompt", tooltip="Wan low positive prompt"),
                io.String.Output(id="four_image_prompt", display_name="four_image_prompt", tooltip="Four-image prompt"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Scene name"),
                io.Int.Output(id="scene_order", display_name="scene_order", tooltip="Scene order"),
                io.String.Output(id="scene_id", display_name="scene_id", tooltip="Scene id"),
                io.String.Output(id="job_id", display_name="job_id", tooltip="Job id"),
                io.String.Output(id="job_scene_dir", display_name="job_scene_dir", tooltip="Job scene directory"),
                io.String.Output(id="input_image_path", display_name="input_image_path", tooltip="Path to first input image (if any)"),
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Fully-loaded SceneInfo for the selected scene"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        scene_batch=None,
        scene_index: int = 0,
    ) -> io.NodeOutput:
        if not scene_batch:
            print("StoryScenePick: scene_batch is empty")
            return io.NodeOutput(None, None, None, None, None, "", "", "", "", "", 0, "", "", "", "", None)

        try:
            scenes_sorted = sorted(scene_batch, key=lambda d: d.get("scene_order", 0))
        except Exception:
            scenes_sorted = scene_batch

        safe_index = max(0, min(len(scenes_sorted) - 1, scene_index))
        descriptor = scenes_sorted[safe_index]

        pose_dir = descriptor.get("pose_dir", "")
        if not pose_dir or not os.path.isdir(pose_dir):
            print(f"StoryScenePick: pose_dir '{pose_dir}' is invalid")
            return io.NodeOutput(None, None, None, None, None, "", "", "", "", descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_scene_dir", ""), descriptor.get("input_image_path", ""), None)

        scene_config = SceneInStory(
            scene_id=descriptor.get("scene_id", ""),
            scene_name=descriptor.get("scene_name", ""),
            scene_order=descriptor.get("scene_order", 0),
            mask_type=descriptor.get("mask_type", "combined"),
            mask_background=descriptor.get("mask_background", True),
            prompt_type=descriptor.get("prompt_type", "girl_pos"),
            custom_prompt=descriptor.get("custom_prompt", ""),
            depth_type=descriptor.get("depth_type", "depth"),
            pose_type=descriptor.get("pose_type", "open"),
        )

        prompt_override = descriptor.get("positive_prompt", "")

        try:
            scene_info, assets, selected_prompt, prompt_data, _ = SceneInfo.from_story_scene(
                scene_config,
                pose_dir_override=pose_dir,
                include_upscale=False,
                include_canny=True,
                prompt_override=prompt_override,
            )
        except Exception as e:
            print(f"StoryScenePick: failed to build SceneInfo for '{scene_config.scene_name}': {e}")
            return io.NodeOutput(None, None, None, None, None, "", "", "", "", descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_scene_dir", ""), descriptor.get("input_image_path", ""), None)

        empty_image = make_empty_image()
        canny_image = assets.get("canny_image", empty_image)
        mask_image = assets.get("mask_image")
        mask = assets.get("mask")
        depth_image = assets.get("depth_image", empty_image)
        pose_image = assets.get("pose_image", empty_image)

        def select_prompt(descriptor, selected, prompt_data):
            return (
                descriptor.get("positive_prompt")
                or selected
                or build_positive_prompt(
                    descriptor.get("prompt_type", ""),
                    prompt_data,
                    descriptor.get("custom_prompt", ""),
                )
            )

        positive_prompt = select_prompt(descriptor, selected_prompt, prompt_data)
        wan_prompt_val = descriptor.get("wan_prompt", prompt_data.get("wan_prompt", ""))
        wan_low_prompt_val = descriptor.get("wan_low_prompt", prompt_data.get("wan_low_prompt", ""))
        four_image_prompt_val = descriptor.get("four_image_prompt", prompt_data.get("four_image_prompt", ""))

        return io.NodeOutput(
            mask_image,
            mask,
            depth_image,
            pose_image,
            canny_image,
            positive_prompt,
            wan_prompt_val,
            wan_low_prompt_val,
            four_image_prompt_val,
            descriptor.get("scene_name", ""),
            descriptor.get("scene_order", 0),
            descriptor.get("scene_id", ""),
            descriptor.get("job_id", ""),
            descriptor.get("job_scene_dir", ""),
            descriptor.get("input_image_path", ""),
            scene_info,
        )


class StorySave(io.ComfyNode):
    """Save the story configuration to a JSON file"""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StorySave"),
            display_name="StorySave",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Custom("STORY_INFO").Input(id="story_info", display_name="story_info", tooltip="Story to save"),
                io.String.Input(id="filename", display_name="filename", default="story.json", tooltip="Filename for the story JSON"),
            ],
            outputs=[
                io.String.Output(id="save_path", display_name="save_path", tooltip="Path where story was saved"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        story_info=None,
        filename="story.json",
    ) -> io.NodeOutput:
        if story_info is None:
            print("StorySave: story_info is None")
            return io.NodeOutput("")
        
        # Ensure story directory exists
        os.makedirs(story_info.story_dir, exist_ok=True)
        
        # Build save path
        save_path = Path(story_info.story_dir) / filename
        
        # Save the story
        save_story(story_info, str(save_path))
        
        print(f"StorySave: Saved story to '{save_path}'")
        
        preview_ui = ui.PreviewText(value=f"Story saved to: {save_path}\nScenes: {len(story_info.scenes)}")
        
        return io.NodeOutput(
            str(save_path),
            ui=preview_ui.as_dict()
        )

class StoryLoad(io.ComfyNode):
    """Load a story from a JSON file"""
    @classmethod
    def define_schema(cls):
        default_stories_dir_path = default_stories_dir()
        stories_subdir_dict = get_subdirectories(default_stories_dir_path)
        available_stories = sorted(stories_subdir_dict.keys()) if stories_subdir_dict else ["default_story"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryLoad"),
            display_name="StoryLoad",
            category="🧊 frost-byte/Story",
            inputs=[
                io.String.Input(id="stories_dir", display_name="stories_dir", default=default_stories_dir_path, tooltip="Directory containing stories"),
                io.Combo.Input(id="story_name", display_name="story_name", options=available_stories, default=available_stories[0], tooltip="Story to load"),
                io.String.Input(id="filename", display_name="filename", default="story.json", tooltip="Filename of the story JSON"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info", display_name="story_info", tooltip="Loaded story information"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        stories_dir="",
        story_name="default_story",
        filename="story.json",
    ) -> io.NodeOutput:
        if not stories_dir:
            stories_dir = default_stories_dir()
        
        story_path = Path(stories_dir) / story_name / filename
        
        if not story_path.exists():
            print(f"StoryLoad: Story file not found at '{story_path}'")
            return io.NodeOutput(None)
        
        story_info = load_story(str(story_path))
        
        if story_info:
            print(f"StoryLoad: Loaded story '{story_info.story_name}' with {len(story_info.scenes)} scenes")
        else:
            print(f"StoryLoad: Failed to load story from '{story_path}'")
        
        return io.NodeOutput(story_info)

class FBTextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("FBTextEncodeQwenImageEditPlus"),
            display_name="FBTextEncodeQwenImageEditPlus",
            category="🧊 frost-byte/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        vae=None,
        image1=None, image2=None, image3=None, image4=None
    ) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


# ============================================================================
# LIBBER NODES
# ============================================================================

class LibberManager(io.ComfyNode):
    """Manage Libber instances - create, load, save, and edit libs with an interactive table."""
    
    @classmethod
    def define_schema(cls):
        libber_dir = default_libber_dir()
        
        # Get available libber files (basenames only, no .json extension)
        libber_names = []
        if os.path.isdir(libber_dir):
            for f in os.listdir(libber_dir):
                if f.endswith('.json'):
                    # Remove .json extension for display
                    libber_names.append(f[:-5])
        
        if not libber_names:
            libber_names = ["none"]
        
        return io.Schema(
            node_id=prefixed_node_id("LibberManager"),
            display_name="LibberManager",
            category="🧊 frost-byte/Libber",
            inputs=[
                io.Combo.Input(
                    id="libber_name",
                    display_name="libber_name",
                    options=sorted(libber_names),
                    default=libber_names[0],
                    tooltip="Select an existing libber or create a new one"
                ),
                io.String.Input(
                    id="libber_dir",
                    display_name="libber_dir",
                    default=libber_dir,
                    tooltip="Directory for libber files"
                ),
                io.String.Input(
                    id="delimiter",
                    display_name="delimiter",
                    default="%",
                    tooltip="Delimiter for lib references"
                ),
                io.Int.Input(
                    id="max_depth",
                    display_name="max_depth",
                    default=10,
                    min=1,
                    max=100,
                    tooltip="Maximum substitution depth"
                ),
            ],
            outputs=[
                io.String.Output(id="status", display_name="status", tooltip="Operation status and info"),
                io.String.Output(id="keys_list", display_name="keys_list", tooltip="List of all lib keys"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, libber_name="my_libber",
                libber_dir="", delimiter="%", max_depth=10):
        
        if not libber_dir:
            libber_dir = default_libber_dir()
        
        # Skip if no libber selected
        if libber_name == "none":
            return io.NodeOutput("Select or create a libber to begin", "")
        
        manager = LibberStateManager.instance()
        
        try:
            # Check if libber file exists and load it to ensure we have the latest data
            libber_filepath = os.path.join(libber_dir, f"{libber_name}.json")
            if os.path.exists(libber_filepath):
                # Reload from file to get latest changes
                libber = manager.load_libber(libber_name, libber_filepath)
                status = f"✓ Reloaded libber '{libber_name}' from file"
            else:
                # Try to get existing in-memory instance or create new one
                libber = manager.get_libber(libber_name)
                if not libber:
                    # Create new libber if it doesn't exist
                    libber = manager.create_libber(libber_name, delimiter, max_depth)
                    status = f"✓ Created new libber '{libber_name}'"
                else:
                    status = f"✓ Libber '{libber_name}' ready (in-memory)"
            
            keys = libber.list_libs()
            
            # Format keys list for display
            keys_display = "\n".join(keys) if keys else "(no libs)"
            
            # Return UI data for dynamic updates
            keys_json = json.dumps(keys)
            
            # Get libber data for UI display
            libber_data = manager.get_libber_data(libber_name)
            if libber_data:
                lib_dict_json = json.dumps(libber_data["lib_dict"])
            else:
                lib_dict_json = json.dumps({})
            
            combined_ui = {
                "text": [keys_json, lib_dict_json, status]
            }
            
            print(f"LibberManager: {status}")
            return io.NodeOutput(status, keys_display, ui=combined_ui)
            
        except Exception as e:
            status = f"✗ Error: {str(e)}"
            print(f"LibberManager error: {status}")
            return io.NodeOutput(status, "")


class LibberApply(io.ComfyNode):
    """Apply Libber substitutions to text with libber selection."""
    
    @classmethod
    def define_schema(cls):
        manager = LibberStateManager.instance()
        available_libbers = manager.list_libbers()
        
        if not available_libbers:
            available_libbers = ["none"]
        
        return io.Schema(
            node_id=prefixed_node_id("LibberApply"),
            display_name="LibberApply",
            category="🧊 frost-byte/Libber",
            inputs=[
                io.Combo.Input(
                    id="libber_name",
                    display_name="libber_name",
                    options=available_libbers,
                    default=available_libbers[0],
                    tooltip="Select which Libber to use"
                ),
                io.String.Input(
                    id="text",
                    display_name="text",
                    default="",
                    multiline=True,
                    tooltip="Input text with lib references (e.g., 'A %chunky% character')"
                ),
            ],
            outputs=[
                io.String.Output(id="result", display_name="result", tooltip="Text with all lib references substituted"),
                io.String.Output(id="info", display_name="info", tooltip="Substitution details and available libs"),
            ],
        )
    
    @classmethod
    def execute(cls, libber_name="my_libber", text=""):
        manager = LibberStateManager.instance()
        
        # Try to reload from file to ensure we have the latest data
        libber_dir = default_libber_dir()
        libber_filepath = os.path.join(libber_dir, f"{libber_name}.json")
        if os.path.exists(libber_filepath):
            try:
                libber = manager.load_libber(libber_name, libber_filepath)
            except Exception as e:
                print(f"LibberApply: Error reloading from file, using in-memory instance: {e}")
                libber = manager.get_libber(libber_name)
        else:
            libber = manager.get_libber(libber_name)
        
        if not libber:
            status = f"✗ Libber '{libber_name}' not found. Create or load it in LibberManager first."
            print(f"LibberApply: {status}")
            return io.NodeOutput(text, status)
        
        if not text:
            # Display available libs when no text provided
            keys = libber.list_libs()
            info_parts = [f"Libber '{libber_name}' ready ({len(keys)} libs)"]
            if keys:
                info_parts.append("\nAvailable libs:")
                for key in keys[:10]:  # Show first 10
                    value = libber.get_lib(key) or ""
                    preview = value[:40] + "..." if len(value) > 40 else value
                    info_parts.append(f"  {key}: {preview}")
                if len(keys) > 10:
                    info_parts.append(f"  ... and {len(keys) - 10} more")
            else:
                info_parts.append("(no libs defined yet)")
            
            info = "\n".join(info_parts)
            return io.NodeOutput("", info)
        
        try:
            result = libber.substitute(text)
            keys = libber.list_libs()
            info = f"✓ Substituted using libber '{libber_name}' ({len(keys)} libs, max_depth={libber.max_depth})"
            print(f"LibberApply: {info}")
            print(f"  Input:  {text[:100]}...")
            print(f"  Output: {result[:100]}...")
            
            # Provide UI data showing available libs
            libber_data = manager.get_libber_data(libber_name)
            if libber_data:
                lib_dict_json = json.dumps(libber_data["lib_dict"])
                combined_ui = {"text": [lib_dict_json, info]}
                return io.NodeOutput(result, info, ui=combined_ui)
            
            return io.NodeOutput(result, info)
            
        except Exception as e:
            result = text
            info = f"✗ Error during substitution: {e}"
            print(f"LibberApply: {info}")
            return io.NodeOutput(result, info)
        
        return io.NodeOutput(result, info)


# ============================================================================
# SCENE PROMPT MANAGEMENT NODES
# ============================================================================

class ScenePromptManager(io.ComfyNode):
    """Manage prompts in a Scene's PromptCollection with an interactive table interface."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("ScenePromptManager"),
            display_name="ScenePromptManager",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info",
                    display_name="scene_info",
                    optional=True,
                    tooltip="Scene to manage prompts for (optional - can create standalone collection)"
                ),
                io.String.Input(
                    id="collection_json",
                    display_name="collection_json",
                    default="",
                    multiline=True,
                    tooltip="Prompt collection JSON (auto-updated by UI table - normally don't edit manually)"
                ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(
                    id="scene_info_out",
                    display_name="scene_info",
                    tooltip="Updated scene with modified prompts"
                ),
                io.Custom("DICT").Output(
                    id="prompt_dict",
                    display_name="prompt_dict",
                    tooltip="Dictionary of composed prompts by name"
                ),
                io.String.Output(
                    id="status",
                    display_name="status",
                    tooltip="Operation status"
                ),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, scene_info=None, collection_json=""):
        # Get or create prompt collection
        if scene_info and scene_info.prompts:
            collection = scene_info.prompts
        elif collection_json:
            try:
                data = json.loads(collection_json)
                collection = PromptCollection.from_dict(data)
            except Exception as e:
                collection = PromptCollection()
                print(f"ScenePromptManager: Error loading collection JSON: {e}")
        else:
            collection = PromptCollection()
        
        # Create or update scene_info
        if scene_info:
            scene_info.prompts = collection
            status = f"✓ Scene '{scene_info.pose_name}' has {len(collection.prompts)} prompts"
        else:
            # Create minimal scene_info if none provided
            scene_info = SceneInfo(
                pose_dir="",
                pose_name="",
                pose_json="[]",
                resolution=512,
                prompts=collection
            )
            status = f"✓ Created new collection with {len(collection.prompts)} prompts"
        
        # Prepare UI data
        collection_data = collection.to_dict()
        prompts_list = []
        for key, metadata in collection.prompts.items():
            prompts_list.append({
                "key": key,
                "value": metadata.value,
                "processing_type": metadata.processing_type,
                "libber_name": metadata.libber_name or "",
                "category": metadata.category or "",
            })
        
        # Get available libbers
        libber_manager = LibberStateManager()
        available_libbers = ["none"] + list(libber_manager.libbers.keys())
        
        # Compose prompts if compositions exist
        prompt_dict = {}
        if collection.compositions:
            prompt_dict = collection.compose_prompts(collection.compositions, libber_manager)
        
        # Prepare compositions list for UI
        compositions_list = []
        for name, prompt_keys in collection.compositions.items():
            compositions_list.append({
                "name": name,
                "prompt_keys": prompt_keys,
                "preview": prompt_dict.get(name, "")[:100] + ("..." if len(prompt_dict.get(name, "")) > 100 else "")
            })
        
        combined_ui = {
            "text": [
                json.dumps(collection_data),
                json.dumps(prompts_list),
                status,
                json.dumps(available_libbers),
                json.dumps(compositions_list),
                json.dumps(prompt_dict)
            ]
        }
        
        print(f"ScenePromptManager: {status}")
        return io.NodeOutput(scene_info, prompt_dict, status, ui=combined_ui)


class PromptComposer(io.ComfyNode):
    """Compose multiple output prompts from a PromptCollection with flexible slot assignment."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("PromptComposer"),
            display_name="PromptComposer",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info",
                    display_name="scene_info",
                    optional=True,
                    tooltip="Scene with prompt collection (from ScenePromptManager or SceneSelect)"
                ),
                io.String.Input(
                    id="composition_json",
                    display_name="composition_json",
                    default='{\n  "qwen_main": ["char1", "char2", "setting", "quality"]\n}',
                    multiline=True,
                    tooltip='Composition map: {"output_name": ["prompt_key1", "prompt_key2"]}. Example: {"main_prompt": ["char1", "setting"], "video_high": ["char1", "quality_high"]}'
                ),
            ],
            outputs=[
                io.Custom("DICT").Output(
                    id="prompt_dict",
                    display_name="prompt_dict",
                    tooltip="Dictionary of composed prompts by name"
                ),
                io.String.Output(
                    id="composition_json_out",
                    display_name="composition_json",
                    tooltip="Composition map for saving/loading"
                ),
                io.String.Output(
                    id="info",
                    display_name="info",
                    tooltip="Composition details"
                ),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, scene_info=None, composition_json=""):
        # Get prompt collection
        collection = None
        if scene_info and scene_info.prompts:
            collection = scene_info.prompts
        
        if not collection:
            status = "✗ No prompt collection provided"
            print(f"PromptComposer: {status}")
            return io.NodeOutput({}, "{}", status)
        
        # Parse composition map
        composition_map = {}
        if composition_json:
            try:
                composition_map = json.loads(composition_json)
            except Exception as e:
                print(f"PromptComposer: Error parsing composition JSON: {e}")
                composition_map = {}
        
        # Default composition if none provided
        if not composition_map:
            # Create default based on legacy prompt names
            available_keys = list(collection.prompts.keys())
            composition_map = {
                "prompt_a": available_keys[:2] if len(available_keys) >= 2 else available_keys,
            }
        
        # Get libber manager for processing
        libber_manager = LibberStateManager.instance()
        
        # Compose prompts
        prompt_dict = collection.compose_prompts(composition_map, libber_manager)
        
        # Generate info
        info_lines = [f"✓ Composed {len(prompt_dict)} output prompts:"]
        for name, value in prompt_dict.items():
            preview = value[:60] + "..." if len(value) > 60 else value
            prompt_count = len(composition_map.get(name, []))
            info_lines.append(f"  {name}: {prompt_count} prompts → \"{preview}\"")
        
        info = "\n".join(info_lines)
        
        # Prepare UI data
        prompts_list = []
        for key, metadata in collection.prompts.items():
            prompts_list.append({
                "key": key,
                "value": metadata.value,
                "processing_type": metadata.processing_type,
                "libber_name": metadata.libber_name or "",
            })
        
        combined_ui = {
            "text": [
                json.dumps(composition_map),
                json.dumps(prompts_list),
                json.dumps(prompt_dict),
                info
            ]
        }
        
        print(f"PromptComposer: {info}")
        return io.NodeOutput(prompt_dict, json.dumps(composition_map, indent=2), info, ui=combined_ui)


# ============================================================================
# REST API STATE MANAGERS
# ============================================================================

from server import PromptServer
from aiohttp import web
import time
from datetime import datetime, timedelta

class PromptCollectionStateManager:
    """
    Manages server-side PromptCollection instances for REST API operations.
    Sessions expire after 30 minutes of inactivity.
    """
    _instance = None
    
    def __init__(self):
        self.sessions = {}  # session_id -> {"collection": PromptCollection, "last_access": datetime}
        self.ttl_minutes = 30
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = datetime.now()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["last_access"] > timedelta(minutes=self.ttl_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]
            print(f"PromptCollectionStateManager: Expired session {sid}")
    
    def create_session(self, session_id: str, collection: PromptCollection):
        """Create or update a session with a PromptCollection."""
        self.cleanup_expired()
        self.sessions[session_id] = {
            "collection": collection,
            "last_access": datetime.now()
        }
        print(f"PromptCollectionStateManager: Created session {session_id}")
    
    def get_collection(self, session_id: str) -> Optional[PromptCollection]:
        """Get PromptCollection for a session, updating last access time."""
        self.cleanup_expired()
        if session_id in self.sessions:
            self.sessions[session_id]["last_access"] = datetime.now()
            return self.sessions[session_id]["collection"]
        return None
    
    def update_collection(self, session_id: str, collection: PromptCollection):
        """Update the PromptCollection for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["collection"] = collection
            self.sessions[session_id]["last_access"] = datetime.now()


class LibberStateManager:
    """
    Manages server-side Libber instances for REST API operations.
    Libbers are stored by name and persist until explicitly deleted or server restart.
    """
    _instance = None
    
    def __init__(self):
        self.libbers = {}  # libber_name -> Libber instance
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def create_libber(self, name: str, delimiter: str = "%", max_depth: int = 10) -> Libber:
        """Create a new Libber instance."""
        libber = Libber(lib_dict={}, delimiter=delimiter, max_depth=max_depth)
        self.libbers[name] = libber
        print(f"LibberStateManager: Created libber '{name}'")
        return libber
    
    def load_libber(self, name: str, filepath: str) -> Libber:
        """Load a Libber from file."""
        libber = Libber.load(filepath)
        self.libbers[name] = libber
        print(f"LibberStateManager: Loaded libber '{name}' from {filepath}")
        return libber
    
    def get_libber(self, name: str) -> Optional[Libber]:
        """Get a Libber by name."""
        return self.libbers.get(name)
    
    def save_libber(self, name: str, filepath: str):
        """Save a Libber to file."""
        if name in self.libbers:
            self.libbers[name].save(filepath)
            print(f"LibberStateManager: Saved libber '{name}' to {filepath}")
        else:
            raise ValueError(f"Libber '{name}' not found")
    
    def list_libbers(self) -> List[str]:
        """List all loaded libber names."""
        return list(self.libbers.keys())
    
    def delete_libber(self, name: str):
        """Remove a Libber from memory."""
        if name in self.libbers:
            del self.libbers[name]
            print(f"LibberStateManager: Deleted libber '{name}'")
    
    def get_libber_data(self, name: str) -> Optional[dict]:
        """Get Libber data for UI display."""
        if name in self.libbers:
            libber = self.libbers[name]
            return {
                "keys": libber.list_libs(),
                "lib_dict": libber.libs.copy(),
                "delimiter": libber.delimiter,
                "max_depth": libber.max_depth
            }
        return None


# Register REST API endpoints
@PromptServer.instance.routes.post("/fbtools/prompts/create")
async def create_prompt_collection(request):
    """Create a new PromptCollection session."""
    try:
        data = await request.json()
        session_id = data.get("session_id", f"prompt_{int(time.time()*1000)}")
        
        # Create new empty collection or from legacy data
        legacy_data = data.get("legacy_data")
        if legacy_data:
            collection = PromptCollection.from_legacy_dict(legacy_data)
        else:
            collection = PromptCollection()
        
        manager = PromptCollectionStateManager.instance()
        manager.create_session(session_id, collection)
        
        return web.json_response({
            "success": True,
            "session_id": session_id,
            "collection": collection.to_dict()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@PromptServer.instance.routes.post("/fbtools/prompts/add")
async def add_prompt(request):
    """Add or update a prompt in a PromptCollection."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        prompt_name = data.get("prompt_name")
        prompt_value = data.get("prompt_value")
        category = data.get("category")
        description = data.get("description")
        tags = data.get("tags")
        
        if not session_id or not prompt_name:
            return web.json_response({
                "success": False,
                "error": "session_id and prompt_name required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        collection.add_prompt(prompt_name, prompt_value, category, description, tags)
        manager.update_collection(session_id, collection)
        
        return web.json_response({
            "success": True,
            "collection": collection.to_dict(),
            "prompt_names": collection.list_prompt_names()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@PromptServer.instance.routes.post("/fbtools/prompts/remove")
async def remove_prompt(request):
    """Remove a prompt from a PromptCollection."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        prompt_name = data.get("prompt_name")
        
        if not session_id or not prompt_name:
            return web.json_response({
                "success": False,
                "error": "session_id and prompt_name required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        removed = collection.remove_prompt(prompt_name)
        if removed:
            manager.update_collection(session_id, collection)
        
        return web.json_response({
            "success": True,
            "removed": removed,
            "collection": collection.to_dict(),
            "prompt_names": collection.list_prompt_names()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@PromptServer.instance.routes.get("/fbtools/prompts/list_names")
async def list_prompt_names(request):
    """Get list of all prompt names in a PromptCollection."""
    try:
        session_id = request.query.get("session_id")
        
        if not session_id:
            return web.json_response({
                "success": False,
                "error": "session_id required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        return web.json_response({
            "success": True,
            "prompt_names": collection.list_prompt_names(),
            "count": len(collection.prompts)
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


# ============================================================================
# LIBBER REST API ENDPOINTS
# ============================================================================

@PromptServer.instance.routes.post("/fbtools/libber/create")
async def libber_create(request):
    """
    Create a new Libber instance.
    Body: {"name": str, "delimiter": str (optional), "max_depth": int (optional)}
    Returns: {"name": str, "keys": [], "status": "created"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        
        delimiter = data.get("delimiter", "%")
        max_depth = data.get("max_depth", 10)
        
        manager = LibberStateManager.instance()
        libber = manager.create_libber(name, delimiter, max_depth)
        
        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "delimiter": libber.delimiter,
            "max_depth": libber.max_depth,
            "status": "created"
        })
    
    except Exception as e:
        print(f"Error creating libber: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/fbtools/libber/load")
async def libber_load(request):
    """
    Load a Libber from file.
    Body: {"name": str, "filepath": str}
    Returns: {"name": str, "keys": [...], "status": "loaded"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        filepath = data.get("filepath")
        
        if not name or not filepath:
            return web.json_response({"error": "name and filepath required"}, status=400)
        
        manager = LibberStateManager.instance()
        libber = manager.load_libber(name, filepath)
        
        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "delimiter": libber.delimiter,
            "max_depth": libber.max_depth,
            "status": "loaded"
        })
    
    except Exception as e:
        print(f"Error loading libber: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/fbtools/libber/add_lib")
async def libber_add_lib(request):
    """
    Add a lib entry to a Libber.
    Body: {"name": str, "key": str, "value": str}
    Returns: {"name": str, "keys": [...], "status": "added"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        key = data.get("key")
        value = data.get("value")
        
        if not all([name, key, value is not None]):
            return web.json_response({"error": "name, key, and value required"}, status=400)
        
        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)
        
        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)
        
        libber.add_lib(key, value)
        
        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "status": "added",
            "key": key
        })
    
    except Exception as e:
        print(f"Error adding lib: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/fbtools/libber/remove_lib")
async def libber_remove_lib(request):
    """
    Remove a lib entry from a Libber.
    Body: {"name": str, "key": str}
    Returns: {"name": str, "keys": [...], "status": "removed"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        key = data.get("key")
        
        if not name or not key:
            return web.json_response({"error": "name and key required"}, status=400)
        
        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)
        
        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)
        
        libber.remove_lib(key)
        
        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "status": "removed",
            "key": key
        })
    
    except Exception as e:
        print(f"Error removing lib: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/fbtools/libber/save")
async def libber_save(request):
    """
    Save a Libber to file.
    Body: {"name": str, "filepath": str}
    Returns: {"name": str, "filepath": str, "status": "saved"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        filepath = data.get("filepath")
        
        if not name or not filepath:
            return web.json_response({"error": "name and filepath required"}, status=400)
        
        manager = LibberStateManager.instance()
        manager.save_libber(name, filepath)
        
        return web.json_response({
            "name": name,
            "filepath": filepath,
            "status": "saved"
        })
    
    except Exception as e:
        print(f"Error saving libber: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/fbtools/libber/list")
async def libber_list(request):
    """
    List all available libbers in memory and on disk.
    Returns: {"libbers": [...], "files": [...]}
    """
    try:
        manager = LibberStateManager.instance()
        libbers = manager.list_libbers()
        
        # Also scan default directory for available files
        libber_dir = default_libber_dir()
        files = []
        if os.path.exists(libber_dir):
            files = [f for f in os.listdir(libber_dir) if f.endswith('.json')]
        
        return web.json_response({
            "libbers": libbers,
            "files": files,
            "count": len(libbers)
        })
    
    except Exception as e:
        print(f"Error listing libbers: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/fbtools/libber/get_data/{name}")
async def libber_get_data(request):
    """
    Get Libber data for UI display.
    Returns: {"keys": [...], "lib_dict": {...}, "delimiter": str, "max_depth": int}
    """
    try:
        name = request.match_info.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        
        manager = LibberStateManager.instance()
        data = manager.get_libber_data(name)
        
        if not data:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)
        
        return web.json_response(data)
    
    except Exception as e:
        print(f"Error getting libber data: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/fbtools/libber/apply")
async def libber_apply(request):
    """
    Apply Libber substitutions to text.
    Body: {"name": str, "text": str}
    Returns: {"result": str, "original": str}
    """
    try:
        data = await request.json()
        name = data.get("name")
        text = data.get("text")
        
        if not name or text is None:
            return web.json_response({"error": "name and text required"}, status=400)
        
        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)
        
        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)
        
        result = libber.substitute(text)
        
        return web.json_response({
            "result": result,
            "original": text,
            "name": name
        })
    
    except Exception as e:
        print(f"Error applying libber: {e}")
        return web.json_response({"error": str(e)}, status=500)


class FBToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FBTextEncodeQwenImageEditPlus,
            SAMPreprocessNHWC,
            QwenAspectRatio,
            SubdirLister,
            # NodeInputSelect,
            SceneCreate,
            SceneUpdate,
            SceneSave,
            SceneInput,
            SceneOutput,
            SceneView,
            SceneSelect,
            StorySceneBatch,
            StoryScenePick,
            StoryCreate,
            StoryEdit,
            StoryView,
            StorySave,
            StoryLoad,
            OpaqueAlpha,
            TailSplit,
            TailEnhancePro,
            # Libber nodes
            LibberManager,
            LibberApply,
            # Scene Prompt Management nodes
            ScenePromptManager,
            PromptComposer,
        ]