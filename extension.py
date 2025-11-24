import node_helpers
import math
from comfy.utils import common_upscale
import comfy.model_management as model_management

from typing_extensions import override
from folder_paths import get_output_directory
from nodes import ImageScaleBy
from .utils.util import draw_pose_json, draw_pose, extend_scalelist, pose_normalized, select_text_by_action, update_ui_widget
from .utils.io import save_json_file, load_prompt_json, load_json_file
from .utils.images import image_resize_ess
from .utils.pose import estimate_dwpose, dense_pose, depth_anything, depth_anything_v2, zoe, zoe_any, openpose, midas, canny

from .utils.images import make_empty_image, _compute_ref_stats, _pick_ref_image, proc_deflicker_luma, proc_deflicker_clahe, proc_color_histmatch, proc_color_meanstd, proc_bilateral_cv2, proc_unsharp, _stack_if_same_shape
from .utils.images import _HAS_KORNIA, _HAS_SKIMAGE, _HAS_CV2, load_image_comfyui, save_image_comfyui, make_placeholder_tensor, normalize_image_tensor
from comfy_api.latest import ComfyExtension, io, ui
from inspect import cleandoc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict

try:
    from westNeighbor_comfyui_ultimate_openpose_editor.openpose_editor_nodes import OpenposeEditorNode  # type: ignore
except Exception:
    OpenposeEditorNode = None


OpenposeJSON = dict

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
            node_id="SAMPreprocessNHWC",
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
            node_id="TailEnhancePro",
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
            node_id="TailSplit",
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
            node_id="OpaqueAlpha",
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
            node_id="SubdirLister",
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

class SceneInfo(BaseModel):
    #metadata
    pose_dir: str
    pose_name: str
    girl_pos: str
    male_pos: str
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

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

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

        # print(f"SceneSelect: default poses_dir='{default_dir}'; default_pose='{default_pose}'; default_options={default_options}")

        return io.Schema(
            node_id="SceneSelect",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("poses_dir", default=default_dir, tooltip="Directory containing pose subdirectories"),
                io.Combo.Input('selected_pose', options=default_options, default=default_pose, tooltip="Select a pose name"),
                io.String.Input(id="girl_pos_in", display_name="girl_pos", multiline=True, default="", tooltip="The positive prompt for the girl"),
                io.Combo.Input(id="girl_action", display_name="action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the girl prompt"),
                io.String.Input(id="male_pos_in", display_name="male_pos", multiline=True, default="", tooltip="The positive prompt for the male"),
                io.Combo.Input(id="male_action", display_name="action", options=["use_file", "use_edit"], default="use_file", tooltip="Action for the male prompt"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images"),
                io.String.Output(id="pose_name", display_name="pose_name", tooltip="Name of the selected pose"),
                io.String.Output(id="pose_dir", display_name="pose_dir", tooltip="Directory of the selected pose"),
                io.String.Output(id="girl_pos", display_name="girl_pos", tooltip="Girl's positive prompt"),
                io.String.Output(id="male_pos", display_name="male_pos", tooltip="Male's positive prompt"),
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
        male_action="use_file"
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

        print(f"{className}: Type of INPUT_TYPES ='{type(inputs)}'; INPUTS ='{inputs}'")
        num_inputs = len(inputs)
        print(f"{className}: INPUT_TYPES ='{input_types}; num_inputs={num_inputs}")
        print(f"{className}: Executing with poses_dir='{poses_dir}'; selected_pose='{selected_pose}'")
        if not poses_dir:
            poses_dir = default_poses_dir()

        if not poses_dir or not selected_pose:
            print(f"{className}: poses_dir or selected_pose is empty")
            return io.NodeOutput(None)
        
        pose_dir = os.path.join(poses_dir, selected_pose)
        print(f"{className}: pose_dir='{pose_dir}'")

        if not os.path.isdir(pose_dir):
            print(f"{className}: pose_dir '{pose_dir}' is not a valid directory")
            return io.NodeOutput(None)
        
        prompt_json_path = os.path.join(pose_dir, "prompts.json")
        prompt_data = load_prompt_json(prompt_json_path)
        print(f"{className}: Loaded prompt_data: {prompt_data}")
        pose_json_path = os.path.join(pose_dir, "pose.json")
        pose_json = load_json_file(pose_json_path)
        if not pose_json:
            pose_json = "[]"

        girl_file_text = prompt_data.get("girl_pos", "")
        girl_pos, girl_widget_text = select_text_by_action(
            girl_pos_in, 
            girl_file_text, 
            girl_action, 
            className
        )

        if girl_widget_text is not None:
            print(f"{className}: Updating UI widget for girl_pos_in with text: '{girl_widget_text[:32]}...'")
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
            print(f"{className}: Updating UI widget for male_pos_in with text: '{male_widget_text[:32]}...'")
            update_ui_widget(className, unique_id, extra_pnginfo, male_widget_text,"male_pos_in", inputs)

        male_pos_ui_text = male_widget_text if male_widget_text is not None else male_file_text

        print(f"{className}: girl_pos: '{girl_pos}'; male_pos: '{male_pos}'")

        depth_image_path = os.path.join(pose_dir, "depth.png")
        if not os.path.exists(depth_image_path):
            print(f"{className}: depth_image_path '{depth_image_path}' does not exist")
        depth_image = load_image_comfyui(depth_image_path)
        print(f"{className}: depth_image shape: {depth_image.shape if depth_image is not None else 'None'} ")
        
        if not depth_image is None:
            H, W = depth_image.shape[1], depth_image.shape[2]
        else:
            H, W = (512, 512)
        resolution = max(H, W)

        depth_any_image_path = os.path.join(pose_dir, "depth_any.png")
        if not os.path.exists(depth_any_image_path):
            print(f"{className}: depth_any_image_path '{depth_any_image_path}' does not exist")
        depth_any_image = load_image_comfyui(depth_any_image_path)
        depth_midas_image_path = os.path.join(pose_dir, "depth_midas.png")
        depth_midas_image = load_image_comfyui(depth_midas_image_path)
        depth_zoe_image_path = os.path.join(pose_dir, "depth_zoe.png")
        depth_zoe_image = load_image_comfyui(depth_zoe_image_path)
        depth_zoe_any_image_path = os.path.join(pose_dir, "depth_zoe_any.png")
        depth_zoe_any_image = load_image_comfyui(depth_zoe_any_image_path)
        pose_dense_image_path = os.path.join(pose_dir, "pose_dense.png")
        pose_dense_image = load_image_comfyui(pose_dense_image_path)
        pose_dw_image_path = os.path.join(pose_dir, "pose_dw.png")
        pose_dw_image = load_image_comfyui(pose_dw_image_path)
        pose_edit_image_path = os.path.join(pose_dir, "pose_edit.png")
        pose_edit_image = load_image_comfyui(pose_edit_image_path)
        pose_face_image_path = os.path.join(pose_dir, "pose_face.png")
        pose_face_image = load_image_comfyui(pose_face_image_path)
        pose_open_image_path = os.path.join(pose_dir, "pose_open.png")
        pose_open_image = load_image_comfyui(pose_open_image_path)
        canny_image_path = os.path.join(pose_dir, "canny.png")
        canny_image = load_image_comfyui(canny_image_path)
        upscale_image_path = os.path.join(pose_dir, "upscale.png")

        if (not os.path.exists(upscale_image_path)):
            print(f"{className}: upscale_image_path '{upscale_image_path}' does not exist")
        upscale_image = load_image_comfyui(upscale_image_path)
        print(f"{className}: upscale_image shape: {upscale_image.shape if upscale_image is not None else 'None'} ")

        preview_image = ui.PreviewImage(image=upscale_image) if upscale_image is not None else None

        ui_data = {
            "images": preview_image.as_dict().get("images", []) if preview_image else None,
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
            "text": [str(girl_pos_ui_text), str(male_pos_ui_text)],
        }
        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=selected_pose,
            girl_pos=girl_pos,
            male_pos=male_pos,
            pose_json=pose_json,
            resolution=resolution,
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
            canny_image=canny_image,
            upscale_image=upscale_image,                       
        )

        return io.NodeOutput(
            scene_info,
            selected_pose,
            pose_dir,
            girl_pos,
            male_pos,
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

class SceneCreate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCreate",
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
                io.String.Input(id="girl_pos", display_name="girl_pos", placeholder="Provide the positive prompt for the female in the scene", tooltip="Positive prompt for the girl", multiline=True),
                io.String.Input(id="male_pos", display_name="male_pos", placeholder="Provide the positive prompt for the male(s) in the scene", tooltip="Positive prompt for the male(s)", multiline=True),
                io.Image.Input(id="base_image", display_name="base_image", tooltip="Base image for the scene"),
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
        girl_pos="",
        male_pos="",
        base_image=None,
    ) -> io.NodeOutput:
        if base_image is None:
            print("SceneCreate: base_image is None")
            return io.NodeOutput(None)
        
        if not poses_dir:
            poses_dir = default_poses_dir()
        
        pose_dir = os.path.join(poses_dir, pose_name)
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir, exist_ok=True)
            print(f"SceneCreate: Created pose_dir='{pose_dir}'")
        
        if not pose_name:
            pose_name = "default_pose"
            
        if not girl_pos:
            girl_pos = ""
        if not male_pos:
            male_pos = ""
        
        upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
        print(f"SceneCreate: upscale_image is of type: {type(upscale_image)} with shape {upscale_image.shape if torch.is_tensor(upscale_image) else 'N/A'}")
        torch_device = model_management.get_torch_device()

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

        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=pose_name,
            resolution=resolution,
            girl_pos=girl_pos,
            male_pos=male_pos,
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
        )
        return io.NodeOutput(
            scene_info,
        )

class SceneUpdate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneUpdate",
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

    ):
        if scene_info_in is None:
            print("SceneUpdate: scene_info is None")
            return io.NodeOutput(None)

        print(f"SceneUpdate: Scene Info In is of type: {type(scene_info_in)} with attributes: {dir(scene_info_in)}")
        scene_info_out = scene_info_in
        if update_prompts:
            if girl_pos is not None:
                scene_info_out.girl_pos = girl_pos
            if male_pos is not None:
                scene_info_out.male_pos = male_pos
                
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
        
        return io.NodeOutput(
            scene_info_out,
        )

class SceneView(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneView",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information" ),
                io.Combo.Input(
                    id="depth_type", options=list(default_depth_options.keys())
                ),
                io.Combo.Input(
                    id="pose_type", options=list(default_pose_options.keys())
                ),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected Depth Image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected Pose Image"),
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
        scene_info=None,
        depth_type="depth",
        pose_type="dense",
    ) -> io.NodeOutput:
        if scene_info is None:
            print("SceneView: scene_info is None")
            return io.NodeOutput(None, None)
        
        depth_attr = default_depth_options.get(depth_type, "depth_image")
        pose_attr = default_pose_options.get(pose_type, "pose_dense_image")
        girl_pos = getattr(scene_info, "girl_pos", "")
        male_pos = getattr(scene_info, "male_pos", "")
        pose_name = getattr(scene_info, "pose_name", "")
        pose_dir = getattr(scene_info, "pose_dir", "")
        depth_any_image = getattr(scene_info, "depth_any_image", None)

        if depth_any_image is None:
            H, W = 512, 512
            depth_any_image = make_placeholder_tensor(H, W)
        else:
            H, W = depth_any_image.shape[1], depth_any_image.shape[2]

        depth_image = getattr(scene_info, depth_attr, make_empty_image())
        pose_image = getattr(scene_info, pose_attr, make_empty_image())
        depth_image = normalize_image_tensor(depth_image, H, W)
        pose_image = normalize_image_tensor(pose_image, H, W)

        combined_batch = torch.cat([depth_image, pose_image], dim=0)
        preview_ui = ui.PreviewImage(image=combined_batch)

        combined_prompt = f"Girl Positive Prompt: {girl_pos}\nMale Positive Prompt: {male_pos}"
        text_ui = ui.PreviewText(value=combined_prompt)
        combined_ui = {
            "text": text_ui.as_dict().get("text", ''),
            "images": preview_ui.as_dict().get("images", []),
            "animated": preview_ui.as_dict().get("animated", False),
        }

        print(
            f"SceneView: depth_type='{depth_type}'; pose_type='{pose_type}'; "
            f"depth_any_image shape: {depth_any_image.shape if depth_any_image is not None else 'None'}; "
            f"depth_image shape: {depth_image.shape if depth_image is not None else 'None'}; "
            f"pose_image shape: {pose_image.shape if pose_image is not None else 'None'}; "
            f"{combined_prompt}"
        )

        return io.NodeOutput(
            depth_image,
            pose_image,
            pose_name,
            pose_dir,
            girl_pos,
            male_pos,
            ui=combined_ui
        )
 
class SceneOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneOutput",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(id="scene_info", display_name="scene_info", tooltip="Scene Information" ),
            ],
            outputs=[
                io.String.Output("pose_dir", display_name="pose_dir", tooltip="Directory where the scene is saved"),
                io.String.Output("pose_name", display_name="pose_name", tooltip="Name of the pose"),
                io.String.Output("girl_pos", display_name="girl_pos", tooltip="Girl Positive Prompt"),
                io.String.Output("male_pos", display_name="male_pos", tooltip="Male Positive Prompt"),
                io.String.Output("pose_json", display_name="pose_json", tooltip="Pose JSON data"),
                io.Image.Output("depth_image", display_name="depth_image", tooltip="Depth Image"),
                io.Image.Output("depth_any_image", display_name="depth_any_image", tooltip="Depth Any Image"),
                io.Image.Output("depth_midas_image", display_name="depth_midas_image", tooltip="Depth Midas Image"),
                io.Image.Output("depth_zoe_image", display_name="depth_zoe_image", tooltip="Depth Zoe Image"),
                io.Image.Output("depth_zoe_any_image", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image"),
                io.Image.Output("pose_dense_image", display_name="pose_dense_image", tooltip="Pose Dense Image"),
                io.Image.Output("pose_dw_image", display_name="pose_dw_image", tooltip="Pose DW Image"),
                io.Image.Output("pose_edit_image", display_name="pose_edit_image", tooltip="Pose Edit Image"),
                io.Image.Output("pose_face_image", display_name="pose_face_image", tooltip="Pose Face Image"),
                io.Image.Output("pose_open_image", display_name="pose_open_image", tooltip="Pose Open Image"),
                io.Image.Output("canny_image", display_name="canny_image", tooltip="Canny Image"),
                io.Image.Output("upscale_image", display_name="upscale_image", tooltip="Upscale Image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info=None,
    ) -> io.NodeOutput:
        if scene_info is None:
            print("SceneOutput: scene_info is None")
            return io.NodeOutput(("", "", "", "", "", None, None, None, None, None, None, None, None, None, None, None, None))
        
        print(
            f"SceneOutput: scene_info.pose_dir='{scene_info.pose_dir}', "
            f"pose_name='{scene_info.pose_name}', "
            f"girl_pos='{scene_info.girl_pos}', "
            f"male_pos='{scene_info.male_pos}', "
            f"depth_image shape: {scene_info.depth_image.shape if scene_info.depth_image is not None else 'None'}"
        )
        return io.NodeOutput(
            scene_info.pose_dir,
            scene_info.pose_name,
            scene_info.girl_pos,
            scene_info.male_pos,
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
        )

class SceneSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveScene",
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
        pose_name = scene_info.pose_name if scene_info else ""
        if pose_name == "" or scene_info is None:
            print("SaveScene: scene_info is None or pose_name is empty")
            return io.NodeOutput(None)

        pose_dir = pose_dir if pose_dir else scene_info.pose_dir
        if not pose_dir:
            pose_dir = Path(default_poses_dir()) / pose_name

        print(f"SaveScene: pose_dir='{pose_dir}'; pose_name='{pose_name}'; dest_dir='{pose_dir}'")
        if pose_dir and not os.path.isdir(pose_dir):
            os.makedirs(pose_dir, exist_ok=True)
            print(f"SaveScene: Created directory '{pose_dir}' for saving scene data")
        if pose_name == "":
            print("SaveScene: pose_name is empty, cannot save pose files")
            return io.NodeOutput(None)
        pose_path = Path(pose_dir)
        if scene_info.depth_image is not None:
            depth_path = pose_path / "depth.png"
            save_image_comfyui(scene_info.depth_image, depth_path)
        if scene_info.depth_any_image is not None:
            depth_any_path = pose_path / "depth_any.png"
            save_image_comfyui(scene_info.depth_any_image, depth_any_path)
        if scene_info.depth_midas_image is not None:
            depth_midas_path = pose_path / "depth_midas.png"
            save_image_comfyui(scene_info.depth_midas_image, depth_midas_path)
        if scene_info.depth_zoe_image is not None:
            depth_zoe_path = pose_path / "depth_zoe.png"
            save_image_comfyui(scene_info.depth_zoe_image, depth_zoe_path)
        if scene_info.depth_zoe_any_image is not None:
            depth_zoe_any_path = pose_path / "depth_zoe_any.png"
            save_image_comfyui(scene_info.depth_zoe_any_image, depth_zoe_any_path)
        if scene_info.pose_dense_image is not None:
            pose_dense_path = pose_path / "pose_dense.png"
            save_image_comfyui(scene_info.pose_dense_image, pose_dense_path)
        if scene_info.pose_dw_image is not None:
            pose_dw_path = pose_path / "pose_dw.png"
            save_image_comfyui(scene_info.pose_dw_image, pose_dw_path)
        if scene_info.pose_edit_image is not None:
            pose_edit_path = pose_path   / "pose_edit.png"
            save_image_comfyui(scene_info.pose_edit_image, pose_edit_path)
        if scene_info.pose_face_image is not None:
            pose_face_path = pose_path / "pose_face.png"
            save_image_comfyui(scene_info.pose_face_image, pose_face_path)
        if scene_info.pose_open_image is not None:
            pose_open_path = pose_path / "pose_open.png"
            save_image_comfyui(scene_info.pose_open_image, pose_open_path)
        if scene_info.canny_image is not None:
            canny_path = pose_path / "canny.png"
            save_image_comfyui(scene_info.canny_image, canny_path)
        if scene_info.upscale_image is not None:
            upscale_path = pose_path / "upscale.png"
            save_image_comfyui(scene_info.upscale_image, upscale_path)
        if scene_info.pose_json:
            pose_json_path = pose_path / "pose.json"
            save_json_file(pose_json_path, json.loads(scene_info.pose_json))
            
        girl_pos = scene_info.girl_pos if scene_info.girl_pos else ""
        male_pos = scene_info.male_pos if scene_info.male_pos else ""
        prompts_path = pose_path / "prompts.json"
        prompt_data = {
            "girl_pos": girl_pos,
            "male_pos": male_pos,
        }
        save_json_file(prompts_path, prompt_data)   
        return io.NodeOutput(
            ui=ui.PreviewText(f"Scene saved to '{pose_dir}' with prompt='The girl {scene_info.girl_pos}, The male {scene_info.male_pos}'"),
        )

class SceneInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneInput",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="pose_dir", display_name="pose_dir", tooltip="Directory where the scene is saved", multiline=False, default=""),
                io.String.Input(id="pose_name", display_name="pose_name", tooltip="Name of the pose", multiline=False, default=""),
                io.String.Input(id="girl_pos", display_name="girl_pos", tooltip="The prompt for the girl in the scene", multiline=True, default=""),
                io.String.Input(id="male_pos", display_name="male_pos", tooltip="The prompt for the male(s) in the scene", multiline=True, default=""),
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
    ) -> io.NodeOutput:
        if not pose_dir or not os.path.isdir(pose_dir):
            print(f"SceneInput: pose_dir '{pose_dir}' is invalid")
            return io.NodeOutput(None)

        print(f"SceneInput: pose_dir='{pose_dir}'; pose_name='{pose_name}'")

        scene_info = SceneInfo(
            pose_dir=pose_dir,
            pose_name=pose_name,
            girl_pos=girl_pos,
            male_pos=male_pos,
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
            canny_image=canny_image,
            upscale_image=upscale_image,
        )

        return io.NodeOutput(
            scene_info
        )

class FBTextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FBTextEncodeQwenImageEditPlus",
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


class FBToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FBTextEncodeQwenImageEditPlus,
            SAMPreprocessNHWC,
            SubdirLister,
            SceneCreate,
            SceneUpdate,
            SceneSave,
            SceneInput,
            SceneOutput,
            SceneView,
            SceneSelect,
            OpaqueAlpha,
            TailSplit,
            TailEnhancePro,
        ]