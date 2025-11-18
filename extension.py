import node_helpers
import math
from comfy.utils import common_upscale
from typing_extensions import override
from folder_paths import get_output_directory

from comfy_api.latest import ComfyExtension, io, ui
from inspect import cleandoc
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Optional
import os
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, ConfigDict

try:
    import kornia
    import kornia.enhance as KE
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False
    
try:
    import numpy as np
    from skimage.exposure import match_histograms
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False



def save_json_file(json_path, data):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"fbTools: Saved JSON to '{json_path}'")
    except Exception as e:
        print(f"fbTools: Error saving JSON to '{json_path}': {e}")

def load_prompt_json(prompt_json_path):
    if not os.path.isfile(prompt_json_path):
        print(f"fbTools: prompt_json_path '{prompt_json_path}' is not a valid file")
        return {"girl_pos": "", "male_pos": ""}

    output = {}
    try:
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output["girl_pos"] = data.get("girl_pos", "")
        output["male_pos"] = data.get("male_pos", "")
        return output
    except Exception as e:
        print(f"fbTools: Error loading prompt JSON from '{prompt_json_path}': {e}")
        return {"girl_pos": "", "male_pos": ""}

def load_json_file(json_path):
    if not os.path.isfile(json_path):
        print(f"fbTools: json_path '{json_path}' is not a valid file")
        return "{}"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"fbTools: Error loading JSON from '{json_path}': {e}")
        return ""

def make_empty_image(batch=1, height=64, width=64, channels=3):
    """
    Returns a blank image tensor in ComfyUI format:
    - Shape: [B, H, W, C]
    - Dtype: torch.float32
    - Values: 0.0 (black) in range [0.0, 1.0]
    """
    return torch.zeros((batch, height, width, channels), dtype=torch.float32)

def save_image_comfyui(image_tensor, save_path):
    """Save ComfyUI IMAGE tensor [1,H,W,C] float32 0..1 to disk as PNG."""
    print(f"fbTools: save_image_comfyui: '{save_path}'; shape {image_tensor.shape}")
    
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must be 4D [B,H,W,C]")
    
    B, H, W, C = image_tensor.shape
    if B != 1:
        print(f"image_tensor batch size > 1; saving only first image of {B}")
        image_tensor = image_tensor[0:1]  # take first image only
        
    if H == 1 or W == 1:
        raise ValueError("image_tensor height and width must be > 1")

    try:
        img = (image_tensor[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()  # [H,W,C] uint8
        print(f"Saving image to '{save_path}' with shape {img.shape}")
        Image.fromarray(img).save(save_path, format='PNG', compress_level=4)
    except Exception as e:
        print(f"Error saving image to '{save_path}': {e}")

def load_image_comfyui(image_path):
    """Load image from disk into ComfyUI IMAGE format [1,H,W,C] float32 0..1"""
    output_images = []
    w, h = None, None
    excluded_formats = ['MPO']

    try:
        img = Image.open(image_path)
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert('RGB')
            
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            image = np.array(image).astype(np.float32) / 255.0  # [H,W,C] float32 0..1
            image = torch.from_numpy(image)[None,]  # [1,H,W,C]
            output_images.append(image)
            
            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)  # [B,H,W,C]
            else:
                output_image = output_images[0]  # [1,H,W,C]
    except (FileNotFoundError, OSError, AttributeError) as e:
        output_image = make_empty_image()
        
    return output_image

def _nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()

def _nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()

def kornia_unsharp(img_nhwc, radius=1.5, amount=0.5):
    # img_nhwc: [B,H,W,C] in 0..1, C=3 or 4
    has_a = (img_nhwc.shape[-1] == 4)
    rgb = img_nhwc[...,:3]
    x = _nhwc_to_nchw(rgb)  # -> [B,3,H,W]
    # gaussian blur (sigma ~ radius), then unsharp
    sigma = max(0.1, float(radius))
    k = KE.gaussian_blur2d(x, kernel_size=int(2*round(3*sigma)+1), sigma=(sigma, sigma))
    sharp = torch.clamp(x + amount*(x - k), 0, 1)
    out = _nchw_to_nhwc(sharp)
    if has_a:
        out = torch.cat([out, img_nhwc[...,3:4]], dim=3)
    return out

def kornia_clahe(img_nhwc, clip_limit=2.0, grid=(8,8)):
    x = _nhwc_to_nchw(img_nhwc[...,:3])
    y = KE.equalize_clahe(x, clip_limit=clip_limit, grid_size=grid)
    out = _nchw_to_nhwc(y)
    return torch.cat([out, img_nhwc[...,3:4]], dim=3) if img_nhwc.shape[-1]==4 else out

def skimage_match_hist(img_nhwc, ref_nhwc, multichannel=True, amount=1.0):
    # Move to CPU/HWC uint8
    def to_u8(hwc):
        if hwc.dtype.is_floating_point:
            hwc = (hwc.clamp(0,1)*255.0).to(torch.uint8)
        return hwc

    i = to_u8(img_nhwc[0,...,:3].detach().cpu())   # [H,W,3]
    r = to_u8(ref_nhwc[0,...,:3].detach().cpu())

    matched = match_histograms(i.numpy(), r.numpy(), channel_axis=-1 if multichannel else None)
    matched = torch.from_numpy(matched).to(dtype=torch.float32)/255.0
    matched = matched.unsqueeze(0)  # [1,H,W,3]

    if img_nhwc.shape[-1] == 4:
        matched = torch.cat([matched, img_nhwc[... ,3:4].detach().cpu()], dim=3)

    # blend with original if amount<1
    out = img_nhwc.detach().cpu()
    out[...,:3] = out[...,:3].lerp(matched[...,:3], float(amount))
    return out.to(img_nhwc.device, dtype=img_nhwc.dtype)

def _stack_if_same_shape(frames: List[torch.Tensor]) -> torch.Tensor:
    if len(frames) == 0 or frames is None:
        raise ValueError("No frames to stack")
    h0, w0, c0 = frames[0].shape[1], frames[0].shape[2], frames[0].shape[3]

    for f in frames:
        hCur, wCur, cCur = f.shape[1], f.shape[2], f.shape[3]
        if f.ndim != 4 or hCur != h0 or wCur != w0 or cCur != c0:
            raise ValueError("All frames must have the same shape to stack")
    return torch.cat(frames, dim=0) # should this use torch.stack?  # [B, H, W, C]

def _compute_ref_stats(frames: List[torch.Tensor], window: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute per-channel mean/std (RGB) and luma mean over the last `window` frames of the list.
    Each frame is [1,H,W,C] NHWC in [0,1].

    Args:
        frames (List[torch.Tensor]): _description_
        window (int): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        mean_c: [1,1,1,3] mean per channel RGB
        std_c: [1,1,1,3] std per channel RGB (clamped to >= 1e-6)
        mean_luma: float mean luma Y in [0,1]
    """
    if not frames:
        # Fallback to neutral stats
        mean_c = torch.tensor([[[[0.5, 0.5, 0.5]]]], dtype=torch.float32)
        std_c = torch.tensor([[[[0.25, 0.25, 0.25]]]], dtype=torch.float32)
        mean_luma = 0.5
        return mean_c, std_c, mean_luma
    
    sub = frames[-window:] if window > 0 else frames
    batch = torch.cat(sub, dim=0)  # [B, H, W, C]
    mean_c = batch.mean(dim=[0,1,2], keepdim=True)  # [1,1,1,C]
    std_c = batch.std(dim=[0,1,2], keepdim=True).clamp_min(1e-6)  # [1,1,1,C]
    
    r, g, b = mean_c[..., 0], mean_c[..., 1], mean_c[..., 2]
    mean_luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).item()  # scalar in [0,1]
    return mean_c, std_c, mean_luma

def _pick_ref_image(frames: List[torch.Tensor], window: int) -> torch.Tensor:
    """Pick one frame (the median index of the last `window`) as histogram reference."""
    if not frames:
        return None

    sub = frames[-window:] if window > 0 else frames
    idx = len(sub) // 2
    return sub[idx]  # [1,H,W,C]


# ------------------ Processing primitives (NHWC) ------------------

def proc_deflicker_luma(img: torch.Tensor, target_luma: float, strength: float) -> torch.Tensor:
    # scale RGB together to match target mean luma
    r, g, b = img[...,:3].unbind(dim=3)
    luma = 0.2126*r + 0.7152*g + 0.0722*b
    mean = luma.mean().item()
    if mean <= 1e-6:
        return img
    s = 1.0 + strength * (target_luma/mean - 1.0)
    out_rgb = (img[...,:3] * s).clamp(0,1)
    return torch.cat([out_rgb, img[...,3:4]], dim=3) if img.shape[3] == 4 else out_rgb

def proc_deflicker_clahe(img: torch.Tensor, clip_limit: float, grid_w: int, grid_h: int) -> torch.Tensor:
    if not _HAS_KORNIA:
        return img  # fallback silently if kornia missing
    # CLAHE works best per-channel in NCHW
    x = img[...,:3].permute(0,3,1,2).contiguous()  # [B,3,H,W]
    y = KE.equalize_clahe(x, clip_limit=float(clip_limit), grid_size=(int(grid_h), int(grid_w)))
    out = y.permute(0,2,3,1).contiguous()
    return torch.cat([out, img[...,3:4]], dim=3) if img.shape[3]==4 else out

def proc_color_meanstd(img: torch.Tensor, tgt_mean: torch.Tensor, tgt_std: torch.Tensor, amount: float) -> torch.Tensor:
    x = img[...,:3]
    im_mean = x.mean(dim=(0,1,2), keepdim=True)
    im_std  = x.std(dim=(0,1,2), keepdim=True).clamp_min(1e-6)
    matched = ((x - im_mean)/im_std) * tgt_std.to(x) + tgt_mean.to(x)
    out = x.lerp(matched, float(amount)).clamp(0,1)
    return torch.cat([out, img[...,3:4]], dim=3) if img.shape[3]==4 else out

def proc_color_histmatch(img: torch.Tensor, ref: torch.Tensor, amount: float) -> torch.Tensor:
    if not _HAS_SKIMAGE:
        return img
    # to CPU uint8 HWC
    i = img[0,...,:3].detach().clamp(0,1).mul(255).to(torch.uint8).cpu().numpy()  # HWC
    r = ref[0,...,:3].detach().clamp(0,1).mul(255).to(torch.uint8).cpu().numpy()
    matched = match_histograms(i, r, channel_axis=-1)
    matched = torch.from_numpy(matched).to(dtype=torch.float32)/255.0
    matched = matched.unsqueeze(0)  # [1,H,W,3]
    out_rgb = img[...,:3].lerp(matched.to(img.device, img.dtype), float(amount)).clamp(0,1)
    return torch.cat([out_rgb, img[...,3:4]], dim=3) if img.shape[3]==4 else out_rgb

def proc_unsharp(img: torch.Tensor, radius: float, amount: float) -> torch.Tensor:
    if not _HAS_KORNIA:
        return img
    x = img[...,:3].permute(0,3,1,2).contiguous()  # NCHW
    sigma = max(0.1, float(radius))
    ksize = int(2*round(3*sigma)+1)
    blur = KE.gaussian_blur2d(x, kernel_size=ksize, sigma=(sigma, sigma))
    sharp = (x + amount*(x - blur)).clamp(0,1)
    out = sharp.permute(0,2,3,1).contiguous()
    return torch.cat([out, img[...,3:4]], dim=3) if img.shape[3]==4 else out

def proc_bilateral_cv2(img: torch.Tensor, d: int, sigma_color: float, sigma_space: float) -> torch.Tensor:
    if not _HAS_CV2:
        return img
    # CPU path for cv2
    hwc = img[0].detach().clamp(0,1).cpu().numpy()
    rgb = (hwc[...,:3]*255.0).astype('uint8')
    out = cv2.bilateralFilter(rgb, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    out = torch.from_numpy(out).to(dtype=img.dtype)/255.0
    out = out.unsqueeze(0)  # [1,H,W,3]
    return torch.cat([out, img[...,3:4].detach().cpu()], dim=3).to(img.device, img.dtype) if img.shape[3]==4 else out.to(img.device, img.dtype)

def get_first_subdirectory(parent_dir: str) -> str:
    """Return the name of the first subdirectory in the given parent directory."""
    try:
        path = Path(parent_dir)
        if not path.is_dir():
            print(f"'{parent_dir}' is not a valid directory.")
            return ""
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        subdirs.sort()
        return subdirs[0] if subdirs else ""

    except FileNotFoundError:
        print(f"Directory '{parent_dir}' not found.")
    return ""

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
                image_rgba = image.clone
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

class SceneInfo(BaseModel):
    #metadata
    save_dir: str
    pose_name: str
    girl_pos: str
    male_pos: str
    pose_json: str

    # Image Tensors (ComfyUI uses torch.Tensor with shape [B,H,W,C] for IMAGE)
    depth_image: Optional[torch.Tensor] = None
    depth_any_image: Optional[torch.Tensor] = None
    depth_midas_image: Optional[torch.Tensor] = None
    depth_zoe_image: Optional[torch.Tensor] = None
    depth_zoe_any_image: Optional[torch.Tensor] = None
    pose_dense_image: Optional[torch.Tensor] = None
    pose_dw_image: Optional[torch.Tensor] = None
    pose_edit_image: Optional[torch.Tensor] = None
    pose_face_image: Optional[torch.Tensor] = None
    pose_open_image: Optional[torch.Tensor] = None
    canny_image: Optional[torch.Tensor] = None
    upscale_image: Optional[torch.Tensor] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

@io.comfytype(io_type="POSE_COMBO")
class PoseCombo(io.ComfyTypeIO):
    Type = str
    class Input(io.WidgetInput):
        """Pose Combo input (dropdown)."""
        Type = str
        def __init__(
            self,
            id: str,
            options: list[str] | list[int] | type[io.Enum] = None,
            display_name: str=None,
            optional=False,
            tooltip: str=None,
            lazy: bool=None,
            default: str | int | io.Enum = None,
            control_after_generate: bool=None,
            upload: io.UploadType=None,
            pose_folder: io.FolderType=None,
            remote: io.RemoteOptions=None,
            socketless: bool=None,
        ):
            if isinstance(options, type) and issubclass(options, io.Enum):
                options = [v.value for v in options]
            if isinstance(default, io.Enum):
                default = default.value
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless)
            self.multiselect = False
            self.options = options
            self.control_after_generate = control_after_generate
            self.upload = upload
            self.pose_folder = pose_folder
            self.remote = remote
            self.default: str

        def as_dict(self):
            return super().as_dict() | io.prune_dict({
                "multiselect": self.multiselect,
                "options": self.options,
                "control_after_generate": self.control_after_generate,
                **({self.upload.value: True} if self.upload is not None else {}),
                "pose_folder": self.pose_folder.value if self.pose_folder else None,
                "remote": self.remote.as_dict() if self.remote else None,
            })

    class Output(io.Output):
        def __init__(self, id: str=None, display_name: str=None, options: list[str]=None, tooltip: str=None, is_output_list=False):
            super().__init__(id, display_name, tooltip, is_output_list)
            self.options = options if options is not None else []

        @property
        def io_type(self):
            return self.options

class SelectScene(io.ComfyNode):
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

        # print(f"SelectScene: default poses_dir='{default_dir}'; default_pose='{default_pose}'; default_options={default_options}")

        return io.Schema(
            node_id="SelectScene",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("poses_dir", default=default_dir, tooltip="Directory containing pose subdirectories"),
                io.Combo.Input('selected_pose', options=default_options, default=default_pose, tooltip="Select a pose directory"),
            ],
            outputs=[
                io.Custom("ScenePipe").Output(id="scene_pipe", display_name="scene_pipe", tooltip="Scene information and images"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        poses_dir="",
        selected_pose="default_pose",
        **kwargs
    ) -> io.NodeOutput:
        print(f"SelectScene: Executing with poses_dir='{poses_dir}'; selected_pose='{selected_pose}'")
        if not poses_dir:
            poses_dir = Path(get_output_directory()) / "poses"

        if not poses_dir or not selected_pose:
            print("SelectScene: poses_dir or selected_pose is empty")
            return io.NodeOutput(None)
        
        pose_dir = os.path.join(poses_dir, selected_pose)
        print(f"SelectScene: pose_dir='{pose_dir}'")

        if not os.path.isdir(pose_dir):
            print(f"SelectScene: pose_dir '{pose_dir}' is not a valid directory")
            return io.NodeOutput(None)
        
        prompt_json_path = os.path.join(pose_dir, "prompts.json")
        prompt_data = load_prompt_json(prompt_json_path)
        print(f"SelectScene: Loaded prompt_data: {prompt_data}")
        pose_json_path = os.path.join(pose_dir, "pose.json")
        pose_json = load_json_file(pose_json_path)
        if not pose_json:
            pose_json = "[]"
        print(f"SelectScene: Loaded pose_json: {pose_json}")

        girl_pos = prompt_data.get("girl_pos", "")
        male_pos = prompt_data.get("male_pos", "")

        depth_image_path = os.path.join(pose_dir, "depth.png")
        if not os.path.exists(depth_image_path):
            print(f"SelectScene: depth_image_path '{depth_image_path}' does not exist")
        depth_image = load_image_comfyui(depth_image_path)
        print(f"SelectScene: depth_image shape: {depth_image.shape if depth_image is not None else 'None'} ")

        depth_any_image_path = os.path.join(pose_dir, "depth_any.png")
        if not os.path.exists(depth_any_image_path):
            print(f"SelectScene: depth_any_image_path '{depth_any_image_path}' does not exist")
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
        upscale_image = load_image_comfyui(upscale_image_path)

        return io.NodeOutput(ScenePipe(
            save_dir=pose_dir,
            pose_name=selected_pose,
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
        ))

class SceneOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneOutput",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input("scene_info"),
            ],
            outputs=[
                io.String.Output("save_dir", display_name="save_dir", tooltip="Directory where the scene is saved"),
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
            f"SceneOutput: scene_info.save_dir='{scene_info.save_dir}', "
            f"pose_name='{scene_info.pose_name}', "
            f"girl_pos='{scene_info.girl_pos}', "
            f"male_pos='{scene_info.male_pos}', "
            f"depth_image shape: {scene_info.depth_image.shape if scene_info.depth_image is not None else 'None'}"
        )
        return io.NodeOutput(
            scene_info.save_dir,
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

class SaveScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveScene",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("save_dir", optional=True, multiline=False, default=""),
                io.String.Input("pose_name", optional=True, multiline=False, default=""),
                io.String.Input("girl_pos", optional=True, multiline=True, default=""),
                io.String.Input("male_pos", optional=True, multiline=True, default=""),
                io.String.Input("pose_json", optional=True, multiline=True, default=""),
                io.Image.Input("depth_image", optional=True),
                io.Image.Input("depth_any_image", optional=True),
                io.Image.Input("depth_midas_image", optional=True),
                io.Image.Input("depth_zoe_image", optional=True),
                io.Image.Input("depth_zoe_any_image", optional=True),
                io.Image.Input("pose_dense_image", optional=True),
                io.Image.Input("pose_dw_image", optional=True),
                io.Image.Input("pose_edit_image", optional=True),
                io.Image.Input("pose_face_image", optional=True),
                io.Image.Input("pose_open_image", optional=True),
                io.Image.Input("canny_image", optional=True),
                io.Image.Input("upscale_image", optional=True),
            ],
            outputs=[],
            is_output_node=True,
        )        

    @classmethod
    def execute(
        cls,
        save_dir="",
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
        dest_dir = Path(save_dir) / pose_name
        print(f"SaveScene: save_dir='{save_dir}'; pose_name='{pose_name}'; dest_dir='{dest_dir}'")
        if dest_dir and not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            print(f"SaveScene: Created directory '{dest_dir}' for saving scene data")
        if pose_name == "":
            print("SaveScene: pose_name is empty, cannot save pose files")
            return io.NodeOutput(None)
        
        if depth_image is not None:
            depth_path = dest_dir / "depth.png"
            save_image_comfyui(depth_image, depth_path)
        if depth_any_image is not None:
            depth_any_path = dest_dir / "depth_any.png"
            save_image_comfyui(depth_any_image, depth_any_path)
        if depth_midas_image is not None:
            depth_midas_path = dest_dir / "depth_midas.png"
            save_image_comfyui(depth_midas_image, depth_midas_path)
        if depth_zoe_image is not None:
            depth_zoe_path = dest_dir / "depth_zoe.png"
            save_image_comfyui(depth_zoe_image, depth_zoe_path)
        if depth_zoe_any_image is not None:
            depth_zoe_any_path = dest_dir / "depth_zoe_any.png"
            save_image_comfyui(depth_zoe_any_image, depth_zoe_any_path)
        if pose_dense_image is not None:
            pose_dense_path = dest_dir / "pose_dense.png"
            save_image_comfyui(pose_dense_image, pose_dense_path)
        if pose_dw_image is not None:
            pose_dw_path = dest_dir / "pose_dw.png"
            save_image_comfyui(pose_dw_image, pose_dw_path)
        if pose_edit_image is not None:
            pose_edit_path = dest_dir / "pose_edit.png"
            save_image_comfyui(pose_edit_image, pose_edit_path)
        if pose_face_image is not None:
            pose_face_path = dest_dir / "pose_face.png"
            save_image_comfyui(pose_face_image, pose_face_path)
        if pose_open_image is not None:
            pose_open_path = dest_dir / "pose_open.png"
            save_image_comfyui(pose_open_image, pose_open_path)
        if canny_image is not None:
            canny_path = dest_dir / "canny.png"
            save_image_comfyui(canny_image, canny_path)
        if upscale_image is not None:
            upscale_path = dest_dir / "upscale.png"
            save_image_comfyui(upscale_image, upscale_path)
        if pose_json:
            pose_json_path = dest_dir / "pose.json"
            save_json_file(pose_json_path, json.loads(pose_json))
        prompts_path = dest_dir / "prompts.json"
        prompt_data = {
            "girl_pos": girl_pos,
            "male_pos": male_pos,
        }
        save_json_file(prompts_path, prompt_data)   
        return io.NodeOutput(
            ui=ui.PreviewText(f"Scene saved to '{save_dir}' with prompt='The girl {girl_pos}, The male {male_pos}'"),
        )

class ScenePipe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScenePipe",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("ScenePipe").Input(id="scene_pipe_in", display_name="scene_pipe", tooltip="Scene Pipe Input", optional=True),
                io.String.Input(id="pose_name", display_name="pose_name", tooltip="The Pose Name", optional=True, multiline=False, default=""),
                io.String.Input(id="pose_dir", display_name="pose_dir", tooltip="The Pose Directory", optional=True, multiline=False, default=""),
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
                io.String.Input(id="pose_json", display_name="pose_json", tooltip="Pose JSON", optional=True, multiline=True, default=""),
                io.Custom("DICT").Input(id="scene_dict", display_name="scene_dict", tooltip="Scene Dictionary", optional=True),
            ],
            outputs=[
                io.Custom("ScenePipe").Output(id="scene_pipe_out",display_name="scene_pipe", tooltip="Scene Pipe Output"),
                io.String.Output(id="pose_name_out", display_name="pose_name", tooltip="The Pose Name"),
                io.String.Output(id="pose_dir_out", display_name="pose_dir", tooltip="The Pose Directory"),
                io.Image.Output(id="depth_image_out", display_name="depth_image", tooltip="Depth Image"),
                io.Image.Output(id="depth_any_image_out", display_name="depth_any_image", tooltip="Depth Any Image"),
                io.Image.Output(id="depth_midas_image_out", display_name="depth_midas_image", tooltip="Depth Midas Image"),
                io.Image.Output(id="depth_zoe_image_out", display_name="depth_zoe_image", tooltip="Depth Zoe Image"),
                io.Image.Output(id="depth_zoe_any_image_out", display_name="depth_zoe_any_image", tooltip="Depth Zoe Any Image"),
                io.Image.Output(id="pose_dense_image_out", display_name="pose_dense_image", tooltip="Pose Dense Image"),
                io.Image.Output(id="pose_dw_image_out", display_name="pose_dw_image", tooltip="Pose DW Image"),
                io.Image.Output(id="pose_edit_image_out", display_name="pose_edit_image", tooltip="Pose Edit Image"),
                io.Image.Output(id="pose_face_image_out", display_name="pose_face_image", tooltip="Pose Face Image"),
                io.Image.Output(id="pose_open_image_out", display_name="pose_open_image", tooltip="Pose Open Image"),
                io.Image.Output(id="canny_image_out", display_name="canny_image", tooltip="Canny Image"),
                io.Image.Output(id="upscale_image_out", display_name="upscale_image", tooltip="Upscale Image"),
                io.String.Output(id="pose_json_out", display_name="pose_json", tooltip="Pose JSON"),
                io.Custom("DICT").Output(id="scene_dict_out", display_name="scene_dict", tooltip="Scene Dictionary"),
            ],
        )        

    @classmethod
    def execute(
        cls,
        scene_pipe_in=None,
        pose_name="",
        pose_dir="",
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
        pose_json="",
        scene_dict={"girl_pos": "", "male_pos": ""},
    ) -> io.NodeOutput:
        if pose_dir and not os.path.isdir(pose_dir):
            pose_dir = ""
            print(f"ScenePipe: pose_dir '{pose_dir}' is not a valid directory, resetting to empty")
        
        if pose_name == "":
            print("ScenePipe: pose_name is empty, cannot load pose files")

        depth_image_out = depth_image if depth_image is not None else scene_pipe_in.get("depth_image") if scene_pipe_in else None
        depth_any_image_out = depth_any_image if depth_any_image is not None else scene_pipe_in.get("depth_any_image") if scene_pipe_in else None
        depth_midas_image_out = depth_midas_image if depth_midas_image is not None else scene_pipe_in.get("depth_midas_image") if scene_pipe_in else None
        depth_zoe_image_out = depth_zoe_image if depth_zoe_image is not None else scene_pipe_in.get("depth_zoe_image") if scene_pipe_in else None
        depth_zoe_any_image_out = depth_zoe_any_image if depth_zoe_any_image is not None else scene_pipe_in.get("depth_zoe_any_image") if scene_pipe_in else None
        pose_dense_image_out = pose_dense_image if pose_dense_image is not None else scene_pipe_in.get("pose_dense_image") if scene_pipe_in else None
        pose_dw_image_out = pose_dw_image if pose_dw_image is not None else scene_pipe_in.get("pose_dw_image") if scene_pipe_in else None
        pose_edit_image_out = pose_edit_image if pose_edit_image is not None else scene_pipe_in.get("pose_edit_image") if scene_pipe_in else None
        pose_face_image_out = pose_face_image if pose_face_image is not None else scene_pipe_in.get("pose_face_image") if scene_pipe_in else None
        pose_open_image_out = pose_open_image if pose_open_image is not None else scene_pipe_in.get("pose_open_image") if scene_pipe_in else None
        canny_image_out = canny_image if canny_image is not None else scene_pipe_in.get("canny_image") if scene_pipe_in else None
        upscale_image_out = upscale_image if upscale_image is not None else scene_pipe_in.get("upscale_image") if scene_pipe_in else None
        pose_json_out = pose_json if pose_json else scene_pipe_in.get("pose_json") if scene_pipe_in else ""
        scene_dict_out = scene_dict if scene_dict else scene_pipe_in.get("scene_dict") if scene_pipe_in else {"girl_pos": "", "male_pos": ""}
        pose_name = pose_name if pose_name else scene_pipe_in.get("pose_name") if scene_pipe_in else ""
        pose_dir = pose_dir if pose_dir else scene_pipe_in.get("pose_dir") if scene_pipe_in else ""

        scene_pipe_out = {
            "pose_name": pose_name,
            "pose_dir": pose_dir,
            "depth_image": depth_image_out,
            "depth_any_image": depth_any_image_out,
            "depth_midas_image": depth_midas_image_out,
            "depth_zoe_image": depth_zoe_image_out,
            "depth_zoe_any_image": depth_zoe_any_image_out,
            "pose_dense_image": pose_dense_image_out,
            "pose_dw_image": pose_dw_image_out,
            "pose_edit_image": pose_edit_image_out,
            "pose_face_image": pose_face_image_out,
            "pose_open_image": pose_open_image_out,
            "canny_image": canny_image_out,
            "upscale_image": upscale_image_out,
            "pose_json": pose_json_out,
            "scene_dict": scene_dict_out,
        }
        
        return io.NodeOutput(
            scene_pipe_out,
            pose_name,
            pose_dir,
            depth_image_out,
            depth_any_image_out,
            depth_midas_image_out,
            depth_zoe_image_out,
            depth_zoe_any_image_out,
            pose_dense_image_out,
            pose_dw_image_out,
            pose_edit_image_out,
            pose_face_image_out,
            pose_open_image_out,
            canny_image_out,
            upscale_image_out,
            pose_json_out,
            scene_dict_out,
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
            ScenePipe,
            SaveScene,
            SceneOutput,
            SelectScene,
            OpaqueAlpha,
            TailSplit,
            TailEnhancePro,
        ]