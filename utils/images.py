
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import comfy.utils
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
import numpy as np
from skimage.exposure import match_histograms
from nodes import MAX_RESOLUTION
import node_helpers

try:
    import kornia
    import kornia.enhance as KE
    import kornia.filters as KF
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

STANDARD_ASPECT_RATIOS = {
    "1:1": 1.0,
    "2:3": 2/3,
    "3:4": 3/4,
    "3:5": 3/5,
    "9:16": 9/16,
    "9:21": 9/21,
    "21:9": 21/9,
    "16:9": 16/9,
    "5:3": 5/3,
    "4:3": 4/3,
    "3:2": 3/2,
}

# divisible by 16
QWEN_ASPECT_RATIOS = {
    "9:21": (672, 1568),
    "9:20": (688, 1504),
    "9:18": (720, 1452), # chinese math is interesting...
    "3:5": (800, 1328),
    "1:1": (1024, 1024),
    "4:5": (704, 880),
    "3:4": (864, 1152),
    "2:3": (832, 1248), # supposedly the preferred aspect ratio for qwen
    "9:16": (720, 1280),
    "16:9": (1280, 720),
    "5:4": (880, 704),
    "4:3": (1152, 864),
    "3:2": (1248, 832),
}

def find_nearest_qwen_aspect_ratio(width: int, height: int) -> Tuple[int, int, str, str, float]:
    round_to = 16
    w_rounded = round_to * round(width / round_to)
    h_rounded = round_to * round(height / round_to)
    input_ratio = w_rounded / h_rounded
    
    best_match = min(
        STANDARD_ASPECT_RATIOS.items(),
        key=lambda x: abs(x[1] - input_ratio)
    )
    matched_name = best_match[0]
    matched_ratio = best_match[1]
    qwen_x, qwen_y = QWEN_ASPECT_RATIOS[matched_name]

    layout = "landscape" if input_ratio > 1 else "portrait" if input_ratio < 1 else "square"

    return (qwen_x, qwen_y, layout, matched_name, matched_ratio)

def make_empty_image(batch=1, height=64, width=64, channels=3):
    """
    Returns a blank image tensor in ComfyUI format:
    - Shape: [B, H, W, C]
    - Dtype: torch.float32
    - Values: 0.0 (black) in range [0.0, 1.0]
    """
    return torch.zeros((batch, height, width, channels), dtype=torch.float32)

def make_placeholder_tensor(h, w, channels=3, color=(0, 0, 0)):
    """Create a [1, H, W, C] tensor filled with a solid color (default: black)."""
    arr = np.full((h, w, channels), color, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]

def normalize_image_tensor(img_tensor, target_h, target_w):
    """Ensure img_tensor is [1, target_h, target_w, 3]. Resize or replace if needed."""
    if img_tensor is None:
        return make_placeholder_tensor(target_h, target_w)

    # Handle batch dim
    if img_tensor.ndim == 4:
        B, H, W, C = img_tensor.shape
        if B != 1:
            img_tensor = img_tensor[:1]  # take first
        if H == target_h and W == target_w and C in (1, 3, 4):
            # Normalize channels to 3 (if grayscale, repeat)
            if C == 1:
                img_tensor = img_tensor.repeat(1, 1, 1, 3)
            elif C == 4:
                img_tensor = img_tensor[..., :3]
            return img_tensor
        else:
            # Optional: resize using PIL (preserves aspect ratio or not)
            pil_img = Image.fromarray(
                np.clip(img_tensor[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            )
            # Resize to exact target (distorts if needed)
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized = np.array(pil_img).astype(np.float32) / 255.0
            if resized.ndim == 2:
                resized = np.stack([resized] * 3, axis=-1)
            return torch.from_numpy(resized).unsqueeze(0)
    else:
        # Unexpected shape — use placeholder
        print(f"Warning: unexpected tensor shape {img_tensor.shape}, using placeholder")
    
    return make_placeholder_tensor(target_h, target_w)

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

def generate_thumbnail(image_tensor, save_path, size=(128, 128)):
    """Generate and save a thumbnail from ComfyUI IMAGE tensor [1,H,W,C] float32 0..1."""
    print(f"fbTools: generate_thumbnail: '{save_path}'; shape {image_tensor.shape}")
    
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must be 4D [B,H,W,C]")
    
    B, H, W, C = image_tensor.shape
    if B != 1:
        print(f"image_tensor batch size > 1; using only first image of {B}")
        image_tensor = image_tensor[0:1]
    
    try:
        # Convert tensor to PIL Image
        img_np = (image_tensor[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(img_np)
        
        # Create thumbnail (maintains aspect ratio and fits within size)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save thumbnail
        print(f"Saving thumbnail to '{save_path}' with size {img.size}")
        img.save(save_path, format='PNG', compress_level=4)
    except Exception as e:
        print(f"Error generating thumbnail to '{save_path}': {e}")

def load_image_comfyui(image_path: str, include_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an image and (optionally) its alpha mask.

    include_mask=True extracts the alpha channel (or palette transparency) into a mask tensor.
    Returns `(image, mask)` where image is [B,H,W,C] float32 0..1 and mask is [B,H,W] float32.
    When include_mask is False or alpha is missing, mask is zero.
    """
    output_images: list[torch.Tensor] = []
    output_masks: list[torch.Tensor] = []
    w, h = None, None
    excluded_formats = ['MPO']
    output_image = make_empty_image()
    output_mask = torch.zeros((1, output_image.shape[1], output_image.shape[2]), dtype=torch.float32)

    try:
        img = node_helpers.pillow(Image.open, image_path)
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i is None:
                continue

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert('RGB')

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0  # [H,W,C] float32 0..1
            image = torch.from_numpy(image)[None,]  # [1,H,W,C]

            if include_mask:
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0  # [H,W] float32 0..1
                    mask = 1. - torch.from_numpy(mask)
                elif i.mode == 'P' and 'transparency' in i.info:
                    mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0  # [H,W] float32 0..1
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")  # [H,W] all 0.0
                output_masks.append(mask.unsqueeze(0))  # [1,H,W]

            output_images.append(image)

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)  # [B,H,W,C]
                if include_mask:
                    output_mask = torch.cat(output_masks, dim=0)  # [B,H,W]
            else:
                output_image = output_images[0]  # [1,H,W,C]
                if include_mask:
                    output_mask = output_masks[0]
                else:
                    output_mask = torch.zeros((1, output_image.shape[1], output_image.shape[2]), dtype=torch.float32, device="cpu")  # [1,H,W]
    except (FileNotFoundError, OSError, AttributeError):
        output_image = make_empty_image()
        output_mask = torch.zeros((1, output_image.shape[1], output_image.shape[2]), dtype=torch.float32)

    return (output_image, output_mask)

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

    k = KF.gaussian_blur2d(x, kernel_size=int(2*round(3*sigma)+1), sigma=(sigma, sigma))
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

def _pick_ref_image(frames: List[torch.Tensor], window: int) -> Optional[torch.Tensor]:
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
    blur = KF.gaussian_blur2d(x, kernel_size=ksize, sigma=(sigma, sigma))
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
        return subdirs[0].name if subdirs else ""

    except FileNotFoundError:
        print(f"Directory '{parent_dir}' not found.")
    return ""

def image_resize_ess(
    image,
    width,
    height,
    method="stretch",
    interpolation="nearest",
    condition="always",
    multiple_of=0,
    keep_proportion=False
):
    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    if keep_proportion:
        method = "keep proportion"

    if multiple_of > 1:
        width = width - (width % multiple_of)
        height = height - (height % multiple_of)

    if method == 'keep proportion' or method == 'pad':
        if width == 0 and oh < height:
            width = MAX_RESOLUTION
        elif width == 0 and oh >= height:
            width = ow

        if height == 0 and ow < width:
            height = MAX_RESOLUTION
        elif height == 0 and ow >= width:
            height = oh

        ratio = min(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)

        if method == 'pad':
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        width = new_width
        height = new_height
    elif method.startswith('fill'):
        width = width if width > 0 else ow
        height = height if height > 0 else oh

        ratio = max(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        x2 = x + width
        y2 = y + height
        if x2 > new_width:
            x -= (x2 - new_width)
        if x < 0:
            x = 0
        if y2 > new_height:
            y -= (y2 - new_height)
        if y < 0:
            y = 0
        width = new_width
        height = new_height
    else:
        width = width if width > 0 else ow
        height = height if height > 0 else oh

    if "always" in condition \
        or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
        or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

        outputs = image.permute(0,3,1,2)

        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if method == 'pad':
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0,2,3,1)

        if method.startswith('fill'):
            if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                outputs = outputs[:, y:y2, x:x2, :]
    else:
        outputs = image

    if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
        width = outputs.shape[2]
        height = outputs.shape[1]
        x = (width % multiple_of) // 2
        y = (height % multiple_of) // 2
        x2 = width - ((width % multiple_of) - x)
        y2 = height - ((height % multiple_of) - y)
        outputs = outputs[:, y:y2, x:x2, :]
    
    outputs = torch.clamp(outputs, 0, 1)

    return outputs


def mask_remove_holes(mask: torch.Tensor, min_hole_size: int = 10) -> torch.Tensor:
    """
    Remove holes from a binary mask using morphological operations.
    
    Args:
        mask: Input mask tensor [H, W] with values 0-1
        min_hole_size: Minimum size of holes to remove (in pixels)
    
    Returns:
        Mask with holes filled
    """
    if not _HAS_CV2:
        print("mask_remove_holes: opencv not available, returning original mask")
        return mask
    
    # Convert to numpy and ensure binary
    mask_np = mask.cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
    
    # Find contours of holes (inverted mask)
    inverted = cv2.bitwise_not(mask_binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fill holes smaller than threshold
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_hole_size:
            cv2.drawContours(mask_binary, [contour], -1, 255, -1)
    
    # Convert back to tensor
    result = torch.from_numpy(mask_binary.astype(np.float32) / 255.0).to(mask.device)
    return result


def mask_grow(mask: torch.Tensor, grow_amount: int = 5) -> torch.Tensor:
    """
    Grow (dilate) a mask by expanding its borders.
    
    Args:
        mask: Input mask tensor [H, W] with values 0-1
        grow_amount: Number of pixels to grow the mask
    
    Returns:
        Dilated mask
    """
    if not _HAS_CV2:
        print("mask_grow: opencv not available, returning original mask")
        return mask
    
    if grow_amount <= 0:
        return mask
    
    # Convert to numpy
    mask_np = mask.cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
    
    # Create circular kernel for dilation
    kernel_size = grow_amount * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    
    # Convert back to tensor
    result = torch.from_numpy(dilated.astype(np.float32) / 255.0).to(mask.device)
    return result


def mask_gaussian_blur(mask: torch.Tensor, blur_radius: float = 5.0) -> torch.Tensor:
    """
    Apply Gaussian blur to a mask for soft edges.
    
    Args:
        mask: Input mask tensor [H, W] with values 0-1
        blur_radius: Radius of Gaussian blur (sigma value)
    
    Returns:
        Blurred mask
    """
    if not _HAS_CV2:
        print("mask_gaussian_blur: opencv not available, returning original mask")
        return mask
    
    if blur_radius <= 0:
        return mask
    
    # Convert to numpy
    mask_np = mask.cpu().numpy()
    
    # Kernel size should be odd and roughly 6*sigma
    kernel_size = int(blur_radius * 6)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)  # Minimum size of 3
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), blur_radius)
    
    # Convert back to tensor
    result = torch.from_numpy(blurred.astype(np.float32)).to(mask.device)
    return result


def mask_smooth(mask: torch.Tensor, smooth_iterations: int = 2) -> torch.Tensor:
    """
    Smooth a mask using morphological operations (opening then closing).
    Preserves grayscale values for soft edges.
    
    Args:
        mask: Input mask tensor [H, W] with values 0-1
        smooth_iterations: Number of smoothing iterations
    
    Returns:
        Smoothed mask
    """
    if not _HAS_CV2:
        print("mask_smooth: opencv not available, returning original mask")
        return mask
    
    if smooth_iterations <= 0:
        return mask
    
    # Convert to numpy (preserve grayscale)
    mask_np = mask.cpu().numpy()
    
    # Create small circular kernel for smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply morphological opening then closing for smoothing
    # Work with grayscale values to preserve soft edges
    for _ in range(smooth_iterations):
        # Opening: erosion followed by dilation (removes small bright spots)
        opened = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        # Closing: dilation followed by erosion (fills small dark spots)
        mask_np = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to tensor (no binarization - preserves soft edges)
    result = torch.from_numpy(mask_np.astype(np.float32)).to(mask.device)
    return result

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

def smooth_masks_region_was(masks: torch.Tensor, sigma=128) -> torch.Tensor:
    """
    Smooth mask region using Gaussian filter and thresholding.
    
    Args:
        masks: Input mask tensor [B, H, W] or [H, W]
        sigma: Gaussian filter sigma value
    
    Returns:
        Smoothed mask tensor [1, H, W] (3D for ComfyUI compatibility)
    """
    if masks.dim() == 3:
        mask_single = masks[0]
    else:
        mask_single = masks
    
    from PIL import Image as PILImage, ImageOps
    
    # Convert to numpy uint8 for PIL
    mask_np = np.clip(255. * mask_single.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    pil_image = PILImage.fromarray(mask_np, mode='L')
    
    # Apply smoothing
    region_mask = smooth_region_was(pil_image, tolerance=sigma)
    
    # Convert back to tensor [H, W] and add batch dimension [1, H, W]
    region_tensor = pil2mask(region_mask).unsqueeze(0)
    
    return region_tensor.to(masks.device)

def smooth_region_was(image, tolerance):
    """
    Apply Gaussian smoothing and thresholding to a PIL image.
    
    Args:
        image: PIL Image
        tolerance: Gaussian filter sigma value
    
    Returns:
        PIL Image (L mode) with smoothed and thresholded mask, inverted
    """
    from scipy.ndimage import gaussian_filter
    
    image = image.convert("L")
    mask_array = np.array(image)
    smoothed_array = gaussian_filter(mask_array, sigma=tolerance)
    threshold = np.max(smoothed_array) / 2
    smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
    smoothed_image = Image.fromarray(smoothed_mask, mode="L")
    
    # Return inverted L mode image (no need to convert to RGB)
    return ImageOps.invert(smoothed_image)

def create_mask_overlay_image(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Create an RGBA image where the masked area is made fully transparent (alpha=0).
    This removes color information from the masked region for inpainting workflows.
    
    Args:
        mask: Mask tensor [H, W] or [B, H, W] with values 0-1 (1=masked, 0=unmasked)
        image: Image tensor [H, W, C] or [B, H, W, C] with values 0-1
    
    Returns:
        RGBA image tensor [1, H, W, 4] where:
        - Masked areas: alpha=0 (transparent), RGB=0 (black, but invisible)
        - Unmasked areas: alpha=1 (opaque), RGB=original image colors
    """
    from PIL import Image as PILImage
    
    # Get first from batch if needed
    if mask.dim() == 3:  # [B, H, W]
        mask_single = mask[0]
    else:
        mask_single = mask
    
    if image.dim() == 4:  # [B, H, W, C]
        img_single = image[0]
    else:
        img_single = image
    
    # Convert mask to numpy (0-1 float)
    mask_np = mask_single.cpu().numpy()  # [H, W]
    
    # Convert image to numpy (0-1 float)
    img_np = img_single.cpu().numpy()  # [H, W, C]
    
    # Create RGBA output: [H, W, 4]
    h, w = mask_np.shape
    overlay_np = np.zeros((h, w, 4), dtype=np.float32)  # Initialize to all 0.0 (black, transparent)
    
    # Set RGB values: black (0.0) where masked, original image where not masked
    # mask_np is 1.0 where masked, 0.0 where not masked
    for c in range(3):  # RGB channels
        overlay_np[:, :, c] = np.where(mask_np > 0.5, 0.0, img_np[:, :, c])
    
    # Set alpha channel: transparent (0.0) where masked, opaque (1.0) where not masked
    # This removes color information from the masked region
    overlay_np[:, :, 3] = np.where(mask_np > 0.5, 0.0, 1.0)
    
    # Convert to tensor [1, H, W, 4]
    overlay_image = torch.from_numpy(overlay_np).unsqueeze(0).to(image.device)
    
    return overlay_image