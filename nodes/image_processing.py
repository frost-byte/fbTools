"""Generic image-processing nodes, extracted from extension.py (Plan 24).

SAMPreprocessNHWC, TailEnhancePro, TailSplit, OpaqueAlpha, MaskProcessor — unrelated to the
composition-engine or nodes/narrative/ systems, pure pixel-pipeline utilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .shared import prefixed_node_id
from ..utils.images import (
    _HAS_KORNIA,
    _HAS_SKIMAGE,
    _HAS_CV2,
    _compute_ref_stats,
    _pick_ref_image,
    proc_deflicker_luma,
    proc_deflicker_clahe,
    proc_color_histmatch,
    proc_color_meanstd,
    proc_bilateral_cv2,
    proc_unsharp,
    _stack_if_same_shape,
    mask_remove_holes,
    mask_grow,
    mask_gaussian_blur,
    mask_smooth,
    create_mask_overlay_image,
    smooth_masks_region_was,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


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
            display_name="SAM Preprocess NHWC",
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

        logger.debug("SAMPreprocessNHWC: image in shape=%s", input_image.shape)
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
        logger.info(info)
        return io.NodeOutput(img, info)

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
            display_name="Tail Enhance Pro",
            category="🧊 frost-byte/Video",
            inputs=[
                io.Image.Input("input_frames", tooltip="Input IMAGE frames to process"),
                io.Int.Input("tail_count", default=6, min=1, max=999,
                    tooltip="Number of frames from the end of the sequence to enhance."),
                io.Int.Input("ref_window", default=24, min=1, max=999,
                    tooltip="Number of frames used as the reference for colour and brightness matching."),
                io.Boolean.Input("ref_from_head", default=True,
                    tooltip="Use the head (start) of the sequence as reference. If False, uses the tail frames themselves."),
                # Deflicker
                io.Boolean.Input("enable_deflicker", default=True,
                    tooltip="Reduce frame-to-frame brightness flicker in the tail frames."),
                io.Combo.Input("deflicker_mode", options=["luma_scale", "clahe"], default="luma_scale",
                    tooltip="Deflicker method: luma_scale adjusts overall brightness; clahe redistributes local contrast."),
                io.Float.Input("deflicker_strength", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Deflicker strength (0=off, 1=full correction)."),
                io.Float.Input("clahe_clip_limit", default=2.0, min=0.1, max=10.0, step=0.1,
                    tooltip="CLAHE clip limit — controls contrast enhancement strength. Only used in CLAHE deflicker mode."),
                io.Int.Input("clahe_grid_w", default=8, min=2, max=64,
                    tooltip="CLAHE tile grid width. Only used in CLAHE deflicker mode."),
                io.Int.Input("clahe_grid_h", default=8, min=2, max=64,
                    tooltip="CLAHE tile grid height. Only used in CLAHE deflicker mode."),
                # Color
                io.Boolean.Input("enable_color_match", default=True,
                    tooltip="Match the colour distribution of tail frames to the reference window."),
                io.Combo.Input("color_mode", options=["histogram", "meanstd"], default="histogram",
                    tooltip="Colour matching algorithm: histogram matches full distribution; meanstd matches mean and standard deviation."),
                io.Float.Input("color_amount", default=0.6, min=0.0, max=1.0, step=0.05,
                    tooltip="Colour match blend amount (0=no correction, 1=full match to reference)."),
                # Sharpen
                io.Boolean.Input("enable_unsharp", default=True,
                    tooltip="Apply unsharp masking to sharpen edges in the tail frames."),
                io.Float.Input("unsharp_radius", default=1.5, min=0.1, max=10.0, step=0.1,
                    tooltip="Gaussian blur radius used to compute the unsharp mask."),
                io.Float.Input("unsharp_amount", default=0.5, min=0.0, max=3.0, step=0.05,
                    tooltip="Unsharp mask strength — higher values produce stronger sharpening."),
                # Denoise
                io.Boolean.Input("enable_bilateral", default=False,
                    tooltip="Apply bilateral filtering to smooth noise while preserving edges."),
                io.Int.Input("bilateral_d", default=5, min=1, max=25,
                    tooltip="Pixel neighbourhood diameter for bilateral filtering. Larger values are slower."),
                io.Float.Input("bilateral_sigma_color", default=25.0, min=1.0, max=250.0, step=1.0,
                    tooltip="Bilateral filter colour sigma — larger values blend more dissimilar colours."),
                io.Float.Input("bilateral_sigma_space", default=7.0, min=1.0, max=100.0, step=1.0,
                    tooltip="Bilateral filter space sigma — larger values blend pixels from a wider spatial neighbourhood."),
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
        if input_frames is None or len(input_frames) == 0:
            return ([], None, "[TailEnhancePro] empty input")

        # ComfyUI hands this a single batched [B, H, W, C] tensor (V3's Input class has no
        # mechanism to deliver a real Python list from one upstream connection) — but everything
        # below, and _compute_ref_stats/_pick_ref_image in utils/images.py, are written for a
        # genuine List[torch.Tensor] of single-frame [1, H, W, C] tensors (per this node's own
        # docstring: "Split last K frames of a LIST[IMAGE]"). Convert once, up front, rather than
        # rewriting the list-based logic below to handle a tensor.
        if isinstance(input_frames, torch.Tensor):
            input_frames = [input_frames[i:i + 1] for i in range(input_frames.shape[0])]

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

        return io.NodeOutput(out_frames, batched, msg)

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
            display_name="Tail Split",
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

        if debug:
            logger.debug(
                "fbTools -> TailSplit: image in shape=%s, tail_size=%s, dtype=%s, device=%s",
                image.shape,
                tail_size,
                image.dtype,
                image.device,
            )
        b, h, w, c = image.shape
        if debug:
            logger.debug("fbTools -> TailSplit: b=%s, h=%s, w=%s, c=%s", b, h, w, c)
        
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
            logger.debug(msg)

        return io.NodeOutput(main_image, tail_image, msg)

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
            display_name="Opaque Alpha",
            category="🧊 frost-byte/Image Processing",
            inputs=[
                io.Image.Input("image", tooltip="Input image, RGB or RGBA"),
                io.Float.Input("alpha_value", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Alpha value to set in the mask"),
                io.Boolean.Input("force_replace_alpha", default=True, tooltip="If true, will replace existing alpha channel if input image is RGBA"),
                io.Boolean.Input("debug", default=False, tooltip="If true, will print debug info to console"),
            ],
            outputs=[
                io.Image.Output("image_rgba", tooltip="Output image with RGBA channels"),
                io.Mask.Output("mask", tooltip="Opaque alpha mask"),
                io.String.Output("debug_info", tooltip="Debug information"),
            ],
        )

    @classmethod
    def execute(cls, image, alpha_value=1.0, force_replace_alpha=True, debug=False):
        # image: torch.FloatTensor [B, H, W, C], C=3 or 4, float 0..1
        if not torch.is_tensor(image):
            raise ValueError("Input 'image' must be a torch tensor")

        if debug:
            logger.debug(
                "OpaqueAlpha: image in shape=%s, alpha_value=%s, force_replace_alpha=%s, dtype=%s, device=%s",
                image.shape,
                alpha_value,
                force_replace_alpha,
                image.dtype,
                image.device,
            )
        b, h, w, c = image.shape
        if debug:
            logger.debug("OpaqueAlpha: b=%s, h=%s, w=%s, c=%s", b, h, w, c)
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
            
        # MASK convention is [B, H, W] (no trailing channel dim) — mask itself stays [B, H, W, 1]
        # above since that's what the RGBA-building math needs (cat/assign against a 4-channel
        # image), only the returned value is squeezed to match.
        mask_out = mask.squeeze(-1)

        msg = (
            f"OpaqueAlpha: image in shape={image.shape}, alpha_value={alpha_value}, force_replace_alpha={force_replace_alpha},dtype={image.dtype}, device={image.device}, "
            f"range=[{mn:.6f},{mx:.6f}] -> image out shape={image_rgba.shape}, mask shape={mask_out.shape}{alpha_summary}"
        )

        if debug:
            logger.debug(msg)

        return io.NodeOutput(image_rgba, mask_out, msg)

class MaskProcessor(io.ComfyNode):
    """
    Processes a mask or batch of masks by applying a sequence of refinement operations:
    1. Remove holes - fills interior holes smaller than threshold
    2. Grow - dilates mask borders
    3. Smooth - applies morphological smoothing
    4. Region smooth - applies Gaussian filter with thresholding (WAS method)
    5. Gaussian blur - softens edges (last step for best blending)
    
    If an image is provided, creates an overlay image where the masked area
    becomes transparent (doesn't retain original colors).
    
    Takes the first mask from batch if multiple masks provided.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("MaskProcessor"),
            display_name="Mask Processor",
            category="🧊 frost-byte/Image Processing",
            inputs=[
                io.Mask.Input("input_mask", tooltip="Input mask or batch of masks"),
                io.Image.Input("image", optional=True, tooltip="Optional: Input image to create overlay with transparent masked area"),
                io.Int.Input("min_hole_size", default=10, min=0, max=10000, step=1, 
                            tooltip="Minimum hole size (in pixels) to fill. Holes smaller than this will be filled."),
                io.Int.Input("grow_amount", default=5, min=0, max=100, step=1,
                            tooltip="Amount to grow (dilate) the mask borders in pixels"),
                io.Int.Input("smooth_iterations", default=0, min=0, max=10, step=1,
                            tooltip="Number of morphological smoothing iterations (can shrink mask)"),
                io.Boolean.Input("enable_region_smooth", default=True, tooltip="Enable region smoothing (Gaussian filter with thresholding - maintains mask size)"),
                io.Int.Input("region_smooth_sigma", default=128, min=1, max=512, step=1,
                            tooltip="Sigma for region smoothing (only used if enabled)"),
                io.Float.Input("blur_radius", default=5.0, min=0.0, max=50.0, step=0.1,
                              tooltip="Gaussian blur radius (sigma value) for edge softening"),
                io.Boolean.Input("debug", default=False, tooltip="Print debug information"),
            ],
            outputs=[
                io.Mask.Output("mask", tooltip="Processed mask"),
                io.Image.Output("overlay_image", tooltip="Image with transparent masked area (if image input provided)"),
                io.String.Output("debug_info", tooltip="Processing information"),
            ],
        )

    @classmethod
    def execute(cls, input_mask, image=None, min_hole_size=10, grow_amount=5, smooth_iterations=2, 
                enable_region_smooth=False, region_smooth_sigma=128, blur_radius=5.0, debug=False):
        
        if not torch.is_tensor(input_mask):
            raise ValueError("Input 'mask' must be a torch tensor")
        
        # Handle batch: select first mask
        if input_mask.dim() == 3:  # [B, H, W]
            mask_single = input_mask[0]  # [H, W]
        elif input_mask.dim() == 2:  # [H, W]
            mask_single = input_mask
        else:
            raise ValueError(f"Expected mask with shape [B, H, W] or [H, W], got {input_mask.shape}")
        
        if debug:
            logger.debug(
                "MaskProcessor: Input shape=%s, selected shape=%s",
                input_mask.shape,
                mask_single.shape,
            )
            logger.debug(
                "MaskProcessor: Parameters - min_hole_size=%s, grow_amount=%s, smooth_iterations=%s, "
                "enable_region_smooth=%s, region_smooth_sigma=%s, blur_radius=%s",
                min_hole_size,
                grow_amount,
                smooth_iterations,
                enable_region_smooth,
                region_smooth_sigma,
                blur_radius,
            )
        
        # Apply operations in sequence
        processed = mask_single
        operations = []
        
        # 1. Remove holes
        if min_hole_size > 0:
            processed = mask_remove_holes(processed, min_hole_size=min_hole_size)
            operations.append(f"remove_holes(min_size={min_hole_size})")
            if debug:
                logger.debug("MaskProcessor: After remove_holes - shape=%s", processed.shape)
        
        # 2. Grow (dilate)
        if grow_amount > 0:
            processed = mask_grow(processed, grow_amount=grow_amount)
            operations.append(f"grow(amount={grow_amount})")
            if debug:
                logger.debug("MaskProcessor: After grow - shape=%s", processed.shape)
        
        # 3. Smooth (morphological cleanup)
        if smooth_iterations > 0:
            processed = mask_smooth(processed, smooth_iterations=smooth_iterations)
            operations.append(f"smooth(iterations={smooth_iterations})")
            if debug:
                logger.debug("MaskProcessor: After smooth - shape=%s", processed.shape)
        
        # 4. Region smooth (Gaussian with thresholding - WAS method)
        if enable_region_smooth:
            # Need to add batch dim temporarily for smooth_masks_region_was
            if processed.dim() == 2:
                processed_batch = processed.unsqueeze(0)
            else:
                processed_batch = processed
            processed_batch = smooth_masks_region_was(processed_batch, sigma=region_smooth_sigma)
            # Extract single mask again
            processed = processed_batch[0] if processed_batch.dim() == 3 else processed_batch
            operations.append(f"region_smooth(sigma={region_smooth_sigma})")
            if debug:
                logger.debug("MaskProcessor: After region_smooth - shape=%s", processed.shape)
        
        # 5. Gaussian blur (LAST - creates soft edges for blending)
        if blur_radius > 0.0:
            processed = mask_gaussian_blur(processed, blur_radius=blur_radius)
            operations.append(f"gaussian_blur(radius={blur_radius})")
            if debug:
                logger.debug("MaskProcessor: After gaussian_blur - shape=%s", processed.shape)
        
        # Ensure output is 3D [B, H, W] for compatibility
        if processed.dim() == 2:
            processed = processed.unsqueeze(0)
        
        operations_str = " -> ".join(operations) if operations else "no operations"
        debug_info = f"MaskProcessor: Applied operations: {operations_str}. Output shape: {processed.shape}"
        
        # Create overlay image if input image provided
        overlay_image = None
        if image is not None:
            try:
                overlay_image = create_mask_overlay_image(processed, image)
                if debug:
                    logger.debug("MaskProcessor: Created overlay_image with shape %s", overlay_image.shape)
            except Exception as e:
                logger.exception("MaskProcessor: Error creating overlay image")
                # Create placeholder RGBA image on error
                h, w = processed.shape[1], processed.shape[2]
                overlay_image = torch.zeros((1, h, w, 4), dtype=torch.float32, device=processed.device)
        else:
            # No image provided - create placeholder RGBA image
            h, w = processed.shape[1], processed.shape[2]
            overlay_image = torch.zeros((1, h, w, 4), dtype=torch.float32, device=processed.device)
        
        if debug:
            logger.debug(debug_info)
        
        # Return io.NodeOutput with positional args matching OUTPUT_TYPES order: mask, overlay_image, debug_info
        return io.NodeOutput(processed, overlay_image, debug_info)
