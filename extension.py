from __future__ import annotations

from PIL import Image
import node_helpers
import math
from comfy.utils import common_upscale
import folder_paths
from folder_paths import get_input_directory, get_output_directory
import comfy.model_management as model_management

from typing_extensions import override
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

from .utils.images import _HAS_KORNIA, _HAS_SKIMAGE, _HAS_CV2, load_image_comfyui, save_image_comfyui, make_placeholder_tensor, normalize_image_tensor, generate_thumbnail
from comfy_api.latest import ComfyExtension, io, ui
from inspect import cleandoc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import os
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import re
import copy
import hashlib
from pydantic import BaseModel, ConfigDict
from .utils.logging_utils import get_logger
from .story_models import SceneInStory, StoryInfo, save_story, load_story
from .captioner import caption_image, get_model, unload_model
from .utils.concept_registry import (
    ConceptRegistry,
    load_registry as _load_concept_registry,
    save_registry as _save_concept_registry,
    resolve_concepts as _resolve_concepts,
    assemble_prompt as _assemble_concept_prompt,
    format_resolved_info as _format_resolved_info,
    parse_concept_ids as _parse_concept_ids,
    build_model_entry as _build_model_entry,
    MODEL_PROFILES as _CONCEPT_MODEL_PROFILES,
    MODEL_TYPE_IDS as _CONCEPT_MODEL_TYPE_IDS,
)
from .utils.subject_profiles import (
    SubjectRegistry,
    load_registry as _load_subject_registry,
    save_registry as _save_subject_registry,
    SUPPORTED_LANGUAGES as _SUBJECT_LANGUAGES,
)
from .utils.scene_templates import (
    SceneTemplate,
    load_template as _load_scene_template,
    scan_templates as _scan_scene_templates,
    template_ids as _scene_template_ids,
    format_template_list as _format_template_list,
    dir_fingerprint as _templates_dir_fingerprint,
)

from .utils.subject_compositor import (
    tensor_to_pil,
    pil_to_tensor,
    parse_color,
    process_layer,
    compute_paste_position,
    composite_onto_canvas,
    snap_to_divisible,
)

logger = get_logger(__name__)

# Status update helper for real-time node feedback
def send_status_update(
    node_id: str,
    status_text: str,
    source: str | None = None,
    level: str = "info",
):
    """Send status update to frontend via websocket."""
    try:
        from server import PromptServer
        server = PromptServer.instance
        payload = {
            "node": node_id,
            "status": status_text,
            "level": level,
        }
        if source:
            payload["source"] = source
        server.send_sync("fbtools.status", payload)
    except Exception as e:
        logger.debug(f"Failed to send status update: {e}")

try:
    from westNeighbor_comfyui_ultimate_openpose_editor.openpose_editor_nodes import OpenposeEditorNode  # type: ignore
except Exception:
    OpenposeEditorNode = None


OpenposeJSON = dict

# Extension-wide node prefix to keep node_id globally unique across ComfyUI
EXTENSION_PREFIX = "fbt"

# Incremented by POST /fbtools/concepts/reload so ConceptRegistryLoad re-executes
_concept_reload_counter: int = 0
# Incremented by POST /fbtools/subjects/reload so SubjectProfileLoad re-executes
_subject_reload_counter: int = 0
# Incremented by POST /fbtools/scene_templates/reload so SceneTemplate nodes re-execute
_scene_template_reload_counter: int = 0

def prefixed_node_id(display_name: str) -> str:
    """Construct a globally-unique node_id using the shared extension prefix."""
    return f"{EXTENSION_PREFIX}_{display_name}"


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

DEFAULT_INSTRUCTION = (
    "Describe this image in precise factual detail. "
    "Include: subject appearance (face shape, eye colour, hair style and colour, "
    "skin tone, any distinguishing features), clothing and accessories worn, "
    "pose and body language, background and setting, and lighting conditions. "
    "Write in flowing prose. Do not use subjective quality descriptors."
)

CAPTIONER_OPTIONS = ["qwen_vl", "qwen_omni", "gemini_flash"]
DEVICE_OPTIONS    = ["auto", "cuda", "cpu"]
DATASET_CAPTION_STATUS_ID = prefixed_node_id("DatasetCaptioner")

# ── Shared helpers ────────────────────────────────────────────────────────────

def _collect_images(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")
    )


def _txt_path(image_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return image_path.with_suffix(".txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (image_path.stem + ".txt")


def _read_caption(image_path: Path, output_dir: Path | None) -> str:
    txt = _txt_path(image_path, output_dir)
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def _write_caption(image_path: Path, caption: str, output_dir: Path | None) -> None:
    _txt_path(image_path, output_dir).write_text(caption.strip(), encoding="utf-8")


def _resolve_relative_to(base_dir: str, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _resolve_dataset_input_directory(raw_path: str, field_name: str = "input_directory") -> Path:
    cleaned = (raw_path or "").strip()
    if not cleaned:
        raise ValueError(
            f"{field_name} is required. Provide an absolute path or a path relative to Comfy input directory."
        )
    return _resolve_relative_to(get_input_directory(), cleaned)


def _resolve_dataset_output_directory(raw_path: str) -> Path | None:
    cleaned = (raw_path or "").strip()
    if not cleaned:
        return None
    return _resolve_relative_to(get_output_directory(), cleaned)


def _directory_fingerprint(path: Path) -> tuple[str, int, int]:
    """Return a stable fingerprint for a directory tree.

    The hash includes relative paths plus mtime_ns/size for every file and directory,
    so edits, creates, deletes, and renames will invalidate cached nodes.
    """
    if not path.exists() or not path.is_dir():
        return ("missing", 0, 0)

    digest = hashlib.sha1()
    dir_count = 0
    file_count = 0

    for root, dirnames, filenames in os.walk(path):
        dirnames.sort()
        filenames.sort()

        root_path = Path(root)
        rel_root = root_path.relative_to(path).as_posix()
        rel_root = rel_root if rel_root else "."

        try:
            root_stat = root_path.stat()
            root_mtime_ns = int(root_stat.st_mtime_ns)
        except Exception:
            root_mtime_ns = 0

        digest.update(f"D|{rel_root}|{root_mtime_ns}\n".encode("utf-8"))
        dir_count += 1

        for filename in filenames:
            file_path = root_path / filename
            rel_file = file_path.relative_to(path).as_posix()
            try:
                st = file_path.stat()
                file_mtime_ns = int(st.st_mtime_ns)
                file_size = int(st.st_size)
            except Exception:
                file_mtime_ns = 0
                file_size = 0

            digest.update(f"F|{rel_file}|{file_mtime_ns}|{file_size}\n".encode("utf-8"))
            file_count += 1

    return (digest.hexdigest(), dir_count, file_count)

# ── Custom type: SUBJECT_LAYER ────────────────────────────────────────────────

SUBJECT_LAYER_TYPE = "SUBJECT_LAYER"


@io.comfytype(io_type=SUBJECT_LAYER_TYPE)
class SubjectLayer:
    """
    Custom type passed between SubjectLayerDefine and SubjectCompositor.
    Carries the raw image tensor plus all per-layer parameters.
    Processing (bg removal, padding, scaling) is deferred to the compositor
    so it has access to the final canvas dimensions.
    """
    Type = dict  # { image, mask, remove_background, bg_model,
                 #   pad_top, pad_bottom, pad_left, pad_right,
                 #   offset_x, offset_y }

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "layer", **kwargs):
            super().__init__(name, **kwargs)


# ── Node 1: SubjectLayerDefine ────────────────────────────────────────────────

BG_MODELS = [
    "BiRefNet-general",
    "BiRefNet-portrait",
    "BiRefNet-general-lite",
    "u2net",
    "u2net_human_seg",
    "isnet-general-use",
]

OUTPUT_MODES = ["composite", "individual", "both"]


class SubjectLayerDefine(io.ComfyNode):
    """
    Define a single subject layer for use with SubjectCompositor.

    Padding is specified as a fraction of the image's longer dimension.
    pad_top=0.2 adds 20% of max(width, height) as transparent space at the top,
    effectively making the subject appear smaller relative to other layers.

    Offset positions the subject's center relative to the canvas center.
    offset_x=0.0, offset_y=0.0 places the subject at the canvas center.
    offset_x=0.5 shifts the subject halfway toward the right edge.
    offset_x=-0.5 shifts the subject halfway toward the left edge.
    offset_y=-0.5 shifts the subject halfway toward the top edge.

    An optional mask input (ComfyUI MASK) can be supplied instead of using
    automatic background removal — useful when an upstream RMBG node is
    already in the workflow.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("SubjectLayerDefine"),
            display_name="Subject Layer Define",
            category="🧊 frost-byte/compositing",
            description=(
                "Define a subject layer for SubjectCompositor. "
                "Specify padding (as fraction of longer dimension), "
                "canvas offset (fraction of half-canvas from center), "
                "and optional background removal."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="Image",
                    tooltip="Input image containing the subject.",
                ),
                io.Mask.Input(
                    "mask",
                    display_name="Mask",
                    optional=True,
                    tooltip=(
                        "Optional pre-computed alpha mask (ComfyUI MASK, 1=keep). "
                        "If provided, overrides background removal."
                    ),
                ),
                io.Float.Input(
                    "pad_top",
                    display_name="Pad Top",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added above the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_bottom",
                    display_name="Pad Bottom",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added below the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_left",
                    display_name="Pad Left",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added to the left of the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "pad_right",
                    display_name="Pad Right",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Transparent padding added to the right of the subject, as fraction of longer dimension.",
                ),
                io.Float.Input(
                    "offset_x",
                    display_name="Offset X",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Horizontal offset of subject center relative to canvas center. "
                        "0.0=center, 1.0=right edge, -1.0=left edge."
                    ),
                ),
                io.Float.Input(
                    "offset_y",
                    display_name="Offset Y",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Vertical offset of subject center relative to canvas center. "
                        "0.0=center, 1.0=bottom edge, -1.0=top edge."
                    ),
                ),
                io.Boolean.Input(
                    "remove_background",
                    display_name="Remove Background",
                    default=True,
                    tooltip="Automatically remove the background. Ignored if a mask is connected.",
                ),
                io.Combo.Input(
                    "bg_model",
                    display_name="BG Removal Model",
                    options=BG_MODELS,
                    default="BiRefNet-general",
                    optional=True,
                    tooltip="Background removal model to use. BiRefNet-portrait works best for people.",
                ),
            ],
            outputs=[
                SubjectLayer.Output(
                    "layer",
                    display_name="Layer",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        pad_top: float = 0.0,
        pad_bottom: float = 0.0,
        pad_left: float = 0.0,
        pad_right: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        remove_background: bool = True,
        bg_model: str = "BiRefNet-general",
        mask: Optional[torch.Tensor] = None,
    ) -> io.NodeOutput:

        layer = {
            "image":             image,
            "mask":              mask,
            "remove_background": remove_background,
            "bg_model":          bg_model,
            "pad_top":           pad_top,
            "pad_bottom":        pad_bottom,
            "pad_left":          pad_left,
            "pad_right":         pad_right,
            "offset_x":          offset_x,
            "offset_y":          offset_y,
        }

        return io.NodeOutput(layer)


# ── Node 2: SubjectCompositor ─────────────────────────────────────────────────

class SubjectCompositor(io.ComfyNode):
    """
    Composite multiple subject layers onto a canvas.

    Accepts 1–20 SUBJECT_LAYER inputs (from SubjectLayerDefine).
    Layers are processed in order — layer_0 is placed first (furthest back),
    later layers are composited on top.

    Output modes:
      composite   — one image with all layers composited together
      individual  — one image per layer, each placed at its offset on its own canvas
      both        — returns both composite and individual images

    The individual images output is a batch tensor [N, H, W, 3] where N is
    the number of connected layers. Each image in the batch corresponds to
    the layer at the same index, and can be fed directly into ReferenceLatent
    or other conditioning nodes.

    The canvas dimensions are snapped to the nearest lower multiple of
    divisible_by (default 32, required by most video/image models).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:

        autogrow_template = io.Autogrow.TemplatePrefix(
            input=SubjectLayer.Input("layer", optional=True),
            prefix="layer",
            min=1,
            max=20,
        )

        return io.Schema(
            node_id=prefixed_node_id("SubjectCompositor"),
            display_name="Subject Compositor",
            category="🧊 frost-byte/compositing",
            description=(
                "Composite multiple subject layers onto a canvas. "
                "Outputs a composite image and/or individual per-subject images "
                "at the target resolution, suitable for ReferenceLatent or "
                "other multi-image conditioning nodes."
            ),
            inputs=[
                io.Int.Input(
                    "canvas_width",
                    display_name="Canvas Width",
                    default=1344,
                    min=64,
                    max=8192,
                    step=1,
                    tooltip="Output canvas width in pixels. Snapped to divisible_by.",
                ),
                io.Int.Input(
                    "canvas_height",
                    display_name="Canvas Height",
                    default=768,
                    min=64,
                    max=8192,
                    step=1,
                    tooltip="Output canvas height in pixels. Snapped to divisible_by.",
                ),
                io.String.Input(
                    "canvas_color",
                    display_name="Canvas Color",
                    default="#222222",
                    multiline=False,
                    tooltip=(
                        "Background color of the canvas. "
                        "Accepts hex (#RRGGBB), named colors, or 'transparent'."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    display_name="Output Mode",
                    options=OUTPUT_MODES,
                    default="both",
                    tooltip=(
                        "composite: one merged image. "
                        "individual: one image per layer. "
                        "both: composite and individual batch."
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    display_name="Divisible By",
                    default=32,
                    min=1,
                    max=256,
                    step=1,
                    tooltip=(
                        "Snap canvas dimensions to nearest lower multiple of this value. "
                        "Use 32 for LTX/most video models, 64 for some diffusion models, "
                        "1 to disable snapping."
                    ),
                ),
                io.Autogrow.Input("layers", template=autogrow_template),
            ],
            outputs=[
                io.Image.Output(
                    "composite",
                    display_name="Composite Image",
                ),
                io.Image.Output(
                    "individual_images",
                    display_name="Individual Images",
                ),
                io.Int.Output(
                    "layer_count",
                    display_name="Layer Count",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        canvas_width: int,
        canvas_height: int,
        canvas_color: str,
        output_mode: str,
        divisible_by: int,
        layers: io.Autogrow.Type,
    ) -> io.NodeOutput:

        # ── 1. Snap canvas dimensions ─────────────────────────────────────────
        cw = snap_to_divisible(canvas_width,  divisible_by)
        ch = snap_to_divisible(canvas_height, divisible_by)

        if cw != canvas_width or ch != canvas_height:
            print(
                f"[SubjectCompositor] Canvas snapped from "
                f"{canvas_width}×{canvas_height} to {cw}×{ch} "
                f"(divisible_by={divisible_by})",
                flush=True,
            )

        # ── 2. Collect connected layers ───────────────────────────────────────
        # Autogrow gives us a dict mapping input names to values.
        # Filter out None entries (unconnected optional slots).
        layer_list = [v for v in layers.values() if v is not None]

        if not layer_list:
            raise ValueError(
                "[SubjectCompositor] No layers connected. "
                "Connect at least one SubjectLayerDefine to layer_0."
            )

        layer_count = len(layer_list)
        bg_color = parse_color(canvas_color)

        # ── 3. Process each layer ─────────────────────────────────────────────
        processed: list[tuple[Image.Image, int, int, float, float]] = []

        for i, layer_def in enumerate(layer_list):
            try:
                img_rgba, scaled_w, scaled_h = process_layer(
                    image_tensor      = layer_def["image"],
                    mask_tensor       = layer_def.get("mask"),
                    remove_background = layer_def.get("remove_background", True),
                    bg_model          = layer_def.get("bg_model", "BiRefNet-general"),
                    pad_top           = layer_def.get("pad_top",    0.0),
                    pad_bottom        = layer_def.get("pad_bottom", 0.0),
                    pad_left          = layer_def.get("pad_left",   0.0),
                    pad_right         = layer_def.get("pad_right",  0.0),
                    canvas_w          = cw,
                    canvas_h          = ch,
                )
                processed.append((
                    img_rgba,
                    scaled_w,
                    scaled_h,
                    layer_def.get("offset_x", 0.0),
                    layer_def.get("offset_y", 0.0),
                ))
            except Exception as e:
                print(f"[SubjectCompositor] Error processing layer {i}: {e}")
                raise

        # ── 4. Build composite image ──────────────────────────────────────────
        composite_tensor = None

        if output_mode in ("composite", "both"):
            canvas = Image.new("RGBA", (cw, ch), bg_color)

            for img_rgba, sw, sh, ox, oy in processed:
                px, py = compute_paste_position(sw, sh, cw, ch, ox, oy)
                canvas = composite_onto_canvas(canvas, img_rgba, px, py)

            # Flatten RGBA to RGB over the background color
            bg = Image.new("RGB", (cw, ch), bg_color[:3])
            bg.paste(canvas.convert("RGB"), mask=canvas.split()[3])
            composite_tensor = pil_to_tensor(bg)

        # ── 5. Build individual images ────────────────────────────────────────
        individual_tensor = None

        if output_mode in ("individual", "both"):
            individual_tensors = []

            for img_rgba, sw, sh, ox, oy in processed:
                ind_canvas = Image.new("RGBA", (cw, ch), bg_color)
                px, py = compute_paste_position(sw, sh, cw, ch, ox, oy)
                ind_canvas = composite_onto_canvas(ind_canvas, img_rgba, px, py)

                # Flatten to RGB
                bg = Image.new("RGB", (cw, ch), bg_color[:3])
                bg.paste(ind_canvas.convert("RGB"), mask=ind_canvas.split()[3])
                individual_tensors.append(pil_to_tensor(bg))  # [1, H, W, 3]

            # Stack into batch [N, H, W, 3]
            individual_tensor = torch.cat(individual_tensors, dim=0)

        # ── 6. Handle output_mode fallbacks ──────────────────────────────────
        # If composite was not generated, create a placeholder (first individual)
        if composite_tensor is None and individual_tensor is not None:
            composite_tensor = individual_tensor[0:1]

        # If individual was not generated, return composite repeated N times
        if individual_tensor is None and composite_tensor is not None:
            individual_tensor = composite_tensor.repeat(layer_count, 1, 1, 1)

        return io.NodeOutput(
            composite_tensor,
            individual_tensor,
            layer_count,
        )

# ── Node: DatasetCaptioner ────────────────────────────────────────────────────

class DatasetCaptioner(io.ComfyNode):
    """
    Captions images in a directory using a local VLM or Gemini Flash.
    Writes one .txt file per image (alongside the image, or into output_directory).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("DatasetCaptioner"),
            display_name="Dataset Captioner",
            category="🧊 frost-byte/Dataset",
            description=(
                "Captions all images in a directory using Qwen2.5-VL, Qwen2.5-Omni, "
                "or Gemini Flash. Writes one .txt caption file per image."
            ),
            inputs=[
                io.String.Input(
                    "input_directory",
                    display_name="Input Directory",
                    default="",
                    multiline=False,
                    tooltip="Absolute path, or path relative to Comfy input directory.",
                ),
                io.Combo.Input(
                    "captioner_type",
                    display_name="Captioner",
                    options=CAPTIONER_OPTIONS,
                    default="qwen_vl",
                    tooltip="qwen_vl is recommended for images. qwen_omni is heavier. gemini_flash requires an API key.",
                ),
                io.String.Input(
                    "instruction",
                    display_name="Instruction",
                    default=DEFAULT_INSTRUCTION,
                    multiline=True,
                    tooltip="What to ask the captioning model about each image.",
                ),
                io.String.Input(
                    "output_directory",
                    display_name="Output Directory",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Absolute path, or path relative to Comfy output directory. Defaults to same directory as images.",
                ),
                io.String.Input(
                    "trigger_word",
                    display_name="Trigger Word",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Prepended deterministically to every caption. More reliable than asking the model to include it.",
                ),
                io.Combo.Input(
                    "device",
                    display_name="Device",
                    options=DEVICE_OPTIONS,
                    default="auto",
                    optional=True,
                ),
                io.Boolean.Input(
                    "use_8bit",
                    display_name="Use 8-bit",
                    default=False,
                    optional=True,
                    tooltip="Load model in 8-bit precision. Halves VRAM usage. Requires bitsandbytes.",
                ),
                io.Boolean.Input(
                    "recursive",
                    display_name="Recursive",
                    default=False,
                    optional=True,
                    tooltip="Descend into subdirectories.",
                ),
                io.Boolean.Input(
                    "override_existing",
                    display_name="Override Existing",
                    default=False,
                    optional=True,
                    tooltip="Re-caption images that already have a .txt file.",
                ),
                io.Boolean.Input(
                    "clean_caption",
                    display_name="Clean Caption",
                    default=True,
                    optional=True,
                    tooltip="Strip common VLM boilerplate phrases from output.",
                ),
                io.Boolean.Input(
                    "unload_after",
                    display_name="Unload Model After",
                    default=False,
                    optional=True,
                    tooltip="Release model from VRAM when done. Useful before running generation nodes.",
                ),
                io.String.Input(
                    "gemini_api_key",
                    display_name="Gemini API Key",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Required only for gemini_flash. Can also be set via GEMINI_API_KEY env var.",
                ),
            ],
            outputs=[
                io.String.Output("dataset_path",  display_name="Dataset Path"),
                io.Int.Output("caption_count",    display_name="Captioned"),
                io.Int.Output("failed_count",     display_name="Failed"),
            ],
        )

    @classmethod
    def execute(
        cls,
        input_directory: str,
        captioner_type: str,
        instruction: str,
        output_directory: str = "",
        trigger_word: str = "",
        device: str = "auto",
        use_8bit: bool = False,
        recursive: bool = False,
        override_existing: bool = False,
        clean_caption: bool = True,
        unload_after: bool = False,
        gemini_api_key: str = "",
    ) -> io.NodeOutput:
        input_dir = _resolve_dataset_input_directory(input_directory)
        output_dir = _resolve_dataset_output_directory(output_directory)
        api_key    = gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "")

        if not input_dir.is_dir():
            raise ValueError(f"input_directory does not exist: {input_dir}")

        images = _collect_images(input_dir, recursive)
        if not override_existing:
            images = [i for i in images if not _txt_path(i, output_dir).exists()]

        if not images:
            send_status_update(
                DATASET_CAPTION_STATUS_ID,
                "Dataset Captioner: no images to process",
                source="dataset_captioner",
                level="warn",
            )
            return io.NodeOutput(str(output_dir or input_dir), 0, 0)

        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: loading {captioner_type} model (first run may download)",
            source="dataset_captioner",
        )
        model, processor = get_model(captioner_type, device, use_8bit)
        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: model ready, captioning {len(images)} image(s)",
            source="dataset_captioner",
        )
        success = failed = 0

        total_images = len(images)
        for idx, img_path in enumerate(images, start=1):
            try:
                send_status_update(
                    DATASET_CAPTION_STATUS_ID,
                    f"Dataset Captioner: processing {idx}/{total_images} ({img_path.name})",
                    source="dataset_captioner",
                )
                caption = caption_image(
                    image_path     = img_path,
                    captioner_type = captioner_type,
                    instruction    = instruction,
                    model          = model,
                    processor      = processor,
                    api_key        = api_key,
                    clean          = clean_caption,
                )
                if trigger_word.strip():
                    caption = f"{trigger_word.strip()}. {caption}"
                _write_caption(img_path, caption, output_dir)
                success += 1
            except Exception as e:
                print(f"[DatasetCaptioner] Error captioning {img_path.name}: {e}")
                send_status_update(
                    DATASET_CAPTION_STATUS_ID,
                    f"Dataset Captioner: error on {img_path.name}",
                    source="dataset_captioner",
                    level="error",
                )
                failed += 1

        if unload_after:
            unload_model()
            send_status_update(
                DATASET_CAPTION_STATUS_ID,
                "Dataset Captioner: model unloaded",
                source="dataset_captioner",
            )

        completion_level = "error" if failed else "success"
        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: completed ({success} ok, {failed} failed)",
            source="dataset_captioner",
            level=completion_level,
        )

        return io.NodeOutput(str(output_dir or input_dir), success, failed)


# ── Node: DatasetCaptionEditor ────────────────────────────────────────────────

class DatasetCaptionEditor(io.ComfyNode):
    """
    Batch-edits caption .txt files in a dataset directory.
    Runs in dry_run mode by default — set dry_run=False to write changes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("DatasetCaptionEditor"),
            display_name="Dataset Caption Editor",
            category="🧊 frost-byte/Dataset",
            description=(
                "Batch-edits all .txt captions in a directory. "
                "Supports prepend, append, and find/replace. "
                "Dry run mode previews changes without writing."
            ),
            inputs=[
                io.String.Input(
                    "dataset_path",
                    display_name="Dataset Path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path, or path relative to Comfy output directory.",
                ),
                io.String.Input(
                    "prepend_text",
                    display_name="Prepend Text",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Added to the start of every caption.",
                ),
                io.String.Input(
                    "append_text",
                    display_name="Append Text",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Added to the end of every caption.",
                ),
                io.String.Input(
                    "find_text",
                    display_name="Find",
                    default="",
                    multiline=False,
                    optional=True,
                ),
                io.String.Input(
                    "replace_text",
                    display_name="Replace With",
                    default="",
                    multiline=False,
                    optional=True,
                ),
                io.Boolean.Input(
                    "recursive",
                    display_name="Recursive",
                    default=False,
                    optional=True,
                ),
                io.Boolean.Input(
                    "dry_run",
                    display_name="Dry Run",
                    default=True,
                    optional=True,
                    tooltip="Preview changes in the console without writing to disk.",
                ),
            ],
            outputs=[
                io.String.Output("dataset_path", display_name="Dataset Path"),
                io.Int.Output("edited_count",    display_name="Edited Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dataset_path: str,
        prepend_text: str = "",
        append_text: str  = "",
        find_text: str    = "",
        replace_text: str = "",
        recursive: bool   = False,
        dry_run: bool     = True,
    ) -> io.NodeOutput:
        base = _resolve_dataset_output_directory(dataset_path)
        if base is None:
            raise ValueError(
                "dataset_path is required. Provide an absolute path or a path relative to Comfy output directory."
            )
        if not base.is_dir():
            raise ValueError(f"dataset_path is not a directory: {base}")

        pattern   = "**/*.txt" if recursive else "*.txt"
        txt_files = list(base.glob(pattern))
        edited    = 0

        for txt in txt_files:
            original = txt.read_text(encoding="utf-8").strip()
            updated  = original

            if find_text:
                updated = updated.replace(find_text, replace_text)
            if prepend_text:
                updated = f"{prepend_text.rstrip()} {updated}".strip()
            if append_text:
                updated = f"{updated.rstrip()} {append_text.lstrip()}".strip()

            if updated != original:
                edited += 1
                if not dry_run:
                    txt.write_text(updated, encoding="utf-8")
                else:
                    print(f"[DatasetCaptionEditor] DRY RUN — {txt.name}")
                    print(f"  BEFORE: {original[:120]}")
                    print(f"  AFTER:  {updated[:120]}")

        if dry_run:
            print(f"[DatasetCaptionEditor] Dry run: {edited} file(s) would be modified.")

        return io.NodeOutput(str(base), edited)


# ── Node: DatasetCaptionViewer ────────────────────────────────────────────────

class DatasetCaptionViewer(io.ComfyNode):
    """
    Loads a dataset directory and passes image+caption data to the
    companion JS frontend widget as a ui payload.

    Caption edits made in the widget are written back via the
    /fbtools/dataset_caption/save API route.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("DatasetCaptionViewer"),
            display_name="Dataset Caption Viewer",
            category="🧊 frost-byte/Dataset",
            description=(
                "Renders an interactive editable caption table in the node. "
                "Supports inline editing, per-image re-captioning, and pagination."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "dataset_path",
                    display_name="Dataset Path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path, or path relative to Comfy input directory.",
                ),
                io.String.Input(
                    "output_directory",
                    display_name="Caption Directory",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Absolute path, or path relative to Comfy output directory. Used when captions are stored separately.",
                ),
                io.Int.Input(
                    "page",
                    display_name="Page",
                    default=1,
                    min=1,
                    max=9999,
                    step=1,
                    optional=True,
                ),
                io.Int.Input(
                    "page_size",
                    display_name="Page Size",
                    default=10,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                ),
                io.Boolean.Input(
                    "recursive",
                    display_name="Recursive",
                    default=False,
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output("dataset_path", display_name="Dataset Path"),
                io.Int.Output("image_count",     display_name="Image Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, dataset_path: str = "", output_directory: str = "", **_):
        """Re-execute when any .txt in the caption directory is modified."""
        base = _resolve_dataset_input_directory(dataset_path, field_name="dataset_path") if dataset_path.strip() else None
        out = _resolve_dataset_output_directory(output_directory) if output_directory.strip() else base
        if out and out.is_dir():
            mtimes = [t.stat().st_mtime for t in out.glob("**/*.txt")]
            return str(sum(mtimes))
        return ""

    @classmethod
    def execute(
        cls,
        dataset_path: str,
        output_directory: str = "",
        page: int       = 1,
        page_size: int  = 10,
        recursive: bool = False,
    ) -> io.NodeOutput:
        if not dataset_path.strip():
            return io.NodeOutput("", 0)

        base = _resolve_dataset_input_directory(dataset_path, field_name="dataset_path")
        output_dir = _resolve_dataset_output_directory(output_directory)

        if not base.is_dir():
            return io.NodeOutput(str(base), 0)

        images    = _collect_images(base, recursive)
        total     = len(images)
        start     = (page - 1) * page_size
        page_imgs = images[start : start + page_size]

        rows = []
        for img in page_imgs:
            caption = _read_caption(img, output_dir)
            txt     = _txt_path(img, output_dir)
            rows.append({
                "filename":    img.name,
                "image_path":  str(img),
                "txt_path":    str(txt),
                "caption":     caption,
                "has_caption": txt.exists(),
            })

        viewer_data = {
            "rows":        rows,
            "total":       total,
            "page":        page,
            "page_size":   page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "base_dir":    str(base),
        }

        return io.NodeOutput(
            str(base),
            total,
            ui={"dataset_viewer": [viewer_data]},
        )


# ── Node: DatasetExportSummary ────────────────────────────────────────────────

class DatasetExportSummary(io.ComfyNode):
    """
    Inspects a dataset directory and reports health statistics.
    Optionally exports a CSV summary.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("DatasetExportSummary"),
            display_name="Dataset Export Summary",
            category="🧊 frost-byte/Dataset",
            description=(
                "Reports dataset health: image count, captioned vs missing, "
                "word count statistics. Optionally exports a CSV."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "dataset_path",
                    display_name="Dataset Path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path, or path relative to Comfy input directory.",
                ),
                io.String.Input(
                    "output_directory",
                    display_name="Caption Directory",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Absolute path, or path relative to Comfy output directory. Used when captions are stored separately.",
                ),
                io.Boolean.Input(
                    "recursive",
                    display_name="Recursive",
                    default=False,
                    optional=True,
                ),
                io.Boolean.Input(
                    "export_csv",
                    display_name="Export CSV",
                    default=False,
                    optional=True,
                    tooltip="Write a dataset_summary.csv alongside the images.",
                ),
            ],
            outputs=[
                io.String.Output("dataset_path",  display_name="Dataset Path"),
                io.String.Output("summary_json",  display_name="Summary JSON"),
                io.Int.Output("image_count",      display_name="Images"),
                io.Int.Output("captioned_count",  display_name="Captioned"),
                io.Int.Output("missing_count",    display_name="Missing"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dataset_path: str,
        output_directory: str = "",
        recursive: bool = False,
        export_csv: bool = False,
    ) -> io.NodeOutput:
        base = _resolve_dataset_input_directory(dataset_path, field_name="dataset_path")
        output_dir = _resolve_dataset_output_directory(output_directory)

        if not base.is_dir():
            raise ValueError(f"dataset_path is not a directory: {base}")

        images    = _collect_images(base, recursive)
        captioned = [i for i in images if _txt_path(i, output_dir).exists()]
        missing   = [i for i in images if not _txt_path(i, output_dir).exists()]

        captions  = [_read_caption(i, output_dir) for i in captioned]
        lengths   = [len(c.split()) for c in captions if c]
        avg_len   = round(sum(lengths) / len(lengths), 1) if lengths else 0

        summary: dict[str, Any] = {
            "image_count":       len(images),
            "captioned_count":   len(captioned),
            "missing_count":     len(missing),
            "avg_caption_words": avg_len,
            "min_caption_words": min(lengths) if lengths else 0,
            "max_caption_words": max(lengths) if lengths else 0,
            "missing_files":     [i.name for i in missing],
        }

        if export_csv:
            import csv
            csv_path = (output_dir or base) / "dataset_summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "has_caption", "word_count", "caption_preview"])
                for img in images:
                    cap     = _read_caption(img, output_dir)
                    words   = len(cap.split()) if cap else 0
                    preview = cap[:80] + "..." if len(cap) > 80 else cap
                    writer.writerow([img.name, bool(cap), words, preview])
            print(f"[DatasetExportSummary] CSV written to {csv_path}")

        summary_json = json.dumps(summary, indent=2)
        print(f"[DatasetExportSummary]\n{summary_json}")

        return io.NodeOutput(
            str(base),
            summary_json,
            len(images),
            len(captioned),
            len(missing),
        )


# ── Node: CaptionModelUnloader ────────────────────────────────────────────────

class CaptionModelUnloader(io.ComfyNode):
    """
    Explicitly releases the cached captioner model from VRAM.
    Connect this after your captioning workflow to free memory
    before running ComfyUI generation nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("CaptionModelUnloader"),
            display_name="Caption Model Unloader",
            category="🧊 frost-byte/Dataset",
            description="Releases the cached captioner VLM from VRAM.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "dataset_path",
                    display_name="Dataset Path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Connect from Dataset Captioner to enforce execution order.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, dataset_path: str = "") -> io.NodeOutput:
        unload_model()
        print("[CaptionModelUnloader] Captioner model unloaded from VRAM.")
        return io.NodeOutput()

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


def user_data_dir() -> str:
    """Return the package-specific persistent data dir under ComfyUI's user dir.

    Falls back to ComfyUI/user/default/<package> if get_user_directory() is unavailable.
    """
    package_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    try:
        base = folder_paths.get_user_directory()
    except AttributeError:
        try:
            base = os.path.join(folder_paths.base_path, "user", "default")
        except AttributeError:
            base = get_output_directory()
    data_dir = os.path.join(base, package_name)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _user_subdir(name: str) -> str:
    """Create and return a named subdirectory under the package user-data dir."""
    path = os.path.join(user_data_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def default_registry_path() -> str:
    """Default path for the concept registry JSON file."""
    return os.path.join(user_data_dir(), "concept_registry.json")


def default_subject_profiles_path() -> str:
    """Default path for the subject profiles JSON file."""
    return os.path.join(user_data_dir(), "subject_profiles.json")


def default_scene_templates_dir() -> str:
    """Return (and create) the user scene_templates directory.

    Seeds bundled example templates on first use when the directory is empty.
    """
    path = _user_subdir("scene_templates")
    if not any(f.endswith(".json") for f in os.listdir(path)):
        _seed_bundled_templates(path)
    return path


def _seed_bundled_templates(dest_dir: str) -> None:
    """Copy bundled example templates into dest_dir (one-time initialisation)."""
    import shutil as _shutil
    src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scene_templates")
    if not os.path.isdir(src_dir):
        return
    for fname in os.listdir(src_dir):
        if fname.endswith(".json"):
            dst = os.path.join(dest_dir, fname)
            if not os.path.exists(dst):
                _shutil.copy2(os.path.join(src_dir, fname), dst)
    logger.info("Seeded %s with bundled scene templates from %s", dest_dir, src_dir)


def default_libber_dir():
    """Get default directory for storing libber files.

    Prefers user data dir; falls back to legacy output/libbers if that
    directory already has content (non-migrated setups).
    """
    new_dir = os.path.join(user_data_dir(), "libbers")
    if os.path.isdir(new_dir) and any(
        f.endswith(".json") for f in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, f))
    ):
        return new_dir
    # Legacy fallback so existing libber files are not lost
    legacy_dir = os.path.join(get_output_directory(), "libbers")
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir, exist_ok=True)
    return legacy_dir



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


class MultiLoraLoader(io.ComfyNode):
    """
    MultiLoraLoader: Load and apply multiple LoRAs to a diffusion model.
    All LoRA slots are always visible (up to 10), but only the number specified
    in 'num_loras_to_apply' will be processed. This preserves user selections
    when changing the count.
    """
    
    def __init__(self):
        super().__init__()
        self.loaded_loras = {}
    
    @classmethod
    def define_schema(cls):
        lora_files = folder_paths.get_filename_list("loras")
        
        # Create all 10 LoRA slots upfront (they'll always be visible)
        lora_inputs = [
            io.Model.Input("model", tooltip="The diffusion model that the LoRAs will be applied to"),
            io.Int.Input(
                "num_loras_to_apply",
                default=1,
                min=0,
                max=10,
                tooltip="How many of the LoRA slots below to actually apply (0-10)"
            ),
        ]
        
        # Add 10 LoRA slots
        for i in range(1, 11):
            lora_inputs.extend([
                io.Combo.Input(
                    f"lora_name_{i}",
                    options=lora_files,
                    optional=True,
                    tooltip=f"LoRA file #{i} (optional - leave empty to skip)"
                ),
                io.Float.Input(
                    f"strength_{i}",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip=f"Strength for LoRA #{i} (can be negative)"
                ),
            ])
        
        return io.Schema(
            node_id=prefixed_node_id("MultiLoraLoader"),
            display_name="Multi LoRA Loader",
            category="🧊 frost-byte/Loaders",
            inputs=lora_inputs,
            outputs=[
                io.Model.Output("model", tooltip="The modified diffusion model with LoRAs applied"),
            ],
        )
    
    @classmethod
    def execute(cls, model, num_loras_to_apply=1, **kwargs):
        """
        Load and apply multiple LoRAs to the model sequentially.
        
        Args:
            model: The input diffusion model
            num_loras_to_apply: How many LoRA slots to process (0-10)
            **kwargs: Contains lora_name_N and strength_N parameters
        
        Returns:
            NodeOutput with the modified model
        """
        import comfy.utils
        import comfy.sd
        
        # Ensure num_loras_to_apply is within bounds
        num_to_apply = max(0, min(10, int(num_loras_to_apply)))
        
        # Apply each LoRA sequentially up to the specified count
        current_model = model
        applied_count = 0
        
        for i in range(1, num_to_apply + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)
            
            # Skip if no LoRA name provided or strength is zero
            if not lora_name or strength == 0:
                logger.debug(f"Skipping LoRA slot {i}: lora_name='{lora_name}', strength={strength}")
                continue
            
            try:
                # Get the full path to the LoRA file
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                
                # Load the LoRA
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                
                # Apply the LoRA to the model (model only, no clip)
                current_model, _ = comfy.sd.load_lora_for_models(
                    current_model, None, lora, strength, 0
                )
                
                applied_count += 1
                logger.info(f"Applied LoRA {applied_count}/{num_to_apply}: {lora_name} (strength: {strength})")
            except Exception as e:
                logger.error(f"Failed to apply LoRA {i} ({lora_name}): {e}")
        
        if applied_count == 0:
            logger.warning("No LoRAs were applied")
        
        return io.NodeOutput(model=current_model)


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
            
        msg = (
            f"OpaqueAlpha: image in shape={image.shape}, alpha_value={alpha_value}, force_replace_alpha={force_replace_alpha},dtype={image.dtype}, device={image.device}, "
            f"range=[{mn:.6f},{mx:.6f}] -> image out shape={image_rgba.shape}, mask shape={mask.shape}{alpha_summary}"
        )
        
        if debug:
            logger.debug(msg)

        return io.NodeOutput({
            "image_rgba": image_rgba,
            "mask": mask,
            "debug_info": msg
        })

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
        from .utils.images import mask_remove_holes, mask_grow, mask_gaussian_blur, mask_smooth, create_mask_overlay_image, smooth_masks_region_was
        
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

def get_subdirectories(directory_path: str) -> dict:
    """Return a dictionary mapping subdirectory names to their full paths."""
    subdir_dict = {}

    if not os.path.isdir(directory_path):
        logger.warning("Directory '%s' does not exist or is not a directory.", directory_path)
        return subdir_dict

    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_dir():
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

def default_scenes_dir():
    """Scenes directory: prefers user data dir; falls back to legacy output/scenes."""
    new_dir = os.path.join(user_data_dir(), "scenes")
    if os.path.isdir(new_dir) and any(
        os.path.isdir(os.path.join(new_dir, x)) for x in os.listdir(new_dir)
    ):
        return new_dir
    # Legacy location (keeps existing scenes accessible without migration)
    legacy_dir = os.path.join(get_output_directory(), "scenes")
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir, exist_ok=True)
        os.makedirs(os.path.join(legacy_dir, "default_scene"), exist_ok=True)
    return legacy_dir

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

        logger.debug("QwenAspectRatio: input image shape=%s -> w=%s, h=%s", input_image.shape, w, h)
        recommended_w, recommended_h, layout, aspect_ratio_str, aspect_ratio_float = find_nearest_qwen_aspect_ratio(w, h)
        logger.debug(
            "QwenAspectRatio: recommended_w=%s, recommended_h=%s, layout=%s, aspect_ratio_str=%s, aspect_ratio_float=%s",
            recommended_w,
            recommended_h,
            layout,
            aspect_ratio_str,
            aspect_ratio_float,
        )

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


# ============================================================================
# MASK SYSTEM - Generic User-Definable Masks
# ============================================================================

RGB = Tuple[int, int, int]

class MaskType(str, Enum):
    """Type of mask definition"""
    TRANSPARENT = "transparent"  # alpha / transparency-based
    COLOR = "color"              # color-defined regions


@dataclass
class MaskDefinition:
    """Definition of a scene mask with arbitrary name and properties"""
    name: str
    type: MaskType
    has_background: bool = True
    color: Optional[RGB] = None  # Only used when type == COLOR

    def validate(self) -> None:
        """Validate mask definition constraints"""
        if self.type == MaskType.TRANSPARENT:
            if self.color is not None:
                raise ValueError(
                    f"Transparent mask '{self.name}' must not define a color."
                )
        elif self.type == MaskType.COLOR:
            if self.color is None:
                raise ValueError(
                    f"Color mask '{self.name}' must define an RGB color."
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "has_background": self.has_background,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaskDefinition":
        """Create from dictionary"""
        mask_type = MaskType(data["type"]) if isinstance(data["type"], str) else data["type"]
        color = tuple(data["color"]) if data.get("color") and isinstance(data["color"], (list, tuple)) else data.get("color")
        return cls(
            name=data["name"],
            type=mask_type,
            has_background=data.get("has_background", True),
            color=color
        )

    def get_filename(self) -> str:
        """Generate filename for this mask"""
        suffix = "_bkgd" if self.has_background else "_no_bkgd"
        return f"{self.name}_mask{suffix}.png"


def load_masks_json(scene_dir: str) -> Dict[str, MaskDefinition]:
    """Load mask definitions from masks.json"""
    masks_path = Path(scene_dir) / "masks.json"
    if not masks_path.exists():
        return {}
    
    try:
        with open(masks_path, 'r') as f:
            data = json.load(f)
        
        masks = {}
        for mask_data in data.get("masks", []):
            mask_def = MaskDefinition.from_dict(mask_data)
            mask_def.validate()
            masks[mask_def.name] = mask_def
        
        return masks
    except Exception as e:
        logger.error(f"Failed to load masks.json from {scene_dir}: {e}")
        return {}


def save_masks_json(scene_dir: str, masks: Dict[str, MaskDefinition]) -> None:
    """Save mask definitions to masks.json"""
    masks_path = Path(scene_dir) / "masks.json"
    
    try:
        data = {
            "version": 1,
            "masks": [mask.to_dict() for mask in masks.values()]
        }
        
        with open(masks_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(masks)} mask definitions to {masks_path}")
    except Exception as e:
        logger.error(f"Failed to save masks.json to {scene_dir}: {e}")


class SceneInfo(BaseModel):
    #metadata
    scene_dir: str
    scene_name: str
    
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
    pose_nlf_image: Optional[torch.Tensor] = None  # NLF pose rendering
    canny_image: Optional[torch.Tensor] = None
    
    # Image hierarchy: base_image → upscale_image → derived images (pose, depth, canny)
    base_image: Optional[torch.Tensor] = None  # Original input image (saved as base.png)
    upscale_image: Optional[torch.Tensor] = None  # Scaled version of base_image (source for derived images)
    
    lora_stack: Optional[list] = None  # list of LORA_ENTRY dicts (LoraStackCollect format)

    # Mask system - generic user-definable masks
    masks: Optional[Dict[str, MaskDefinition]] = None  # Mask definitions by name
    mask_images: Optional[Dict[str, torch.Tensor]] = None  # Mask image tensors by name
    
    # Legacy mask fields - maintained for backward compatibility with existing scenes
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
        return os.path.join(self.scene_dir, "input") + "/*.png"

    def input_img_dir(self) -> str:
        return f"scenes/{self.scene_name}/input/img"

    def output_dir(self) -> str:
        return f"scenes/{self.scene_name}/output"

    @classmethod
    def load_depth_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load depth images from a scene directory, optionally filtering by keys."""
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
            images[key] = _img(os.path.join(scene_dir, filename))
        return images

    @classmethod
    def load_pose_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> dict:
        """Load pose images from a scene directory, optionally filtering by keys."""
        mapping = {
            'base_image': "base.png",  # Original input image
            'pose_dense_image': "pose_dense.png",
            'pose_dw_image': "pose_dw.png",
            'pose_edit_image': "pose_edit.png",
            'pose_face_image': "pose_face.png",
            'pose_open_image': "pose_open.png",
            'pose_nlf_image': "pose_nlf.png",  # NLF pose rendering
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
            images[key] = _img(os.path.join(scene_dir, filename))
        return images

    @classmethod
    def load_mask_images(cls, scene_dir: str, mask_names: Optional[list[str]] = None) -> tuple[dict, dict]:
        """Load mask images and their alpha masks from a scene directory.

        Returns a tuple `(images, masks)` where images maps mask names to IMAGE tensors,
        and masks maps the same names to [B,H,W] float masks (1.0 means masked-out).
        
        Args:
            scene_dir: Path to scene directory
            mask_names: Optional list of mask names to load. If None, loads all from masks.json
        
        Returns:
            (images_dict, masks_dict) tuple
        """
        # Load mask definitions
        mask_defs = load_masks_json(scene_dir)
        
        # If no masks.json exists, try legacy format
        if not mask_defs:
            logger.debug(f"No masks.json found in {scene_dir}, trying legacy mask format")
            return cls._load_legacy_mask_images(scene_dir, mask_names)
        
        images = {}
        masks = {}
        
        # Determine which masks to load
        names_to_load = mask_names if mask_names else list(mask_defs.keys())
        
        for name in names_to_load:
            if name not in mask_defs:
                # Only warn if it's not the "combined" fallback
                if name != "combined":
                    logger.warning(f"Mask '{name}' not found in mask definitions")
                continue
            
            mask_def = mask_defs[name]
            filename = mask_def.get_filename()
            filepath = os.path.join(scene_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Mask file not found: {filepath}")
                continue
            
            try:
                image, mask = load_image_comfyui(filepath, include_mask=True)
                images[name] = image
                masks[name] = mask
            except Exception as e:
                logger.error(f"Failed to load mask '{name}' from {filepath}: {e}")
        
        return images, masks

    @classmethod
    def _load_legacy_mask_images(cls, scene_dir: str, keys: Optional[list[str]] = None) -> tuple[dict, dict]:
        """Load legacy hardcoded mask images for backward compatibility.
        
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
            filepath = os.path.join(scene_dir, filename)
            if not os.path.exists(filepath):
                continue
            
            try:
                image, mask = load_image_comfyui(filepath, include_mask=True)
                images[key] = image
                masks[key] = mask
            except Exception as e:
                logger.debug(f"Could not load legacy mask {filename}: {e}")

        return images, masks

    @classmethod
    def load_all_images(cls, scene_dir: str) -> dict:
        """Load all images (depth, pose, mask) from a scene directory"""
        all_images = {}
        all_images.update(cls.load_depth_images(scene_dir))
        all_images.update(cls.load_pose_images(scene_dir))
        mask_images, _ = cls.load_mask_images(scene_dir)
        all_images.update(mask_images)
        return all_images

    @classmethod
    def load_preview_assets(
            cls,
            scene_dir: str,
            depth_attr: str,
            pose_attr: str,
            mask_name: str,
            mask_background: Optional[bool] = None,  # None = use mask name directly (new system), True/False = legacy behavior
            include_upscale: bool = False,
            include_canny: bool = False,
    ) -> dict:
        """Load a minimal, normalized bundle for preview/output (depth, pose, mask, base_image, optional canny).

        Returns dict keys:
            depth_image, pose_image, mask_image, mask (B,H,W,1), mask_preview (B,H,W,3),
            base_image, canny_image, preview_batch (list of tensors), H, W, resolution,
            plus raw dictionaries depth_images/pose_images/mask_images for downstream SceneInfo population.
        
        Args:
            scene_dir: Path to scene directory
            depth_attr: Depth image attribute name to load
            pose_attr: Pose image attribute name to load
            mask_name: Name of mask to load
            mask_background: For legacy masks - whether to include background. If None, uses mask_name directly
            include_upscale: Whether to include upscale image (kept for compatibility)
            include_canny: Whether to include canny image
        
        Note: include_upscale parameter is kept for compatibility but base_image is always loaded for previews.
        """
        mask_key = resolve_mask_key(mask_name, mask_background)

        depth_keys = {depth_attr, "depth_image"}
        pose_keys = {pose_attr, "pose_open_image", "base_image"}  # Always load base_image for preview
        if include_canny:
            pose_keys.add("canny_image")
        
        # For masks, try to load the requested mask
        # For legacy compatibility, also try to load "combined" as fallback (only if mask_key is not empty)
        mask_names_to_load = []
        if mask_key:  # Only load masks if mask_key is not empty
            mask_names_to_load.append(mask_key)
            if mask_key != "combined":
                mask_names_to_load.append("combined")

        depth_images = cls.load_depth_images(scene_dir, keys=list(depth_keys))
        pose_images = cls.load_pose_images(scene_dir, keys=list(pose_keys))
        mask_images, mask_tensors = cls.load_mask_images(scene_dir, mask_names=mask_names_to_load) if mask_names_to_load else ({}, {})

        # Determine spatial size from available images
        empty_image = make_empty_image(1, 512, 512)
        base_image = pose_images.get("base_image")  # Always load base for preview
        depth_image_raw = depth_images.get("depth_image")
        pose_image_raw = pose_images.get(pose_attr, pose_images.get("pose_open_image", empty_image))
        mask_image_raw = mask_images.get(mask_key, mask_images.get("combined", empty_image))

        if depth_image_raw is not None:
            H, W = depth_image_raw.shape[1], depth_image_raw.shape[2]
        elif pose_image_raw is not None:
            H, W = pose_image_raw.shape[1], pose_image_raw.shape[2]
        elif mask_image_raw is not None:
            H, W = mask_image_raw.shape[1], mask_image_raw.shape[2]
        elif base_image is not None:
            H, W = base_image.shape[1], base_image.shape[2]
        else:
            H, W = 512, 512

        # Normalize images to a consistent size
        depth_image = normalize_image_tensor(depth_images.get(depth_attr, depth_images.get("depth_image", empty_image)), H, W)
        pose_image = normalize_image_tensor(pose_image_raw, H, W)
        base_image = normalize_image_tensor(base_image, H, W) if base_image is not None else None
        mask_image = normalize_image_tensor(mask_image_raw, H, W)

        # Build mask output (single-channel) and preview (3-channel)
        mask = None
        mask_tensor = mask_tensors.get(mask_key)
        unsqueeze_me = False
        if mask_tensor is not None:
            logger.debug("SceneInfo.load_preview_assets: using mask tensor for key '%s'", mask_key)
            mask = mask_tensor
            unsqueeze_me = True
        elif mask_image is not None:
            logger.debug("SceneInfo.load_preview_assets: building empty mask matching mask_image shape")
            b, hh, ww, _ = mask_image.shape
            mask = torch.zeros((b, hh, ww, 1), device=mask_image.device, dtype=torch.float32)
        else:
            logger.debug(
                "SceneInfo.load_preview_assets: building empty mask of size (1,%s,%s,1)",
                H,
                W,
            )
            mask = torch.zeros((1, H, W, 1), dtype=torch.float32)

        if mask is not None and mask.dtype != torch.float32:
            logger.debug("SceneInfo.load_preview_assets: converting mask to float32")
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
            scenes_dir: Optional[str] = None,
            prompt_in: str = "",
            prompt_action: str = "use_file",
            include_upscale: bool = False,
            include_canny: bool = False,
            prompt_override: Optional[str] = None,
            scene_dir_override: Optional[str] = None,
    ) -> tuple["SceneInfo", dict, str, dict, Optional[str]]:
        """Build SceneInfo + assets from a SceneInStory configuration.

        Returns (scene_info, assets, selected_prompt, prompt_data, prompt_widget_text).
        """

        scenes_dir = scenes_dir or default_scenes_dir()
        scene_dir = scene_dir_override if scene_dir_override else os.path.join(scenes_dir, scene.scene_name)

        if not os.path.isdir(scene_dir):
            raise ValueError(f"from_story_scene: scene_dir '{scene_dir}' is invalid")

        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_data_raw = load_prompt_json(prompt_json_path) or {}
        
        # Load the scene's PromptCollection to get prompt_dict and composition_dict
        if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
            prompt_collection = PromptCollection.from_dict(prompt_data_raw)
        else:
            # Legacy format - migrate
            prompt_collection = PromptCollection.from_legacy_dict(prompt_data_raw)
        
        logger.debug(
            "SceneInfo.from_story_scene: Loaded PromptCollection with %d prompts and %d compositions",
            len(prompt_collection.prompts),
            len(prompt_collection.compositions),
        )
        if prompt_collection.compositions:
            logger.debug("  -> compositions: %s", list(prompt_collection.compositions.keys()))
        else:
            logger.debug("  -> compositions: None/Empty")
        
        # Use shared LibberStateManager so loaded libbers (e.g., story_libber) are applied
        libber_manager = LibberStateManager.instance()
        
        # Build prompt_dict: just the raw individual prompts (not composed)
        prompt_dict = {}
        for key, metadata in prompt_collection.prompts.items():
            value = metadata.value
            # Process libber substitution if needed
            if metadata.processing_type == "libber" and metadata.libber_name:
                libber = libber_manager.ensure_libber(metadata.libber_name)
                if libber:
                    value = libber.substitute(value)
            prompt_dict[key] = value
        
        # Build composition_dict: composed prompts from compositions
        # compositions is dict[str, List[str]] where key is output name, value is list of prompt keys
        composition_dict = {}
        if prompt_collection.compositions:
            composition_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
            logger.debug("  -> Composed %d compositions: %s", len(composition_dict), list(composition_dict.keys()))
        else:
            logger.debug("  -> No compositions to compose")
        
        # Determine the selected prompt based on prompt_source and prompt_key
        prompt_file_text = ""
        if scene.prompt_source == "custom":
            prompt_file_text = scene.custom_prompt
        elif scene.prompt_source == "prompt" and scene.prompt_key:
            prompt_file_text = prompt_dict.get(scene.prompt_key, "")
        elif scene.prompt_source == "composition" and scene.prompt_key:
            prompt_file_text = composition_dict.get(scene.prompt_key, "")
        
        logger.debug(
            "SceneInfo.from_story_scene: scene=%s, prompt_source=%s, prompt_key=%s",
            scene.scene_name,
            scene.prompt_source,
            scene.prompt_key,
        )
        logger.debug("  -> prompt_dict has %d keys", len(prompt_dict))
        logger.debug("  -> composition_dict has %d keys", len(composition_dict))
        logger.debug("  -> prompt_file_text length: %d", len(prompt_file_text))
        
        class_name = f"{cls.__name__}.from_story_scene"
        selected_prompt, prompt_widget_text = select_text_by_action(
            prompt_in,
            prompt_file_text,
            prompt_action,
            class_name,
        )
        if prompt_override:
            selected_prompt = prompt_override

        pose_json_path = os.path.join(scene_dir, "pose.json")
        pose_json_obj = load_json_file(pose_json_path)
        pose_json = json.dumps(pose_json_obj) if pose_json_obj else "[]"

        lora_stack = load_lora_stack(scene_dir)

        depth_attr = default_depth_options.get(scene.depth_type, "depth_image")
        pose_attr = default_pose_options.get(scene.pose_type, "pose_open_image")

        assets = cls.load_preview_assets(
            scene_dir,
            depth_attr=depth_attr,
            pose_attr=pose_attr,
            mask_name=scene.mask_name,
            mask_background=scene.mask_background,
            include_upscale=include_upscale,
            include_canny=include_canny,
        )

        depth_images = assets.get("depth_images", {})
        pose_images = assets.get("pose_images", {})
        mask_images = assets.get("mask_images", {})
        
        # For backwards compatibility, keep the old fields but they'll be empty
        # since we no longer use them in the new system
        scene_info = cls(
            scene_dir=scene_dir,
            scene_name=scene.scene_name,
            girl_pos="",  # Deprecated
            male_pos="",  # Deprecated
            four_image_prompt="",  # Deprecated
            wan_prompt="",  # Deprecated
            wan_low_prompt="",  # Deprecated
            pose_json=pose_json,
            resolution=assets.get("resolution", 0),
            prompts=prompt_collection,  # Now using PromptCollection
            lora_stack=lora_stack,
            **depth_images,
            **pose_images,
            **mask_images,
        )
        
        # Return prompt_dict in the prompt_data for compatibility
        return_prompt_data = {
            "prompt_dict": prompt_dict,
            "composition_dict": composition_dict,
        }

        return scene_info, assets, selected_prompt or "", return_prompt_data, prompt_widget_text

    @classmethod
    def from_scene_directory(cls, scene_dir: str, scene_name: str, prompt_data: Optional[dict] = None,
                           pose_json: str = "", lora_stack: Optional[list] = None):
        """Create a SceneInfo instance by loading all data from a scene directory"""
        if prompt_data is None:
            prompt_json_path = os.path.join(scene_dir, "prompts.json")
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
                logger.info(
                    "SceneInfo.from_scene_directory: Migrated %d legacy prompts",
                    len(prompt_collection.prompts),
                )
        else:
            # No prompts file - create empty collection
            prompt_collection = PromptCollection()
        
        # Load all images
        all_images = cls.load_all_images(scene_dir)
        
        # Load mask definitions and separate out mask images
        mask_defs = load_masks_json(scene_dir)
        mask_images_dict = {}
        
        # Extract mask images from all_images based on mask definitions
        for mask_name in list(mask_defs.keys()):
            if mask_name in all_images:
                mask_images_dict[mask_name] = all_images.pop(mask_name)
        
        # Determine resolution from depth_image
        depth_image = all_images.get('depth_image')
        if depth_image is not None:
            H, W = depth_image.shape[1], depth_image.shape[2]
            resolution = max(H, W)
        else:
            resolution = 512
        
        return cls(
            scene_dir=scene_dir,
            scene_name=scene_name,
            prompts=prompt_collection,
            pose_json=pose_json,
            resolution=resolution,
            lora_stack=lora_stack,
            masks=mask_defs if mask_defs else None,
            mask_images=mask_images_dict if mask_images_dict else None,
            **all_images
        )

    def save_all_images(self, scene_dir: Optional[str] = None):
        """Save all images to the scene directory"""
        from pathlib import Path
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        
        # Save depth images
        if self.depth_image is not None:
            save_image_comfyui(self.depth_image, scene_path / "depth.png")
        if self.depth_any_image is not None:
            save_image_comfyui(self.depth_any_image, scene_path / "depth_any.png")
        if self.depth_midas_image is not None:
            save_image_comfyui(self.depth_midas_image, scene_path / "depth_midas.png")
        if self.depth_zoe_image is not None:
            save_image_comfyui(self.depth_zoe_image, scene_path / "depth_zoe.png")
        if self.depth_zoe_any_image is not None:
            save_image_comfyui(self.depth_zoe_any_image, scene_path / "depth_zoe_any.png")
        
        # Handle base.webp conversion to base.png
        base_webp_path = scene_path / "base.webp"
        base_png_path = scene_path / "base.png"
        if base_webp_path.exists() and not base_png_path.exists():
            try:
                from PIL import Image
                webp_img = Image.open(base_webp_path)
                webp_img.save(base_png_path, format='PNG')
                logger.info("SceneInfo: Converted base.webp to base.png")
            except Exception as e:
                logger.error("SceneInfo: Failed to convert base.webp to base.png: %s", e)
        
        # Save pose images
        if self.base_image is not None:
            save_image_comfyui(self.base_image, base_png_path)
            # Generate thumbnail from base image
            generate_thumbnail(self.base_image, scene_path / "thumbnail.png", size=(128, 128))
        elif self.upscale_image is not None:
            # Check if we should create base.png from upscale.png using depth.png dimensions
            depth_png_path = scene_path / "depth.png"
            if depth_png_path.exists() and not base_png_path.exists():
                try:
                    # Load depth to get target dimensions
                    depth_img, _ = load_image_comfyui(str(depth_png_path), include_mask=False)
                    _, depth_h, depth_w, _ = depth_img.shape
                    
                    # Resize upscale image to match depth dimensions
                    from PIL import Image
                    upscale_np = (self.upscale_image[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    upscale_pil = Image.fromarray(upscale_np)
                    resized_pil = upscale_pil.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
                    
                    # Convert back to tensor and save
                    import numpy as np
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    base_tensor = torch.from_numpy(resized_np).unsqueeze(0)
                    save_image_comfyui(base_tensor, base_png_path)
                    logger.info("SceneInfo: Created base.png from upscale.png at depth.png resolution")
                except Exception as e:
                    logger.error("SceneInfo: Failed to create base.png from upscale: %s", e)
            
            # Fallback: use upscale image for thumbnail if base doesn't exist
            generate_thumbnail(self.upscale_image, scene_path / "thumbnail.png", size=(128, 128))
        
        if self.pose_dense_image is not None:
            save_image_comfyui(self.pose_dense_image, scene_path / "pose_dense.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_dense_image")
        if self.pose_dw_image is not None:
            save_image_comfyui(self.pose_dw_image, scene_path / "pose_dw.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_dw_image")
        if self.pose_edit_image is not None:
            save_image_comfyui(self.pose_edit_image, scene_path / "pose_edit.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_edit_image")
        if self.pose_face_image is not None:
            save_image_comfyui(self.pose_face_image, scene_path / "pose_face.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_face_image")
        if self.pose_open_image is not None:
            save_image_comfyui(self.pose_open_image, scene_path / "pose_open.png")
            logger.debug("SceneInfo.save_all_images: Saved pose_open_image")
        if self.pose_nlf_image is not None:
            save_image_comfyui(self.pose_nlf_image, scene_path / "pose_nlf.png")
            logger.info("SceneInfo.save_all_images: Saved pose_nlf_image to pose_nlf.png")
        else:
            logger.debug("SceneInfo.save_all_images: pose_nlf_image is None, skipping")
        if self.canny_image is not None:
            save_image_comfyui(self.canny_image, scene_path / "canny.png")
        if self.upscale_image is not None:
            save_image_comfyui(self.upscale_image, scene_path / "upscale.png")
        
        # Save new mask system images and definitions
        if self.masks and self.mask_images:
            # Save mask definitions
            save_masks_json(str(scene_path), self.masks)
            
            # Save mask image files
            for mask_name, mask_tensor in self.mask_images.items():
                if mask_tensor is not None and mask_name in self.masks:
                    mask_def = self.masks[mask_name]
                    filename = mask_def.get_filename()
                    save_image_comfyui(mask_tensor, scene_path / filename)
                    logger.debug(f"Saved mask '{mask_name}' to {filename}")
        
        # Save legacy mask images (for backward compatibility)
        if self.girl_mask_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_bkgd_image, scene_path / "girl_mask_bkgd.png")
        if self.male_mask_bkgd_image is not None:
            save_image_comfyui(self.male_mask_bkgd_image, scene_path / "male_mask_bkgd.png")
        if self.combined_mask_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_bkgd_image, scene_path / "combined_mask_bkgd.png")
        if self.girl_mask_no_bkgd_image is not None:
            save_image_comfyui(self.girl_mask_no_bkgd_image, scene_path / "girl_mask_no_bkgd.png")
        if self.male_mask_no_bkgd_image is not None:
            save_image_comfyui(self.male_mask_no_bkgd_image, scene_path / "male_mask_no_bkgd.png")
        if self.combined_mask_no_bkgd_image is not None:
            save_image_comfyui(self.combined_mask_no_bkgd_image, scene_path / "combined_mask_no_bkgd.png")

    def save_prompts(self, scene_dir: Optional[str] = None):
        """Save prompts to prompts.json in v2 format with v1_backup"""
        from pathlib import Path
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        prompts_path = scene_path / "prompts.json"
        
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

    def save_pose_json(self, scene_dir: Optional[str] = None):
        """Save pose_json to pose.json in the pose directory"""
        from pathlib import Path
        import json
        
        if not self.pose_json:
            return
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        pose_json_path = scene_path / "pose.json"
        save_json_file(pose_json_path, json.loads(self.pose_json))

    def save_loras(self, scene_dir: Optional[str] = None):
        """Save LoRA stack to lora_stack.json in the scene directory."""
        from pathlib import Path

        if self.lora_stack is None:
            return

        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        lora_stack_path = scene_path / "lora_stack.json"
        save_json_file(str(lora_stack_path), self.lora_stack)

    def ensure_directories(self, scene_dir: Optional[str] = None):
        """Ensure scene directory and input/output subdirectories exist"""
        import os
        
        scene_path = scene_dir if scene_dir else self.scene_dir
        
        if not os.path.exists(scene_path):
            os.makedirs(scene_path, exist_ok=True)
            logger.info("SceneInfo: Created scene_dir='%s'", scene_path)
        
        input_dir = os.path.join(scene_path, "input")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            logger.info("SceneInfo: Created input_dir='%s'", input_dir)
        
        output_dir = os.path.join(scene_path, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info("SceneInfo: Created output_dir='%s'", output_dir)

    def save_all(self, scene_dir: Optional[str] = None):
        """Save all scene data (images, prompts, pose_json, loras) to the scene directory"""
        target_dir = scene_dir if scene_dir else self.scene_dir
        self.ensure_directories(target_dir)
        self.save_all_images(target_dir)
        self.save_prompts(target_dir)
        self.save_pose_json(target_dir)
        self.save_loras(target_dir)
    
    def regenerate_thumbnail(self, scene_dir: Optional[str] = None, force: bool = False):
        """Regenerate thumbnail if missing or base/upscale image changed
        
        Args:
            scene_dir: Target scene directory (uses self.scene_dir if None)
            force: If True, regenerate thumbnail even if it already exists
        """
        from pathlib import Path
        from PIL import Image
        import numpy as np
        
        scene_path = Path(scene_dir) if scene_dir else Path(self.scene_dir)
        thumbnail_path = scene_path / "thumbnail.png"
        
        # Check if thumbnail already exists (skip if not forcing)
        if thumbnail_path.exists() and not force:
            logger.debug("SceneInfo: Thumbnail already exists at '%s'", thumbnail_path)
            return
        
        # Define paths for disk-based operations
        base_png_path = scene_path / "base.png"
        base_webp_path = scene_path / "base.webp"
        upscale_path = scene_path / "upscale.png"
        depth_path = scene_path / "depth.png"
        
        # Priority 1: Check for base.webp and convert to base.png
        if base_webp_path.exists() and not base_png_path.exists():
            try:
                webp_img = Image.open(base_webp_path)
                webp_img.save(base_png_path, format='PNG')
                logger.info("SceneInfo: Converted base.webp to base.png at '%s'", base_png_path)
            except Exception as e:
                logger.error("SceneInfo: Failed to convert base.webp to base.png: %s", e)
        
        # Priority 2: Use base_image from memory if available
        if self.base_image is not None:
            generate_thumbnail(self.base_image, thumbnail_path, size=(128, 128))
            logger.info("SceneInfo: Generated thumbnail from base_image in memory at '%s'", thumbnail_path)
            return
        
        # Priority 3: Use base.png from disk
        if base_png_path.exists():
            try:
                img, _ = load_image_comfyui(str(base_png_path), include_mask=False)
                generate_thumbnail(img, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from base.png at '%s'", thumbnail_path)
                return
            except Exception as e:
                logger.error("SceneInfo: Failed to generate thumbnail from base.png: %s", e)
        
        # Priority 4: Use upscale_image from memory if available
        if self.upscale_image is not None:
            generate_thumbnail(self.upscale_image, thumbnail_path, size=(128, 128))
            logger.info("SceneInfo: Generated thumbnail from upscale_image in memory at '%s'", thumbnail_path)
            return
        
        # Priority 5: Create base.png from upscale.png if base.png doesn't exist
        if upscale_path.exists() and not base_png_path.exists():
            try:
                img, _ = load_image_comfyui(str(upscale_path), include_mask=False)
                
                # Determine target resolution for base.png
                target_width, target_height = 1024, 1024  # Default resolution
                
                # Check if depth.png exists and is not empty (64x64)
                if depth_path.exists():
                    try:
                        depth_img, _ = load_image_comfyui(str(depth_path), include_mask=False)
                        _, depth_h, depth_w, _ = depth_img.shape
                        
                        # Only use depth dimensions if not the empty 64x64 size
                        if depth_w != 64 or depth_h != 64:
                            target_width, target_height = depth_w, depth_h
                            logger.info("SceneInfo: Using depth.png resolution for base.png: %dx%d", target_width, target_height)
                        else:
                            logger.info("SceneInfo: depth.png is 64x64 (empty), using default 1024x1024 for base.png")
                    except Exception as e:
                        logger.warning("SceneInfo: Failed to read depth.png dimensions, using default 1024x1024: %s", e)
                else:
                    logger.info("SceneInfo: No depth.png found, using default 1024x1024 for base.png")
                
                # Resize upscale to target dimensions and save as base.png
                upscale_np = (img[0] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                upscale_pil = Image.fromarray(upscale_np)
                resized_pil = upscale_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Convert back to tensor and save
                resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                base_tensor = torch.from_numpy(resized_np).unsqueeze(0)
                save_image_comfyui(base_tensor, base_png_path)
                logger.info("SceneInfo: Created base.png from upscale.png at %dx%d resolution", target_width, target_height)
                
                # Now generate thumbnail from the newly created base.png
                generate_thumbnail(base_tensor, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from newly created base.png at '%s'", thumbnail_path)
                return
                
            except Exception as e:
                logger.error("SceneInfo: Failed to create base.png and thumbnail from upscale.png: %s", e)
        
        # Priority 6: If upscale.png exists but base.png already exists, use upscale for thumbnail
        if upscale_path.exists():
            try:
                img, _ = load_image_comfyui(str(upscale_path), include_mask=False)
                generate_thumbnail(img, thumbnail_path, size=(128, 128))
                logger.info("SceneInfo: Generated thumbnail from upscale.png at '%s'", thumbnail_path)
                return
            except Exception as e:
                logger.error("SceneInfo: Failed to generate thumbnail from upscale.png: %s", e)
        
        # Priority 7: Create empty/default thumbnail
        try:
            # Create a small empty gray image as default
            default_img = Image.new('RGB', (128, 128), color=(64, 64, 64))
            default_img.save(thumbnail_path, format='PNG')
            logger.info("SceneInfo: Created default empty thumbnail at '%s'", thumbnail_path)
        except Exception as e:
            logger.error("SceneInfo: Failed to create default thumbnail: %s", e)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

# ============================================================================
# STORY MODELS - Imported from story_models.py
# ============================================================================
# SceneInStory and StoryInfo have been extracted to story_models.py for easier
# testing and reusability. See story_models.py for the full definitions.

def _migrate_loras_json_to_stack(loras_json_path: str) -> list:
    """Convert a legacy loras.json (high/low WANVIDLORA format) to a lora_stack list.

    Each 'high' entry becomes a Wan2.2-Wrapper-High LORA_ENTRY dict;
    each 'low' entry becomes a Wan2.2-Wrapper-Low entry.
    Per-entry blocks, layer_filter, low_mem_load, and merge_loras are preserved
    so WANVIDLORA output remains bit-for-bit identical to the old output.
    """
    data = load_json_file(loras_json_path)
    if not data or not isinstance(data, dict):
        return []

    entries: list[dict] = []
    for target, lora_type in [("Wan2.2-Wrapper-High", "high"), ("Wan2.2-Wrapper-Low", "low")]:
        for item in data.get(lora_type, []):
            lora_name = item.get("lora_name", "")
            strength  = item.get("strength", 1.0)
            if not lora_name or lora_name.lower() == "none":
                continue
            entries.append({
                "lora":           lora_name,
                "model_target":   target,
                "strength_model": strength,
                "strength_clip":  1.0,
                "enabled":        True,
                # Preserve WanVideoWrapper-specific fields verbatim
                "blocks":         item.get("blocks", {}),
                "layer_filter":   item.get("layer_filter", ""),
                "low_mem_load":   item.get("low_mem_load", False),
                "merge_loras":    item.get("merge_loras", False),
            })
    return entries


def load_lora_stack(scene_dir: str) -> Optional[list]:
    """Load the LoRA stack for a scene.

    Checks for lora_stack.json first (new format).
    Falls back to migrating loras.json (old Wan-only WANVIDLORA format) if not found.
    Returns None if neither file exists.
    """
    lora_stack_path = os.path.join(scene_dir, "lora_stack.json")
    if os.path.isfile(lora_stack_path):
        data = load_json_file(lora_stack_path)
        return data if isinstance(data, list) else None

    # Legacy migration path — does not write anything; StorySceneBatch/SceneSelect
    # will transparently use migrated data.  Run the migration script to persist.
    loras_json_path = os.path.join(scene_dir, "loras.json")
    if os.path.isfile(loras_json_path):
        return _migrate_loras_json_to_stack(loras_json_path)

    return None


def save_lora_stack(scene_dir: str, lora_stack: list) -> None:
    """Persist a lora_stack list to lora_stack.json in the given scene directory."""
    lora_stack_path = os.path.join(scene_dir, "lora_stack.json")
    save_json_file(lora_stack_path, lora_stack)


# ── Legacy helpers kept for backward compat with any external callers ─────────

def load_loras(loras_json_path: str) -> tuple[list, list] | tuple[None, None]:
    """DEPRECATED: use load_lora_stack(scene_dir) instead.
    Retained so any external callers don't break immediately."""
    entries = _migrate_loras_json_to_stack(loras_json_path) if os.path.isfile(loras_json_path) else []
    high = [e for e in entries if e.get("model_target") == "Wan2.2-Wrapper-High"]
    low  = [e for e in entries if e.get("model_target") == "Wan2.2-Wrapper-Low"]
    # Re-shape back to old path/strength/blocks/layer_filter/low_mem_load/merge_loras dicts
    def _to_wanvid(entry: dict) -> dict:
        try:
            path = folder_paths.get_full_path_or_raise("loras", entry["lora"])
        except Exception:
            path = folder_paths.get_full_path("loras", entry["lora"]) or entry["lora"]
        return {
            "path": path,
            "strength": entry["strength_model"],
            "name": os.path.splitext(entry["lora"])[0],
            "blocks": entry.get("blocks", {}),
            "layer_filter": entry.get("layer_filter", ""),
            "low_mem_load": entry.get("low_mem_load", False),
            "merge_loras": entry.get("merge_loras", False),
        }
    loras_high = [_to_wanvid(e) for e in high]
    loras_low  = [_to_wanvid(e) for e in low]
    return (loras_high or None, loras_low or None)


def save_loras(loras_high: list, loras_low: list, loras_json_path: str):
    """DEPRECATED: use save_lora_stack(scene_dir, lora_stack) instead.
    Retained so SceneWanVideoLoraMultiSave continues to function unchanged."""
    high, low = [], []
    for lora in (loras_high or []):
        high.append({
            "lora_name":    os.path.basename(lora["path"]),
            "strength":     lora["strength"],
            "blocks":       lora.get("blocks", {}),
            "layer_filter": lora.get("layer_filter", ""),
            "low_mem_load": lora.get("low_mem_load", False),
            "merge_loras":  lora.get("merge_loras", False),
        })
    for lora in (loras_low or []):
        low.append({
            "lora_name":    os.path.basename(lora["path"]),
            "strength":     lora["strength"],
            "blocks":       lora.get("blocks", {}),
            "layer_filter": lora.get("layer_filter", ""),
            "low_mem_load": lora.get("low_mem_load", False),
            "merge_loras":  lora.get("merge_loras", False),
        })
    save_json_file(loras_json_path, {"high": high, "low": low})

def get_available_stories():
    stories_dir = default_stories_dir() if callable(globals().get('default_stories_dir')) else os.path.join(get_output_directory(), "stories")
    if not os.path.isdir(stories_dir):
        return ["default_story"]
    story_names = []
    for entry in os.listdir(stories_dir):
        entry_path = os.path.join(stories_dir, entry)
        if os.path.isdir(entry_path):
            story_names.append(entry)
    return story_names if story_names else ["default_story"]

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

        logger.debug("%s: node='%s'; input_name_in='%s'", class_name, node_name, input_name_in)

        # All nodes for the workflow
        nodes_data = get_workflow_all_nodes(cls.__name__)

        if nodes_data is None or not isinstance(nodes_data, dict) or len(nodes_data) == 0:
            logger.warning("%s: No nodes available.", class_name)
            return io.NodeOutput(
                input_name_in,
                ""
            )

        logger.debug("%s: nodes_data keys=%s", class_name, list(nodes_data.keys()) if nodes_data else "None")

        # List of node names for the dropdown
        nodes = listify_nodes_data(nodes_data)
        logger.debug("%s: available nodes=%s", class_name, nodes)

        # The default is the first node, if available
        node_id = list(nodes_data.keys())[0] if nodes_data and len(nodes_data) > 0 else None

        # If a node name is provided, extract the node id
        if node_name != "1_Unknown_Node":
            node_id = node_name.split("_", 1)[0] if "_" in node_name else None

        if node_id is None:
            logger.warning("%s: Could not determine node_id from node_name='%s'", class_name, node_name)
            return io.NodeOutput(
                input_name_in,
                ""
            )

        logger.debug("%s: selected node_id=%s", class_name, node_id)

        if isinstance(nodes_data, dict):
            node_data = nodes_data.get(str(node_id), None)
            
            if node_data is None:
                logger.warning("%s: No data found for node_id=%s", class_name, node_id)
                return io.NodeOutput(
                    input_name_in,
                    ""
                )

            node_inputs = node_input_details(cls.__name__, node_data)

            if node_inputs and isinstance(node_inputs, dict):
                logger.debug("%s: node_inputs keys=%s", class_name, list(node_inputs.keys()))
                input_name_out = input_name_in if input_name_in and input_name_in in node_inputs.keys() else None
                
                # If the specified input name is not found, default to the first input
                if input_name_out == "No Inputs" or input_name_out is None:
                    input_name_out = list(node_inputs.keys())[0]

                input_value = node_inputs.get(input_name_out, "")

        logger.info(
            "%s: selected input_name='%s'; input_value='%s'",
            class_name,
            input_name_out,
            input_value,
        )

        return io.NodeOutput(
            input_name_out,
            input_value,
        )

class SceneSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_dir = os.path.join(output_dir, "scenes")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            os.makedirs(os.path.join(default_dir, "default_scene"), exist_ok=True)

        subdir_dict = get_subdirectories(default_dir)
        default_options = sorted(subdir_dict.keys()) if subdir_dict else ["default_scene"]
        default_scene = default_options[0]
        pose_options = list(default_pose_options.keys())
        depth_options = list(default_depth_options.keys())
        
        # Load mask names from default scene
        default_scene_dir = os.path.join(default_dir, default_scene)
        masks_dict = load_masks_json(default_scene_dir)
        mask_options = ["(none)"] + sorted(masks_dict.keys()) if masks_dict else ["(none)"]

        return io.Schema(
            node_id=prefixed_node_id("SceneSelect"),
            display_name="SceneSelect",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("scenes_dir", default=default_dir, tooltip="Directory containing scene subdirectories"),
                io.Combo.Input('selected_scene', options=default_options, default=default_scene, tooltip="Select a scene name"),
                io.Combo.Input(id="depth_image_type", display_name="depth_image_type", options=depth_options, default="depth", tooltip="Type of depth image to use from the scene"),
                io.Combo.Input(id="pose_image_type", display_name="pose_image_type", options=pose_options, default="open", tooltip="Type of pose image to use from the scene"),
                io.Combo.Input(id="mask_name", display_name="mask_name", options=mask_options, default="(none)", tooltip="Name of the mask to use from the scene (select '(none)' to skip mask selection)"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Whether to use the background variant of the mask (True=with background, False=no background)"),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images with PromptCollection"),
                DictType.Output(id="prompt_dict", display_name="prompt_dict", tooltip="Dictionary of composed prompts from the scene"),
                DictType.Output(id="comp_dict", display_name="comp_dict", tooltip="Dictionary of composition names to their fully processed prompt values"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory of the selected scene"),
                io.String.Output(id="input_img_glob", display_name="input_img_glob", tooltip="Input image glob pattern for the scene"),
                io.String.Output(id="output_image_prefix", display_name="output_image_prefix", tooltip="Output image prefix for the scene"),
                io.String.Output(id="output_video_prefix", display_name="output_video_prefix", tooltip="Output video prefix for the scene"),
                io.Image.Output(id="base_image", display_name="base_image", tooltip="Base IMAGE from the scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth IMAGE from the scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask IMAGE from the scene"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from the selected mask image"),
                io.Image.Output(id='canny_image', display_name='canny_image', tooltip='Canny IMAGE from the scene'),
                io.Image.Output(id='pose_image', display_name='pose_image', tooltip='Pose IMAGE from the scene'),
                io.Image.Output(id='upscale_image', display_name='upscale_image', tooltip='Upscaled base IMAGE from the scene'),
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Full multi-target LoRA stack (LORA_STACK_DATA). Feed into LoraStackApply."),
                io.Custom("WANVIDLORA").Output(id="loras_high_out", display_name="loras_high", tooltip="WanVideoWrapper WANVIDLORA list for the Wan2.2-Wrapper-High (first-pass) model"),
                io.Custom("WANVIDLORA").Output(id="loras_low_out",  display_name="loras_low",  tooltip="WanVideoWrapper WANVIDLORA list for the Wan2.2-Wrapper-Low (second-pass) model"),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo 
            ],
            is_output_node=True,
        )
    
    @classmethod
    def fingerprint_inputs(
        cls,
        scenes_dir: str = "",
        selected_scene: str = "",
        depth_image_type: str = "",
        pose_image_type: str = "",
        mask_name: str = "",
        mask_background: bool = True,
        **_,
    ):
        """Invalidate cache whenever any file inside the selected scene folder changes."""
        resolved_dir = scenes_dir if scenes_dir else default_scenes_dir()
        if not resolved_dir or not selected_scene:
            return None
        scene_path = Path(resolved_dir) / selected_scene
        scene_hash, dir_count, file_count = _directory_fingerprint(scene_path)
        return (
            str(scene_path),
            scene_hash,
            dir_count,
            file_count,
            depth_image_type,
            pose_image_type,
            mask_name,
            mask_background,
        )

    @classmethod
    def execute(
        cls,
        scenes_dir="",
        selected_scene="default_scene",
        depth_image_type="depth",
        pose_image_type="open",
        mask_name="",
        mask_background=True,
    ) -> io.NodeOutput:
        className = cls.__name__
        input_types = cls.INPUT_TYPES()
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        logger.debug("%s: unique_id='%s'; extra_pnginfo='%s'", className, unique_id, extra_pnginfo)
        logger.debug("%s: selected_scene input='%s'", className, selected_scene)

        if not scenes_dir:
            scenes_dir = default_scenes_dir()

        if not scenes_dir or not selected_scene:
            logger.warning("%s: scenes_dir or selected_scene is empty", className)
            return io.NodeOutput(None)
        
        scene_dir = os.path.join(scenes_dir, selected_scene)
        logger.debug("%s: using scene_dir='%s' for selected_scene='%s'", className, scene_dir, selected_scene)

        if not os.path.isdir(scene_dir):
            logger.error("%s: scene_dir '%s' is not a valid directory", className, scene_dir)
            return io.NodeOutput(None)
        
        # Load prompts.json for PromptCollection
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_collection = PromptCollection.load_from_json(prompt_json_path)
        
        # Load pose.json
        pose_json_path = os.path.join(scene_dir, "pose.json")
        pose_json = load_json_file(pose_json_path)
        if not pose_json:
            pose_json = "[]"
        else:
            pose_json = json.dumps(pose_json)

        # Load LoRA stack (new format, with automatic legacy migration)
        lora_stack = load_lora_stack(scene_dir)
        if lora_stack is None:
            logger.warning("%s: no lora_stack.json or loras.json found in '%s'", className, scene_dir)

        # Derive WANVIDLORA outputs for backward-compatible workflow wiring
        wan_high_entries = _lora_entries_for_target(lora_stack or [], "Wan2.2-Wrapper-High")
        wan_low_entries  = _lora_entries_for_target(lora_stack or [], "Wan2.2-Wrapper-Low")
        loras_high, _ = _lora_build_wanvid(wan_high_entries, None, False, True)
        loras_low,  _ = _lora_build_wanvid(wan_low_entries,  None, False, True)

        # Load selected/normalized assets (and mask preview/output separation)
        selected_depth_attr = default_depth_options.get(depth_image_type, "depth_image")
        selected_pose_attr = default_pose_options.get(pose_image_type, "pose_open_image")
        
        # Load assets with mask_name (supports both new and legacy mask systems)
        # Treat "(none)" as empty string to skip mask loading
        actual_mask_name = "" if mask_name == "(none)" else mask_name
        logger.info(
            "%s: Loading assets from scene_dir='%s'; mask_name='%s', mask_background=%s",
            className,
            scene_dir,
            actual_mask_name,
            mask_background,
        )
        assets = SceneInfo.load_preview_assets(
            scene_dir,
            depth_attr=selected_depth_attr,
            pose_attr=selected_pose_attr,
            mask_name=actual_mask_name,
            mask_background=mask_background,
            include_upscale=True,
            include_canny=True,
        )

        # Also load full images for SceneInfo completeness (depth variants, masks, canny)
        depth_images_full = SceneInfo.load_depth_images(scene_dir)
        pose_images_full = SceneInfo.load_pose_images(scene_dir)
        
        # Load new mask system if available
        masks_dict = load_masks_json(scene_dir)
        mask_images_dict = {}
        if masks_dict:
            # Load images for all masks in masks.json
            mask_names = list(masks_dict.keys())
            mask_images_full, _ = SceneInfo.load_mask_images(scene_dir, mask_names=mask_names)
            mask_images_dict = mask_images_full
            logger.info(f"%s: Loaded {len(masks_dict)} masks from new system", className)
        else:
            # Fall back to legacy mask loading
            mask_images_full, _ = SceneInfo.load_mask_images(scene_dir)
            logger.info(f"%s: Loaded masks from legacy system", className)
        
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

        logger.debug(
            "%s: depth_image shape: %s",
            className,
            selected_depth_image.shape if selected_depth_image is not None else "None",
        )
        logger.debug(
            "%s: upscale_image shape: %s",
            className,
            base_image.shape if base_image is not None else "None",
        )

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None

        ui_data = {
            "images": preview_image.as_dict().get("images", []) if preview_image else None,
            "animated": preview_image.as_dict().get("animated", False) if preview_image else False,
        }

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=selected_scene,
            pose_json=pose_json,
            resolution=resolution,
            prompts=prompt_collection,
            masks=masks_dict,
            mask_images=mask_images_dict,
            base_image=pose_images_full.get("base_image"),  # Load base image from scene
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
            girl_mask_bkgd_image=mask_images_full.get('girl') if not masks_dict else None,
            male_mask_bkgd_image=mask_images_full.get('male') if not masks_dict else None,
            combined_mask_bkgd_image=mask_images_full.get('combined') if not masks_dict else None,
            girl_mask_no_bkgd_image=mask_images_full.get('girl_no_bg') if not masks_dict else None,
            male_mask_no_bkgd_image=mask_images_full.get('male_no_bg') if not masks_dict else None,
            combined_mask_no_bkgd_image=mask_images_full.get('combined_no_bg') if not masks_dict else None,
            lora_stack=lora_stack,
        )

        # Build prompt_dict and comp_dict from PromptCollection
        prompt_dict = {}  # Individual prompts processed
        comp_dict = {}    # Compositions processed
        
        if prompt_collection:
            libber_manager = LibberStateManager.instance()
            
            # Process individual prompts
            for key, metadata in prompt_collection.prompts.items():
                value = metadata.value
                
                # Apply libber substitution if needed
                if metadata.processing_type == "libber" and metadata.libber_name and libber_manager:
                    libber = libber_manager.ensure_libber(metadata.libber_name)
                    if libber:
                        value = libber.substitute(value)
                
                prompt_dict[key] = value
            
            # Process compositions
            if prompt_collection.compositions:
                comp_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
        
        return io.NodeOutput(
            scene_info,
            prompt_dict,
            comp_dict,
            selected_scene,
            scene_dir,
            scene_info.input_img_glob(),
            scene_info.input_img_dir(),
            os.path.join(scene_info.output_dir(), "vid_"),
            base_image,
            selected_depth_image,
            mask_image,
            mask,
            canny_image,
            pose_image,
            base_image,
            lora_stack,   # LORA_STACK_DATA — full multi-target stack
            loras_high,   # WANVIDLORA — Wan2.2-Wrapper-High entries (backward compat)
            loras_low,    # WANVIDLORA — Wan2.2-Wrapper-Low entries (backward compat)
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
    "nlf": "pose_nlf_image",
}

default_mask_options = {
    "girl": "girl_mask_bkgd",
    "male": "male_mask_bkgd",
    "combined": "combined_mask_bkgd",
    "girl_no_bg": "girl_mask_no_bkgd",
    "male_no_bg": "male_mask_no_bkgd",
    "combined_no_bg": "combined_mask_no_bkgd",
}

def resolve_mask_key(mask_name: str, mask_background: Optional[bool] = None) -> str:
    """Return the mask key to use. 
    
    For new mask system: just returns mask_name
    For legacy system: handles _no_bg suffix based on mask_background flag
    
    Args:
        mask_name: Name of the mask
        mask_background: Optional flag for legacy masks. If provided, adds/removes _no_bg suffix
    
    Returns:
        Mask key/name to use for lookups
    """
    # If mask_background is not specified, return name as-is (new system)
    if mask_background is None:
        return mask_name
    
    # Legacy behavior: add/remove _no_bg suffix
    key = mask_name or "combined"
    if not mask_background and not key.endswith("_no_bg"):
        key = f"{key}_no_bg"
    elif mask_background and key.endswith("_no_bg"):
        # Remove _no_bg if background is requested
        key = key.replace("_no_bg", "")
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

        scene_dir = info_in.scene_dir
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("%s: Invalid scene_dir '%s' in SceneInfo", className, scene_dir)
            return io.NodeOutput(None)

        if not loras_high is None:
            logger.info("%s: Saving %d High LoRA entries to scene_dir '%s'", className, len(loras_high), scene_dir)
            loras_high_path = os.path.join(scene_dir, "loras_high.json")
        else:
            loras_high = []
        if not loras_low is None:
            logger.info("%s: Saving %d Low LoRA entries to scene_dir '%s'", className, len(loras_low), scene_dir)
            loras_low_path = os.path.join(scene_dir, "loras_low.json")
        else:
            loras_low = []

        loras_path = os.path.join(scene_dir, "loras.json")
        save_loras(loras_high, loras_low, loras_path)
        logger.info("%s: Saved LoRA preset to: %s", className, loras_path)

        return io.NodeOutput(info_in)


class SceneLoraStackSave(io.ComfyNode):
    """Save a LoRA stack (from LoraStackCollect) to the given scene directory.

    Writes lora_stack.json — the new multi-target format that replaces the
    Wan-only loras.json.  Connect the lora_stack_data output of LoraStackCollect
    here, or supply a raw stack_json STRING if you prefer text storage.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneLoraStackSave"),
            display_name="SceneLoraStackSave",
            category="🧊 frost-byte/Scene",
            description=(
                "Save a LoRA stack to the scene directory as lora_stack.json. "
                "Connect LoraStackCollect's lora_stack_data output here. "
                "SceneSelect will load this file automatically."
            ),
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info",
                    display_name="scene_info",
                    tooltip="SceneInfo providing the scene directory path.",
                ),
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from LoraStackCollect. Takes priority over stack_json.",
                ),
                io.String.Input(
                    "stack_json",
                    display_name="Stack JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip="Raw JSON string (stack_json output of LoraStackCollect). Used when Stack Data is not connected.",
                ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(
                    id="scene_info_out",
                    display_name="scene_info",
                    tooltip="Pass-through scene_info.",
                ),
                io.Int.Output("entry_count", display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info,
        lora_stack_data: Optional[list] = None,
        stack_json: str = "[]",
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneLoraStackSave: scene_info is None")
            return io.NodeOutput(None, 0)

        scene_dir = scene_info.scene_dir
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("SceneLoraStackSave: invalid scene_dir '%s'", scene_dir)
            return io.NodeOutput(scene_info, 0)

        stack = lora_stack_data if lora_stack_data is not None else _lora_json_to_stack(stack_json)

        save_lora_stack(scene_dir, stack)
        logger.info("SceneLoraStackSave: saved %d entries to '%s/lora_stack.json'", len(stack), scene_dir)

        return io.NodeOutput(scene_info, len(stack))


class SceneCreate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneCreate"),
            display_name="SceneCreate",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input(id="scenes_dir", display_name="scenes_dir", tooltip="Root Directory where all scene subdirectories are saved"),
                io.String.Input(id="scene_name", display_name="scene_name", tooltip="Name of the pose"),
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
                io.Boolean.Input(
                    id="generate_nlf_pose",
                    display_name="generate_nlf_pose",
                    default=False,
                    tooltip="Generate NLF (Neural Lifting Framework) pose from base image"
                ),
                io.Combo.Input(
                    id="nlf_model",
                    display_name="nlf_model",
                    options=["nlf_l_multi_0.3.2.torchscript", "nlf_l_multi_0.2.2.torchscript"],
                    default="nlf_l_multi_0.3.2.torchscript",
                    tooltip="NLF model to use (will auto-download if not present)"
                ),
                io.Boolean.Input(
                    id="nlf_draw_face",
                    display_name="nlf_draw_face",
                    default=True,
                    tooltip="Draw face keypoints in NLF pose rendering"
                ),
                io.Boolean.Input(
                    id="nlf_draw_hands",
                    display_name="nlf_draw_hands",
                    default=True,
                    tooltip="Draw hand keypoints in NLF pose rendering"
                ),
                io.Combo.Input(
                    id="nlf_render_device",
                    display_name="nlf_render_device",
                    options=["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"],
                    default="gpu",
                    tooltip="Device to use for NLF pose rendering (Taichi backend)"
                ),
                io.Boolean.Input(
                    id="nlf_scale_hands",
                    display_name="nlf_scale_hands",
                    default=True,
                    tooltip="Scale hand keypoints in NLF pose rendering"
                ),
                io.Combo.Input(
                    id="nlf_render_backend",
                    display_name="nlf_render_backend",
                    options=["torch", "taichi"],
                    default="torch",
                    tooltip="Rendering backend for NLF poses (torch=more compatible, taichi=faster if installed)"
                ),
                io.Image.Input(id="base_image", display_name="base_image", tooltip="Base image for the scene"),
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="Optional LoRA stack to assign to this scene (from LoraStackCollect)."),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene Information"),
                io.String.Output(id="scene_name_out", display_name="scene_name", tooltip="Name of the created scene")
            ],
        )

    @classmethod
    async def execute(
        cls,
        scenes_dir="",
        scene_name="default_scene",
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
        generate_nlf_pose=False,
        nlf_model="nlf_l_multi_0.3.2.torchscript",
        nlf_draw_face=True,
        nlf_draw_hands=True,
        nlf_render_device="gpu",
        nlf_scale_hands=True,
        nlf_render_backend="torch",
        base_image=None,
        lora_stack_data=None,
    ) -> io.NodeOutput:
        if base_image is None:
            logger.error("SceneCreate: base_image is None")
            return io.NodeOutput(None)
        
        if not scenes_dir:
            scenes_dir = default_scenes_dir()
        
        if not scene_name:
            scene_name = "default_scene"

        scene_dir = os.path.join(scenes_dir, scene_name)

        # Create upscale_image from base_image
        upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
        logger.info(
            "SceneCreate: Created upscale_image from base_image - shape %s",
            upscale_image.shape if torch.is_tensor(upscale_image) else "N/A",
        )

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
        pose_dw_image, pose_json = estimate_dwpose(upscale_image, detect_face=False, resolution=resolution)
        pose_face_image = openpose(upscale_image, include_hand=False, include_face=True, include_body=False, resolution=resolution)
        normalized_upscale_image = image_resize_ess(upscale_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)
        base_image_normalized = image_resize_ess(base_image, W, H, method="keep proportion", interpolation="nearest", multiple_of=16)

        pose_open_image = openpose(normalized_upscale_image, include_face=False, resolution=resolution)
        canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)

        # NLF Pose Generation
        pose_nlf_image = None
        if generate_nlf_pose:
            try:
                from .utils.nlf_pose import load_nlf_model, predict_nlf_pose, render_nlf_pose, nlfpred_to_pose_keypoint
                
                logger.info("SceneCreate: Generating NLF pose...")
                
                # Load NLF model
                nlf_model = load_nlf_model(nlf_model, warmup=True)
                
                # Predict poses from upscale image
                nlf_pred, bboxes = predict_nlf_pose(nlf_model, upscale_image)
                logger.info(f"SceneCreate: NLF detected {len(bboxes)} person(s)")
                
                # Render NLF poses
                pose_nlf_image, nlf_mask = render_nlf_pose(
                    nlf_pred, W, H,
                    draw_face=nlf_draw_face,
                    draw_hands=nlf_draw_hands,
                    render_device=nlf_render_device,
                    scale_hands=nlf_scale_hands,
                    render_backend=nlf_render_backend
                )
                
                # Convert to POSE_KEYPOINT format for potential editing
                pose_keypoints = nlfpred_to_pose_keypoint(nlf_pred, W, H)
                
                # Update pose_json with NLF-derived keypoints
                # This allows users to edit the pose with OpenposeEditorNode
                if pose_keypoints:
                    pose_json = pose_keypoints  # Keep as list for now, will be converted to JSON string below
                    logger.info("SceneCreate: Updated pose.json with NLF-derived keypoints")
                
                logger.info("SceneCreate: NLF pose generation complete")
                
            except ImportError as e:
                logger.error(f"SceneCreate: Failed to import NLF utilities: {e}")
                logger.error("Make sure utils/nlf_pose.py exists and dependencies are installed")
            except Exception as e:
                logger.error(f"SceneCreate: NLF pose generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"
        
        # Convert pose_json (list of dicts) to JSON string for storage
        if isinstance(pose_json, list):
            pose_dwpose_json = json.dumps({'people': pose_json})
        else:
            pose_dwpose_json = json.dumps(pose_json)

        # Create empty PromptCollection for new scenes
        # Users will add prompts via ScenePromptManager
        prompt_collection = PromptCollection()

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=scene_name,
            resolution=resolution,
            prompts=prompt_collection,
            base_image=base_image_normalized,
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
            pose_face_image=pose_face_image,
            pose_nlf_image=pose_nlf_image,
            pose_json=pose_dwpose_json,  # Store as JSON string
            canny_image=canny_image,
            lora_stack=lora_stack_data,
        )
        
        # Save all scene data using the helper method
        scene_info.save_all(scene_dir)
        logger.info("SceneCreate: Saved all scene data to '%s'", scene_dir)
        
        return io.NodeOutput(
            scene_info,
            scene_name,
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
                io.Image.Input(id="base_image", display_name="base_image", tooltip="New base image (if update_base=True)", optional=True),
                io.Boolean.Input(id="update_base", display_name="update_base", tooltip="If true, will replace base_image and regenerate upscale_image and all derived images", default=False),
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
                io.Boolean.Input(id="update_nlf_pose", display_name="update_nlf_pose", tooltip="If true, will update the NLF pose in the scene_info", default=False),
                io.Image.Input(id="pose_image", display_name="pose_image", tooltip="Custom pose image to use (if provided with pose_keypoint, skips generation)", optional=True),
                io.Custom("POSE_KEYPOINT").Input(id="pose_keypoint", display_name="pose_keypoint", tooltip="Custom pose keypoints (if provided with pose_image, skips generation)", optional=True),
                io.Boolean.Input(id="nlf_draw_face", display_name="nlf_draw_face", default=True, tooltip="Draw face keypoints in NLF pose rendering"),
                io.Boolean.Input(id="nlf_draw_hands", display_name="nlf_draw_hands", default=True, tooltip="Draw hand keypoints in NLF pose rendering"),
                io.Combo.Input(
                    id="nlf_render_device",
                    display_name="nlf_render_device",
                    options=["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"],
                    default="gpu",
                    tooltip="Device to use for NLF pose rendering (Taichi backend)"
                ),
                io.Boolean.Input(id="nlf_scale_hands", display_name="nlf_scale_hands", default=True, tooltip="Scale hand keypoints in NLF pose rendering"),
                io.Combo.Input(
                    id="nlf_render_backend",
                    display_name="nlf_render_backend",
                    options=["torch", "taichi"],
                    default="torch",
                    tooltip="Rendering backend for NLF poses (torch=more compatible, taichi=faster if installed)"
                ),
                io.Combo.Input(
                    id="nlf_model",
                    display_name="nlf_model",
                    options=["nlf_l_multi_0.3.2.torchscript", "nlf_l_multi_0.2.2.torchscript"],
                    default="nlf_l_multi_0.3.2.torchscript",
                    tooltip="NLF model to use (will auto-download if not present)"
                ),
                io.Boolean.Input(id="update_loras", display_name="update_loras", tooltip="If true, replaces the scene's LoRA stack with the provided lora_stack_data", default=False),
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
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="New LoRA stack to replace the scene's existing stack (requires update_loras=True)."),
            ],
            hidden=[
                io.Hidden.unique_id,
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info_out", display_name="scene_info", tooltip="Updated Scene Information"),
            ],
        )
        
    @classmethod
    async def execute(
        cls,
        scene_info_in=None,
        base_image=None,
        update_base=False,
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
        update_nlf_pose=False,
        pose_image=None,
        pose_keypoint=None,
        nlf_draw_face=True,
        nlf_draw_hands=True,
        nlf_render_device="gpu",
        nlf_scale_hands=True,
        nlf_render_backend="torch",
        nlf_model="nlf_l_multi_0.3.2.torchscript",
        update_loras=False,
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
        lora_stack_data=None,
    ):
        # Get node ID for status updates
        node_id = cls.hidden.unique_id
        
        send_status_update(node_id, "Starting scene update...")
        logger.info("="*60)
        logger.info("SceneUpdate: Node execution started")
        logger.info("SceneUpdate: update_nlf_pose=%s, update_base=%s, update_upscale=%s", 
                   update_nlf_pose, update_base, update_upscale)
        
        if scene_info_in is None:
            logger.error("SceneUpdate: scene_info is None")
            return io.NodeOutput(None)

        scene_info_out = scene_info_in
        
        # Handle base_image update first (triggers full regeneration)
        if update_base:
            if base_image is None:
                logger.warning("SceneUpdate: update_base=True but base_image is None - attempting to use existing base_image")
                logger.debug("SceneUpdate: scene_info_in.base_image is None: %s", scene_info_in.base_image is None)
                logger.debug("SceneUpdate: scene_info_in.scene_dir: %s", scene_info_in.scene_dir)
                base_image = scene_info_in.base_image
                if base_image is None:
                    logger.error("SceneUpdate: Cannot update - both input base_image and scene_info.base_image are None")
            else:
                logger.info("SceneUpdate: Replacing base_image with new input")
                logger.debug("SceneUpdate: New base_image shape: %s", base_image.shape if hasattr(base_image, 'shape') else 'N/A')
                scene_info_out.base_image = base_image
            
            # Regenerate upscale_image from base_image
            if base_image is not None:
                logger.info(
                    "SceneUpdate: Regenerating upscale_image from base_image using factor %s",
                    upscale_factor,
                )
                upscale_image, = ImageScaleBy().upscale(base_image, upscale_method=upscale_method, scale_by=upscale_factor)
                scene_info_out.upscale_image = upscale_image
                # Force regeneration of all derived images
                update_upscale = True
            else:
                logger.error("SceneUpdate: base_image is None, cannot regenerate upscale_image")
                upscale_image = scene_info_in.upscale_image
        else:
            # Start with existing upscale_image from scene
            upscale_image = scene_info_in.upscale_image
            
            # If user wants to refresh upscale_image, prefer regenerating from base_image.
            # This ensures updates to base.png are reflected even when update_base is False.
            if update_upscale:
                source_base = base_image if base_image is not None else scene_info_in.base_image

                if source_base is not None:
                    logger.info(
                        "SceneUpdate: Regenerating upscale_image from base_image using factor %s and method %s",
                        upscale_factor,
                        upscale_method,
                    )
                    if base_image is not None:
                        logger.info("SceneUpdate: Applying provided base_image while update_base=False")
                        scene_info_out.base_image = base_image
                    base_image = source_base
                    upscale_image, = ImageScaleBy().upscale(source_base, upscale_method=upscale_method, scale_by=upscale_factor)
                    scene_info_out.upscale_image = upscale_image
                elif upscale_image is not None:
                    logger.warning(
                        "SceneUpdate: base_image unavailable; falling back to rescaling existing upscale_image by factor %s using %s",
                        upscale_factor,
                        upscale_method,
                    )
                    upscale_image, = ImageScaleBy().upscale(upscale_image, upscale_method=upscale_method, scale_by=upscale_factor)
                    scene_info_out.upscale_image = upscale_image
                else:
                    logger.error("SceneUpdate: Cannot update upscale_image - no base_image or existing upscale_image available")
        
        if upscale_image is None:
            logger.error("SceneUpdate: upscale_image is None, cannot regenerate derived images")
            return io.NodeOutput(scene_info_out)
        
        # upscale_image is now the source for regenerating all other images

        if update_facepose:
            pose_face_image = openpose(upscale_image, include_hand=False, include_face=True, include_body=False, resolution=resolution)
            scene_info_out.pose_face_image = pose_face_image
            scene_info_out.pose_json = pose_json
        if update_densepose:
            send_status_update(node_id, f"Generating DensePose ({densepose_model})...")
            scene_info_out.pose_dense_image = dense_pose(upscale_image, densepose_model, densepose_cmap, resolution)

        if update_depth:
            send_status_update(node_id, f"Generating depth maps ({depth_any_v2_ckpt})...")
            # Depth Anything
            scene_info_out.depth_any_image = depth_anything(upscale_image, ckpt=depth_any_ckpt, resolution=resolution)
            scene_info_out.depth_image = depth_anything_v2(upscale_image, ckpt=depth_any_v2_ckpt, resolution=resolution)

        # MiDas
        if update_midas:
            send_status_update(node_id, "Generating Midas depth map...")
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
            send_status_update(node_id, "Generating Canny edges...")
            canny_image = canny(upscale_image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, resolution=resolution)
            scene_info_out.canny_image = canny_image

        if update_dwpose:
            send_status_update(node_id, "Generating DWPose...")
            pose_dw_image, pose_json = estimate_dwpose(upscale_image, detect_face=False, resolution=resolution)
            scene_info_out.pose_dw_image = pose_dw_image
            #scene_info_out.pose_json = pose_json

        # Update NLF pose
        if update_nlf_pose:
            send_status_update(node_id, "Processing NLF pose...")
            logger.info("SceneUpdate: NLF pose update requested")
            logger.info("SceneUpdate: pose_image provided: %s", pose_image is not None)
            logger.info("SceneUpdate: pose_keypoint provided: %s", pose_keypoint is not None)
            logger.info("SceneUpdate: base_image available: %s", base_image is not None)
            logger.info("SceneUpdate: upscale_image available: %s", 'upscale_image' in locals())
            
            from .utils.nlf_pose import (
                load_nlf_model,
                predict_nlf_pose,
                render_nlf_pose,
                nlfpred_to_pose_keypoint
            )
            
            # Check if custom pose image and keypoint were provided (edited workflow)
            if pose_image is not None and pose_keypoint is not None:
                logger.info("SceneUpdate: Using provided pose_image and pose_keypoint for NLF pose")
                logger.info("SceneUpdate: pose_image shape: %s", pose_image.shape if hasattr(pose_image, 'shape') else 'unknown')
                logger.info("SceneUpdate: pose_keypoint type: %s, length: %s", 
                           type(pose_keypoint).__name__, 
                           len(pose_keypoint) if isinstance(pose_keypoint, list) else 'N/A')
                scene_info_out.pose_nlf_image = pose_image
                # Update pose.json with custom keypoints for editing support
                # pose_keypoint is a list of dicts in OpenPose format
                if isinstance(pose_keypoint, list):
                    # Store as JSON string
                    import json
                    scene_info_out.pose_json = json.dumps({
                        'people': pose_keypoint
                    })
                    logger.info("SceneUpdate: Updated pose.json with custom pose keypoints")
            else:
                # Regenerate NLF pose from base_image (or upscale_image if base not available)
                logger.debug("SceneUpdate: Checking source images for NLF generation")
                logger.debug("SceneUpdate: base_image is None: %s", base_image is None)
                logger.debug("SceneUpdate: upscale_image defined: %s", 'upscale_image' in locals())
                logger.debug("SceneUpdate: scene_info_in.base_image is None: %s", scene_info_in.base_image is None)
                logger.debug("SceneUpdate: scene_info_in.upscale_image is None: %s", scene_info_in.upscale_image is None)
                
                # Try to get source image from multiple sources
                if base_image is None:
                    base_image = scene_info_in.base_image
                    logger.debug("SceneUpdate: Using scene_info_in.base_image as source")
                
                if 'upscale_image' not in locals():
                    upscale_image = scene_info_in.upscale_image
                    logger.debug("SceneUpdate: Using scene_info_in.upscale_image as fallback")
                
                source_image = base_image if base_image is not None else upscale_image
                logger.info("SceneUpdate: Source image selection - using base_image: %s", base_image is not None)
                
                if source_image is None:
                    logger.error("SceneUpdate: Cannot generate NLF pose - no source image available")
                    logger.error("SceneUpdate: base_image is None: %s", base_image is None)
                    logger.error("SceneUpdate: upscale_image is None: %s", upscale_image is None if 'upscale_image' in locals() else 'not defined')
                else:
                    logger.info("SceneUpdate: Regenerating NLF pose from source image")
                    logger.info("SceneUpdate: Source image shape: %s", source_image.shape)
                    logger.info("SceneUpdate: NLF model: %s", nlf_model)
                    logger.info("SceneUpdate: NLF config - draw_face=%s, draw_hands=%s, render_device=%s, render_backend=%s",
                               nlf_draw_face, nlf_draw_hands, nlf_render_device, nlf_render_backend)
                    try:
                        # Load NLF model
                        send_status_update(node_id, f"Loading NLF model ({nlf_model})...")
                        logger.info("SceneUpdate: Loading NLF model...")
                        nlf_model_obj = load_nlf_model(nlf_model, warmup=True)
                        send_status_update(node_id, "Running NLF prediction...")
                        logger.info("SceneUpdate: NLF model loaded successfully")
                        
                        # Generate NLF predictions (returns tuple of dict and list)
                        logger.info("SceneUpdate: Generating NLF predictions...")
                        nlf_pred_dict, nlf_pred_list = predict_nlf_pose(nlf_model_obj, source_image, per_batch=1)
                        logger.info("SceneUpdate: NLF predictions generated - dict keys: %s, list length: %s",
                                   list(nlf_pred_dict.keys()) if nlf_pred_dict else 'None',
                                   len(nlf_pred_list) if nlf_pred_list else 'None')
                        
                        # Debug NLF prediction structure
                        if nlf_pred_dict and 'joints3d_nonparam' in nlf_pred_dict:
                            joints = nlf_pred_dict['joints3d_nonparam']
                            logger.debug("SceneUpdate: joints3d_nonparam type: %s", type(joints))
                            logger.debug("SceneUpdate: joints3d_nonparam length: %s", len(joints) if hasattr(joints, '__len__') else 'N/A')
                            if isinstance(joints, list) and len(joints) > 0:
                                logger.debug("SceneUpdate: joints[0] type: %s", type(joints[0]))
                                logger.debug("SceneUpdate: joints[0] length: %s", len(joints[0]) if hasattr(joints[0], '__len__') else 'N/A')
                                if len(joints[0]) > 0:
                                    logger.debug("SceneUpdate: joints[0][0] shape: %s", joints[0][0].shape if hasattr(joints[0][0], 'shape') else 'N/A')
                        
                        # Check if any persons were detected
                        num_detections = len(nlf_pred_list) if nlf_pred_list else 0
                        logger.info("SceneUpdate: NLF detected %d person(s)", num_detections)
                        
                        if num_detections == 0:
                            logger.warning("SceneUpdate: No persons detected by NLF - pose_nlf will be black")
                            send_status_update(node_id, "⚠️ NLF: No persons detected in image")
                        else:
                            send_status_update(node_id, f"✓ NLF: Detected {num_detections} person(s)")
                        
                        # Get dimensions from source image
                        h, w = source_image.shape[1], source_image.shape[2]
                        logger.info("SceneUpdate: Target dimensions - width=%s, height=%s", w, h)
                        
                        # Render NLF pose (returns tuple of image tensor and mask tensor)
                        send_status_update(node_id, f"Rendering NLF pose ({nlf_render_backend})...")
                        logger.info("SceneUpdate: Rendering NLF pose...")
                        pose_nlf_image, nlf_mask = render_nlf_pose(
                            nlf_pred_dict,
                            w, h,
                            draw_face=nlf_draw_face,
                            draw_hands=nlf_draw_hands,
                            render_device=nlf_render_device,
                            scale_hands=nlf_scale_hands,
                            render_backend=nlf_render_backend
                        )
                        
                        scene_info_out.pose_nlf_image = pose_nlf_image
                        logger.info("SceneUpdate: NLF pose rendered successfully")
                        logger.info("SceneUpdate: pose_nlf_image shape: %s", pose_nlf_image.shape)
                        logger.info("SceneUpdate: Generated NLF pose image (%sx%s)", w, h)
                        logger.info("SceneUpdate: pose_nlf_image tensor id: %s", id(pose_nlf_image))
                        
                        # Verify it's not being aliased to other pose fields
                        if scene_info_out.pose_dense_image is not None:
                            logger.warning("SceneUpdate: pose_dense_image is also set (tensor id: %s)", id(scene_info_out.pose_dense_image))
                        if scene_info_out.pose_dw_image is not None:
                            logger.warning("SceneUpdate: pose_dw_image is also set (tensor id: %s)", id(scene_info_out.pose_dw_image))
                        if scene_info_out.pose_edit_image is not None:
                            logger.warning("SceneUpdate: pose_edit_image is also set (tensor id: %s)", id(scene_info_out.pose_edit_image))
                        
                        # Convert NLF prediction to POSE_KEYPOINT format for editing
                        # nlfpred_to_pose_keypoint expects just the dict, not the tuple
                        logger.info("SceneUpdate: Converting NLF predictions to POSE_KEYPOINT format...")
                        
                        # Only convert if we have detections
                        if num_detections > 0:
                            try:
                                pose_keypoint_list = nlfpred_to_pose_keypoint(nlf_pred_dict, w, h)
                                logger.info("SceneUpdate: Converted to %s pose keypoint entries", len(pose_keypoint_list) if pose_keypoint_list else 0)
                            except Exception as e:
                                logger.error("SceneUpdate: Failed to convert NLF to POSE_KEYPOINT: %s", str(e))
                                logger.debug("SceneUpdate: Conversion error details:", exc_info=True)
                                pose_keypoint_list = []
                        else:
                            logger.info("SceneUpdate: Skipping POSE_KEYPOINT conversion - no detections")
                            pose_keypoint_list = []
                        
                        # Store as JSON string
                        import json
                        scene_info_out.pose_json = json.dumps({
                            'people': pose_keypoint_list
                        })
                        logger.info("SceneUpdate: Updated pose.json with NLF-derived keypoints for editing")
                        logger.info("SceneUpdate: NLF pose update completed successfully")
                        
                    except ImportError as e:
                        error_msg = (
                            "NLF pose generation failed: ComfyUI-SCAIL-Pose not found. "
                            "Install via ComfyUI-Manager or from https://github.com/kijai/ComfyUI-SCAIL-Pose"
                        )
                        logger.error("SceneUpdate: %s", error_msg)
                        logger.debug("SceneUpdate: Import error details: %s", str(e))
                        send_status_update(node_id, f"⚠ {error_msg}")
                    except Exception as e:
                        logger.error("SceneUpdate: Failed to generate NLF pose: %s", str(e), exc_info=True)
                        send_status_update(node_id, f"⚠ NLF pose generation failed: {str(e)}")

        # Determine target dimensions from reference images
        # Use upscale_image dimensions as the reference since it's the source
        ref_h, ref_w = upscale_image.shape[1], upscale_image.shape[2]
        logger.debug("SceneUpdate: Using upscale_image dimensions as reference: %sx%s", ref_w, ref_h)
        
        # Normalize midas image to match reference dimensions (typically half size)
        if scene_info_out.depth_midas_image is not None and torch.is_tensor(scene_info_out.depth_midas_image):
            midas_h, midas_w = scene_info_out.depth_midas_image.shape[1], scene_info_out.depth_midas_image.shape[2]
            if midas_h != ref_h or midas_w != ref_w:
                logger.debug(
                    "SceneUpdate: Normalizing midas image from %sx%s to %sx%s",
                    midas_w,
                    midas_h,
                    ref_w,
                    ref_h,
                )
                scene_info_out.depth_midas_image = image_resize_ess(
                    scene_info_out.depth_midas_image, ref_w, ref_h,
                    method="keep proportion", interpolation="nearest", multiple_of=16
                )
        
        # Normalize all depth images to reference dimensions
        for depth_attr in ['depth_image', 'depth_any_image', 'depth_zoe_image', 'depth_zoe_any_image']:
            img = getattr(scene_info_out, depth_attr, None)
            if img is not None and torch.is_tensor(img):
                img_h, img_w = img.shape[1], img.shape[2]
                if img_h != ref_h or img_w != ref_w:
                    logger.debug(
                        "SceneUpdate: Normalizing %s from %sx%s to %sx%s",
                        depth_attr,
                        img_w,
                        img_h,
                        ref_w,
                        ref_h,
                    )
                    setattr(scene_info_out, depth_attr, image_resize_ess(
                        img, ref_w, ref_h,
                        method="keep proportion", interpolation="nearest", multiple_of=16
                    ))
        
        # Normalize all pose images to reference dimensions
        for pose_attr in ['pose_dense_image', 'pose_dw_image', 'pose_edit_image', 'pose_face_image', 'pose_open_image', 'pose_nlf_image']:
            img = getattr(scene_info_out, pose_attr, None)
            if img is not None and torch.is_tensor(img):
                img_h, img_w = img.shape[1], img.shape[2]
                if img_h != ref_h or img_w != ref_w:
                    logger.debug(
                        "SceneUpdate: Normalizing %s from %sx%s to %sx%s",
                        pose_attr,
                        img_w,
                        img_h,
                        ref_w,
                        ref_h,
                    )
                    setattr(scene_info_out, pose_attr, image_resize_ess(
                        img, ref_w, ref_h,
                        method="keep proportion", interpolation="nearest", multiple_of=16
                    ))

        normalized_upscale_image = image_resize_ess(upscale_image, ref_w, ref_h, method="keep proportion", interpolation="nearest", multiple_of=16)

        if update_openpose or update_editpose:
            pose_open_image = openpose(normalized_upscale_image, include_face=False, resolution=resolution)
            scene_info_out.pose_open_image = pose_open_image

        # todo: consider whether or not the Face Detection using onnx is even worth it (WanAnimatePreprocess (v2) modified based upon post on github)
        # would require specifying params for ONNX detection model: vitpose, yolo, onnx_device and then all the params for "Pose and Face Detection"

        # Resize existing masks if dimensions changed
        if scene_info_out.masks and scene_info_out.mask_images:
            # Get new dimensions from depth, pose, or base image
            new_H, new_W = None, None
            if scene_info_out.depth_image is not None:
                new_H, new_W = scene_info_out.depth_image.shape[1], scene_info_out.depth_image.shape[2]
            elif scene_info_out.pose_dense_image is not None:
                new_H, new_W = scene_info_out.pose_dense_image.shape[1], scene_info_out.pose_dense_image.shape[2]
            elif scene_info_out.base_image is not None:
                new_H, new_W = scene_info_out.base_image.shape[1], scene_info_out.base_image.shape[2]
            
            if new_H and new_W:
                for mask_name, mask_image in scene_info_out.mask_images.items():
                    old_H, old_W = mask_image.shape[1], mask_image.shape[2]
                    if old_H != new_H or old_W != new_W:
                        logger.info(f"SceneUpdate: Resizing mask '{mask_name}' from {old_W}x{old_H} to {new_W}x{new_H}")
                        scene_info_out.mask_images[mask_name] = normalize_image_tensor(mask_image, new_H, new_W)

        # Update LoRA stack
        if update_loras and lora_stack_data is not None:
            scene_info_out.lora_stack = lora_stack_data

        if update_loras:
            scene_info_out.save_loras()
            logger.info(
                "SceneUpdate: Saved LoRA stack (%d entries) to: %s/lora_stack.json",
                len(scene_info_out.lora_stack or []),
                scene_info_in.scene_dir,
            )

        # Log which pose images are set for debugging
        pose_status = {
            "pose_dense": scene_info_out.pose_dense_image is not None,
            "pose_dw": scene_info_out.pose_dw_image is not None,
            "pose_edit": scene_info_out.pose_edit_image is not None,
            "pose_face": scene_info_out.pose_face_image is not None,
            "pose_open": scene_info_out.pose_open_image is not None,
            "pose_nlf": scene_info_out.pose_nlf_image is not None,
        }
        logger.info("SceneUpdate: Final pose image status: %s", pose_status)

        # Save all updated scene data to disk
        send_status_update(node_id, "Saving scene data...")
        scene_info_out.save_all(scene_info_out.scene_dir)
        logger.info("SceneUpdate: Saved all scene data to '%s'", scene_info_out.scene_dir)

        send_status_update(node_id, "✓ Scene update completed")
        logger.info("SceneUpdate: Node execution completed successfully")
        logger.info("="*60)
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
                io.Combo.Input(
                    id="mask_name", 
                    display_name="mask_name", 
                    options=["none"], 
                    default="none", 
                    tooltip="Name of mask to preview (dynamically updated from scene)"
                ),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Selected Depth Image"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Selected Pose Image"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Selected Mask Image"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask derived from selected mask image"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the selected scene"),
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory of the selected scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    async def execute(
        cls,
        scene_info=Optional[SceneInfo],
        depth_type="depth",
        pose_type="dense",
        mask_name="none",
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneView: scene_info is None")
            return io.NodeOutput(None, None, None, None, None, None)
        
        if not isinstance(scene_info, SceneInfo):
            logger.error("SceneView: scene_info is not of type SceneInfo")
            return io.NodeOutput(None, None, None, None, None, None)

        # Auto-select first mask if mask_name is "none" and masks are available
        if (mask_name == "none" or not mask_name) and scene_info.masks:
            available_masks = sorted(scene_info.masks.keys())
            if available_masks:
                mask_name = available_masks[0]
                logger.info(f"SceneView: Auto-selected first available mask: {mask_name}")

        # Determine include_mask_bg from mask definition
        include_mask_bg = True
        if mask_name and mask_name != "none" and scene_info.masks and mask_name in scene_info.masks:
            mask_def = scene_info.masks[mask_name]
            include_mask_bg = mask_def.has_background
        
        assets = scene_info.load_preview_assets(
            scene_info.scene_dir,
            depth_attr=depth_type,
            pose_attr=pose_type,
            mask_name=mask_name if mask_name != "none" else "",
            mask_background=include_mask_bg,
            include_canny=True,
        )

        mask_image = assets["mask_image"]
        mask = assets["mask"]
        depth_image = assets["depth_image"]
        pose_image = assets["pose_image"]
        girl_pos = getattr(scene_info, "girl_pos", "")
        male_pos = getattr(scene_info, "male_pos", "")
        scene_name = getattr(scene_info, "scene_name", "")
        scene_dir = getattr(scene_info, "scene_dir", "")

        preview_batch = assets.get("preview_batch", [])
        preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None
        
        # Show scene info instead of deprecated prompts
        info_text = f"Scene: {scene_name}\nDepth: {depth_type}\nPose: {pose_type}"
        if mask_name and mask_name != "none":
            info_text += f"\nMask: {mask_name}"
        text_ui = ui.PreviewText(value=info_text)
 
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
            scene_name,
            scene_dir,
            ui=ui_data
        )

class SceneMaskDefinition(io.ComfyNode):
    """
    Define and generate masks for scenes using SAM3 segmentation.
    Outputs an updated scene_info with the mask definition added.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SceneMaskDefinition"),
            display_name="SceneMaskDefinition",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info", 
                    display_name="scene_info", 
                    tooltip="Scene Information (mask will be generated from scene base_image)"
                ),
                io.String.Input(
                    id="mask_name", 
                    display_name="mask_name", 
                    default="mask_1",
                    tooltip="Name of the mask (must be unique within the scene)"
                ),
                io.Combo.Input(
                    id="mask_type",
                    display_name="mask_type",
                    options=["transparent", "color"],
                    default="transparent",
                    tooltip="Type of mask: transparent (alpha-based) or color (colored regions)"
                ),
                io.Boolean.Input(
                    id="has_background",
                    display_name="has_background",
                    default=True,
                    tooltip="Whether the mask includes background (True) or masks it out (False)"
                ),
                io.String.Input(
                    id="mask_color",
                    display_name="mask_color",
                    default="255,255,255",
                    tooltip="RGB color for the mask as r,g,b (e.g., '255,0,0' for red). Only used if mask_type is 'color'"
                ),
                # SAM3 Segmentation parameters
                io.String.Input(
                    id="sam3_prompt",
                    display_name="sam3_prompt",
                    default="",
                    tooltip="Text prompt describing what to segment (e.g., 'person', 'face', 'clothing')"
                ),
                io.Float.Input(
                    id="confidence_threshold",
                    display_name="confidence_threshold",
                    default=0.30,
                    min=0.00,
                    max=1.00,
                    step=0.01,
                    tooltip="Confidence threshold for segmentation (lower = more permissive)"
                ),
                io.Combo.Input(
                    id="background_mode",
                    display_name="background_mode",
                    options=["Alpha", "Color"],
                    default="Alpha",
                    tooltip="Background mode: Alpha (transparent) or Color (solid color)"
                ),
                io.String.Input(
                    id="background_color",
                    display_name="background_color",
                    default="#222222",
                    tooltip="Background color as hex code (only used if background_mode is 'Color')"
                ),
                io.Int.Input(
                    id="max_segments",
                    display_name="max_segments",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Maximum number of segments to keep (0 = no limit)"
                ),
                io.Int.Input(
                    id="segment_pick",
                    display_name="segment_pick",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Pick a specific segment by index (0 = use all segments)"
                ),
                io.Int.Input(
                    id="mask_blur",
                    display_name="mask_blur",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Amount of blur to apply to mask edges"
                ),
                io.Int.Input(
                    id="mask_offset",
                    display_name="mask_offset",
                    default=0,
                    min=-64,
                    max=64,
                    step=1,
                    tooltip="Offset to grow (+) or shrink (-) the mask"
                ),
                io.Combo.Input(
                    id="device",
                    display_name="device",
                    options=["Auto", "CPU", "GPU"],
                    default="Auto",
                    tooltip="Device to run segmentation on"
                ),
                io.Boolean.Input(
                    id="invert_output",
                    display_name="invert_output",
                    default=False,
                    tooltip="Invert the mask output"
                ),
                io.Boolean.Input(
                    id="unload_model",
                    display_name="unload_model",
                    default=False,
                    tooltip="Unload the model after processing to free memory"
                ),
                # MaskProcessor parameters
                io.Int.Input(
                    id="min_hole_size",
                    display_name="min_hole_size",
                    default=10,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Minimum hole size (in pixels) to fill. Holes smaller than this will be filled."
                ),
                io.Int.Input(
                    id="grow_amount",
                    display_name="grow_amount",
                    default=5,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Amount to grow (dilate) the mask borders in pixels"
                ),
                io.Int.Input(
                    id="smooth_iterations",
                    display_name="smooth_iterations",
                    default=0,
                    min=0,
                    max=10,
                    step=1,
                    tooltip="Number of morphological smoothing iterations (can shrink mask)"
                ),
                io.Boolean.Input(
                    id="enable_region_smooth",
                    display_name="enable_region_smooth",
                    default=True,
                    tooltip="Enable region smoothing (Gaussian filter with thresholding - maintains mask size)"
                ),
                io.Int.Input(
                    id="region_smooth_sigma",
                    display_name="region_smooth_sigma",
                    default=128,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Sigma for region smoothing (only used if enabled)"
                ),
                io.Float.Input(
                    id="blur_radius",
                    display_name="blur_radius",
                    default=5.0,
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    tooltip="Gaussian blur radius (sigma value) for edge softening"
                ),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(
                    id="scene_info_out", 
                    display_name="scene_info", 
                    tooltip="Updated Scene Information with mask definition added"
                ),
                io.Image.Output(
                    id="image_out", 
                    display_name="IMAGE", 
                    tooltip="Segmented image with background applied"
                ),
                io.Mask.Output(
                    id="mask_out", 
                    display_name="MASK", 
                    tooltip="Binary mask of segmented region"
                ),
                io.Image.Output(
                    id="mask_image_out", 
                    display_name="MASK_IMAGE", 
                    tooltip="Grayscale visualization of the mask"
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        scene_info=None,
        mask_name="mask_1",
        mask_type="transparent",
        has_background=True,
        mask_color="255,255,255",
        sam3_prompt="",
        confidence_threshold=0.30,
        background_mode="Alpha",
        background_color="#222222",
        max_segments=0,
        segment_pick=0,
        mask_blur=0,
        mask_offset=0,
        device="Auto",
        invert_output=False,
        unload_model=False,
        min_hole_size=10,
        grow_amount=5,
        smooth_iterations=0,
        enable_region_smooth=True,
        region_smooth_sigma=128,
        blur_radius=5.0,
    ) -> io.NodeOutput:
        """Execute mask definition and segmentation"""
        
        if scene_info is None:
            logger.error("SceneMaskDefinition: scene_info is required")
            return io.NodeOutput(None, None, None, None)
        
        if not isinstance(scene_info, SceneInfo):
            logger.error("SceneMaskDefinition: scene_info is not of type SceneInfo")
            return io.NodeOutput(None, None, None, None)
        
        # Get base_image from scene_info
        original_image = scene_info.base_image
        if original_image is None:
            logger.error("SceneMaskDefinition: scene_info.base_image is None - base_image is required for segmentation")
            return io.NodeOutput(None, None, None, None)
        
        # Clone and convert base_image to RGB for SAM3
        # SAM3 requires RGB format (no alpha channel)
        logger.info(f"SceneMaskDefinition: Converting base_image to RGB for SAM3 (original shape: {original_image.shape})")
        
        # Image tensor is [B, H, W, C] in ComfyUI format
        if original_image.shape[-1] == 4:
            # Has alpha channel - extract RGB only
            image_rgb = original_image[..., :3].clone()
            logger.info(f"SceneMaskDefinition: Extracted RGB channels from RGBA image")
        elif original_image.shape[-1] == 3:
            # Already RGB
            image_rgb = original_image.clone()
            logger.info(f"SceneMaskDefinition: Image already in RGB format")
        else:
            logger.error(f"SceneMaskDefinition: Unexpected image channel count: {original_image.shape[-1]}")
            return io.NodeOutput(None, None, None, None)
        
        # Import SAM3 segmentation from ComfyUI-RMBG
        try:
            from .utils.util import import_virtual_package, add_custom_node_to_syspath
            candidates = ["ComfyUI-RMBG", "comfyui-rmbg"]
            rmbg_path = add_custom_node_to_syspath(candidates)
            
            if rmbg_path is None:
                logger.error(
                    "SceneMaskDefinition: ComfyUI-RMBG not found. "
                    "Install from ComfyUI-Manager or https://github.com/AInsert/ComfyUI-RMBG"
                )
                return io.NodeOutput(None, None, None, None)
            
            import_virtual_package("rmbg", rmbg_path)
            from rmbg.py.AILab_SAM3Segment import SAM3Segment # type: ignore
            
            logger.info("SceneMaskDefinition: Successfully imported SAM3Segment")
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to import SAM3Segment: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Run SAM3 segmentation on RGB image
        try:
            logger.info(f"SceneMaskDefinition: Running SAM3 segmentation for mask '{mask_name}'")
            logger.info(f"SceneMaskDefinition: Prompt: '{sam3_prompt}', Confidence: {confidence_threshold}")
            
            sam3_node = SAM3Segment()
            result = sam3_node.segment(
                image=image_rgb,
                prompt=sam3_prompt or "object",
                output_mode="Merged",
                confidence_threshold=confidence_threshold,
                max_segments=max_segments,
                segment_pick=segment_pick,
                mask_blur=mask_blur,
                mask_offset=mask_offset,
                device=device,
                invert_output=invert_output,
                unload_model=unload_model,
                background=background_mode,
                background_color=background_color,
            )
            
            segmented_image, sam3_mask, mask_image = result
            logger.info(f"SceneMaskDefinition: SAM3 segmentation complete")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Segmentation failed: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Process mask using MaskProcessor with original image (can be RGBA)
        try:
            from .utils.images import (
                mask_remove_holes, 
                mask_grow, 
                mask_gaussian_blur, 
                mask_smooth, 
                create_mask_overlay_image, 
                smooth_masks_region_was
            )
            
            logger.info(f"SceneMaskDefinition: Processing mask with MaskProcessor")
            
            # Handle batch: select first mask
            if sam3_mask.dim() == 3:  # [B, H, W]
                mask_single = sam3_mask[0]  # [H, W]
            elif sam3_mask.dim() == 2:  # [H, W]
                mask_single = sam3_mask
            else:
                logger.error(f"SceneMaskDefinition: Unexpected mask shape: {sam3_mask.shape}")
                return io.NodeOutput(None, None, None, None)
            
            # Apply MaskProcessor operations in sequence
            processed_mask = mask_single
            operations = []
            
            # 1. Remove holes
            if min_hole_size > 0:
                processed_mask = mask_remove_holes(processed_mask, min_hole_size=min_hole_size)
                operations.append(f"remove_holes(min_size={min_hole_size})")
            
            # 2. Grow (dilate)
            if grow_amount > 0:
                processed_mask = mask_grow(processed_mask, grow_amount=grow_amount)
                operations.append(f"grow(amount={grow_amount})")
            
            # 3. Smooth (morphological cleanup)
            if smooth_iterations > 0:
                processed_mask = mask_smooth(processed_mask, smooth_iterations=smooth_iterations)
                operations.append(f"smooth(iterations={smooth_iterations})")
            
            # 4. Region smooth (Gaussian with thresholding - WAS method)
            if enable_region_smooth:
                # Need to add batch dim temporarily for smooth_masks_region_was
                if processed_mask.dim() == 2:
                    processed_mask_batch = processed_mask.unsqueeze(0)
                else:
                    processed_mask_batch = processed_mask
                processed_mask_batch = smooth_masks_region_was(processed_mask_batch, sigma=region_smooth_sigma)
                # Extract single mask again
                processed_mask = processed_mask_batch[0] if processed_mask_batch.dim() == 3 else processed_mask_batch
                operations.append(f"region_smooth(sigma={region_smooth_sigma})")
            
            # 5. Gaussian blur (LAST - creates soft edges for blending)
            if blur_radius > 0.0:
                processed_mask = mask_gaussian_blur(processed_mask, blur_radius=blur_radius)
                operations.append(f"gaussian_blur(radius={blur_radius})")
            
            # Ensure output is 3D [B, H, W] for compatibility
            if processed_mask.dim() == 2:
                processed_mask = processed_mask.unsqueeze(0)
            
            operations_str = " -> ".join(operations) if operations else "no operations"
            logger.info(f"SceneMaskDefinition: Applied MaskProcessor operations: {operations_str}")
            
            # Create overlay image using original_image (can be RGBA)
            overlay_image = create_mask_overlay_image(processed_mask, original_image)
            logger.info(f"SceneMaskDefinition: Created overlay_image with shape {overlay_image.shape}")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: MaskProcessor failed: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Parse mask color
        parsed_color: Optional[RGB] = None
        try:
            if mask_type == "color":
                color_parts = [int(x.strip()) for x in mask_color.split(',')]
                if len(color_parts) != 3:
                    raise ValueError("Color must be in format 'r,g,b'")
                parsed_color = (color_parts[0], color_parts[1], color_parts[2])
            else:
                parsed_color = None
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Invalid mask_color format '{mask_color}': {e}")
            return io.NodeOutput(None, None, None, None)
        
        # Create MaskDefinition
        try:
            mask_def = MaskDefinition(
                name=mask_name,
                type=MaskType(mask_type),
                has_background=has_background,
                color=parsed_color
            )
            mask_def.validate()
            logger.info(f"SceneMaskDefinition: Created mask definition for '{mask_name}'")
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to create mask definition: {e}")
            return io.NodeOutput(None, None, None, None)
        
        # Save mask and overlay image to scene directory
        try:
            import numpy as np
            from PIL import Image
            
            scene_dir = scene_info.scene_dir
            if not scene_dir or not os.path.exists(scene_dir):
                logger.error(f"SceneMaskDefinition: Invalid scene_dir: {scene_dir}")
                return io.NodeOutput(None, None, None, None)
            
            # Build save path based on mask definition filename
            mask_filename = mask_def.get_filename()
            save_path = os.path.join(scene_dir, mask_filename)
            
            logger.info(f"SceneMaskDefinition: Saving mask to {save_path}")
            
            # Use PathSaveImageRGBA logic to save the overlay image with mask as alpha
            # Extract first image and mask from batch
            img_tensor = overlay_image[0].cpu().numpy()
            mask_tensor = processed_mask[0].cpu()
            
            # Convert to alpha channel (0-255)
            # Note: invert_mask=False, so we don't invert
            alpha_np = (255.0 * (1.0 - mask_tensor.numpy())).astype(np.uint8)
            
            # Convert to uint8 format for PIL
            img_np = (img_tensor * 255).astype(np.uint8)
            
            # Create PIL image - handle both RGB and RGBA input
            if img_np.shape[-1] == 4:
                # Already RGBA - extract RGB only
                pil_img = Image.fromarray(img_np[..., :3])
            elif img_np.shape[-1] == 3:
                # RGB
                pil_img = Image.fromarray(img_np)
            else:
                logger.error(f"SceneMaskDefinition: Unexpected image channel count: {img_np.shape[-1]}")
                return io.NodeOutput(None, None, None, None)
            
            # Create alpha channel image
            alpha_img = Image.fromarray(alpha_np, mode='L')
            
            # Convert to RGBA and add alpha channel
            pil_img_rgba = pil_img.convert("RGBA")
            pil_img_rgba.putalpha(alpha_img)
            
            # Save the image (format=png, quality=95, create_dirs=False per requirements)
            pil_img_rgba.save(save_path, format="PNG")
            
            logger.info(f"SceneMaskDefinition: Successfully saved mask image to {save_path}")
            
        except Exception as e:
            logger.error(f"SceneMaskDefinition: Failed to save mask image: {e}", exc_info=True)
            return io.NodeOutput(None, None, None, None)
        
        # Add mask definition to scene_info
        scene_info_out = copy.deepcopy(scene_info)
        if scene_info_out.masks is None:
            scene_info_out.masks = {}
        if scene_info_out.mask_images is None:
            scene_info_out.mask_images = {}
        
        # Add or update the mask definition
        scene_info_out.masks[mask_name] = mask_def
        scene_info_out.mask_images[mask_name] = overlay_image
        
        logger.info(f"SceneMaskDefinition: Added mask '{mask_name}' to scene_info")
        logger.info(f"SceneMaskDefinition: Scene now has {len(scene_info_out.masks)} mask(s)")
        
        return io.NodeOutput(
            scene_info_out,
            overlay_image,
            processed_mask,
            overlay_image  # Return overlay_image as MASK_IMAGE output
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
                io.String.Output(id="scene_dir", display_name="scene_dir", tooltip="Directory where the scene is saved"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Name of the pose"),
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
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Multi-target LoRA stack for this scene. Feed into LoraStackApply."),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_info=None,
    ) -> io.NodeOutput:
        if scene_info is None:
            logger.error("SceneOutput: scene_info is None")
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
            ))
        
        logger.info(
            "SceneOutput: scene_dir='%s', scene_name='%s', girl_pos='%s', male_pos='%s', wan_prompt='%s', wan_low_prompt='%s', depth_image shape=%s",
            scene_info.scene_dir,
            scene_info.scene_name,
            scene_info.girl_pos[:32],
            scene_info.male_pos[:32],
            scene_info.wan_prompt[:32],
            scene_info.wan_low_prompt[:32],
            scene_info.depth_image.shape if scene_info.depth_image is not None else "None",
        )
        return io.NodeOutput(
            scene_info.scene_dir,
            scene_info.scene_name,
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
            scene_info.lora_stack,
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
                io.String.Input(id="scene_dir", display_name="scene_dir", optional=True, tooltip="The Pose directory for the scene, overrides the scene_info", multiline=False, default=""),
            ],
            outputs=[],
            is_output_node=True,
        )        

    @classmethod
    def execute(
        cls,
        scene_info=None,
        scene_dir="",
    ) -> io.NodeOutput:
        if scene_info is None or not scene_info.scene_name:
            logger.error("SaveScene: scene_info is None or scene_name is empty")
            return io.NodeOutput(None)

        # Use provided scene_dir or fall back to scene_info's scene_dir
        target_dir = scene_dir if scene_dir else scene_info.scene_dir
        if not target_dir:
            target_dir = str(Path(default_scenes_dir()) / scene_info.scene_name)

        logger.info("SaveScene: scene_name='%s'; dest_dir='%s'", scene_info.scene_name, target_dir)
        
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
                io.String.Input(id="scene_dir", display_name="scene_dir", tooltip="Directory where the scene is saved", multiline=False, default=""),
                io.String.Input(id="scene_name", display_name="scene_name", tooltip="Name of the pose", multiline=False, default=""),
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
                LoraStackData.Input("lora_stack_data", display_name="LoRA Stack", optional=True, tooltip="Multi-target LoRA stack. If omitted, loaded from scene directory on disk."),
            ],
            outputs=[
                io.Custom("SCENE_INFO").Output(id="scene_info", display_name="scene_info", tooltip="Scene information and images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_dir="",
        scene_name="",
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
        lora_stack_data=None,
    ) -> io.NodeOutput:
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("SceneInput: scene_dir '%s' is invalid", scene_dir)
            return io.NodeOutput(None)

        logger.info("SceneInput: scene_dir='%s'; scene_name='%s'", scene_dir, scene_name)
        resolution = min(depth_image.shape[1], depth_image.shape[2]) if depth_image is not None else 512

        scene_info = SceneInfo(
            scene_dir=scene_dir,
            scene_name=scene_name,
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
            lora_stack=lora_stack_data if lora_stack_data is not None else load_lora_stack(scene_dir),
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
        default_scenes_dir_path = default_scenes_dir()
        
        # Get available scenes
        scenes_subdir_dict = get_subdirectories(default_scenes_dir_path)
        available_scenes = sorted(scenes_subdir_dict.keys()) if scenes_subdir_dict else ["default_scene"]
        
        # Placeholder for prompt keys (dynamically populated in UI)
        prompt_key_options = ["(select a key from scene)"]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryCreate"),
            display_name="StoryCreate",
            category="🧊 frost-byte/Story",
            inputs=[
                io.String.Input(id="story_name", display_name="story_name", default="my_story", tooltip="Name of the story"),
                io.String.Input(id="story_dir", display_name="story_dir", default=default_stories_dir_path, tooltip="Directory to save the story"),
                io.Combo.Input(id="initial_scene", display_name="initial_scene", options=available_scenes, default=available_scenes[0], tooltip="First scene to add to the story"),
                io.String.Input(id="mask_name", display_name="mask_name", default="", tooltip="Name of mask from the scene (leave empty to skip)"),
                io.Boolean.Input(id="mask_background", display_name="mask_background", default=True, tooltip="Include background in mask"),
                io.Combo.Input(id="prompt_source", display_name="prompt_source", options=["prompt", "composition", "custom"], default="prompt", tooltip="Source of the prompt: 'prompt' (from prompt_dict), 'composition' (from composition_dict), or 'custom'"),
                io.String.Input(id="prompt_key", display_name="prompt_key", default="", tooltip="Key from scene's prompt_dict or composition_dict (leave empty for custom)"),
                io.String.Input(id="custom_prompt", display_name="custom_prompt", default="", multiline=True, tooltip="Custom prompt (only used if prompt_source is 'custom')"),
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
        initial_scene="default_scene",
        mask_name="",
        mask_background=True,
        prompt_source="prompt",
        prompt_key="",
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
            mask_name=mask_name,
            mask_background=mask_background,
            prompt_source=prompt_source,
            prompt_key=prompt_key,
            custom_prompt=custom_prompt,
            depth_type=depth_type,
            pose_type=pose_type,
        )
        
        story_info = StoryInfo(
            version=2,
            story_name=story_name,
            story_dir=str(story_path),
            scenes=[initial_scene_obj]
        )
        
        logger.info(
            "StoryCreate: Created story '%s' (v2) with initial scene '%s' using %s:%s",
            story_name,
            initial_scene,
            prompt_source,
            prompt_key or "custom",
        )
        
        return io.NodeOutput(story_info)

class StoryEdit(io.ComfyNode):
    """View and preview a story with scene selection. CRUD operations handled via frontend REST API."""
    @classmethod
    def define_schema(cls):
        available_stories = get_available_stories() if callable(globals().get('get_available_stories')) else ["default_story"]
        
        # Get scene names from first story for initial preview_scene options
        default_scene_options = [""]  # Empty option means "use first scene"
        if available_stories:
            first_story_info = cls._load_story_info(available_stories[0])
            if first_story_info and hasattr(first_story_info, 'scenes') and first_story_info.scenes:
                sorted_scenes = sorted(first_story_info.scenes, key=lambda s: s.scene_order)
                default_scene_options.extend([scene.scene_name for scene in sorted_scenes])
        
        return io.Schema(
            node_id=prefixed_node_id("StoryEdit"),
            display_name="StoryEdit",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_select", display_name="Story", options=available_stories, default=available_stories[0], tooltip="Select a story to view/edit"),
                io.Combo.Input(id="preview_scene_name", display_name="Preview Scene", options=default_scene_options, default=default_scene_options[0], tooltip="Scene within the story to preview (empty selects the first scene)"),
            ],
            outputs=[
                io.Custom("STORY_INFO").Output(id="story_info_out", display_name="story_info", tooltip="Loaded story information"),
                io.Image.Output(id="base_image", display_name="base_image", tooltip="Base/upscale image for preview scene"),
                io.Image.Output(id="mask_image", display_name="mask_image", tooltip="Mask image for preview scene"),
                io.Mask.Output(id="mask", display_name="mask", tooltip="Alpha mask for preview scene"),
                io.Image.Output(id="pose_image", display_name="pose_image", tooltip="Pose image for preview scene"),
                io.Image.Output(id="depth_image", display_name="depth_image", tooltip="Depth image for preview scene"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def validate_inputs(cls, story_select: str = "default_story", preview_scene_name: str = ""):
        """Validate that story_select exists in the stories directory."""
        if not story_select:
            return "Story selection is required"
        
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_select / "story.json"
        
        if not story_json_path.exists():
            return f"Story '{story_select}' not found at {story_json_path}"
        
        # Validate preview_scene_name if provided
        if preview_scene_name:
            story_info = cls._load_story_info(story_select)
            if story_info and hasattr(story_info, 'scenes'):
                scene_names = [scene.scene_name for scene in story_info.scenes]
                if preview_scene_name not in scene_names:
                    return f"Scene '{preview_scene_name}' not found in story '{story_select}'"
        
        return True
    
    @classmethod
    def fingerprint_inputs(cls, story_select: str = "default_story", preview_scene_name: str = ""):
        """Generate fingerprint based on stories directory modification time to trigger combo refresh."""
        try:
            stories_dir = default_stories_dir()
            stories_path = Path(stories_dir)
            
            # Collect all story.json modification times and sizes
            story_fingerprints = []
            if stories_path.exists():
                for story_dir in stories_path.iterdir():
                    if story_dir.is_dir():
                        story_json = story_dir / "story.json"
                        if story_json.exists():
                            st = os.stat(story_json)
                            story_fingerprints.append((story_dir.name, int(st.st_mtime), int(st.st_size)))
            
            # Sort by name for consistent fingerprinting
            story_fingerprints.sort()
            
            logger.debug("StoryEdit: Fingerprint includes %d stories", len(story_fingerprints))
            return tuple(story_fingerprints) if story_fingerprints else None
            
        except Exception as e:
            logger.warning("StoryEdit: Failed to generate fingerprint: %s", e)
            return None
    
    @classmethod
    def execute(
        cls,
        story_select="default_story",
        preview_scene_name="",
    ) -> io.NodeOutput:
        # Load story from file system
        story_info = cls._load_story_info(story_select)
        if story_info is None:
            logger.error("StoryEdit: Story '%s' could not be loaded", story_select)
            return io.NodeOutput(None, None, None, None, None, None)
        
        # Resolve which scene to preview
        preview_scene = cls._resolve_preview_scene(story_info, preview_scene_name)
        
        # Initialize preview outputs
        base_image = None
        mask_image = None
        mask = None
        pose_image = None
        depth_image = None
        selected_prompt_text = ""
        preview_image_ui = None
        
        # Load preview assets if we have a scene
        if preview_scene:
            assets = cls._load_scene_assets(preview_scene)
            base_image = assets.get("base_image")
            mask_image = assets.get("mask_image")
            mask = assets.get("mask")
            pose_image = assets.get("pose_image")
            depth_image = assets.get("depth_image")
            selected_prompt_text = cls._load_prompt_text(
                preview_scene.scene_name,
                preview_scene.prompt_source,
                preview_scene.prompt_key,
                preview_scene.custom_prompt,
            )
            
            # Build preview image UI
            preview_batch = assets.get("preview_batch", [])
            if preview_batch:
                try:
                    preview_image_ui = ui.PreviewImage(image=torch.cat(preview_batch, dim=0))
                except Exception as exc:
                    logger.exception("StoryEdit: Failed to build preview image UI")
        
        # Build summary text and metadata
        summary_text = cls._build_summary_text(story_info, preview_scene)
        meta_payload = cls._build_meta_payload(story_info, preview_scene)
        
        # Combine UI elements
        ui_payload = {
            "text": [summary_text, selected_prompt_text, meta_payload],
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        return io.NodeOutput(
            story_info,
            base_image,
            mask_image,
            mask,
            pose_image,
            depth_image,
            ui=ui_payload
        )
    
    @staticmethod
    def _load_story_info(story_select: str) -> Optional[StoryInfo]:
        """Load story from filesystem"""
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_select / "story.json"
        if not story_json_path.exists():
            logger.warning("StoryEdit: Story file not found at '%s'", story_json_path)
            return None
        return load_story(str(story_json_path))
    
    @staticmethod
    def _resolve_preview_scene(story_info: StoryInfo, preview_scene_name: str) -> Optional[SceneInStory]:
        """Determine which scene to preview"""
        if not story_info or not getattr(story_info, "scenes", None):
            logger.warning("StoryEdit: Story has no scenes to preview")
            return None
        
        # If a specific scene name is provided, find it
        if preview_scene_name:
            for scene in story_info.scenes:
                if scene.scene_name == preview_scene_name:
                    return scene
        
        # Default to first scene by order
        return sorted(story_info.scenes, key=lambda s: s.scene_order)[0]
    
    @staticmethod
    def _load_scene_assets(scene: SceneInStory) -> dict:
        """Load preview assets for a scene"""
        scenes_dir = default_scenes_dir()
        scene_dir = os.path.join(scenes_dir, scene.scene_name)
        if not os.path.isdir(scene_dir):
            logger.warning("StoryEdit: Scene directory '%s' missing for preview", scene_dir)
            return {}
        
        depth_attr = default_depth_options.get(scene.depth_type, "depth_image")
        pose_attr = default_pose_options.get(scene.pose_type, "pose_open_image")
        
        try:
            assets = SceneInfo.load_preview_assets(
                scene_dir,
                depth_attr=depth_attr,
                pose_attr=pose_attr,
                mask_name=scene.mask_name,
                mask_background=scene.mask_background,
                include_upscale=True,
                include_canny=False,
            )
            assets["scene_dir"] = scene_dir
            return assets
        except Exception as exc:
            logger.exception("StoryEdit: Failed to load preview assets for '%s'", scene.scene_name)
            return {}
    
    @staticmethod
    def _load_prompt_text(scene_name: str, prompt_source: str, prompt_key: str, custom_prompt: str) -> str:
        """Load prompt text for preview"""
        if prompt_source == "custom":
            return custom_prompt or ""
        
        scene_dir = os.path.join(default_scenes_dir(), scene_name)
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        prompt_data_raw = load_prompt_json(prompt_json_path) or {}
        
        if prompt_data_raw.get("version") == 2:            
            prompt_collection = PromptCollection.from_dict(prompt_data_raw)
            libber_manager = LibberStateManager.instance()
            
            # Build individual prompts
            prompt_dict = {}
            for key, metadata in prompt_collection.prompts.items():
                value = metadata.value
                if metadata.processing_type == "libber" and metadata.libber_name:
                    libber = libber_manager.ensure_libber(metadata.libber_name)
                    if libber:
                        value = libber.substitute(value)
                prompt_dict[key] = value
            
            # Build compositions
            compositions = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager) if prompt_collection.compositions else {}
            
            if prompt_source == "prompt" and prompt_key:
                return prompt_dict.get(prompt_key, "")
            if prompt_source == "composition" and prompt_key:
                return compositions.get(prompt_key, "")
            return ""
        
        # Legacy format fallback
        if prompt_key:
            return prompt_data_raw.get(prompt_key, "")
        return ""
    
    @staticmethod
    def _build_summary_text(story_info: StoryInfo, preview_scene: Optional[SceneInStory]) -> str:
        """Build text summary of story and scenes"""
        selected_id = preview_scene.scene_id if preview_scene else ""
        lines = []
        for scene in sorted(getattr(story_info, "scenes", []), key=lambda s: s.scene_order):
            marker = "▶ " if selected_id and scene.scene_id == selected_id else "  "
            mask_suffix = "" if scene.mask_background else " (no bg)"
            prompt_display = f"{scene.prompt_source}:{scene.prompt_key}" if scene.prompt_key else scene.prompt_source
            lines.append(
                f"{marker}{scene.scene_order}: {scene.scene_name} | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={prompt_display} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
        
        summary_header = (
            f"Story: {story_info.story_name}\n"
            f"Dir: {story_info.story_dir}\n"
            f"Scenes: {len(getattr(story_info, 'scenes', []))}\n"
            f"Preview: {preview_scene.scene_name if preview_scene else '(none)'}\n\n"
            "Scenes:\n"
        )
        return summary_header + ("\n".join(lines) if lines else "No scenes available")
    
    @staticmethod
    def _build_meta_payload(story_info: StoryInfo, preview_scene: Optional[SceneInStory]) -> str:
        """Build JSON metadata for frontend"""
        # Include full scene data for frontend table
        scenes_dir = default_scenes_dir()
        scenes_data = []
        for scene in getattr(story_info, "scenes", []):
            # Load available masks for this scene
            available_masks = ["none"]
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if os.path.isdir(scene_dir):
                try:
                    # Load new mask system masks
                    masks_dict = load_masks_json(scene_dir)
                    available_masks.extend(masks_dict.keys())
                    
                    # Add legacy masks if they exist
                    legacy_mask_names = ["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"]
                    for legacy_name in legacy_mask_names:
                        mask_file = f"{legacy_name.replace('_no_bg', '_mask_no_bkgd' if '_no_bg' in legacy_name else '_mask_bkgd')}.png"
                        mask_path = os.path.join(scene_dir, mask_file)
                        if os.path.exists(mask_path) and legacy_name not in available_masks:
                            available_masks.append(legacy_name)
                except Exception as e:
                    logger.debug(f"StoryEdit: Could not load masks for scene '{scene.scene_name}': {e}")
            
            scenes_data.append({
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_type": scene.mask_type,
                "mask_background": scene.mask_background,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key or "",
                "custom_prompt": scene.custom_prompt or "",
                "video_prompt_source": getattr(scene, "video_prompt_source", "auto"),
                "video_prompt_key": getattr(scene, "video_prompt_key", ""),
                "video_custom_prompt": getattr(scene, "video_custom_prompt", ""),
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
                "use_depth": getattr(scene, "use_depth", False),
                "use_mask": getattr(scene, "use_mask", False),
                "use_pose": getattr(scene, "use_pose", False),
                "use_canny": getattr(scene, "use_canny", False),
                "available_masks": available_masks,
            })
        
        payload = {
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scene_count": len(getattr(story_info, "scenes", [])),
            "preview_scene": preview_scene.scene_name if preview_scene else None,
            "scenes": scenes_data,
        }
        return json.dumps(payload)

class StoryView(io.ComfyNode):
    """View and select scenes from a story with preview capabilities"""
    @classmethod
    def define_schema(cls):
        # Get default scene options for when no story is loaded
        default_scenes_dir_path = default_scenes_dir()
        scenes_subdir_dict = get_subdirectories(default_scenes_dir_path)
        default_scene_options = sorted(scenes_subdir_dict.keys()) if scenes_subdir_dict else ["default_scene"]
        
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
        selected_scene="default_scene",
        prompt_in="",
        prompt_action="use_file",
    ) -> io.NodeOutput:
        className = cls.__name__
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        
        if story_info is None:
            logger.error("StoryView: story_info is None")
            return io.NodeOutput(None, None, "", "", 0, "", "", None, None, None)
        
        # Find the selected scene configuration in the story
        scene_config = None
        for scene in story_info.scenes:
            if scene.scene_name == selected_scene:
                scene_config = scene
                break
        
        if scene_config is None and story_info.scenes:
            logger.warning(
                "StoryView: Scene '%s' not found in story, defaulting to first scene '%s'",
                selected_scene,
                story_info.scenes[0].scene_name,
            )
            scene_config = story_info.scenes[0]
            selected_scene = scene_config.scene_name

        # If scene not found in story, create a default configuration
        if scene_config is None:
            logger.warning("StoryView: Scene '%s' not found in story, using defaults", selected_scene)
            scene_config = SceneInStory(
                scene_name=selected_scene,
                scene_order=0,
                mask_name="",
                mask_background=True,
                prompt_source="prompt",
                prompt_key="",
                custom_prompt="",
                depth_type="depth",
                pose_type="open",
            )
        
        # Load scene data from scene directory
        scenes_dir = default_scenes_dir()
        scene_dir = os.path.join(scenes_dir, selected_scene)
        
        if not os.path.isdir(scene_dir):
            logger.error("StoryView: scene_dir '%s' is not a valid directory", scene_dir)
            return io.NodeOutput(story_info, None, story_info.story_name, story_info.story_dir, len(story_info.scenes), selected_scene, "", None, None, None)
        
        try:
            scene_info, assets, selected_prompt, prompt_data, prompt_widget_text = SceneInfo.from_story_scene(
                scene_config,
                scenes_dir=scenes_dir,
                prompt_in=prompt_in,
                prompt_action=prompt_action,
                include_upscale=False,
                include_canny=False,
            )
        except Exception as e:
            logger.exception("StoryView: failed to build SceneInfo for '%s'", selected_scene)
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
            
            # Display prompt_source:prompt_key or custom
            prompt_display = f"{scene.prompt_source}:{scene.prompt_key}" if scene.prompt_key else scene.prompt_source
            
            scene_line = (
                f"{marker}{scene.scene_order}: {scene.scene_name} [{scene.scene_id[:8]}] | "
                f"mask={scene.mask_type}{mask_suffix} | "
                f"prompt={prompt_display} | "
                f"depth={scene.depth_type} | "
                f"pose={scene.pose_type}"
            )
            if scene.prompt_source == "custom" and scene.custom_prompt:
                scene_line += f" | custom='{scene.custom_prompt[:30]}...'"
            scene_list_lines.append(scene_line)
        
        scene_list_text = "\n".join(scene_list_lines) if scene_list_lines else "No scenes"
        
        prompt_display = f"{scene_config.prompt_source}:{scene_config.prompt_key}" if scene_config.prompt_key else scene_config.prompt_source
        
        preview_text = (
            f"Story: {story_info.story_name}\n"
            f"Dir: {story_info.story_dir}\n"
            f"Scenes: {len(story_info.scenes)}\n"
            f"Selected: {selected_scene} (order {scene_config.scene_order})\n"
            f"Prompt: {prompt_display}\n"
            f"Prompt Text: {selected_prompt}\n\n"
            f"All Scenes:\n{scene_list_text}"
        )
        text_ui = ui.PreviewText(value=preview_text)
        
        # Combine UI elements
        combined_ui = {
            "text": text_ui.as_dict().get("text", []),
            "images": preview_image_ui.as_dict().get("images", []) if preview_image_ui else [],
            "animated": preview_image_ui.as_dict().get("animated", False) if preview_image_ui else False,
        }
        
        logger.info(
            "StoryView: Story '%s' - Selected scene '%s' with prompt '%s'",
            story_info.story_name,
            selected_scene,
            prompt_display,
        )
        
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
        # Get available stories for dropdown
        stories_dir = default_stories_dir()
        available_stories = get_subdirectories(stories_dir)
        story_names = list(available_stories.keys()) if available_stories else [""]
        
        # Job ID options will be populated dynamically by frontend when story_name changes
        # Empty string means auto-generate a new unique job_id
        job_id_options = [""]
        
        return io.Schema(
            node_id=prefixed_node_id("StorySceneBatch"),
            display_name="StorySceneBatch",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_name", display_name="story_name", options=story_names, default=story_names[0] if story_names else "", tooltip="Select story to batch process"),
                io.Combo.Input(id="job_id", display_name="job_id", options=job_id_options, default="", tooltip="Select existing job_id or leave empty to auto-generate. Options update when story changes."),
            ],
            outputs=[
                io.Int.Output(id="scene_count", display_name="scene_count", tooltip="Total number of scenes"),
                io.Custom("SCENE_BATCH").Output(id="scene_batch", display_name="scene_batch", tooltip="Ordered list of scene dictionaries"),
                io.String.Output(id="job_id_out", display_name="job_id", tooltip="Job id used for this batch"),
                io.String.Output(id="job_root_dir_out", display_name="job_root_dir", tooltip="Resolved job root directory"),
            ],
        )

    @classmethod
    def validate_inputs(cls, story_name: str = "", job_id: str = ""):
        """Validate that job_id is valid for the selected story."""
        if not story_name:
            return "Story name is required"
        
        if not job_id:
            # Empty job_id is allowed - will auto-generate
            return True
        
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            return f"Story '{story_name}' not found"
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return f"Failed to load story '{story_name}'"
        
        available_jobs = list_job_ids(story_info.story_dir)
        if job_id not in available_jobs:
            return f"Job ID '{job_id}' not found in story '{story_name}'. Available jobs: {', '.join(available_jobs) if available_jobs else '(none)'}"
        
        return True

    @classmethod
    def fingerprint_inputs(cls, story_name: str = "", job_id: str = ""):
        """Generate fingerprint based on story.json and all referenced scene directory content."""
        if story_name:
            logger.debug("StorySceneBatch: Generating fingerprint for story '%s'", story_name)
            stories_dir = default_stories_dir()
            story_json_path = Path(stories_dir) / story_name / "story.json"
            if story_json_path.exists():
                try:
                    story_stat = os.stat(story_json_path)
                    story_info = load_story(str(story_json_path))

                    if not story_info or not getattr(story_info, "scenes", None):
                        return (
                            str(story_json_path),
                            int(story_stat.st_mtime_ns),
                            int(story_stat.st_size),
                            (),
                            job_id.strip(),
                        )

                    scenes_dir = Path(default_scenes_dir())
                    scene_fingerprints: list[tuple[str, str, int, int]] = []

                    for scene in sorted(story_info.scenes, key=lambda s: (s.scene_order, s.scene_name)):
                        scene_dir = scenes_dir / scene.scene_name
                        scene_hash, dir_count, file_count = _directory_fingerprint(scene_dir)
                        scene_fingerprints.append((scene.scene_name, scene_hash, dir_count, file_count))

                    return (
                        str(story_json_path),
                        int(story_stat.st_mtime_ns),
                        int(story_stat.st_size),
                        tuple(scene_fingerprints),
                        job_id.strip(),
                    )
                except Exception as e:
                    logger.warning("StorySceneBatch: Failed to stat story.json for fingerprinting: %s", e)
        # Return None to use default fingerprinting behavior
        return None

    @classmethod
    def execute(
        cls,
        story_name: str = "",
        job_id: str = "",
    ) -> io.NodeOutput:
        if not story_name:
            logger.warning("StorySceneBatch: story_name is empty")
            return io.NodeOutput(0, [], "", "")
        
        # Load story from filesystem
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            logger.error("StorySceneBatch: Story '%s' not found at '%s'", story_name, story_json_path)
            return io.NodeOutput(0, [], "", "")
        
        story_info = load_story(str(story_json_path))
        if not story_info or not getattr(story_info, "scenes", None):
            logger.error("StorySceneBatch: Failed to load story '%s' or story has no scenes", story_name)
            return io.NodeOutput(0, [], "", "")

        # Auto-generate unique job_id - check for collisions with existing jobs
        jobs_dir = Path(story_info.story_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided job_id or generate a new unique one
        resolved_job_id = job_id.strip() if job_id else ""
        
        if resolved_job_id:
            # User provided job_id - validate and use it
            logger.info("StorySceneBatch: Using user-provided job_id='%s'", resolved_job_id)
            job_root = jobs_dir / resolved_job_id
            
            # Check if this job already exists
            if job_root.exists():
                logger.warning(
                    "StorySceneBatch: Job directory '%s' already exists, will reuse it",
                    job_root
                )
        else:
            # Auto-generate unique job_id
            existing_job_ids = set()
            if jobs_dir.exists():
                existing_job_ids = {d.name for d in jobs_dir.iterdir() if d.is_dir()}
            
            # Generate unique job_id (should succeed on first try, but be safe)
            max_attempts = 100
            for _ in range(max_attempts):
                candidate_id = uuid.uuid4().hex[:12]
                if candidate_id not in existing_job_ids:
                    resolved_job_id = candidate_id
                    break
            
            if not resolved_job_id:
                logger.error("StorySceneBatch: Failed to generate unique job_id after %d attempts", max_attempts)
                return io.NodeOutput(0, [], "", "")
            
            logger.info("StorySceneBatch: Auto-generated unique job_id='%s'", resolved_job_id)
            job_root = jobs_dir / resolved_job_id
        
        # Create job root directory if it doesn't exist
        job_root.mkdir(parents=True, exist_ok=True)

        scenes_dir = default_scenes_dir()
        batch: list[dict] = []

        scenes_sorted = sorted(story_info.scenes, key=lambda s: s.scene_order)
        logger.info(
            "StorySceneBatch: Preparing batch for story '%s' with %d scenes under job_id='%s' at '%s'",
            story_info.story_name,
            len(scenes_sorted),
            resolved_job_id,
            job_root,
        )
        for scene in scenes_sorted:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            prompt_path = os.path.join(scene_dir, "prompts.json")
            prompt_data_raw = load_prompt_json(prompt_path) or {}
            
            logger.debug(
                "StorySceneBatch: Processing scene '%s' (order %s)",
                scene.scene_name,
                scene.scene_order,
            )
            
            # Log scene configuration for debugging
            logger.info(
                "StorySceneBatch: Scene '%s' - depth_type='%s', pose_type='%s', mask_type='%s', use_pose=%s, use_depth=%s",
                scene.scene_name,
                scene.depth_type,
                scene.pose_type,
                scene.mask_type,
                scene.use_pose,
                scene.use_depth,
            )
            
            # Load PromptCollection and compose prompts using the new system
            if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
                logger.debug("StorySceneBatch: Detected v2 prompt format for scene '%s'", scene.scene_name)
                prompt_collection = PromptCollection.from_dict(prompt_data_raw)
                # Use shared LibberStateManager so any loaded libbers are applied across nodes
                libber_manager = LibberStateManager.instance()
                
                # Build prompt_dict: individual prompts (not composed)
                prompt_dict = {}
                for key, metadata in prompt_collection.prompts.items():
                    value = metadata.value
                    # Process libber substitution if needed
                    if metadata.processing_type == "libber" and metadata.libber_name:
                        libber = libber_manager.ensure_libber(metadata.libber_name)
                        if libber:
                            value = libber.substitute(value)
                    prompt_dict[key] = value
                
                # Build composition_dict: composed prompts from compositions
                composition_dict = {}
                if prompt_collection.compositions:
                    composition_dict = prompt_collection.compose_prompts(prompt_collection.compositions, libber_manager)
                # Determine positive_prompt based on prompt_source and prompt_key
                if scene.prompt_source == "custom":
                    positive_prompt = scene.custom_prompt
                elif scene.prompt_source == "prompt" and scene.prompt_key:
                    positive_prompt = prompt_dict.get(scene.prompt_key, "")
                elif scene.prompt_source == "composition" and scene.prompt_key:
                    positive_prompt = composition_dict.get(scene.prompt_key, "")
                else:
                    positive_prompt = ""
                    logger.warning("StorySceneBatch: No valid prompt configuration for scene '%s'", scene.scene_name)
                
                # Warn if prompt is empty
                if not positive_prompt:
                    logger.warning(
                        "StorySceneBatch: Scene '%s' order=%d has EMPTY positive_prompt!",
                        scene.scene_name, scene.scene_order
                    )
                
                # For backwards compatibility, keep old prompt fields
                prompt_data = {
                    "girl_pos": prompt_dict.get("girl_pos", ""),
                    "male_pos": prompt_dict.get("male_pos", ""),
                    "four_image_prompt": prompt_dict.get("four_image_prompt", ""),
                    "wan_prompt": prompt_dict.get("wan_prompt", ""),
                    "wan_low_prompt": prompt_dict.get("wan_low_prompt", ""),
                }
            else:
                logger.debug("StorySceneBatch: Detected legacy prompt format for scene '%s'", scene.scene_name)
                # Legacy format
                prompt_data = prompt_data_raw
                # Use old build_positive_prompt for backwards compatibility if needed
                # But we should still respect the new fields if they exist
                if hasattr(scene, 'prompt_source') and scene.prompt_source:
                    if scene.prompt_source == "custom":
                        positive_prompt = scene.custom_prompt
                    elif scene.prompt_key:
                        if scene.prompt_source == "prompt":
                            positive_prompt = prompt_data.get(scene.prompt_key, "")
                        elif scene.prompt_source == "composition":
                            positive_prompt = prompt_data
                        else:
                            positive_prompt = prompt_data.get(scene.prompt_key, "")
                    else:
                        positive_prompt = ""
                else:
                    # Very old data - fallback
                    positive_prompt = build_positive_prompt(getattr(scene, 'prompt_type', 'girl_pos'), prompt_data, scene.custom_prompt)

            mask_key = resolve_mask_key(scene.mask_name, scene.mask_background)
            depth_key = default_depth_options.get(scene.depth_type, "depth_image")
            pose_key = default_pose_options.get(scene.pose_type, "pose_open_image")

            # Use flat structure: job_root/input/ for all scene images
            job_input_dir = job_root / "input"
            job_output_dir = job_root / "output"
            job_input_dir.mkdir(parents=True, exist_ok=True)
            job_output_dir.mkdir(parents=True, exist_ok=True)

            source_input_dir = Path(scene_dir) / "input"
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
                "mask_name": scene.mask_name,
                "mask_background": scene.mask_background,
                "mask_key": mask_key,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key,
                "custom_prompt": scene.custom_prompt,
                # Legacy fields for backwards compatibility
                "prompt_type": getattr(scene, 'prompt_type', ''),
                "depth_type": scene.depth_type,
                "depth_key": depth_key,
                "pose_type": scene.pose_type,
                "pose_key": pose_key,
                # Control flags for which inputs to use
                "use_depth": scene.use_depth,
                "use_mask": scene.use_mask,
                "use_pose": scene.use_pose,
                "use_canny": scene.use_canny,
                "scene_dir": scene_dir,
                "story_dir": story_info.story_dir,
                "job_id": resolved_job_id,
                "job_root": str(job_root),
                "job_input_dir": str(job_input_dir),
                "job_output_dir": str(job_output_dir),
                "source_input_dir": str(source_input_dir),
                "source_output_dir": str(Path(scene_dir) / "output"),
                "positive_prompt": positive_prompt,
                "wan_prompt": prompt_data.get("wan_prompt", ""),
                "wan_low_prompt": prompt_data.get("wan_low_prompt", ""),
                "four_image_prompt": prompt_data.get("four_image_prompt", ""),
                "girl_pos": prompt_data.get("girl_pos", ""),
                "male_pos": prompt_data.get("male_pos", ""),
                "input_image_path": first_input_image,
                "prompt_data": prompt_data,
            }

            logger.debug("StorySceneBatch: Added descriptor for scene '%s'", scene.scene_name)
            logger.info(
                "StorySceneBatch: Descriptor for '%s' has positive_prompt: '%s...'",
                scene.scene_name,
                descriptor.get("positive_prompt", "")[:100]
            )
            batch.append(descriptor)

        logger.info(
            "StorySceneBatch: Prepared %d scenes with job_id=%s at %s",
            len(batch),
            resolved_job_id,
            job_root,
        )

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
                io.String.Output(id="prompt", display_name="prompt", tooltip="Selected prompt for this scene (composition/custom/prompt)"),
                io.Boolean.Output(id="use_pose", display_name="use_pose", tooltip="Whether the pose image should be used for this scene"),
                io.Boolean.Output(id="use_depth", display_name="use_depth", tooltip="Whether the depth image should be used for this scene"),
                io.Boolean.Output(id="use_canny", display_name="use_canny", tooltip="Whether the canny image should be used for this scene"),
                io.Boolean.Output(id="use_mask", display_name="use_mask", tooltip="Whether the mask image should be used for this scene"),
                io.String.Output(id="scene_name", display_name="scene_name", tooltip="Scene name"),
                io.Int.Output(id="scene_order", display_name="scene_order", tooltip="Scene order"),
                io.String.Output(id="scene_id", display_name="scene_id", tooltip="Scene id"),
                io.String.Output(id="job_id", display_name="job_id", tooltip="Job id"),
                io.String.Output(id="job_input_dir", display_name="job_input_dir", tooltip="Job input directory (where images are saved)"),
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
            logger.warning("StoryScenePick: scene_batch is empty")
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, "", 0, "", "", "", "", None)

        try:
            scenes_sorted = sorted(scene_batch, key=lambda d: d.get("scene_order", 0))
        except Exception:
            scenes_sorted = scene_batch

        safe_index = max(0, min(len(scenes_sorted) - 1, scene_index))
        descriptor = scenes_sorted[safe_index]

        scene_dir = descriptor.get("scene_dir", "")
        if not scene_dir or not os.path.isdir(scene_dir):
            logger.error("StoryScenePick: scene_dir '%s' is invalid", scene_dir)
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_input_dir", ""), descriptor.get("input_image_path", ""), None)
        prompt_key = descriptor.get("prompt_key", "")
        scene_config = SceneInStory(
            scene_id=descriptor.get("scene_id", ""),
            scene_name=descriptor.get("scene_name", ""),
            scene_order=descriptor.get("scene_order", 0),
            mask_name=descriptor.get("mask_name", descriptor.get("mask_type", "")),  # Support both new and legacy
            mask_background=descriptor.get("mask_background", True),
            prompt_source=descriptor.get("prompt_source", "prompt"),
            prompt_key=prompt_key,
            custom_prompt=descriptor.get("custom_prompt", ""),
            # Include legacy prompt_type for backwards compatibility
            prompt_type=descriptor.get("prompt_type", ""),
            depth_type=descriptor.get("depth_type", "depth"),
            pose_type=descriptor.get("pose_type", "open"),
            use_depth=descriptor.get("use_depth", False),
            use_mask=descriptor.get("use_mask", False),
            use_pose=descriptor.get("use_pose", False),
            use_canny=descriptor.get("use_canny", False),
        )
        
        # Log the pose configuration for debugging
        pose_attr = default_pose_options.get(scene_config.pose_type, "pose_open_image")
        logger.info(
            "StoryScenePick: Scene '%s' - pose_type='%s' -> pose_attr='%s' (from descriptor: '%s')",
            scene_config.scene_name,
            scene_config.pose_type,
            pose_attr,
            descriptor.get("pose_type", "NOT_IN_DESCRIPTOR")
        )
        logger.debug("StoryScenePick: Processing scene '%s'", scene_config.scene_name)

        # Use the pre-computed positive_prompt from StorySceneBatch
        # It's already been processed with compositions and libbers applied
        prompt = descriptor.get("positive_prompt", "")
        
        if not prompt:
            logger.warning(
                "StoryScenePick: Scene '%s' order=%d - No positive_prompt in descriptor; available keys: %s",
                descriptor.get("scene_name", "unknown"), descriptor.get("scene_order", -1), list(descriptor.keys())
            )

        try:
            # Use the descriptor's pre-computed prompt - don't let from_story_scene override it
            scene_info, assets, selected_prompt, prompt_data, _ = SceneInfo.from_story_scene(
                scene_config,
                scene_dir_override=scene_dir,
                include_upscale=False,
                include_canny=True,
                prompt_override=prompt,  # Use the descriptor's positive_prompt
            )
        except Exception as e:
            logger.error("StoryScenePick: failed to build SceneInfo for '%s': %s", scene_config.scene_name, e)
            return io.NodeOutput(None, None, None, None, None, "", False, False, False, False, descriptor.get("scene_name", ""), descriptor.get("scene_order", 0), descriptor.get("scene_id", ""), descriptor.get("job_id", ""), descriptor.get("job_input_dir", ""), descriptor.get("input_image_path", ""), None)

        empty_image = make_empty_image()
        canny_image = assets.get("canny_image", empty_image)
        mask_image = assets.get("mask_image")
        mask = assets.get("mask")
        depth_image = assets.get("depth_image", empty_image)
        pose_image = assets.get("pose_image", empty_image)
        
        logger.debug(
            "StoryScenePick: Scene '%s' (order %s) - prompt length: %d",
            scene_config.scene_name,
            scene_config.scene_order,
            len(prompt),
        )

        return io.NodeOutput(
            mask_image,
            mask,
            depth_image,
            pose_image,
            canny_image,
            prompt,
            scene_config.use_pose,
            scene_config.use_depth,
            scene_config.use_canny,
            scene_config.use_mask,
            descriptor.get("scene_name", ""),
            descriptor.get("scene_order", 0),
            descriptor.get("scene_id", ""),
            descriptor.get("job_id", ""),
            descriptor.get("job_input_dir", ""),
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
            logger.warning("StorySave: story_info is None")
            return io.NodeOutput("")
        
        # Ensure story directory exists
        os.makedirs(story_info.story_dir, exist_ok=True)
        
        # Build save path
        save_path = Path(story_info.story_dir) / filename
        
        # Save the story
        save_story(story_info, str(save_path))
        
        logger.info("StorySave: Saved story to '%s'", save_path)
        
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
            logger.warning("StoryLoad: Story file not found at '%s'", story_path)
            return io.NodeOutput(None)
        
        story_info = load_story(str(story_path))
        
        if story_info is None:
            logger.error("StoryLoad: Failed to load story from '%s'", story_path)
        
        return io.NodeOutput(story_info)

# ============================================================================
# TESTABLE IMAGE SAVE HELPERS - imported from utils module
# ============================================================================

from .utils.scene_image_save import (
    SceneImageSaveConfig,
    ImageSaver,
    select_scene_descriptor,
    generate_preview_text
)


class StorySceneImageSave(io.ComfyNode):
    """Save generated image for a story scene with automatic naming and path management"""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("StorySceneImageSave"),
            display_name="StorySceneImageSave",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Image.Input(id="image", display_name="image", tooltip="Generated image to save"),
                io.Custom("SCENE_BATCH").Input(id="scene_batch", display_name="scene_batch", tooltip="Scene batch from StorySceneBatch"),
                io.Int.Input(id="scene_index", display_name="scene_index", default=0, tooltip="Index into scene_batch (0-based); images saved to job_root/input/"),
                io.Combo.Input(id="image_format", display_name="image_format", options=["png", "jpg", "jpeg", "webp"], default="png", tooltip="Output image format"),
                io.Int.Input(id="quality", display_name="quality", default=95, min=1, max=100, tooltip="JPEG/WebP quality (1-100)"),
            ],
            outputs=[
                io.Image.Output(id="image_out", display_name="image", tooltip="Pass-through of the input image"),
                io.String.Output(id="filename", display_name="filename", tooltip="Name of the saved file"),
                io.String.Output(id="filepath", display_name="filepath", tooltip="Full path to the saved file"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        image=None,
        scene_batch=None,
        scene_index: int = 0,
        image_format: str = "png",
        quality: int = 95,
    ) -> io.NodeOutput:
        """Main execution - thin orchestration layer over testable components"""
        if image is None:
            logger.warning("StorySceneImageSave: No image provided")
            return io.NodeOutput(None, "", "")

        if scene_batch is None or not isinstance(scene_batch, list) or not scene_batch:
            logger.warning("StorySceneImageSave: scene_batch is missing or invalid")
            return io.NodeOutput(image, "", "")

        # Use testable functions
        descriptor = select_scene_descriptor(scene_batch, scene_index)
        if descriptor is None:
            logger.warning("StorySceneImageSave: Could not select descriptor")
            return io.NodeOutput(image, "", "")
        
        config = SceneImageSaveConfig.from_descriptor(descriptor, scene_index, image_format, quality)
        if config is None:
            logger.warning("StorySceneImageSave: Invalid configuration from descriptor")
            return io.NodeOutput(image, "", "")

        filepath = config.generate_filepath()
        
        try:
            # Use injected/mockable ImageSaver
            ImageSaver.ensure_directory(config.target_dir)
            pil_image = ImageSaver.tensor_to_pil(image)
            ImageSaver.save_pil_image(pil_image, filepath, config.image_format, config.quality)
            
            logger.info("StorySceneImageSave: Saved image to '%s'", filepath)
            
            preview_text = generate_preview_text(config, filepath)
            preview_ui = ui.PreviewText(value=preview_text)
            
            return io.NodeOutput(
                image,
                config.generate_filename(),
                filepath,
                ui=preview_ui.as_dict()
            )
        except Exception as e:
            logger.exception("StorySceneImageSave: Error saving image: %s", e)
            return io.NodeOutput(image, "", "")


# Import video generation utilities
from .utils.story_video import (
    list_job_ids,
    find_scene_image,
    pair_consecutive_scenes,
    generate_video_filename,
    resolve_video_prompt,
    build_video_descriptor,
)


class StoryVideoBatch(io.ComfyNode):
    """Generate video prompts and aggregate LoRAs for story scene transitions.

    Self-contained node with story and job selection via combo widgets.
    Use `lora_target_filter` to select which model pipeline will be used;
    the aggregated `lora_stack_data` output can be fed directly into
    `LoraStackApply` (which handles High/Low routing internally for Wan2.2).

    Outputs:
    1. input_folder_path - Path to the job's input folder with ordered scene images
    2. video_prompts     - Multiline string, one prompt per scene transition
    3. lora_stack_data   - Aggregated unique LoRA stack filtered by lora_target_filter
    4. video_count       - Total number of video transitions
    5. story_name_out    - Selected story name
    """
    
    @classmethod
    def define_schema(cls):
        # Get available stories for combo widget
        available_stories = get_available_stories()
        default_story = available_stories[0] if available_stories else "default_story"
        
        # Get job IDs for the first story as default options
        default_jobs = [""]
        if available_stories:
            stories_dir = default_stories_dir()
            first_story_path = os.path.join(stories_dir, default_story)
            if os.path.isdir(first_story_path):
                story_info = load_story(os.path.join(first_story_path, "story.json"))
                if story_info:
                    jobs = list_job_ids(story_info.story_dir)
                    default_jobs = jobs if jobs else [""]
        
        return io.Schema(
            node_id=prefixed_node_id("StoryVideoBatch"),
            display_name="StoryVideoBatch",
            category="🧊 frost-byte/Story",
            inputs=[
                io.Combo.Input(id="story_name", display_name="story_name", options=available_stories, default=default_story, tooltip="Select story from available stories"),
                io.Combo.Input(id="job_id", display_name="job_id", options=default_jobs, default=default_jobs[0] if default_jobs else "", tooltip="Select job ID from available jobs (leave empty to use most recent)"),
                io.Combo.Input(
                    id="lora_target_filter",
                    display_name="LoRA Target Filter",
                    options=["All"] + LORA_MODEL_TARGETS,
                    default="All",
                    tooltip=(
                        "Filter the aggregated LoRA stack to a specific model target. "
                        "'All' includes every entry regardless of target. "
                        "Select e.g. 'Wan2.2-Wrapper-High' to output only entries for that pass, "
                        "or 'LTX2.3' for LTX inference. "
                        "Feed the output into LoraStackApply."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(id="input_folder_path", display_name="input_folder_path", tooltip="Path to job input folder with ordered scene images"),
                io.String.Output(id="video_prompts", display_name="video_prompts", tooltip="Multiline string with one prompt per transition"),
                LoraStackData.Output("lora_stack_data", display_name="lora_stack_data", tooltip="Aggregated unique LoRA stack across all story scenes, filtered by lora_target_filter. Feed into LoraStackApply."),
                io.Int.Output(id="video_count", display_name="video_count", tooltip="Total number of video transitions"),
                io.String.Output(id="story_name_out", display_name="story_name", tooltip="Selected story name"),
            ],
        )
    
    @classmethod
    def validate_inputs(cls, story_name: str = "default_story", job_id: str = ""):
        """Validate that job_id is valid for the selected story."""
        if not job_id:
            # Empty job_id is allowed - will use most recent
            return True
        
        stories_dir = default_stories_dir()
        story_json_path = os.path.join(stories_dir, story_name, "story.json")
        
        if not os.path.isfile(story_json_path):
            return f"Story '{story_name}' not found"
        
        story_info = load_story(story_json_path)
        if not story_info:
            return f"Failed to load story '{story_name}'"
        
        available_jobs = list_job_ids(story_info.story_dir)
        if job_id not in available_jobs:
            return f"Job ID '{job_id}' not found in story '{story_name}'. Available jobs: {', '.join(available_jobs)}"
        
        return True
    
    @classmethod
    def fingerprint_inputs(cls, story_name: str = "default_story", job_id: str = ""):
        """Generate fingerprint based on story.json and all referenced scene directory content."""
        if story_name:
            stories_dir = default_stories_dir()
            story_json_path = os.path.join(stories_dir, story_name, "story.json")
            if os.path.isfile(story_json_path):
                try:
                    story_stat = os.stat(story_json_path)
                    story_info = load_story(story_json_path)

                    if not story_info or not getattr(story_info, "scenes", None):
                        return (
                            str(story_json_path),
                            int(story_stat.st_mtime_ns),
                            int(story_stat.st_size),
                            (),
                            job_id.strip(),
                        )

                    scenes_dir = Path(default_scenes_dir())
                    scene_fingerprints: list[tuple[str, str, int, int]] = []

                    for scene in sorted(story_info.scenes, key=lambda s: (s.scene_order, s.scene_name)):
                        scene_dir = scenes_dir / scene.scene_name
                        scene_hash, dir_count, file_count = _directory_fingerprint(scene_dir)
                        scene_fingerprints.append((scene.scene_name, scene_hash, dir_count, file_count))

                    return (
                        str(story_json_path),
                        int(story_stat.st_mtime_ns),
                        int(story_stat.st_size),
                        tuple(scene_fingerprints),
                        job_id.strip(),
                    )
                except Exception as e:
                    logger.warning("StoryVideoBatch: Failed to stat story.json for fingerprinting: %s", e)
        # Return None to use default fingerprinting behavior
        return None
    
    @classmethod
    def execute(
        cls,
        story_name: str = "default_story",
        job_id: str = "",
        lora_target_filter: str = "All",
    ) -> io.NodeOutput:
        # Debug logging to track what parameters are being received
        logger.info("StoryVideoBatch.execute called with story_name='%s', job_id='%s', lora_target_filter='%s'", story_name, job_id, lora_target_filter)
        
        # Load story from story_name
        stories_dir = default_stories_dir()
        story_json_path = os.path.join(stories_dir, story_name, "story.json")
        
        if not os.path.isfile(story_json_path):
            logger.warning("StoryVideoBatch: Story file not found: '%s'", story_json_path)
            return io.NodeOutput("", "", None, 0, story_name)
        
        story_info = load_story(story_json_path)
        if story_info is None or not getattr(story_info, "scenes", None):
            logger.warning("StoryVideoBatch: Failed to load story or story has no scenes")
            return io.NodeOutput("", "", None, 0, story_name)
        
        # List available job IDs
        available_jobs = list_job_ids(story_info.story_dir)
        if not available_jobs:
            logger.warning("StoryVideoBatch: No jobs found in story directory '%s'", story_info.story_dir)
            return io.NodeOutput("", "", None, 0, story_name)
        
        logger.info("StoryVideoBatch: Available job_ids for story '%s': %s", story_name, available_jobs)
        
        # Select job ID (use first/newest if not specified or not found)
        if job_id and job_id in available_jobs:
            selected_job = job_id
            logger.info("StoryVideoBatch: Using specified job_id='%s'", selected_job)
        else:
            selected_job = available_jobs[0]
            if job_id:
                logger.warning("StoryVideoBatch: Specified job_id='%s' not found, using most recent: '%s'", job_id, selected_job)
            else:
                logger.info("StoryVideoBatch: No job_id specified, using most recent: '%s'", selected_job)
        
        job_root = Path(story_info.story_dir) / "jobs" / selected_job
        job_input_dir = str(job_root / "input")
        
        if not Path(job_input_dir).exists():
            logger.warning("StoryVideoBatch: Job input directory does not exist: '%s'", job_input_dir)
            return io.NodeOutput("", "", None, 0, story_name)
        
        scenes_dir = default_scenes_dir()
        scenes_sorted = sorted(story_info.scenes, key=lambda s: s.scene_order)
        
        logger.info(
            "StoryVideoBatch: Preparing video prompts for story '%s' with %d scenes from job_id='%s'",
            story_info.story_name,
            len(scenes_sorted),
            selected_job,
        )
        
        # Dict keyed by (lora, model_target) to aggregate unique entries across scenes
        loras_stack_dict: dict[tuple[str, str], dict] = {}
        
        # Build scene descriptors with processed prompts
        scene_descriptors = []
        libber_manager = LibberStateManager.instance()
        
        for scene in scenes_sorted:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            prompt_path = os.path.join(scene_dir, "prompts.json")
            prompt_data_raw = load_prompt_json(prompt_path) or {}
            
            # Process prompts using the v2 system with compositions
            prompt_dict = {}
            composition_dict = {}
            
            if "version" in prompt_data_raw and prompt_data_raw.get("version") == 2:
                prompt_collection = PromptCollection.from_dict(prompt_data_raw)
                
                # Process individual prompts
                for key, metadata in prompt_collection.prompts.items():
                    value = metadata.value
                    if metadata.processing_type == "libber" and metadata.libber_name:
                        libber = libber_manager.get_libber(metadata.libber_name)
                        if libber:
                            value = libber.substitute(value)
                    prompt_dict[key] = value
                
                # Process compositions
                if prompt_collection.compositions:
                    composition_dict = prompt_collection.compose_prompts(
                        prompt_collection.compositions,
                        libber_manager
                    )
            else:
                # Legacy format
                prompt_dict = prompt_data_raw
            
            # Load LoRA stack and aggregate unique entries
            scene_lora_stack = load_lora_stack(scene_dir) or []
            for entry in scene_lora_stack:
                key = (entry.get("lora", ""), entry.get("model_target", ""))
                if key[0] and key[0].lower() != "none":
                    loras_stack_dict[key] = entry  # last scene's value wins

            scene_descriptors.append({
                "scene": scene,
                "prompt_dict": prompt_dict,
                "composition_dict": composition_dict,
            })
        
        # Generate video prompts for consecutive scene transitions
        video_prompts = []
        scene_pairs = pair_consecutive_scenes(scene_descriptors)
        
        for current_desc, next_desc in scene_pairs:
            current_scene = current_desc["scene"]
            prompt_dict = current_desc["prompt_dict"]
            composition_dict = current_desc["composition_dict"]
            
            # Resolve video prompt based on video_prompt_source
            video_prompt = ""
            
            if current_scene.video_prompt_source == "auto":
                # Use the image prompt based on prompt_source and prompt_key
                if current_scene.prompt_source == "custom":
                    video_prompt = current_scene.custom_prompt
                elif current_scene.prompt_source == "prompt" and current_scene.prompt_key:
                    video_prompt = prompt_dict.get(current_scene.prompt_key, "")
                elif current_scene.prompt_source == "composition" and current_scene.prompt_key:
                    video_prompt = composition_dict.get(current_scene.prompt_key, "")
            
            elif current_scene.video_prompt_source == "custom":
                video_prompt = current_scene.video_custom_prompt
            
            elif current_scene.video_prompt_source == "prompt" and current_scene.video_prompt_key:
                video_prompt = prompt_dict.get(current_scene.video_prompt_key, "")
            
            elif current_scene.video_prompt_source == "composition" and current_scene.video_prompt_key:
                video_prompt = composition_dict.get(current_scene.video_prompt_key, "")
            
            video_prompts.append(video_prompt)
            
            logger.debug(
                "StoryVideoBatch: Added video prompt for transition '%s' -> '%s': %s",
                current_scene.scene_name,
                next_desc["scene"].scene_name if next_desc else "end",
                video_prompt[:50] + "..." if len(video_prompt) > 50 else video_prompt,
            )
        
        # Build the aggregated stack, applying the target filter
        aggregated_stack = list(loras_stack_dict.values())
        if lora_target_filter and lora_target_filter != "All":
            aggregated_stack = [e for e in aggregated_stack if e.get("model_target") == lora_target_filter]
        lora_stack_out = aggregated_stack if aggregated_stack else None
        
        # Join video prompts into multiline string
        video_prompts_multiline = "\n".join(video_prompts)
        
        logger.info(
            "StoryVideoBatch: Generated %d video prompts, %d unique LoRA entries (filter=%s) for story '%s'",
            len(video_prompts),
            len(aggregated_stack),
            lora_target_filter,
            story_name,
        )
        
        return io.NodeOutput(
            job_input_dir,
            video_prompts_multiline,
            lora_stack_out,
            len(video_prompts),
            story_name,
        )


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
            
            logger.info("LibberManager: %s", status)
            return io.NodeOutput(status, keys_display, ui=combined_ui)
            
        except Exception as e:
            status = f"✗ Error: {str(e)}"
            logger.error("LibberManager error: %s", status)
            return io.NodeOutput(status, "")


class LibberApply(io.ComfyNode):
    """Apply Libber substitutions to text with libber selection."""
    
    @classmethod
    def define_schema(cls):
        libber_dir = default_libber_dir()
        manager = LibberStateManager.instance()
        available_libbers = set(manager.list_libbers())

        # Include libbers available on disk (same source behavior as LibberManager)
        if os.path.isdir(libber_dir):
            for f in os.listdir(libber_dir):
                if f.endswith('.json'):
                    available_libbers.add(f[:-5])

        available_libbers = sorted(available_libbers)
        
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

        if libber_name == "none":
            return io.NodeOutput(text, "Select a libber in LibberManager or create one first.")
        
        # Try to reload from file to ensure we have the latest data
        libber_dir = default_libber_dir()
        libber_filepath = os.path.join(libber_dir, f"{libber_name}.json")
        if os.path.exists(libber_filepath):
            try:
                libber = manager.load_libber(libber_name, libber_filepath)
            except Exception as e:
                logger.warning("LibberApply: Error reloading from file, using in-memory instance: %s", e)
                libber = manager.get_libber(libber_name)
        else:
            libber = manager.get_libber(libber_name)
        
        if not libber:
            status = f"✗ Libber '{libber_name}' not found. Create or load it in LibberManager first."
            logger.warning("LibberApply: %s", status)
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
            logger.info("LibberApply: %s", info)
            logger.debug("LibberApply input preview: %s", text[:100])
            logger.debug("LibberApply output preview: %s", result[:100])
            
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
            logger.error("LibberApply: %s", info)
            return io.NodeOutput(result, info)
        
        return io.NodeOutput(result, info)


# ============================================================================
# SCENE PROMPT MANAGEMENT NODES
# ============================================================================

class ScenePromptManager(io.ComfyNode):
    """Manage prompts in a Scene's PromptCollection with an interactive table interface."""
    
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_dir = os.path.join(output_dir, "scenes")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            os.makedirs(os.path.join(default_dir, "default_scene"), exist_ok=True)
        
        subdir_dict = get_subdirectories(default_dir)
        all_scenes = sorted(subdir_dict.keys()) if subdir_dict else ["default_scene"]
        
        # Find scenes with valid v2 prompts.json files
        valid_scenes = []
        for scene_name in all_scenes:
            scene_dir = os.path.join(default_dir, scene_name)
            prompts_path = os.path.join(scene_dir, "prompts.json")
            if os.path.exists(prompts_path):
                try:
                    # Check if it's a valid v2 format
                    with open(prompts_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'prompts' in data:
                            valid_scenes.append(scene_name)
                except:
                    pass
        
        # Use valid scenes if any exist, otherwise show all scenes
        default_options = valid_scenes if valid_scenes else all_scenes
        default_scene = default_options[0] if default_options else "default_scene"
        
        return io.Schema(
            node_id=prefixed_node_id("ScenePromptManager"),
            display_name="ScenePromptManager",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("scenes_dir", default=default_dir, tooltip="Directory containing pose subdirectories"),
                io.Combo.Input('scene_name', options=default_options, default=default_scene, tooltip="Select a scene to manage prompts"),
                io.String.Input(
                    id="collection_json",
                    display_name="collection_json",
                    default="",
                    multiline=True,
                    tooltip="Prompt collection JSON (auto-updated by UI table - normally don't edit manually)"
                ),
            ],
            outputs=[
                io.Custom("DICT").Output(
                    id="prompt_dict",
                    display_name="prompt_dict",
                    tooltip="Dictionary of individual prompts (raw or libber-processed)"
                ),
                io.Custom("DICT").Output(
                    id="comp_dict",
                    display_name="comp_dict",
                    tooltip="Dictionary of composed prompts by composition name"
                ),
                io.String.Output(
                    id="scene_name_out",
                    display_name="scene_name",
                    tooltip="Name of the managed scene"
                ),
                io.String.Output(
                    id="scene_dir",
                    display_name="scene_dir",
                    tooltip="Directory path of the managed scene"
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
    def execute(cls, scenes_dir="", scene_name="", collection_json=""):
        if not scenes_dir:
            scenes_dir = default_scenes_dir()
        
        if not scene_name:
            status = "✗ No scene selected"
            logger.warning("ScenePromptManager: %s", status)
            combined_ui = {"text": ["{}", "[]", status, "[]", "[]", "{}", "{}"]}
            return io.NodeOutput({}, {}, "", "", status, ui=combined_ui)
        
        scene_dir = os.path.join(scenes_dir, scene_name)
        
        if not os.path.isdir(scene_dir):
            status = f"✗ Scene directory not found: {scene_dir}"
            logger.error("ScenePromptManager: %s", status)
            combined_ui = {"text": ["{}", "[]", status, "[]", "[]", "{}", "{}"]}
            return io.NodeOutput({}, {}, scene_name, scene_dir, status, ui=combined_ui)
        
        # Load prompt collection from file or JSON
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        
        # Check if prompts.json exists
        if not os.path.exists(prompt_json_path) and not collection_json:
            status = f"⚠ Scene '{scene_name}' has no prompts.json file. Create prompts using the UI table."
            logger.warning("ScenePromptManager: %s", status)
            collection = PromptCollection()
            # Save empty collection to create the file
            try:
                with open(prompt_json_path, 'w', encoding='utf-8') as f:
                    json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
                status += " (Created empty prompts.json)"
            except Exception as e:
                status += f" (Failed to create file: {e})"
        else:
            # Priority: collection_json (user edits) > prompts.json file
            if collection_json:
                try:
                    data = json.loads(collection_json)
                    collection = PromptCollection.from_dict(data)
                    logger.info(
                        "ScenePromptManager: Loaded collection from UI JSON with %d prompts",
                        len(collection.prompts),
                    )
                    
                    # Save to file
                    try:
                        with open(prompt_json_path, 'w', encoding='utf-8') as f:
                            json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
                        status = f"✓ Saved {len(collection.prompts)} prompts to '{scene_name}'"
                        logger.error("ScenePromptManager: %s", status)
                    except Exception as e:
                        status = f"⚠ Loaded {len(collection.prompts)} prompts but failed to save: {e}"
                        logger.error("ScenePromptManager: %s", status)
                        
                except Exception as e:
                    # Fall back to file
                    status = f"✗ Error parsing UI JSON: {e}. Loading from file instead."
                    logger.error("ScenePromptManager: %s", status)
                    try:
                        collection = PromptCollection.load_from_json(prompt_json_path)
                    except Exception as e2:
                        status = f"✗ Failed to load from file: {e2}"
                        logger.error("ScenePromptManager: %s", status)
                        collection = PromptCollection()
            else:
                # Load from file
                try:
                    collection = PromptCollection.load_from_json(prompt_json_path)
                    
                    # Check if it's v2 format
                    if len(collection.prompts) == 0:
                        status = f"⚠ Scene '{scene_name}' has empty or v1 format prompts.json. Use UI to add prompts."
                    else:
                        status = f"✓ Loaded {len(collection.prompts)} prompts from '{scene_name}'"
                    
                    logger.info("ScenePromptManager: %s", status)
                except Exception as e:
                    status = f"✗ Error loading prompts.json: {e}"
                    logger.error("ScenePromptManager: %s", status)
                    collection = PromptCollection()
        
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
        libber_manager = LibberStateManager.instance()
        available_libbers = ["none"] + list(libber_manager.libbers.keys())
        
        # Build prompt_dict (individual prompts processed)
        prompt_dict = {}
        for key, metadata in collection.prompts.items():
            value = metadata.value
            
            # Apply libber substitution if needed
            if metadata.processing_type == "libber" and metadata.libber_name and libber_manager:
                libber = libber_manager.ensure_libber(metadata.libber_name)
                if libber:
                    value = libber.substitute(value)
            
            prompt_dict[key] = value
        
        # Build comp_dict (compositions processed)
        comp_dict = {}
        if collection.compositions:
            comp_dict = collection.compose_prompts(collection.compositions, libber_manager)
        
        # Prepare compositions list for UI
        compositions_list = []
        for name, prompt_keys in collection.compositions.items():
            compositions_list.append({
                "name": name,
                "prompt_keys": prompt_keys,
                "preview": comp_dict.get(name, "")[:100] + ("..." if len(comp_dict.get(name, "")) > 100 else "")
            })
        
        combined_ui = {
            "text": [
                json.dumps(collection_data, indent=2),
                json.dumps(prompts_list),
                status,
                json.dumps(available_libbers),
                json.dumps(compositions_list),
                json.dumps(prompt_dict),
                json.dumps(comp_dict)
            ]
        }
        
        logger.info("ScenePromptManager: %s", status)
        return io.NodeOutput(prompt_dict, comp_dict, scene_name, scene_dir, status, ui=combined_ui)


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
            logger.warning("PromptComposer: %s", status)
            return io.NodeOutput({}, "{}", status)
        
        # Parse composition map
        composition_map = {}
        if composition_json:
            try:
                composition_map = json.loads(composition_json)
            except Exception as e:
                logger.error("PromptComposer: Error parsing composition JSON: %s", e)
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
        
        logger.info("PromptComposer: %s", info)
        return io.NodeOutput(prompt_dict, json.dumps(composition_map, indent=2), info, ui=combined_ui)


# ============================================================================
# REST API STATE MANAGERS
# ============================================================================

from server import PromptServer
from aiohttp import web
import time
from datetime import datetime, timedelta

routes = PromptServer.instance.routes

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
            logger.info("PromptCollectionStateManager: Expired session %s", sid)
    
    def create_session(self, session_id: str, collection: PromptCollection):
        """Create or update a session with a PromptCollection."""
        self.cleanup_expired()
        self.sessions[session_id] = {
            "collection": collection,
            "last_access": datetime.now()
        }
        logger.info("PromptCollectionStateManager: Created session %s", session_id)
    
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
        logger.info("LibberStateManager: Created libber '%s'", name)
        return libber
    
    def load_libber(self, name: str, filepath: str) -> Libber:
        """Load a Libber from file."""
        libber = Libber.load(filepath)
        self.libbers[name] = libber
        logger.info("LibberStateManager: Loaded libber '%s' from %s", name, filepath)
        return libber
    
    def get_libber(self, name: str) -> Optional[Libber]:
        """Get a Libber by name."""
        return self.libbers.get(name)

    def ensure_libber(self, name: str, base_dir: Optional[str] = None) -> Optional[Libber]:
        """Get a Libber if loaded; otherwise try loading from disk (base_dir/name.json)."""
        libber = self.get_libber(name)
        if libber:
            return libber
        base_dir = base_dir or default_libber_dir()
        filepath = os.path.join(base_dir, f"{name}.json")
        if os.path.exists(filepath):
            try:
                return self.load_libber(name, filepath)
            except Exception as exc:
                logger.warning(
                    "LibberStateManager: Failed to auto-load libber '%s' from %s: %s",
                    name,
                    filepath,
                    exc,
                )
        else:
            logger.warning(
                "LibberStateManager: Libber '%s' not loaded and file not found at %s",
                name,
                filepath,
            )
        return None
    
    def save_libber(self, name: str, filepath: str):
        """Save a Libber to file."""
        if name in self.libbers:
            self.libbers[name].save(filepath)
            logger.info("LibberStateManager: Saved libber '%s' to %s", name, filepath)
        else:
            raise ValueError(f"Libber '{name}' not found")
    
    def list_libbers(self) -> List[str]:
        """List all loaded libber names."""
        return list(self.libbers.keys())
    
    def delete_libber(self, name: str):
        """Remove a Libber from memory."""
        if name in self.libbers:
            del self.libbers[name]
            logger.info("LibberStateManager: Deleted libber '%s'", name)
    
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
@routes.post("/fbtools/prompts/create")
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


@routes.post("/fbtools/prompts/add")
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


@routes.post("/fbtools/prompts/remove")
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


@routes.get("/fbtools/prompts/list_names")
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
# DATASET CAPTION API ENDPOINTS
# ============================================================================

@routes.get("/fbtools/dataset_caption/image")
async def serve_image(request: web.Request) -> web.Response:
    """
    Serve an arbitrary local image by absolute path for viewer thumbnails.
    Only serves files with recognised image extensions.
    Query param: path — absolute path to image file.
    """
    import mimetypes
    SAFE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    path_str  = request.rel_url.query.get("path", "").strip()
    if not path_str:
        return web.Response(status=400, text="Missing path parameter")
    p = Path(path_str)
    if p.suffix.lower() not in SAFE_EXTS:
        return web.Response(status=403, text="Forbidden file extension")
    if not p.is_file():
        return web.Response(status=404, text="File not found")
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return web.Response(body=p.read_bytes(), content_type=mime)


@routes.get("/fbtools/dataset_caption/list")
async def list_dataset(request: web.Request) -> web.Response:
    """
    Return a paginated page of image+caption rows for a given directory.
    Query params:
      path       — dataset directory (required)
      output_dir — optional separate .txt directory
      page       — 1-based page number (default 1)
      page_size  — items per page (default 10)
      recursive  — "true" / "false" (default "false")
    """
    params = request.rel_url.query
    path_s = params.get("path", "").strip()
    if not path_s:
        return web.json_response({"error": "Missing path parameter"}, status=400)

    try:
        base = _resolve_dataset_input_directory(path_s, field_name="path")
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    output_dir = _resolve_dataset_output_directory(params.get("output_dir", "").strip())
    page         = int(params.get("page", 1))
    page_size    = int(params.get("page_size", 10))
    recursive    = params.get("recursive", "false").lower() == "true"

    if not base.is_dir():
        return web.json_response({"error": f"Not a directory: {base}"}, status=400)

    images    = _collect_images(base, recursive)
    total     = len(images)
    start     = (page - 1) * page_size
    page_imgs = images[start : start + page_size]

    rows = []
    for img in page_imgs:
        caption = _read_caption(img, output_dir)
        txt     = _txt_path(img, output_dir)
        rows.append({
            "filename":    img.name,
            "image_path":  str(img),
            "txt_path":    str(txt),
            "caption":     caption,
            "has_caption": txt.exists(),
        })

    return web.json_response({
        "rows":        rows,
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "base_dir":    str(base),
    })


@routes.post("/fbtools/dataset_caption/edit")
async def edit_dataset_captions(request: web.Request) -> web.Response:
    """
    Batch-edit dataset caption .txt files via API.
    JSON body:
            {
                "dataset_path": "captions" | "/abs/path",
                "caption_directory": "optional captions dir, abs or relative to Comfy output",
                "output_directory": "alias of caption_directory",
        "prepend_text": "",
        "append_text": "",
        "find_text": "",
        "replace_text": "",
        "recursive": false,
        "dry_run": true
      }
    """
    try:
        body = await request.json()

        dataset_path = str(body.get("dataset_path", "")).strip()
        if not dataset_path:
            return web.json_response({"error": "dataset_path is required"}, status=400)

        base = _resolve_dataset_output_directory(dataset_path)
        if base is None:
            return web.json_response(
                {
                    "error": "dataset_path is required. Provide an absolute path or a path relative to Comfy output directory."
                },
                status=400,
            )

        if not base.is_dir():
            return web.json_response({"error": f"dataset_path is not a directory: {base}"}, status=400)

        caption_dir_raw = str(
            body.get("caption_directory", "") or body.get("output_directory", "")
        ).strip()
        if caption_dir_raw:
            caption_dir = _resolve_dataset_output_directory(caption_dir_raw)
            if caption_dir is None:
                return web.json_response(
                    {
                        "error": "caption_directory must be an absolute path or a path relative to Comfy output directory."
                    },
                    status=400,
                )
        else:
            caption_dir = base
        if not caption_dir.is_dir():
            return web.json_response({"error": f"caption_directory is not a directory: {caption_dir}"}, status=400)

        prepend_text = str(body.get("prepend_text", ""))
        append_text = str(body.get("append_text", ""))
        find_text = str(body.get("find_text", ""))
        replace_text = str(body.get("replace_text", ""))
        recursive = bool(body.get("recursive", False))
        dry_run = bool(body.get("dry_run", True))

        pattern = "**/*.txt" if recursive else "*.txt"
        txt_files = list(caption_dir.glob(pattern))

        edited = 0
        preview_changes = []
        max_preview = 25

        for txt in txt_files:
            original = txt.read_text(encoding="utf-8").strip()
            updated = original

            if find_text:
                updated = updated.replace(find_text, replace_text)
            if prepend_text:
                updated = f"{prepend_text.rstrip()} {updated}".strip()
            if append_text:
                updated = f"{updated.rstrip()} {append_text.lstrip()}".strip()

            if updated != original:
                edited += 1
                if len(preview_changes) < max_preview:
                    preview_changes.append({
                        "filename": txt.name,
                        "before": original[:180],
                        "after": updated[:180],
                    })

                if not dry_run:
                    txt.write_text(updated, encoding="utf-8")

        return web.json_response({
            "ok": True,
            "dataset_path": str(base),
            "caption_directory": str(caption_dir),
            "dry_run": dry_run,
            "total_txt_files": len(txt_files),
            "edited_count": edited,
            "preview_truncated": edited > len(preview_changes),
            "preview_changes": preview_changes,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/dataset_caption/save")
async def save_caption(request: web.Request) -> web.Response:
    """
    Save an edited caption back to disk.
    JSON body: { "txt_path": "/abs/path/to/image.txt", "caption": "..." }
    """
    try:
        body     = await request.json()
        txt_path = Path(body["txt_path"])
        caption  = body["caption"].strip()
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(caption, encoding="utf-8")
        return web.json_response({"ok": True, "txt_path": str(txt_path)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/dataset_caption/recaption")
async def recaption_single(request: web.Request) -> web.Response:
    """
    Re-caption a single image on demand from the viewer widget.
    JSON body: {
        "image_path":     "/abs/path/to/image.jpg",
        "txt_path":       "/abs/path/to/image.txt",
        "captioner_type": "qwen_vl",
        "instruction":    "...",
        "trigger_word":   "MYTOKEN",
        "device":         "auto",
        "use_8bit":       false,
        "clean_caption":  true,
        "gemini_api_key": ""
    }
    """
    import os
    from .captioner import caption_image, get_model

    try:
        body           = await request.json()
        image_path     = Path(body["image_path"])
        txt_path       = Path(body["txt_path"])
        captioner_type = body.get("captioner_type", "qwen_vl")
        instruction    = body.get("instruction", "Describe this image in detail.")
        trigger_word   = body.get("trigger_word", "")
        device         = body.get("device", "auto")
        use_8bit       = bool(body.get("use_8bit", False))
        clean          = bool(body.get("clean_caption", True))
        api_key        = body.get("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", "")

        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: loading {captioner_type} for re-caption",
            source="dataset_captioner",
        )
        model, processor = get_model(captioner_type, device, use_8bit)
        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            "Dataset Captioner: generating caption",
            source="dataset_captioner",
        )
        caption = caption_image(
            image_path     = image_path,
            captioner_type = captioner_type,
            instruction    = instruction,
            model          = model,
            processor      = processor,
            api_key        = api_key,
            clean          = clean,
        )
        if trigger_word.strip():
            caption = f"{trigger_word.strip()}. {caption}"

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(caption, encoding="utf-8")
        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            "Dataset Captioner: re-caption complete",
            source="dataset_captioner",
            level="success",
        )
        return web.json_response({"ok": True, "caption": caption})
    except Exception as e:
        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: re-caption failed ({e})",
            source="dataset_captioner",
            level="error",
        )
        return web.json_response({"error": str(e)}, status=500)

# ============================================================================
# LIBBER REST API ENDPOINTS
# ============================================================================

@routes.post("/fbtools/libber/create")
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
        logger.exception("Error creating libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/load")
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
        logger.exception("Error loading libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/add_lib")
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
        logger.exception("Error adding lib")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/remove_lib")
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
        logger.exception("Error removing lib")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/save")
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
        logger.exception("Error saving libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/libber/list")
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
            "libber_dir": libber_dir,
            "count": len(libbers)
        })
    
    except Exception as e:
        logger.exception("Error listing libbers")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/libber/get_data/{name}")
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
        logger.exception("Error getting libber data")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/apply")
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
        logger.exception("Error applying libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/scene/process_compositions")
async def scene_process_compositions(request):
    """
    Process compositions from a prompt collection and return composed prompts.
    Body: {"collection": dict}
    Returns: {"prompt_dict": dict, "status": str}
    """
    try:
        data = await request.json()
        collection_data = data.get("collection")
        
        if not collection_data:
            return web.json_response({"error": "collection data required"}, status=400)
        
        # Parse collection data
        try:
            collection = PromptCollection.from_dict(collection_data)
        except Exception as e:
            return web.json_response({"error": f"Invalid collection data: {str(e)}"}, status=400)
        
        # Get libber manager for substitutions
        libber_manager = LibberStateManager.instance()
        
        # Compose prompts
        prompt_dict = collection.compose_prompts(collection.compositions, libber_manager)
        
        return web.json_response({
            "prompt_dict": prompt_dict,
            "status": f"Processed {len(prompt_dict)} compositions"
        })
    
    except Exception as e:
        logger.exception("Error processing compositions")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/scene/get_scene_prompts")
async def scene_get_prompts(request):
    """
    Get prompts and compositions from a scene's prompts.json file.
    Query param: scene_dir
    Returns: {"prompts": [...], "compositions": {...}}
    """
    try:
        scene_dir = request.query.get("scene_dir")
        
        if not scene_dir:
            return web.json_response({"error": "scene_dir parameter required"}, status=400)
        
        if not os.path.isdir(scene_dir):
            return web.json_response({"error": f"scene_dir '{scene_dir}' is not a valid directory"}, status=400)
        
        # Load prompts.json
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        if not os.path.isfile(prompt_json_path):
            return web.json_response({"prompts": [], "compositions": {}})
        
        try:
            collection = PromptCollection.load_from_json(prompt_json_path)
        except Exception as e:
            return web.json_response({"error": f"Failed to load prompts.json: {str(e)}"}, status=500)
        
        # Convert prompts to list format for UI
        prompts_list = [
            {
                "key": key,
                "value": prompt.value,
                "category": prompt.category,
                "processing_type": prompt.processing_type,
                "libber_name": prompt.libber_name
            }
            for key, prompt in collection.prompts.items()
        ]
        
        # Get available libbers (merge in-memory and on-disk)
        libbers_set = set()
        try:
            manager = LibberStateManager.instance()
            libbers_set.update(manager.list_libbers())

            libbers_dir = default_libber_dir()
            if os.path.isdir(libbers_dir):
                for filename in os.listdir(libbers_dir):
                    filepath = os.path.join(libbers_dir, filename)
                    if os.path.isfile(filepath) and filename.endswith('.json'):
                        libbers_set.add(filename[:-5])
        except Exception as e:
            logger.warning("Warning: Could not load libbers list: %s", e)

        libbers_list = ["none"] + sorted(libbers_set)
        
        # Get scene_flags from collection if present
        scene_flags = {}
        collection_dict = collection.to_dict()
        if 'scene_flags' in collection_dict:
            scene_flags = collection_dict['scene_flags']
        
        # Load masks from masks.json
        masks_dict = load_masks_json(scene_dir)
        masks_data = {}
        if masks_dict:
            # Convert MaskDefinition objects to dict format
            masks_data = {name: mask.to_dict() for name, mask in masks_dict.items()}
        
        # Return compositions as dict with scene_flags and masks
        return web.json_response({
            "prompts": prompts_list,
            "compositions": collection.compositions,
            "scene_flags": scene_flags,
            "libbers": libbers_list,
            "masks": masks_data
        })
    
    except Exception as e:
        logger.exception("Error getting scene prompts")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/scene/save_scene_prompts")
async def scene_save_prompts(request):
    """
    Save prompts and compositions to a scene's prompts.json file.
    Body: {"scene_dir": str, "collection": dict}
    Returns: {"success": bool, "message": str}
    """
    try:
        data = await request.json()
        scene_dir = data.get("scene_dir")
        collection_data = data.get("collection")
        
        logger.info("ScenePromptManager API: Received save request for scene_dir='%s'", scene_dir)
        
        if not scene_dir:
            return web.json_response({"error": "scene_dir required"}, status=400)
        
        if not collection_data:
            return web.json_response({"error": "collection data required"}, status=400)
        
        if not os.path.isdir(scene_dir):
            logger.error("ScenePromptManager API: scene_dir '%s' is not a valid directory", scene_dir)
            return web.json_response({"error": f"scene_dir '{scene_dir}' is not a valid directory"}, status=400)
        
        # Parse and validate collection
        try:
            logger.debug("ScenePromptManager API: Incoming collection_data keys: %s", list(collection_data.keys()) if isinstance(collection_data, dict) else "not a dict")
            logger.debug("ScenePromptManager API: collection_data type: %s", type(collection_data))
            
            # Check if collection_data has the expected structure
            if not isinstance(collection_data, dict):
                raise ValueError(f"collection_data must be a dict, got {type(collection_data)}")
            
            # Log the structure
            if 'prompts' in collection_data:
                logger.debug("ScenePromptManager API: prompts type: %s, count: %d", 
                           type(collection_data['prompts']), 
                           len(collection_data['prompts']) if isinstance(collection_data['prompts'], (dict, list)) else 0)
            if 'compositions' in collection_data:
                logger.debug("ScenePromptManager API: compositions type: %s, count: %d",
                           type(collection_data['compositions']),
                           len(collection_data['compositions']) if isinstance(collection_data['compositions'], (dict, list)) else 0)
            if 'scene_flags' in collection_data:
                logger.debug("ScenePromptManager API: scene_flags: %s", collection_data['scene_flags'])
            
            collection = PromptCollection.from_dict(collection_data)
            logger.info(
                "ScenePromptManager API: Parsed collection with %d prompts and %d compositions",
                len(collection.prompts),
                len(collection.compositions),
            )
        except Exception as e:
            logger.exception("ScenePromptManager API: Error parsing collection data: %s", str(e))
            logger.error("ScenePromptManager API: collection_data content: %s", str(collection_data)[:500])
            return web.json_response({"error": f"Invalid collection data: {str(e)}"}, status=400)
        
        # Save to file
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        logger.info("ScenePromptManager API: Attempting to save to: %s", prompt_json_path)
        logger.debug(
            "ScenePromptManager API: File exists before save: %s",
            os.path.exists(prompt_json_path),
        )
        
        try:
            # Convert to dict - scene_flags now preserved automatically
            collection_dict = collection.to_dict()
            
            logger.debug(
                "ScenePromptManager API: Collection dict keys: %s",
                list(collection_dict.keys()),
            )
            logger.debug(
                "ScenePromptManager API: Prompt keys in dict: %s",
                list(collection_dict.get('prompts', {}).keys()),
            )
            logger.debug(
                "ScenePromptManager API: Composition keys in dict: %s",
                list(collection_dict.get('compositions', {}).keys()),
            )
            
            with open(prompt_json_path, 'w', encoding='utf-8') as f:
                json.dump(collection_dict, f, indent=2, ensure_ascii=False)
            
            logger.debug(
                "ScenePromptManager API: File written successfully; exists=%s; size=%s",
                os.path.exists(prompt_json_path),
                os.path.getsize(prompt_json_path),
            )
            
            # Read back to verify
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            logger.debug(
                "ScenePromptManager API: Verification - read back %d prompts",
                len(saved_data.get('prompts', {})),
            )
            
            message = f"Saved {len(collection.prompts)} prompts and {len(collection.compositions)} compositions to {os.path.basename(scene_dir)}"
            logger.info("ScenePromptManager API: %s", message)
            return web.json_response({
                "success": True,
                "message": message
            })
        except Exception as e:
            logger.exception("ScenePromptManager API: Error saving to file")
            return web.json_response({"error": f"Failed to save prompts.json: {str(e)}"}, status=500)
    
    except Exception as e:
        logger.exception("Error saving scene prompts")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/story/load/{story_name}")
async def story_load(request):
    """
    Load story data from filesystem.
    Returns: {"story_name": str, "story_dir": str, "scenes": [...]}
    """
    try:
        story_name = request.match_info.get("story_name")
        
        if not story_name:
            return web.json_response({"error": "story_name required"}, status=400)
        
        # Load story from filesystem
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            return web.json_response({"error": f"Story '{story_name}' not found"}, status=404)
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return web.json_response({"error": f"Failed to load story '{story_name}'"}, status=500)
        
        # Convert scenes to dict format for frontend
        scenes_dir = default_scenes_dir()
        scenes_data = []
        for scene in getattr(story_info, "scenes", []):
            # Load available masks for this scene
            available_masks = ["none"]
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if os.path.isdir(scene_dir):
                try:
                    # Load new mask system masks
                    masks_dict = load_masks_json(scene_dir)
                    available_masks.extend(masks_dict.keys())
                    
                    # Add legacy masks if they exist
                    legacy_mask_names = ["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"]
                    for legacy_name in legacy_mask_names:
                        mask_file = f"{legacy_name.replace('_no_bg', '_mask_no_bkgd' if '_no_bg' in legacy_name else '_mask_bkgd')}.png"
                        mask_path = os.path.join(scene_dir, mask_file)
                        if os.path.exists(mask_path) and legacy_name not in available_masks:
                            available_masks.append(legacy_name)
                except Exception as e:
                    logger.debug(f"story_load API: Could not load masks for scene '{scene.scene_name}': {e}")
            
            scenes_data.append({
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_name,
                "scene_order": scene.scene_order,
                "mask_name": getattr(scene, "mask_name", getattr(scene, "mask_type", "")),  # Use mask_name, fall back to mask_type for old data
                "mask_background": scene.mask_background,
                "prompt_source": scene.prompt_source,
                "prompt_key": scene.prompt_key or "",
                "custom_prompt": scene.custom_prompt or "",
                "video_prompt_source": getattr(scene, "video_prompt_source", "auto"),
                "video_prompt_key": getattr(scene, "video_prompt_key", ""),
                "video_custom_prompt": getattr(scene, "video_custom_prompt", ""),
                "depth_type": scene.depth_type,
                "pose_type": scene.pose_type,
                "use_depth": getattr(scene, "use_depth", False),
                "use_mask": getattr(scene, "use_mask", False),
                "use_pose": getattr(scene, "use_pose", False),
                "use_canny": getattr(scene, "use_canny", False),
                "available_masks": available_masks,
            })
        
        return web.json_response({
            "story_name": story_info.story_name,
            "story_dir": story_info.story_dir,
            "scene_count": len(scenes_data),
            "scenes": scenes_data,
        })
    
    except Exception as e:
        logger.exception("Error loading story")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/story/job_ids")
async def story_get_job_ids(request):
    """
    Get list of job IDs for a specific story.
    Query params: story_name (required)
    Returns: {"job_ids": [str]} - List of job IDs sorted by modification time (newest first)
    """
    try:
        story_name = request.query.get('story_name', '')
        
        if not story_name:
            return web.json_response({'error': 'story_name parameter required'}, status=400)
        
        stories_dir = default_stories_dir()
        story_dir = os.path.join(stories_dir, story_name)
        
        if not os.path.isdir(story_dir):
            logger.warning("fbTools API: story_dir '%s' not found for story_name='%s'", story_dir, story_name)
            return web.json_response({'job_ids': []})
        
        job_ids = list_job_ids(story_dir)
        logger.info("fbTools API: story_name='%s' has %d job_ids: %s", story_name, len(job_ids), job_ids)
        
        return web.json_response({'job_ids': job_ids})
    except Exception as e:
        logger.exception("fbTools API: Error getting job_ids for story_name='%s'", story_name)
        return web.json_response({'error': str(e)}, status=500)


@routes.get("/fbtools/scene/list")
async def scene_list(request):
    """
    Get list of available scene names.
    Returns: {"scenes": [str]}
    """
    try:
        scenes_dir = default_scenes_dir()
        available_scenes = get_subdirectories(scenes_dir)
        scene_names = sorted(available_scenes.keys()) if available_scenes else []
        
        return web.json_response({'scenes': scene_names})
    except Exception as e:
        logger.exception("fbTools API: Error listing scenes")
        return web.json_response({'error': str(e)}, status=500)


@routes.get("/fbtools/story/list")
async def story_list(request):
    """
    Get list of available story names.
    Returns: {"stories": [str]}
    """
    try:
        stories_dir = default_stories_dir()
        available_stories = get_subdirectories(stories_dir)
        story_names = sorted(available_stories.keys()) if available_stories else []
        
        return web.json_response({'stories': story_names})
    except Exception as e:
        logger.exception("fbTools API: Error listing stories")
        return web.json_response({'error': str(e)}, status=500)


@routes.post("/fbtools/story/regenerate_thumbnails")
async def story_regenerate_thumbnails(request):
    """
    Regenerate thumbnails for all scenes in a story that don't have them.
    Body: {"story_name": str}
    Returns: {"success": bool, "regenerated": int, "message": str}
    """
    try:
        data = await request.json()
        story_name = data.get("story_name")
        
        if not story_name:
            return web.json_response({'success': False, 'error': 'story_name is required'}, status=400)
        
        stories_dir = default_stories_dir()
        story_dir = os.path.join(stories_dir, story_name)
        
        if not os.path.isdir(story_dir):
            return web.json_response({'success': False, 'error': f'Story "{story_name}" not found'}, status=404)
        
        # Load story
        story_json_path = os.path.join(story_dir, "story.json")
        story_info = load_story(story_json_path)
        if not story_info:
            return web.json_response({'success': False, 'error': f'Failed to load story "{story_name}"'}, status=500)
        
        regenerated_count = 0
        scenes_dir = default_scenes_dir()
        
        # Regenerate thumbnails for each scene (force=True to regenerate all)
        for scene in story_info.scenes:
            scene_dir = os.path.join(scenes_dir, scene.scene_name)
            if not os.path.isdir(scene_dir):
                logger.warning("Scene directory not found: %s", scene_dir)
                continue
            
            thumbnail_path = os.path.join(scene_dir, "thumbnail.png")
            
            # Load scene info and regenerate thumbnail (force=True)
            scene_info = SceneInfo.from_scene_directory(scene_dir, scene.scene_name)
            scene_info.regenerate_thumbnail(scene_dir, force=True)
            
            # Check if thumbnail was actually created
            if os.path.exists(thumbnail_path):
                regenerated_count += 1
                logger.info("Generated thumbnail for scene '%s'", scene.scene_name)
            else:
                logger.warning("Failed to generate thumbnail for scene '%s'", scene.scene_name)
        
        return web.json_response({
            'success': True,
            'regenerated': regenerated_count,
            'message': f'Regenerated {regenerated_count} thumbnails for story "{story_name}"'
        })
        
    except Exception as e:
        logger.exception("fbTools API: Error regenerating thumbnails")
        return web.json_response({'error': str(e)}, status=500)


@routes.get("/fbtools/scene/thumbnail/{scene_name}")
async def get_scene_thumbnail(request):
    """
    Serve thumbnail image for a scene.
    Returns: thumbnail PNG image or 404 if not found
    """
    try:
        scene_name = request.match_info.get("scene_name")
        
        if not scene_name:
            return web.json_response({"error": "scene_name required"}, status=400)
        
        scenes_dir = default_scenes_dir()
        thumbnail_path = os.path.join(scenes_dir, scene_name, "thumbnail.png")
        
        if not os.path.exists(thumbnail_path):
            logger.warning("Thumbnail not found: %s", thumbnail_path)
            return web.json_response({"error": f"Thumbnail not found for scene '{scene_name}'"}, status=404)
        
        # Serve the image file
        return web.FileResponse(
            thumbnail_path,
            headers={
                'Content-Type': 'image/png',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        logger.exception("Error serving thumbnail")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/story/save")
async def story_save(request):
    """
    Save story data to filesystem.
    Body: {"story_name": str, "scenes": [...]}
    Returns: {"success": bool, "message": str}
    """
    try:
        data = await request.json()
        story_name = data.get("story_name")
        scenes_data = data.get("scenes", [])
        
        logger.info(
            "fb_tools -> StoryEdit: Received save request for story '%s' with %d scenes",
            story_name,
            len(scenes_data),
        )
        
        if not story_name:
            return web.json_response({"error": "story_name required"}, status=400)
        
        # Load existing story
        stories_dir = default_stories_dir()
        story_json_path = Path(stories_dir) / story_name / "story.json"
        
        if not story_json_path.exists():
            logger.warning("fb_tools -> StoryEdit: Story not found at %s", story_json_path)
            return web.json_response({"error": f"Story '{story_name}' not found"}, status=404)
        
        story_info = load_story(str(story_json_path))
        if not story_info:
            return web.json_response({"error": f"Failed to load story '{story_name}'"}, status=500)
        
        # Update scenes from received data
        updated_scenes = []
        for scene_data in scenes_data:
            scene = SceneInStory(
                scene_id=scene_data.get("scene_id", ""),
                scene_name=scene_data.get("scene_name", ""),
                scene_order=scene_data.get("scene_order", 0),
                mask_name=scene_data.get("mask_name", scene_data.get("mask_type", "")),  # Support both new and legacy
                mask_background=scene_data.get("mask_background", True),
                prompt_source=scene_data.get("prompt_source", "prompt"),
                prompt_key=scene_data.get("prompt_key", ""),
                custom_prompt=scene_data.get("custom_prompt", ""),
                video_prompt_source=scene_data.get("video_prompt_source", "auto"),
                video_prompt_key=scene_data.get("video_prompt_key", ""),
                video_custom_prompt=scene_data.get("video_custom_prompt", ""),
                depth_type=scene_data.get("depth_type", "depth"),
                pose_type=scene_data.get("pose_type", "open"),
                use_depth=scene_data.get("use_depth", False),
                use_mask=scene_data.get("use_mask", False),
                use_pose=scene_data.get("use_pose", False),
                use_canny=scene_data.get("use_canny", False),
            )
            updated_scenes.append(scene)
        
        # Update story info with new scenes
        story_info.scenes = updated_scenes
        
        # Save to disk
        save_story(story_info, str(story_json_path))
        
        return web.json_response({
            "success": True,
            "message": f"Saved story '{story_name}' with {len(updated_scenes)} scenes"
        })

    except Exception as e:
        logger.exception("Error saving story")
        return web.json_response({"error": str(e)}, status=500)


# ── LoRA civitai info endpoint ────────────────────────────────────────────────

_lora_info_cache: dict[str, dict] = {}


def _compute_lora_hash(path: str) -> str:
    """Full-file SHA256 — matches the hash CivitAI indexes in its by-hash API."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@routes.get("/fbtools/lora/civitai_info")
async def get_lora_civitai_info(request):
    """Fetch model version info from civitai by lora filename.

    Query params: lora=<filename relative to loras folder>
    Returns civitai model-version JSON or {"error": "..."}.
    Results are cached in memory and a .civitai.info sidecar is checked first
    (compatible with rgthree-comfy sidecar files).
    """
    import aiohttp as _aiohttp

    lora_name = request.rel_url.query.get("lora", "").strip()
    if not lora_name or lora_name == "None":
        return web.json_response({"error": "lora parameter required"}, status=400)

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return web.json_response({"error": f"LoRA not found: {lora_name}"}, status=404)

    if lora_path in _lora_info_cache:
        return web.json_response(_lora_info_cache[lora_path])

    # Honour existing rgthree-style sidecar without re-fetching.
    # rgthree names sidecars "<filename>.civitai.info" (full filename preserved),
    # e.g. "my_lora.safetensors.civitai.info".
    sidecar = Path(lora_path).parent / (Path(lora_path).name + ".civitai.info")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            _lora_info_cache[lora_path] = data
            return web.json_response(data)
        except Exception:
            pass

    try:
        sha256 = _compute_lora_hash(lora_path)
    except Exception as e:
        return web.json_response({"error": f"Hash computation failed: {e}"}, status=500)

    civitai_url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256}"
    try:
        async with _aiohttp.ClientSession() as session:
            async with session.get(
                civitai_url,
                headers={"User-Agent": "comfyui-fbTools/1.0"},
                timeout=_aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    _lora_info_cache[lora_path] = data
                    return web.json_response(data)
                elif resp.status == 404:
                    return web.json_response(
                        {"error": "LoRA not found on Civitai"}, status=404
                    )
                else:
                    return web.json_response(
                        {"error": f"Civitai returned HTTP {resp.status}"}, status=502
                    )
    except Exception as e:
        return web.json_response({"error": f"Civitai request failed: {e}"}, status=502)


# ============================================================================
# LORA SCENE NODES — persist and apply LoRA settings per model target
# ============================================================================

# Model targets supported by the LoRA scene system.
# Wan2.2 uses two separate diffusion models — one for the high-pass (structure) sampler
# and one for the low-pass (detail) sampler — so each has its own target.
# All other models have a single pass and need no High/Low distinction.
LORA_MODEL_TARGETS = [
    "LTX2.3",
    "Wan2.2-Native-High",   # first-pass (structure) model — outputs LORA_STACK for easy-use
    "Wan2.2-Native-Low",    # second-pass (detail) model — outputs LORA_STACK for easy-use
    "Wan2.2-Wrapper-High",  # first-pass (structure) model — outputs WANVIDLORA for WanVideoWrapper
    "Wan2.2-Wrapper-Low",   # second-pass (detail) model  — outputs WANVIDLORA for WanVideoWrapper
    "Flux2/Klein",
    "Qwen",
    "MiniMaxH3",
    "Z-Image",
]

# Weight key fragments that belong to LTX2.3 audio layers
LORA_AUDIO_KEYWORDS = [
    "audio", "vocoder", "speech", "audio_stream",
    "cross_modal", "video_to_audio", "av_ca",
]

LORA_ENTRY_TYPE      = "LORA_ENTRY"
LORA_STACK_DATA_TYPE = "LORA_STACK_DATA"


@io.comfytype(io_type=LORA_ENTRY_TYPE)
class LoraEntry:
    """
    Carries a single LoRA definition between LoraEntryDefine and LoraStackCollect.
    Internal dict structure:
      {
        "lora":           str,    # filename from loras folder
        "strength_model": float,
        "strength_clip":  float,
        "enabled":        bool,
        "model_target":   str,    # one of LORA_MODEL_TARGETS
        "audio_enabled":  bool,   # LTX2.3 only — include audio weights
      }
    """
    Type = dict

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "lora_entry", **kwargs):
            super().__init__(name, **kwargs)


@io.comfytype(io_type=LORA_STACK_DATA_TYPE)
class LoraStackData:
    """
    Carries the collected stack (list of LORA_ENTRY dicts) between nodes.
    Also serialisable to/from JSON for scene persistence.
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "lora_stack_data", **kwargs):
            super().__init__(name, **kwargs)


# ── LoRA scene helpers ────────────────────────────────────────────────────────

def _lora_get_list() -> list[str]:
    return ["None"] + folder_paths.get_filename_list("loras")


# mtime-keyed in-memory cache: {full_path: (mtime, weights_dict, metadata_dict)}
_lora_weight_cache: dict[str, tuple[float, dict, dict]] = {}


def _lora_load_weights(lora_name: str) -> tuple[dict, dict]:
    """Load raw LoRA weights + safetensors metadata from disk (cached by mtime).

    Returns (weights, metadata).  metadata is the safetensors header dict
    (may be empty {}) — pass it to comfy.sd.load_lora_for_models as
    lora_metadata so downstream nodes can inspect which LoRAs are applied.
    """
    path = folder_paths.get_full_path("loras", lora_name)
    if not path:
        raise FileNotFoundError(f"LoRA not found: {lora_name}")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    cached = _lora_weight_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    import comfy.utils as _comfy_utils
    weights, metadata = _comfy_utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    _lora_weight_cache[path] = (mtime, weights, metadata)
    return weights, metadata


def _lora_entries_for_target(entries: list[dict], model_target: str) -> list[dict]:
    """Filter to enabled entries matching the given model_target."""
    return [
        e for e in entries
        if e.get("enabled", True)
        and e.get("lora", "None") != "None"
        and e.get("model_target") == model_target
    ]


def _lora_stack_to_json(entries: list[dict]) -> str:
    return json.dumps(entries, indent=2, ensure_ascii=False)


def _lora_json_to_stack(json_str: str) -> list[dict]:
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ── Node: LoraEntryDefine ─────────────────────────────────────────────────────

class LoraEntryDefine(io.ComfyNode):
    """
    Define a single LoRA entry for a specific model target.
    Connect one or more of these to LoraStackCollect.

    video/audio/cross-attention strength controls only have effect when
    model_target is LTX2.3.  Set audio=0 and audio_to_video=0 to fully
    mute audio layers (equivalent to the old audio_enabled=False).
    strength_clip is ignored for model targets that have no CLIP encoder.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraEntryDefine"),
            display_name="LoRA Entry Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one LoRA for a specific model target. "
                "Connect to LoraStackCollect to build a persisted stack."
            ),
            inputs=[
                io.Combo.Input(
                    "lora",
                    display_name="LoRA",
                    options=_lora_get_list(),
                    default="None",
                    tooltip="Select the LoRA file.",
                ),
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline this LoRA applies to.",
                ),
                io.Float.Input(
                    "strength_model",
                    display_name="Strength (Model)",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.0001,
                    tooltip="LoRA strength applied to the UNet/transformer model weights.",
                ),
                io.Float.Input(
                    "strength_clip",
                    display_name="Strength (CLIP)",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.0001,
                    tooltip=(
                        "LoRA strength applied to the text encoder (CLIP). "
                        "Ignored for model targets that have no CLIP component."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enabled",
                    default=True,
                    tooltip="Disable to skip this LoRA without removing it from the stack.",
                ),
                io.Float.Input(
                    "video_strength",
                    display_name="Video Strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="LTX2.3: multiplier for all video layers (attn, feedforward, video→audio cross-attn). Set to 0 to skip video weights entirely.",
                ),
                io.Float.Input(
                    "audio_strength",
                    display_name="Audio Strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="LTX2.3: multiplier for all audio layers (attn, feedforward, audio→video cross-attn). Set to 0 to fully mute audio weights.",
                ),
            ],
            outputs=[
                LoraEntry.Output("lora_entry", display_name="LoRA Entry"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora: str,
        model_target: str,
        strength_model: float,
        strength_clip: float,
        enabled: bool,
        video_strength: float = 1.0,
        audio_strength: float = 1.0,
    ) -> io.NodeOutput:
        entry = {
            "lora":           lora,
            "model_target":   model_target,
            "strength_model": strength_model,
            "strength_clip":  strength_clip,
            "enabled":        enabled,
            "video_strength": video_strength,
            "audio_strength": audio_strength,
        }
        return io.NodeOutput(entry)


# ── Node: LoraStackCollect ────────────────────────────────────────────────────

class LoraStackCollect(io.ComfyNode):
    """
    Collect multiple LoRA entries into a persisted stack.

    Outputs a JSON string suitable for storing in a scene node,
    and a LORA_STACK_DATA object for direct connection to LoraStackApply.

    Merging/override rules (last-write-wins on (lora, model_target) key):
      1. existing_json entries are loaded first (lowest priority).
      2. prev_stack entries override existing_json duplicates.
      3. Connected LoraEntryDefine entries override everything.

    To edit a single entry in an existing scene stack without rebuilding it:
      SceneSelect.lora_stack_data → prev_stack
      LoraEntryDefine (same lora + model_target, new params) → entry_0
      → SceneLoraStackSave
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=1,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("LoraStackCollect"),
            display_name="LoRA Stack Collect",
            category="🧊 frost-byte/lora",
            description=(
                "Collect LoRA entries into a deduplicated stack. "
                "New entries (entry_0, entry_1…) override prev_stack/existing_json "
                "entries with the same (lora, model_target) key."
            ),
            inputs=[
                io.Autogrow.Input("entries", template=autogrow_template),
                LoraStackData.Input(
                    "prev_stack",
                    display_name="Prev Stack",
                    optional=True,
                    tooltip=(
                        "Existing LORA_STACK_DATA to merge into (e.g. from SceneSelect). "
                        "New entries on entry_0/entry_1/… override any duplicate "
                        "(lora, model_target) pairs from this stack."
                    ),
                ),
                io.String.Input(
                    "existing_json",
                    display_name="Existing JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Existing stack as a JSON string. Used if Prev Stack is not "
                        "connected. New entries override duplicates here too."
                    ),
                ),
            ],
            outputs=[
                LoraStackData.Output("lora_stack_data", display_name="Stack Data"),
                io.String.Output("stack_json",          display_name="Stack JSON"),
                io.Int.Output("entry_count",            display_name="Entry Count"),
                io.Custom("LORA_STACK").Output(
                    "lora_stack",
                    display_name="LoRA Stack",
                    tooltip=(
                        "Easy-use compatible LORA_STACK: list of (lora_name, model_strength, clip_strength) tuples. "
                        "Connect to EasyLoraStack, PowerLoraLoader, or any node that accepts LORA_STACK."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        entries: io.Autogrow.Type,
        prev_stack: Optional[list] = None,
        existing_json: str = "[]",
    ) -> io.NodeOutput:
        # Build ordered base: existing_json first, prev_stack overrides it
        base: list[dict] = _lora_json_to_stack(existing_json)
        if prev_stack:
            base.extend(prev_stack)

        # New entries (from LoraEntryDefine) have highest priority
        new_entries = [v for v in entries.values() if v is not None]
        base.extend(new_entries)

        # Deduplicate: last entry for each (lora, model_target) key wins
        seen: dict[tuple, dict] = {}
        for entry in base:
            key = (entry.get("lora", ""), entry.get("model_target", ""))
            seen[key] = entry
        merged = list(seen.values())

        stack_json = _lora_stack_to_json(merged)
        easy_stack = [
            (e["lora"], e["strength_model"], e["strength_clip"])
            for e in merged
            if e.get("enabled", True) and e.get("lora", "None") != "None"
        ]
        return io.NodeOutput(merged, stack_json, len(merged), easy_stack)


# ── Node: LoraStackView ──────────────────────────────────────────────────────

class LoraStackView(io.ComfyNode):
    """
    Inspect the contents of a LORA_STACK_DATA without modifying it.

    Outputs a human-readable summary string (connect to a Show Text node)
    and the raw stack_json STRING (connect to LoraStackCollect.existing_json
    if you need to extend the stack without the prev_stack input).
    Pass-through lora_stack_data output lets you chain this inline.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraStackView"),
            display_name="LoRA Stack View",
            category="🧊 frost-byte/lora",
            description=(
                "Display the contents of a LoRA stack as readable text. "
                "Also outputs the raw JSON string and a pass-through stack "
                "so it can sit inline between SceneSelect and LoraStackCollect."
            ),
            inputs=[
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from SceneSelect, StoryVideoBatch, or LoraStackCollect.",
                ),
            ],
            outputs=[
                io.String.Output("summary",         display_name="Summary",    tooltip="Human-readable list of LoRA entries."),
                io.String.Output("stack_json",       display_name="Stack JSON", tooltip="Raw JSON — connect to LoraStackCollect.existing_json if needed."),
                LoraStackData.Output("lora_stack_data", display_name="Stack Data", tooltip="Pass-through — same stack, unchanged."),
                io.Int.Output("entry_count",         display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora_stack_data: Optional[list] = None,
    ) -> io.NodeOutput:
        stack = lora_stack_data or []
        lines: list[str] = [f"LoRA Stack ({len(stack)} entries):"]
        for i, entry in enumerate(stack):
            lora        = entry.get("lora", "?") or "?"
            target      = entry.get("model_target", "?") or "?"
            strength    = entry.get("strength_model", 1.0)
            enabled     = entry.get("enabled", True)
            status      = "" if enabled else "  [DISABLED]"
            lines.append(f"  [{i}] {lora}  |  {target}  |  strength={strength:.4f}{status}")
        summary = "\n".join(lines) if stack else "(empty stack)"
        stack_json = _lora_stack_to_json(stack)
        return io.NodeOutput(summary, stack_json, stack if stack else None, len(stack))


# ── Node: LoraStackApply ──────────────────────────────────────────────────────

class LoraStackApply(io.ComfyNode):
    """
    Apply a persisted LoRA stack to the active model at inference time.

    Set model_target to match the pipeline you are running.
    Only LoRA entries tagged for that target will be loaded and applied.

    Output behaviour by target:
      LTX2.3        → MODEL, CLIP (audio weights filtered per entry flag)
      Wan2.2-Native → MODEL, CLIP, LORA_STACK (easy-use compatible tuple list)
      Wan2.2-Wrapper→ MODEL, CLIP, WANVIDLORA (WanVideoWrapper compatible dict list)
      Flux2/Klein   → MODEL, CLIP (standard load_lora_for_models)
      Qwen          → MODEL, CLIP (standard load_lora_for_models)
      Z-Image       → MODEL, CLIP (standard load_lora_for_models)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraStackApply"),
            display_name="LoRA Stack Apply",
            category="🧊 frost-byte/lora",
            description=(
                "Apply a persisted LoRA stack to the active model. "
                "Set model_target to match the pipeline you are running. "
                "Only LoRAs tagged for that target are applied."
            ),
            inputs=[
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline to apply LoRAs for.",
                ),
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from LoraStackCollect.",
                ),
                io.String.Input(
                    "stack_json",
                    display_name="Stack JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "JSON string from a scene node. "
                        "Used if Stack Data input is not connected."
                    ),
                ),
                io.Model.Input(
                    "model",
                    display_name="Model",
                    optional=True,
                    tooltip="Connect the MODEL to patch with LoRA weights.",
                ),
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    optional=True,
                    tooltip="Connect the CLIP/text encoder to patch with LoRA weights.",
                ),
                io.Custom("LORA_STACK").Input(
                    "prev_lora_stack",
                    display_name="Prev LoRA Stack",
                    optional=True,
                    tooltip=(
                        "Wan2.2-Native only: chain from another LORA_STACK output "
                        "to prepend existing entries."
                    ),
                ),
                io.Custom("WANVIDLORA").Input(
                    "prev_wanvid_lora",
                    display_name="Prev WanVid LoRA",
                    optional=True,
                    tooltip=(
                        "Wan2.2-Wrapper only: chain from another WANVIDLORA output "
                        "to prepend existing entries."
                    ),
                ),
                io.Boolean.Input(
                    "low_mem_load",
                    display_name="Low VRAM Load",
                    default=False,
                    optional=True,
                    tooltip="Wan2.2-Wrapper only: load LoRA with reduced VRAM usage.",
                ),
                io.Boolean.Input(
                    "merge_loras",
                    display_name="Merge LoRAs",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Wan2.2-Wrapper only: merge LoRAs into model weights. "
                        "Disable for GGUF / scaled fp8 models."
                    ),
                ),
                io.Boolean.Input(
                    "int8_model",
                    display_name="INT8 Model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Enable when the model is INT8-quantized (ComfyUI-INT8-Fast). "
                        "LoRAs are applied directly to the model via INT8ModelPatcher's "
                        "dequantize→apply→requantize cycle, bypassing Wan2.2 stack outputs."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model",        display_name="Model"),
                io.Clip.Output("clip",          display_name="CLIP"),
                io.Custom("LORA_STACK").Output("lora_stack",   display_name="LoRA Stack"),
                io.Custom("WANVIDLORA").Output("wanvid_lora",  display_name="WanVid LoRA"),
                io.Int.Output("applied_count",  display_name="Applied Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_target: str,
        lora_stack_data: Optional[list] = None,
        stack_json: str = "[]",
        model: Optional[object] = None,
        clip: Optional[object] = None,
        prev_lora_stack: Optional[list] = None,
        prev_wanvid_lora: Optional[list] = None,
        low_mem_load: bool = False,
        merge_loras: bool = True,
        int8_model: bool = False,
    ) -> io.NodeOutput:
        all_entries = lora_stack_data if lora_stack_data is not None else _lora_json_to_stack(stack_json)
        target_entries = _lora_entries_for_target(all_entries, model_target)

        if int8_model:
            # INT8 models (ComfyUI-INT8-Fast INT8ModelPatcher) require direct patching.
            # Wan2.2 stack outputs (LORA_STACK / WANVIDLORA) are not usable with INT8 because
            # their downstream consumers have no knowledge of INT8 dequant/requant.
            # For LTX2.3 targets, use per-layer filtering; for all others, use flat apply.
            if model_target == "LTX2.3":
                model, clip, count = _lora_apply_ltx23(model, clip, target_entries)
            else:
                model, clip, count = _lora_apply_standard(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)

        if model_target == "LTX2.3":
            model, clip, count = _lora_apply_ltx23(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)
        elif model_target in ("Wan2.2-Native-High", "Wan2.2-Native-Low"):
            lora_stack, count = _lora_build_stack(target_entries, prev_lora_stack)
            return io.NodeOutput(model, clip, lora_stack, None, count)
        elif model_target in ("Wan2.2-Wrapper-High", "Wan2.2-Wrapper-Low"):
            wanvid_lora, count = _lora_build_wanvid(
                target_entries, prev_wanvid_lora, low_mem_load, merge_loras
            )
            return io.NodeOutput(model, clip, None, wanvid_lora, count)
        else:
            # Flux2/Klein, Qwen, Z-Image — standard load_lora_for_models
            model, clip, count = _lora_apply_standard(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)


# ── LoRA apply implementations ────────────────────────────────────────────────

def _lora_apply_ltx23(model, clip, entries: list[dict]) -> tuple:
    """LTX2.3 apply: per-key strength scaling for audio/video/cross-attn layers.

    New format (video_strength / audio_strength): video_strength scales all video
    and video-side cross-attn keys; audio_strength scales all audio and audio-side
    cross-attn keys.

    Legacy format (video / video_to_audio / audio / audio_to_video / other) and
    the old audio_enabled=False boolean are still accepted for backward compat.
    """
    import comfy.lora as _comfy_lora
    m, c = model, clip
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue

        if "video_strength" in entry or "audio_strength" in entry:
            # New 2-param format
            video_s    = float(entry.get("video_strength", 1.0))
            audio_s    = float(entry.get("audio_strength", 1.0))
            video_to_a = video_s   # video drives video→audio cross-attn
            audio_to_v = audio_s   # audio drives audio→video cross-attn
            other_s    = video_s
        else:
            # Legacy 5-param format (+ audio_enabled boolean compat)
            _old_audio = entry.get("audio_enabled", None)
            if _old_audio is False and "audio" not in entry:
                audio_s    = 0.0
                audio_to_v = 0.0
            else:
                audio_s    = float(entry.get("audio",          1.0))
                audio_to_v = float(entry.get("audio_to_video", 1.0))
            video_s    = float(entry.get("video",          1.0))
            video_to_a = float(entry.get("video_to_audio", 1.0))
            other_s    = float(entry.get("other",          1.0))

        strength_model = entry.get("strength_model", 1.0)
        strength_clip  = entry.get("strength_clip",  1.0)

        try:
            import comfy.lora_convert as _comfy_lora_convert
            raw, _meta = _lora_load_weights(lora_name)
            raw = _comfy_lora_convert.convert_lora(raw)

            key_map = {}
            if m is not None:
                key_map = _comfy_lora.model_lora_keys_unet(m.model, key_map)
            if c is not None:
                key_map = _comfy_lora.model_lora_keys_clip(c.cond_stage_model, key_map)

            loaded = _comfy_lora.load_lora(raw, key_map)

            if not (video_s == 1.0 and video_to_a == 1.0
                    and audio_s == 1.0 and audio_to_v == 1.0 and other_s == 1.0):
                keys_to_delete = []
                for key, value in loaded.items():
                    ks = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) else str(key))
                    if   "video_to_audio_attn" in ks: mult = video_to_a
                    elif "audio_to_video_attn" in ks: mult = audio_to_v
                    elif "audio_attn" in ks or "audio_ff.net" in ks: mult = audio_s
                    elif "attn" in ks or "ff.net" in ks: mult = video_s
                    else: mult = other_s
                    if mult == 0.0:
                        keys_to_delete.append(key)
                    elif mult != 1.0 and hasattr(value, "weights"):
                        wl = list(value.weights)
                        wl[2] = (wl[2] if wl[2] is not None else 1.0) * mult
                        loaded[key].weights = tuple(wl)
                for key in keys_to_delete:
                    loaded.pop(key, None)

            if m is not None:
                new_m = m.clone()
                new_m.add_patches(loaded, strength_model)
                m = new_m
            if c is not None:
                new_c = c.clone()
                new_c.add_patches(loaded, strength_clip)
                c = new_c
            count += 1
        except Exception as e:
            print(f"[LoraStackApply] LTX2.3: failed to load '{lora_name}': {e}")
    return m, c, count


def _lora_apply_standard(model, clip, entries: list[dict]) -> tuple:
    """Standard apply via load_lora_for_models (Flux2/Klein, Qwen, MiniMaxH3, Z-Image)."""
    import comfy.sd as _comfy_sd
    m, c = model, clip
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        try:
            weights, metadata = _lora_load_weights(lora_name)
            m, c = _comfy_sd.load_lora_for_models(
                m, c, weights,
                entry.get("strength_model", 1.0),
                entry.get("strength_clip",  1.0),
                lora_metadata=metadata or None,
            )
            count += 1
        except Exception as e:
            print(f"[LoraStackApply] Standard: failed to load '{lora_name}': {e}")
    return m, c, count


def _lora_build_stack(entries: list[dict], prev_lora_stack: Optional[list]) -> tuple:
    """Build a LORA_STACK compatible with easy-use loraStack."""
    stack: list[tuple] = []
    if prev_lora_stack:
        stack.extend([l for l in prev_lora_stack if l[0] != "None"])
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        stack.append((lora_name, entry.get("strength_model", 1.0), entry.get("strength_clip", 1.0)))
        count += 1
    return stack if stack else None, count


def _lora_build_wanvid(
    entries: list[dict],
    prev_wanvid_lora: Optional[list],
    low_mem_load: bool,
    merge_loras: bool,
) -> tuple:
    """Build a WANVIDLORA compatible with WanVideoWrapper.

    Per-entry 'low_mem_load', 'merge_loras', 'blocks', and 'layer_filter' fields
    take precedence over the node-level arguments when present (preserved from
    migrated loras.json data or explicitly set by the user).
    """
    loras_list: list[dict] = []
    if prev_wanvid_lora:
        loras_list.extend(list(prev_wanvid_lora))
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        # Per-entry values override node-level infrastructure settings when present
        entry_merge    = entry.get("merge_loras",  merge_loras)
        entry_low_mem  = entry.get("low_mem_load", low_mem_load)
        if not entry_merge:
            entry_low_mem = False  # matches WanVideoWrapper behaviour
        try:
            path = folder_paths.get_full_path_or_raise("loras", lora_name)
        except Exception:
            path = folder_paths.get_full_path("loras", lora_name)
            if not path:
                print(f"[LoraStackApply] WanVid: LoRA not found: {lora_name}")
                continue
        loras_list.append({
            "path":         path,
            "strength":     round(entry.get("strength_model", 1.0), 4),
            "name":         os.path.splitext(os.path.basename(lora_name))[0],
            "blocks":       entry.get("blocks", {}),
            "layer_filter": entry.get("layer_filter", ""),
            "low_mem_load": entry_low_mem,
            "merge_loras":  entry_merge,
        })
        count += 1
    return loras_list if loras_list else None, count


# ── Node: WanVidLoraStack ─────────────────────────────────────────────────────

class WanVidLoraStack(io.ComfyNode):
    """
    Build a WANVIDLORA list for WanVideoWrapper directly from LORA_ENTRY inputs.

    Accepts 1–20 LORA_ENTRY inputs (from LoraEntryDefine) and converts them into a
    WANVIDLORA list compatible with WanVideoWrapper's sampler nodes. All enabled,
    non-None entries are included regardless of their model_target tag, making this
    node the Wan-specific alternative to LoraStackApply.

    Use prev_wanvid_lora to chain with upstream WanVideoLoraSelect nodes.
    Per-entry low_mem_load/merge_loras values (from migrated loras.json data) take
    precedence over the node-level settings when present.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=1,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("WanVidLoraStack"),
            display_name="WanVid LoRA Stack",
            category="🧊 frost-byte/lora",
            description=(
                "Build a WANVIDLORA list from LoRA entries for WanVideoWrapper. "
                "Connect LoraEntryDefine nodes to entry_0, entry_1… inputs. "
                "Outputs WANVIDLORA compatible with WanVideoWrapper sampler nodes."
            ),
            inputs=[
                io.Autogrow.Input("entries", template=autogrow_template),
                io.Custom("WANVIDLORA").Input(
                    "prev_wanvid_lora",
                    display_name="Prev WanVid LoRA",
                    optional=True,
                    tooltip="Chain from another WANVIDLORA output to prepend existing entries.",
                ),
                io.Boolean.Input(
                    "low_mem_load",
                    display_name="Low VRAM Load",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Load LoRAs with reduced VRAM usage, at the cost of slower loading. "
                        "No effect when merge_loras is False."
                    ),
                ),
                io.Boolean.Input(
                    "merge_loras",
                    display_name="Merge LoRAs",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Merge LoRAs into model weights before sampling. "
                        "Disable for GGUF or scaled fp8 models."
                    ),
                ),
            ],
            outputs=[
                io.Custom("WANVIDLORA").Output(
                    "wanvid_lora",
                    display_name="WanVid LoRA",
                    tooltip="Connect to WanVideoWrapper sampler nodes.",
                ),
                io.Int.Output("entry_count", display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        entries: io.Autogrow.Type,
        prev_wanvid_lora: Optional[list] = None,
        low_mem_load: bool = False,
        merge_loras: bool = True,
    ) -> io.NodeOutput:
        all_entries = [v for v in entries.values() if v is not None]
        enabled = [e for e in all_entries if e.get("enabled", True)]
        wanvid_lora, count = _lora_build_wanvid(enabled, prev_wanvid_lora, low_mem_load, merge_loras)
        return io.NodeOutput(wanvid_lora, count)


# ── Node: LoraStackBuilder ───────────────────────────────────────────────────

_LORA_BUILDER_ROWS = 8


def _lora_builder_inline_inputs(num_rows: int = _LORA_BUILDER_ROWS) -> list:
    """Generate the flat per-row LoRA widget inputs for LoraStackBuilder."""
    lora_list = _lora_get_list()
    inputs = []
    for i in range(num_rows):
        inputs.extend([
            io.Combo.Input(
                f"lora_{i}",
                display_name=f"LoRA {i}",
                options=lora_list,
                default="None",
                optional=True,
                tooltip=f"LoRA file for slot {i}. Leave as None to skip.",
            ),
            io.Float.Input(
                f"strength_model_{i}",
                display_name=f"Strength (Model)",
                default=1.0, min=-10.0, max=10.0, step=0.0001,
                optional=True,
                tooltip=f"Model weight strength for slot {i}.",
            ),
            io.Float.Input(
                f"strength_clip_{i}",
                display_name=f"Strength (CLIP)",
                default=1.0, min=-10.0, max=10.0, step=0.0001,
                optional=True,
                tooltip=f"CLIP weight strength for slot {i}. Ignored for model targets without a CLIP encoder.",
            ),
            io.Boolean.Input(
                f"enabled_{i}",
                display_name=f"Enabled",
                default=True,
                optional=True,
                tooltip=f"Uncheck to skip slot {i} without removing it.",
            ),
            io.Float.Input(
                f"video_{i}",
                display_name=f"Video Strength",
                default=1.0, min=0.0, max=1.0, step=0.01,
                optional=True,
                tooltip=f"LTX2.3 only: multiplier for video and video-side cross-attn layers in slot {i}.",
            ),
            io.Float.Input(
                f"audio_{i}",
                display_name=f"Audio Strength",
                default=1.0, min=0.0, max=1.0, step=0.01,
                optional=True,
                tooltip=f"LTX2.3 only: multiplier for audio and audio-side cross-attn layers in slot {i}.",
            ),
        ])
    return inputs


class LoraStackBuilder(io.ComfyNode):
    """
    Build a LORA_STACK_DATA from up to 8 inline LoRA rows — no separate
    LoraEntryDefine / LoraStackCollect nodes required.

    Select model_target once at the top; JS hides the video/audio sliders for
    any target that isn't LTX2.3.  An optional autogrow input accepts LORA_ENTRY
    connections from LoraEntryDefine nodes for power-user workflows.
    A prev_stack input lets you merge onto an existing stack (e.g. from a scene).

    Dedup rule: last-write-wins on (lora, model_target) key.
    Priority (lowest → highest): prev_stack → inline rows → autogrow connections.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=0,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("LoraStackBuilder"),
            display_name="LoRA Stack Builder",
            category="🧊 frost-byte/lora",
            description=(
                "Build a LORA_STACK_DATA from inline LoRA rows. "
                "Select model_target to auto-show LTX2.3 video/audio sliders. "
                "Connect prev_stack to merge onto an existing stack."
            ),
            inputs=[
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline these LoRAs apply to.",
                ),
                LoraStackData.Input(
                    "prev_stack",
                    display_name="Prev Stack",
                    optional=True,
                    tooltip="Existing LORA_STACK_DATA to merge into (lowest priority).",
                ),
                io.Autogrow.Input(
                    "entries",
                    template=autogrow_template,
                    optional=True,
                    tooltip="Optional LORA_ENTRY connections from LoraEntryDefine nodes (highest priority).",
                ),
                *_lora_builder_inline_inputs(_LORA_BUILDER_ROWS),
            ],
            outputs=[
                LoraStackData.Output("lora_stack_data", display_name="Stack Data"),
                io.String.Output("stack_json",          display_name="Stack JSON"),
                io.Int.Output("entry_count",            display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_target: str,
        entries: io.Autogrow.Type,
        prev_stack: Optional[list] = None,
        **kwargs,
    ) -> io.NodeOutput:
        is_ltx = model_target == "LTX2.3"

        # Build inline entries from flat kwargs (rows 0 .. _LORA_BUILDER_ROWS-1)
        inline: list[dict] = []
        for i in range(_LORA_BUILDER_ROWS):
            lora = kwargs.get(f"lora_{i}") or "None"
            if lora == "None":
                continue
            entry: dict = {
                "lora":           lora,
                "model_target":   model_target,
                "strength_model": kwargs.get(f"strength_model_{i}", 1.0),
                "strength_clip":  kwargs.get(f"strength_clip_{i}",  1.0),
                "enabled":        kwargs.get(f"enabled_{i}",        True),
            }
            if is_ltx:
                entry["video_strength"] = kwargs.get(f"video_{i}", 1.0)
                entry["audio_strength"] = kwargs.get(f"audio_{i}", 1.0)
            inline.append(entry)

        # Collect autogrow LORA_ENTRY connections (highest priority)
        connected: list[dict] = [v for v in entries.values() if v is not None]

        # Merge: prev_stack → inline → connected; last-write-wins on (lora, model_target)
        base = list(prev_stack) if prev_stack else []
        seen: dict[tuple, dict] = {}
        for e in base + inline + connected:
            key = (e.get("lora", ""), e.get("model_target", ""))
            seen[key] = e
        merged = list(seen.values())

        stack_json = _lora_stack_to_json(merged)
        return io.NodeOutput(merged, stack_json, len(merged))


# ── Preset scene-image loader ─────────────────────────────────────────────────

def _load_preset_scene_images(
    scene_name: str,
    pose_image_type: str,
) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
    """Return (base_image, pose_image) tensors for a preset's linked scene.

    Returns (None, None) when scene_name is "none"/empty or the directory
    doesn't exist.  Callers should substitute a placeholder before wiring
    these to io.Image outputs.
    """
    if not scene_name or scene_name == "none":
        return None, None
    scene_dir = os.path.join(default_scenes_dir(), scene_name)
    if not os.path.isdir(scene_dir):
        logger.warning("_load_preset_scene_images: scene_dir '%s' not found", scene_dir)
        return None, None
    pose_attr = default_pose_options.get(pose_image_type, "pose_open_image")
    try:
        assets = SceneInfo.load_preview_assets(
            scene_dir,
            depth_attr="depth_image",
            pose_attr=pose_attr,
            mask_name="",
        )
        return assets.get("base_image"), assets.get("pose_image")
    except Exception:
        logger.exception("_load_preset_scene_images: error loading assets for '%s'", scene_name)
        return None, None


def _preset_scene_ui_and_images(
    preset: dict,
    names: list[str],
) -> "tuple[torch.Tensor, torch.Tensor, dict]":
    """Build (base_image, pose_image, ui_data) for a *PresetSelect execute().

    Always returns tensors (placeholder when no scene is linked).
    ui_data includes preset_names and any preview images for is_output_node.
    """
    base_image, pose_image = _load_preset_scene_images(
        preset.get("scene_name", "none"),
        preset.get("pose_image_type", "open"),
    )

    placeholder = make_empty_image(1, 64, 64)
    base_out = base_image if base_image is not None else placeholder
    pose_out = pose_image if pose_image is not None else placeholder

    preview_batch = [t for t in [base_image, pose_image] if t is not None]
    preview_image = ui.PreviewImage(image=torch.cat(preview_batch, dim=0)) if preview_batch else None

    ui_data: dict = {"preset_names": names}
    if preview_image:
        pd = preview_image.as_dict()
        ui_data["images"] = pd.get("images", [])
        ui_data["animated"] = pd.get("animated", False)

    return base_out, pose_out, ui_data


# ── Custom type: LORA_PRESET_LIST ────────────────────────────────────────────

LORA_PRESET_LIST_TYPE = "LORA_PRESET_LIST"


@io.comfytype(io_type=LORA_PRESET_LIST_TYPE)
class LoraPresetList:
    """
    Carries an ordered list of LoRA presets between nodes.
    Each entry is a dict: { name, lora_stack, prompt, scene_name, pose_image_type }.
    lora_stack holds a LORA_STACK_DATA value (list of dicts from LoraStackCollect).
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "preset_list", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: LoraPresetDefine ────────────────────────────────────────────────────

class LoraPresetDefine(io.ComfyNode):
    """
    Define one LoRA preset and append it to an optional incoming preset list.
    Chain multiple LoraPresetDefine nodes sequentially to build a collection;
    leave preset_list unconnected on the first node.

    Each preset holds a name, a single LORA_STACK_DATA, optional prompt, and
    an optional linked scene (for base/pose image output from LoraPresetSelect).
    Use this instead of WanPresetDefine for models with a single sampler stage.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraPresetDefine"),
            display_name="LoRA Preset Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one LoRA preset (name, LoRA stack, prompt, optional scene) "
                "and append it to an optional incoming preset list. "
                "Chain multiple nodes to build a preset collection."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    display_name="Preset Name",
                    default="Preset",
                    tooltip="Human-readable name for this preset.",
                ),
                LoraStackData.Input(
                    "lora_stack",
                    display_name="LoRA Stack (LORA_STACK_DATA)",
                    optional=True,
                    tooltip="Rich per-target stack from LoraStackCollect's 'Stack Data' output. Auto-generates the native LORA_STACK output.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_stack_native",
                    display_name="LoRA Stack (Native)",
                    optional=True,
                    tooltip="Native (name, model_str, clip_str) stack from any easy-use compatible source. Use instead of or alongside the LORA_STACK_DATA input.",
                ),
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    default="",
                    multiline=True,
                    tooltip="Positive prompt text for this preset.",
                ),
                io.Combo.Input(
                    "scene_name",
                    display_name="Scene",
                    options=["none"],
                    default="none",
                    tooltip=(
                        "Optional scene to associate with this preset. "
                        "When selected, LoraPresetSelect outputs the scene's base and pose images."
                    ),
                ),
                io.Combo.Input(
                    "pose_image_type",
                    display_name="Pose Image Type",
                    options=list(default_pose_options.keys()),
                    default="open",
                    tooltip="Which pose image variant to load from the scene.",
                ),
                LoraPresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    optional=True,
                    tooltip="Incoming list from a previous LoraPresetDefine node. Leave unconnected on the first node in the chain.",
                ),
            ],
            outputs=[
                LoraPresetList.Output("preset_list", display_name="Preset List"),
            ],
        )

    @classmethod
    def validate_inputs(cls, scene_name: str = "none", **kwargs) -> bool | str:
        # scene_name is populated dynamically by the frontend; bypass static validation.
        return True

    @classmethod
    def execute(
        cls,
        name: str,
        lora_stack: Optional[list] = None,
        lora_stack_native: Optional[list] = None,
        prompt: str = "",
        scene_name: str = "none",
        pose_image_type: str = "open",
        preset_list: Optional[list] = None,
    ) -> io.NodeOutput:
        from .utils.lora_presets import preset_define
        return io.NodeOutput(
            preset_define(name, lora_stack, prompt, preset_list, scene_name, pose_image_type, lora_stack_native)
        )


# ── Node: LoraPresetSelect ────────────────────────────────────────────────────

class LoraPresetSelect(io.ComfyNode):
    """
    Select one preset from a LoraPresetDefine chain by name.
    Outputs the preset's LoRA stack, prompt, scene images, and a summary of
    all available presets (wire to a Show Text node).

    Falls back to the first preset if the selected name is not found.
    If the preset has a linked scene, base_image and pose_image are loaded
    from that scene; otherwise placeholder 64×64 black images are returned.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraPresetSelect"),
            display_name="LoRA Preset Select",
            category="🧊 frost-byte/lora",
            description=(
                "Select one preset from a completed LoraPresetDefine chain. "
                "Outputs the LoRA stack, prompt, and optional scene images."
            ),
            inputs=[
                LoraPresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    tooltip="The complete preset list from the end of a LoraPresetDefine chain.",
                ),
                io.Combo.Input(
                    "selected_preset",
                    display_name="Preset",
                    options=["none"],
                    default="none",
                    tooltip="Select a preset by name. Connect a Preset List and run this node to populate the dropdown.",
                ),
            ],
            outputs=[
                io.String.Output("name",              display_name="Name"),
                LoraStackData.Output("lora_stack",    display_name="LoRA Stack (LORA_STACK_DATA)",
                    tooltip="Rich per-target stack. Connect to LoraStackApply."),
                io.String.Output("prompt",            display_name="Prompt"),
                io.String.Output("available_presets", display_name="Available Presets"),
                io.Image.Output("base_image",         display_name="Base Image",
                    tooltip="Base image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Image.Output("pose_image",         display_name="Pose Image",
                    tooltip="Pose image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Custom("LORA_STACK").Output("lora_stack_native", display_name="LoRA Stack (Native)",
                    tooltip="Native (name, model_str, clip_str) stack. Connect to EasyLoraStack, PowerLoraLoader, or any easy-use compatible node."),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, selected_preset: str, **kwargs) -> bool | str:
        # Accept any string — options are populated dynamically by the frontend
        # after execution, so the static schema list ["none"] is just a placeholder.
        return True

    @classmethod
    def execute(
        cls,
        preset_list: list,
        selected_preset: str,
    ) -> io.NodeOutput:
        from .utils.lora_presets import preset_select
        name, lora_stack, lora_stack_native, prompt, available = preset_select(preset_list, selected_preset)
        names = [p.get("name", "") for p in preset_list] if preset_list else []
        selected = next((p for p in preset_list if p.get("name") == name), {}) if preset_list else {}
        base_image, pose_image, ui_data = _preset_scene_ui_and_images(selected, names)
        return io.NodeOutput(name, lora_stack, prompt, available, base_image, pose_image, lora_stack_native, ui=ui_data)


# ── Custom type: PRESET_LIST ──────────────────────────────────────────────────

PRESET_LIST_TYPE = "PRESET_LIST"


@io.comfytype(io_type=PRESET_LIST_TYPE)
class PresetList:
    """
    Carries an ordered list of Wan video generation presets between nodes.
    Each entry is a dict: { name, lora_h, lora_l, prompt, scene_name, pose_image_type }.
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "preset_list", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: WanPresetDefine ─────────────────────────────────────────────────────

class WanPresetDefine(io.ComfyNode):
    """
    Define one Wan video generation preset and append it to an optional
    incoming preset list.  Chain multiple WanPresetDefine nodes sequentially
    to build a collection; leave preset_list unconnected on the first node.

    lora_h / lora_l carry LoRA stacks for the high-noise and low-noise model
    stages respectively.  An optional linked scene provides base/pose images
    that WanPresetSelect outputs when this preset is selected.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("WanPresetDefine"),
            display_name="Wan Preset Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one Wan video preset (name, high/low LoRA, prompt, optional scene) "
                "and append it to an optional incoming preset list. "
                "Chain multiple nodes to build a preset collection."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    display_name="Preset Name",
                    default="Preset",
                    tooltip="Human-readable name for this preset.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_h",
                    display_name="LoRA Stack (High Noise)",
                    optional=True,
                    tooltip="LoRA stack for the high-noise model stage. Connect from LoraStackCollect, EasyLoraStack, PowerLoraLoader, or any LORA_STACK source.",
                ),
                io.Custom("LORA_STACK").Input(
                    "lora_l",
                    display_name="LoRA Stack (Low Noise)",
                    optional=True,
                    tooltip="LoRA stack for the low-noise model stage. Connect from LoraStackCollect, EasyLoraStack, PowerLoraLoader, or any LORA_STACK source.",
                ),
                io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    default="",
                    multiline=True,
                    tooltip="Positive prompt text for this preset.",
                ),
                io.Combo.Input(
                    "scene_name",
                    display_name="Scene",
                    options=["none"],
                    default="none",
                    tooltip=(
                        "Optional scene to associate with this preset. "
                        "When selected, WanPresetSelect outputs the scene's base and pose images."
                    ),
                ),
                io.Combo.Input(
                    "pose_image_type",
                    display_name="Pose Image Type",
                    options=list(default_pose_options.keys()),
                    default="open",
                    tooltip="Which pose image variant to load from the scene.",
                ),
                PresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    optional=True,
                    tooltip="Incoming list from a previous WanPresetDefine node. Leave unconnected on the first node in the chain.",
                ),
            ],
            outputs=[
                PresetList.Output("preset_list", display_name="Preset List"),
            ],
        )

    @classmethod
    def validate_inputs(cls, scene_name: str = "none", **kwargs) -> bool | str:
        # scene_name is populated dynamically by the frontend; bypass static validation.
        return True

    @classmethod
    def execute(
        cls,
        name: str,
        lora_h: Optional[list] = None,
        lora_l: Optional[list] = None,
        prompt: str = "",
        scene_name: str = "none",
        pose_image_type: str = "open",
        preset_list: Optional[list] = None,
    ) -> io.NodeOutput:
        from .utils.wan_presets import preset_define
        return io.NodeOutput(preset_define(name, lora_h, lora_l, prompt, preset_list, scene_name, pose_image_type))


# ── Node: WanPresetSelect ─────────────────────────────────────────────────────

class WanPresetSelect(io.ComfyNode):
    """
    Select one preset from a WanPresetDefine chain by name.
    Outputs individual fields for downstream consumption, a formatted summary
    of all available presets, and scene images if the preset has a linked scene.

    Falls back to the first preset if the selected name is not found.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("WanPresetSelect"),
            display_name="Wan Preset Select",
            category="🧊 frost-byte/lora",
            description=(
                "Select one preset from a completed WanPresetDefine chain. "
                "Outputs individual fields, scene images, and an available-presets summary."
            ),
            inputs=[
                PresetList.Input(
                    "preset_list",
                    display_name="Preset List",
                    tooltip="The complete preset list from the end of a WanPresetDefine chain.",
                ),
                io.Combo.Input(
                    "selected_preset",
                    display_name="Preset",
                    options=["none"],
                    default="none",
                    tooltip="Select a preset by name. Connect a Preset List and run this node to populate the dropdown.",
                ),
            ],
            outputs=[
                io.String.Output("name",              display_name="Name"),
                io.Custom("LORA_STACK").Output("lora_h", display_name="LoRA Stack (High Noise)"),
                io.Custom("LORA_STACK").Output("lora_l", display_name="LoRA Stack (Low Noise)"),
                io.String.Output("prompt",            display_name="Prompt"),
                io.String.Output("available_presets", display_name="Available Presets"),
                io.Image.Output("base_image",         display_name="Base Image",
                    tooltip="Base image from the preset's linked scene, or a placeholder if no scene is set."),
                io.Image.Output("pose_image",         display_name="Pose Image",
                    tooltip="Pose image from the preset's linked scene, or a placeholder if no scene is set."),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, selected_preset: str, **kwargs) -> bool | str:
        # Accept any string — options are populated dynamically by the frontend
        # after execution, so the static schema list ["none"] is just a placeholder.
        return True

    @classmethod
    def execute(
        cls,
        preset_list: list,
        selected_preset: str,
    ) -> io.NodeOutput:
        from .utils.wan_presets import preset_select
        name, lora_h, lora_l, prompt, available = preset_select(preset_list, selected_preset)
        names = [p.get("name", "") for p in preset_list] if preset_list else []
        selected = next((p for p in preset_list if p.get("name") == name), {}) if preset_list else {}
        base_image, pose_image, ui_data = _preset_scene_ui_and_images(selected, names)
        return io.NodeOutput(name, lora_h, lora_l, prompt, available, base_image, pose_image, ui=ui_data)


# ── Node: AudioFixShape ───────────────────────────────────────────────────────

class AudioFixShape(io.ComfyNode):
    """Fixes audio waveform tensor shape by ensuring the batch dimension exists.

    Useful after nodes that strip the batch dimension, leaving a 1-D or 2-D
    tensor instead of the expected (batch, channels, samples) layout.
    """
    node_id = prefixed_node_id("AudioFixShape")
    display_name = "Audio Fix Shape"
    category = "🧊 frost-byte/Audio"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            description=cls.__doc__.split("\n")[0].strip(),
            category=cls.category,
            inputs=[
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(),
            ],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        if audio is None:
            return io.NodeOutput(None)

        waveform = audio["waveform"]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        return io.NodeOutput({"waveform": waveform, "sample_rate": audio["sample_rate"]})


# ── Custom type: CONCEPT_REGISTRY ────────────────────────────────────────────

CONCEPT_REGISTRY_TYPE = "CONCEPT_REGISTRY"


@io.comfytype(io_type=CONCEPT_REGISTRY_TYPE)
class ConceptRegistryIOType:
    """Carries a ConceptRegistry instance between Load → Define → Resolve nodes."""
    Type = object  # ConceptRegistry

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "registry", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: ConceptRegistryLoad ─────────────────────────────────────────────────

class ConceptRegistryLoad(io.ComfyNode):
    """Load the concept registry from disk.

    Connects to one or more ConceptDefine or ConceptResolve nodes.
    Use the "Reload Registry" button (added by JS) to force re-execution
    after the file has been edited externally.
    """
    node_id = prefixed_node_id("ConceptRegistryLoad")
    display_name = "Concept Registry Load"
    category = "🧊 frost-byte/lora"
    is_output_node = True  # allow standalone execution to preview available concepts

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.String.Input(
                    "registry_file",
                    display_name="Registry File (leave empty for default)",
                    default="",
                    tooltip=(
                        "Absolute path to a concept_registry.json file. "
                        "Leave empty to use the default user-data location."
                    ),
                    multiline=False,
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Concept registry to wire into ConceptDefine or ConceptResolve nodes.",
                ),
                io.String.Output(
                    "available_concepts",
                    display_name="Available Concepts",
                    tooltip="Human-readable list of all defined concepts.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, registry_file: str = "", **_):
        path = registry_file.strip() or default_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, _concept_reload_counter)

    @classmethod
    def execute(cls, registry_file: str = ""):
        path = registry_file.strip() or default_registry_path()
        registry = _load_concept_registry(path)
        available = registry.list_concepts()
        return io.NodeOutput(registry, available, ui={"available_concepts": available})


# ── Node: ConceptDefine ───────────────────────────────────────────────────────

class ConceptDefine(io.ComfyNode):
    """Define (or update) one concept entry for a specific model type.

    Chain multiple ConceptDefine nodes to build a complete registry before
    passing it to ConceptResolve.  If auto_save is enabled the updated
    registry is written back to its source file (with backup).
    """
    node_id = prefixed_node_id("ConceptDefine")
    display_name = "Concept Define"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or a previous ConceptDefine.",
                ),
                io.String.Input(
                    "concept_id",
                    display_name="Concept ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or style_cinematic.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Display Name",
                    default="",
                    tooltip="Human-readable label shown in ConceptList.",
                    multiline=False,
                ),
                io.String.Input(
                    "description",
                    display_name="Description",
                    default="",
                    multiline=True,
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Target model family for this LoRA entry.",
                ),
                io.Combo.Input(
                    "lora",
                    display_name="LoRA (or High LoRA for split models)",
                    options=_lora_get_list(),
                    default="None",
                    tooltip="LoRA file. For split models (Wan 2.2, BerniniR) this is the HIGH model LoRA.",
                ),
                io.Combo.Input(
                    "lora_low",
                    display_name="Low LoRA (split models only)",
                    options=_lora_get_list(),
                    default="None",
                    optional=True,
                    tooltip="Low-model LoRA for Wan 2.2 / BerniniR. Hidden by JS for single-model types.",
                ),
                io.Float.Input(
                    "weight",
                    display_name="Weight (or High Weight)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="LoRA strength. For split models this applies to the HIGH LoRA.",
                ),
                io.Float.Input(
                    "weight_low",
                    display_name="Low Weight (split models only)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    optional=True,
                    tooltip="LoRA strength for the LOW model LoRA. Hidden by JS for single-model types.",
                ),
                io.String.Input(
                    "trigger",
                    display_name="Trigger Words",
                    default="",
                    multiline=False,
                    tooltip="Trigger text appended/prepended to the prompt by ConceptResolve.",
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=False,
                    tooltip="If enabled, persist the updated registry to disk after each execution.",
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Updated registry with this concept entry added or merged.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, lora="None", lora_low="None", **kwargs):
        return True

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concept_id: str,
        name: str,
        description: str,
        model_type: str,
        lora: str = "None",
        lora_low: str = "None",
        weight: float = 1.0,
        weight_low: float = 1.0,
        trigger: str = "",
        auto_save: bool = False,
    ):
        model_entry = _build_model_entry(model_type, lora, lora_low, weight, weight_low, trigger)
        updated = registry.define(concept_id.strip(), name.strip(), description, model_type, model_entry)
        if auto_save:
            save_path = registry.file_path or default_registry_path()
            try:
                _save_concept_registry(updated, save_path, backup=True)
            except Exception as exc:
                logger.warning("ConceptDefine: auto_save failed: %s", exc)
        return io.NodeOutput(updated)


# ── Node: ConceptResolve ──────────────────────────────────────────────────────

class ConceptResolve(io.ComfyNode):
    """Resolve concept IDs against the registry and apply their LoRAs.

    For split-model types (Wan 2.2, BerniniR) the HIGH LoRA is applied to
    *model* and the LOW LoRA is applied to *model_low*.  For single-model
    types only *model* is used.

    Trigger words from each concept are collected and assembled into the
    output prompt according to the trigger_position setting.
    """
    node_id = prefixed_node_id("ConceptResolve")
    display_name = "Concept Resolve"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or ConceptDefine chain.",
                ),
                io.String.Input(
                    "concepts",
                    display_name="Concepts",
                    default="",
                    multiline=True,
                    tooltip="Concept IDs to resolve — one per line or comma-separated.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Select the model family so the correct LoRAs are chosen.",
                ),
                io.Model.Input(
                    "model",
                    display_name="Model",
                    tooltip="Primary model (or HIGH model for split types).",
                ),
                io.Model.Input(
                    "model_low",
                    display_name="Model (Low)",
                    optional=True,
                    tooltip="Low model for split types (Wan 2.2 / BerniniR). Leave unconnected for single-model types.",
                ),
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP encoder. Both high and low LoRAs are applied to CLIP for split models.",
                ),
                io.String.Input(
                    "base_prompt",
                    display_name="Base Prompt",
                    default="",
                    multiline=True,
                    tooltip="Starting prompt text. Concept triggers are merged in via trigger_position.",
                ),
                io.Combo.Input(
                    "trigger_position",
                    display_name="Trigger Position",
                    options=["prepend", "append"],
                    default="prepend",
                    tooltip="Where to place concept trigger words relative to base_prompt.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    "model",
                    display_name="Model",
                    tooltip="Primary model with HIGH (or single) LoRAs applied.",
                ),
                io.Model.Output(
                    "model_low",
                    display_name="Model (Low)",
                    tooltip="Low model with LOW LoRAs applied (passthrough for single-model types).",
                ),
                io.Clip.Output(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP with all concept LoRAs applied.",
                ),
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Base prompt with concept trigger words merged in.",
                ),
                io.String.Output(
                    "resolved_info",
                    display_name="Resolved Info",
                    tooltip="Summary of which concepts were resolved and which LoRAs were applied.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concepts: str,
        model_type: str,
        model,
        clip,
        base_prompt: str = "",
        trigger_position: str = "prepend",
        model_low=None,
    ):
        import comfy.sd as _comfy_sd
        import comfy.utils as _comfy_utils

        concept_ids = _parse_concept_ids(concepts)
        resolved = _resolve_concepts(registry, concept_ids, model_type)
        is_split = _CONCEPT_MODEL_PROFILES.get(model_type, {}).get("split", False)

        cur_model = model
        cur_model_low = model_low
        cur_clip = clip
        triggers: list[str] = []

        for r in resolved:
            if r.error:
                logger.warning("ConceptResolve [%s]: %s", r.concept_id, r.error)
                continue

            if r.trigger:
                triggers.append(r.trigger)

            # Apply high (or single) LoRA to primary model + CLIP
            if r.lora_high:
                path = folder_paths.get_full_path("loras", r.lora_high)
                if path:
                    weights = _comfy_utils.load_torch_file(path, safe_load=True)
                    cur_model, cur_clip = _comfy_sd.load_lora_for_models(
                        cur_model, cur_clip, weights, r.weight_high, r.weight_high
                    )
                else:
                    logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_high)
                    r.error = f"file missing: {r.lora_high}"

            # Apply low LoRA for split model types
            if is_split and r.lora_low:
                if cur_model_low is not None:
                    path = folder_paths.get_full_path("loras", r.lora_low)
                    if path:
                        weights = _comfy_utils.load_torch_file(path, safe_load=True)
                        cur_model_low, cur_clip = _comfy_sd.load_lora_for_models(
                            cur_model_low, cur_clip, weights, r.weight_low, r.weight_low
                        )
                    else:
                        logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_low)
                        r.error = f"file missing: {r.lora_low}"
                else:
                    logger.warning(
                        "ConceptResolve [%s]: model_low not connected for split type %s — "
                        "low LoRA '%s' skipped",
                        r.concept_id, model_type, r.lora_low,
                    )

        prompt = _assemble_concept_prompt(triggers, base_prompt, trigger_position)
        resolved_info = _format_resolved_info(model_type, resolved, prompt)

        return io.NodeOutput(cur_model, cur_model_low, cur_clip, prompt, resolved_info)


# ── Node: ConceptList ─────────────────────────────────────────────────────────

class ConceptList(io.ComfyNode):
    """Display a summary of all concepts in the registry.

    Optionally filter by model_type to show only concepts defined for that
    target.  Useful for quickly reviewing which concepts are available before
    wiring up ConceptResolve.
    """
    node_id = prefixed_node_id("ConceptList")
    display_name = "Concept List"
    category = "🧊 frost-byte/lora"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry to inspect.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Filter by Model Type",
                    options=["all"] + _CONCEPT_MODEL_TYPE_IDS,
                    default="all",
                    tooltip="Show only concepts that have an entry for this model type, or 'all'.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "concept_list",
                    display_name="Concept List",
                    tooltip="Formatted list of matching concepts.",
                ),
                io.Int.Output(
                    "concept_count",
                    display_name="Count",
                    tooltip="Number of matching concepts.",
                ),
            ],
        )

    @classmethod
    def execute(cls, registry: ConceptRegistry, model_type: str = "all"):
        filter_type = None if model_type == "all" else model_type
        listing = registry.list_concepts(filter_type)
        count = len([
            cid for cid, c in registry.concepts.items()
            if filter_type is None or filter_type in c.get("models", {})
        ])
        return io.NodeOutput(listing, count, ui={"concept_list": listing, "concept_count": count})


# ── Subject Profile helpers ───────────────────────────────────────────────────

def _subject_get_ids() -> list[str]:
    """Read subject_profiles.json and return subject IDs for combo widgets."""
    try:
        reg = _load_subject_registry(default_subject_profiles_path())
        ids = reg.subject_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


def _load_subject_images(filenames: list[str]) -> "torch.Tensor | None":
    """Load character sheet images from the ComfyUI input directory.

    Returns a [N, H, W, 3] float32 tensor (batch), or None if no images load.
    Images that fail to load are silently skipped.
    Images with different sizes are resized to match the first loaded image.
    """
    tensors: list = []
    target_h = target_w = None
    input_dir = get_input_directory()
    for fname in filenames:
        if not fname:
            continue
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            logger.debug("Subject image not found: %s", path)
            continue
        try:
            img, _ = load_image_comfyui(path, include_mask=False)  # [1, H, W, 3]
            h, w = img.shape[1], img.shape[2]
            if target_h is None:
                target_h, target_w = h, w
            if h != target_h or w != target_w:
                img = normalize_image_tensor(img, target_h, target_w)
            tensors.append(img)
        except Exception as exc:
            logger.warning("Failed to load subject image %s: %s", fname, exc)
    if not tensors:
        return None
    import torch as _torch
    return _torch.cat(tensors, dim=0)  # [N, H, W, 3]


def _load_subject_audio(filename: str) -> "dict | None":
    """Load an audio reference file from the ComfyUI input directory."""
    if not filename:
        return None
    path = os.path.join(get_input_directory(), filename)
    if not os.path.exists(path):
        logger.debug("Subject audio not found: %s", path)
        return None
    try:
        import torchaudio as _torchaudio
        waveform, sample_rate = _torchaudio.load(path)
        # ComfyUI AUDIO dict: waveform is [batch, channels, samples]
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as exc:
        logger.warning("Failed to load subject audio %s: %s", filename, exc)
        return None


# ── Custom type: SUBJECT_PROFILE ──────────────────────────────────────────────

SUBJECT_PROFILE_TYPE = "SUBJECT_PROFILE"


@io.comfytype(io_type=SUBJECT_PROFILE_TYPE)
class SubjectProfileIOType:
    """Carries a subject profile dict between Load/Define → SceneCompose nodes."""
    Type = object  # dict with name, appearance, voice, character_sheet_images, concept_id

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "subject_profile", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: SubjectProfileLoad ──────────────────────────────────────────────────

class SubjectProfileLoad(io.ComfyNode):
    """Load a subject profile from disk and expose its fields as outputs.

    Character sheet images are loaded from the ComfyUI input directory and
    stacked into an IMAGE batch.  Audio reference is loaded if defined.
    Use the Reload button in the REST endpoint to force re-execution after
    editing subject_profiles.json externally.
    """
    node_id = prefixed_node_id("SubjectProfileLoad")
    display_name = "Subject Profile Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        subject_ids = _subject_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "subject_id",
                    options=subject_ids,
                    display_name="Subject ID",
                    tooltip="Subject to load.  Refresh the page after adding new subjects via SubjectProfileDefine.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Full subject profile dict for wiring into SceneCompose.",
                ),
                io.String.Output("name", display_name="Name"),
                io.String.Output("appearance_summary", display_name="Appearance Summary"),
                io.Image.Output(
                    "character_sheet_images",
                    display_name="Character Sheet Images",
                    tooltip="Batch of all character sheet images (N×H×W×3). None if no images defined.",
                ),
                io.Audio.Output(
                    "audio_reference",
                    display_name="Audio Reference",
                    tooltip="Voice reference audio, or None if not defined.",
                ),
                io.String.Output("concept_id", display_name="Concept ID"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, subject_id: str = "", **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, subject_id, mtime, _subject_reload_counter)

    @classmethod
    def execute(cls, subject_id: str = "") -> io.NodeOutput:
        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)

        subject = registry.get_subject(subject_id) if subject_id and subject_id != "(none)" else None
        if subject is None:
            logger.warning("SubjectProfileLoad: subject_id %r not found in %s", subject_id, path)
            return io.NodeOutput(None, "", "", None, None, "")

        name = subject.get("name", subject_id)
        appearance = subject.get("appearance", {})
        appearance_summary = appearance.get("summary", "")
        voice = subject.get("voice", {})
        audio_file = voice.get("audio_reference_file", "")
        concept_id = subject.get("concept_id", "")
        sheet_files = subject.get("character_sheet_images", [])

        images = _load_subject_images(sheet_files)
        audio = _load_subject_audio(audio_file)

        send_status_update(
            cls.node_id,
            f"Loaded: {name} | {len(sheet_files)} sheet images | audio: {'yes' if audio else 'no'}",
        )
        return io.NodeOutput(subject, name, appearance_summary, images, audio, concept_id)


# ── Node: SubjectProfileDefine ────────────────────────────────────────────────

class SubjectProfileDefine(io.ComfyNode):
    """Create or update a subject profile entry.

    When auto_save is enabled the updated profiles file is written back to
    subject_profiles.json immediately.  Character sheet images are managed
    separately (edit the JSON directly to update the file list).
    """
    node_id = prefixed_node_id("SubjectProfileDefine")
    display_name = "Subject Profile Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "subject_id",
                    display_name="Subject ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or narrator.",
                    multiline=False,
                ),
                io.String.Input("name", display_name="Name", default="", multiline=False),
                io.String.Input(
                    "appearance_summary",
                    display_name="Appearance Summary",
                    default="",
                    tooltip="One-sentence description used in compact prompts and as the baseline for detailed descriptions.",
                    multiline=True,
                ),
                io.String.Input("face", display_name="Face", default="", multiline=True),
                io.String.Input("hair", display_name="Hair", default="", multiline=False),
                io.String.Input("body", display_name="Body", default="", multiline=False),
                io.String.Input(
                    "default_outfit",
                    display_name="Default Outfit",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "voice_description",
                    display_name="Voice Description",
                    default="",
                    multiline=True,
                    tooltip="Textual description of vocal quality for use in prompts referencing audio.",
                ),
                io.String.Input(
                    "audio_reference_file",
                    display_name="Audio Reference File",
                    default="",
                    multiline=False,
                    tooltip="Filename of the voice reference clip in the ComfyUI input directory.",
                ),
                io.Combo.Input(
                    "language",
                    options=_SUBJECT_LANGUAGES,
                    display_name="Language",
                    tooltip="BCP-47 language tag for dialogue tags in H3 prompts.",
                ),
                io.String.Input(
                    "concept_id",
                    display_name="Concept ID",
                    default="",
                    multiline=False,
                    tooltip="Links to the concept registry entry for LoRA resolution.",
                ),
                io.Bool.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=True,
                    tooltip="Write subject_profiles.json immediately after defining this subject.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Defined subject profile dict.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        subject_id: str,
        name: str = "",
        appearance_summary: str = "",
        face: str = "",
        hair: str = "",
        body: str = "",
        default_outfit: str = "",
        voice_description: str = "",
        audio_reference_file: str = "",
        language: str = "en-us",
        concept_id: str = "",
        auto_save: bool = True,
    ) -> io.NodeOutput:
        if not subject_id.strip():
            raise ValueError("SubjectProfileDefine: subject_id cannot be empty")

        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)
        registry = registry.define(
            subject_id=subject_id.strip(),
            name=name,
            appearance_summary=appearance_summary,
            face=face,
            hair=hair,
            body=body,
            default_outfit=default_outfit,
            voice_description=voice_description,
            audio_reference_file=audio_reference_file,
            language=language,
            concept_id=concept_id,
        )

        if auto_save:
            _save_subject_registry(registry, path, backup=True)
            logger.info("SubjectProfileDefine: saved %r to %s", subject_id, path)
            send_status_update(cls.node_id, f"Saved subject: {subject_id}")

        return io.NodeOutput(registry.get_subject(subject_id.strip()))


# ── Node: SubjectProfileList ──────────────────────────────────────────────────

class SubjectProfileList(io.ComfyNode):
    """Display all defined subject profiles.

    Useful for quickly reviewing what subjects are available without opening
    the JSON file.
    """
    node_id = prefixed_node_id("SubjectProfileList")
    display_name = "Subject Profile List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "subject_list",
                    display_name="Subject List",
                    tooltip="Formatted list of all defined subjects.",
                ),
                io.Int.Output("subject_count", display_name="Subject Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, _subject_reload_counter)

    @classmethod
    def execute(cls) -> io.NodeOutput:
        registry = _load_subject_registry(default_subject_profiles_path())
        listing = registry.list_subjects()
        count = len(registry.subjects)
        return io.NodeOutput(listing, count, ui={"subject_list": listing, "subject_count": count})


# ── Subject REST API endpoints ────────────────────────────────────────────────

@routes.post("/fbtools/subjects/reload")
async def _subjects_reload(request):
    """Increment reload counter so SubjectProfileLoad/List nodes re-execute."""
    global _subject_reload_counter
    _subject_reload_counter += 1
    logger.info("Subject profiles reload requested (counter=%d)", _subject_reload_counter)
    return web.json_response({"success": True, "counter": _subject_reload_counter})


@routes.get("/fbtools/subjects/profiles")
async def _subjects_get_profiles(request):
    """Return the current subject profiles as JSON for the frontend."""
    try:
        registry = _load_subject_registry(default_subject_profiles_path())
        return web.json_response(registry.to_dict())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Custom type: SCENE_TEMPLATE ──────────────────────────────────────────────

SCENE_TEMPLATE_TYPE = "SCENE_TEMPLATE"


@io.comfytype(io_type=SCENE_TEMPLATE_TYPE)
class SceneTemplateIOType:
    """Carries a SceneTemplate instance between Load → SceneCompose nodes."""
    Type = object  # SceneTemplate

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "template", **kwargs):
            super().__init__(name, **kwargs)


# ── Scene Template helpers ─────────────────────────────────────────────────────

def _template_get_ids() -> list[str]:
    """Return available template IDs for combo population at schema time."""
    try:
        ids = _scene_template_ids(default_scene_templates_dir())
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: SceneTemplateLoad ───────────────────────────────────────────────────

class SceneTemplateLoad(io.ComfyNode):
    """Load a scene template from the scene_templates directory.

    The combo is populated at extension load time from the user's
    scene_templates/ directory (seeded with bundled examples on first use).
    Refresh the page after adding new templates to see them in the dropdown.
    """
    node_id = prefixed_node_id("SceneTemplateLoad")
    display_name = "Scene Template Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        template_id_options = _template_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "template_id",
                    options=template_id_options,
                    display_name="Template ID",
                    tooltip="Scene template to load.  Refresh the page after adding new templates.",
                ),
            ],
            outputs=[
                SceneTemplateIOType.Output(
                    "template",
                    display_name="Scene Template",
                    tooltip="Template object for wiring into SceneCompose.",
                ),
                io.String.Output(
                    "slot_info",
                    display_name="Slot Info",
                    tooltip="Formatted summary of slot requirements.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, template_id: str = "", **_):
        templates_dir = default_scene_templates_dir()
        path = os.path.join(templates_dir, f"{template_id}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, _scene_template_reload_counter)

    @classmethod
    def execute(cls, template_id: str = "") -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        if not template_id or template_id == "(none)":
            logger.warning("SceneTemplateLoad: no template_id selected")
            return io.NodeOutput(None, "")
        path = os.path.join(templates_dir, f"{template_id}.json")
        if not os.path.exists(path):
            logger.warning("SceneTemplateLoad: template not found: %s", path)
            return io.NodeOutput(None, f"Template not found: {template_id}")
        template = _load_scene_template(path)
        slot_info = template.format_slot_info()
        send_status_update(
            cls.node_id,
            f"Loaded: {template.name} | {template.slot_count} slot(s) | {len(template.shots)} shots",
        )
        return io.NodeOutput(
            template,
            slot_info,
            ui={"slot_info": slot_info},
        )


# ── Node: SceneTemplateList ───────────────────────────────────────────────────

class SceneTemplateList(io.ComfyNode):
    """List all available scene templates.

    Scans the scene_templates/ directory and returns a formatted summary.
    Useful for quickly reviewing available templates.
    """
    node_id = prefixed_node_id("SceneTemplateList")
    display_name = "Scene Template List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "template_list",
                    display_name="Template List",
                    tooltip="Formatted list of all available templates.",
                ),
                io.Int.Output("template_count", display_name="Template Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        templates_dir = default_scene_templates_dir()
        return (_templates_dir_fingerprint(templates_dir), _scene_template_reload_counter)

    @classmethod
    def execute(cls) -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        listing = _format_template_list(templates_dir)
        count = len(_scan_scene_templates(templates_dir))
        return io.NodeOutput(listing, count, ui={"template_list": listing, "template_count": count})


# ── Scene Template REST API endpoints ─────────────────────────────────────────

@routes.post("/fbtools/scene_templates/reload")
async def _scene_templates_reload(request):
    """Increment reload counter so SceneTemplate nodes re-execute."""
    global _scene_template_reload_counter
    _scene_template_reload_counter += 1
    logger.info("Scene templates reload requested (counter=%d)", _scene_template_reload_counter)
    return web.json_response({"success": True, "counter": _scene_template_reload_counter})


@routes.get("/fbtools/scene_templates/list")
async def _scene_templates_list(request):
    """Return the list of available template metadata as JSON."""
    try:
        templates_dir = default_scene_templates_dir()
        templates = _scan_scene_templates(templates_dir)
        return web.json_response({"templates": templates})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Concept REST API endpoints ─────────────────────────────────────────────────

@routes.post("/fbtools/concepts/reload")
async def _concepts_reload(request):
    """Increment reload counter so ConceptRegistryLoad nodes re-execute."""
    global _concept_reload_counter
    _concept_reload_counter += 1
    logger.info("Concept registry reload requested (counter=%d)", _concept_reload_counter)
    return web.json_response({"success": True, "counter": _concept_reload_counter})


@routes.get("/fbtools/concepts/registry")
async def _concepts_get_registry(request):
    """Return the current default registry as JSON for the frontend."""
    try:
        registry = _load_concept_registry(default_registry_path())
        return web.json_response(registry.to_dict())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# =============================================================================


class FBToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SubjectLayerDefine,
            SubjectCompositor,
            CaptionModelUnloader,
            DatasetCaptioner,
            DatasetCaptionEditor,
            DatasetCaptionViewer,
            DatasetExportSummary,
            FBTextEncodeQwenImageEditPlus,
            SAMPreprocessNHWC,
            QwenAspectRatio,
            SubdirLister,
            MultiLoraLoader,
            # NodeInputSelect,
            SceneCreate,
            SceneUpdate,
            SceneMaskDefinition,
            SceneSave,
            SceneInput,
            SceneOutput,
            SceneView,
            SceneSelect,
            SceneWanVideoLoraMultiSave,
            SceneLoraStackSave,
            StorySceneBatch,
            StoryScenePick,
            StoryVideoBatch,
            StoryCreate,
            StoryEdit,
            StoryView,
            StorySave,
            StoryLoad,
            StorySceneImageSave,
            OpaqueAlpha,
            MaskProcessor,
            TailSplit,
            TailEnhancePro,
            # Libber nodes
            LibberManager,
            LibberApply,
            # Scene Prompt Management nodes
            ScenePromptManager,
            PromptComposer,
            # LoRA scene nodes
            LoraStackBuilder,
            LoraStackApply,
            LoraEntryDefine,
            LoraStackCollect,
            WanVidLoraStack,
            # Wan preset nodes
            LoraPresetDefine,
            LoraPresetSelect,
            WanPresetDefine,
            WanPresetSelect,
            # Audio nodes
            AudioFixShape,
            # Concept Registry nodes
            ConceptRegistryLoad,
            ConceptDefine,
            ConceptResolve,
            ConceptList,
            # Subject Profile nodes
            SubjectProfileLoad,
            SubjectProfileDefine,
            SubjectProfileList,
            # Scene Template nodes
            SceneTemplateLoad,
            SceneTemplateList,
        ]