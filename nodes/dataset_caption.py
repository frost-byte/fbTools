"""Dataset captioning: image+caption batch tools (DatasetCaptioner, DatasetCaptionEditor,
DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader) and the
/fbtools/dataset_caption/* REST routes.

Moved out of extension.py (pure code motion; Plan 18 continues Plan 1/11/17's extension.py ->
nodes/ package split). Body order preserved as in extension.py: shared constants -> 7 owned
helpers -> the 5 node classes -> the 5 routes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aiohttp import web
from comfy_api.latest import io
from folder_paths import get_input_directory, get_output_directory

from ..captioner import unload_model
from .llm_assistant import _run_vision_inference
from .shared import routes, prefixed_node_id, send_status_update
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

DEFAULT_INSTRUCTION = (
    "Describe this image in precise factual detail. "
    "Include: subject appearance (face shape, eye colour, hair style and colour, "
    "skin tone, any distinguishing features), clothing and accessories worn, "
    "pose and body language, background and setting, and lighting conditions. "
    "Write in flowing prose. Do not use subjective quality descriptors."
)

CAPTIONER_OPTIONS = ["llm_client", "gemini_flash", "modal", "unsloth"]
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


# ── Node: DatasetCaptioner ────────────────────────────────────────────────────

class DatasetCaptioner(io.ComfyNode):
    """
    Captions images in a directory using the loaded LLM panel model or Gemini Flash.
    Writes one .txt file per image (alongside the image, or into output_directory).
    Load a vision-capable model in the fbTools Compose -> LLM panel before running.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("DatasetCaptioner"),
            display_name="Dataset Captioner",
            category="🧊 frost-byte/Dataset",
            description=(
                "Captions all images in a directory using the model loaded in the "
                "fbTools LLM panel, or Gemini Flash via API key. "
                "Writes one .txt caption file per image."
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
                    default="llm_client",
                    tooltip=(
                        "llm_client: uses the model loaded in the fbTools Compose -> LLM panel. "
                        "gemini_flash: uses the Gemini Flash API (set GEMINI_API_KEY env var — see docs)."
                    ),
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
        recursive: bool = False,
        override_existing: bool = False,
        clean_caption: bool = True,
    ) -> io.NodeOutput:
        input_dir = _resolve_dataset_input_directory(input_directory)
        output_dir = _resolve_dataset_output_directory(output_directory)
        api_key    = os.environ.get("GEMINI_API_KEY", "")

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
            f"Dataset Captioner: captioning {len(images)} image(s) via {captioner_type}",
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
                caption = _run_vision_inference(
                    str(img_path),
                    instruction,
                    captioner_type=captioner_type,
                    clean=clean_caption,
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


# ── DATASET CAPTION API ENDPOINTS ───────────────────────────────────────────────

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
    }
    """
    try:
        body           = await request.json()
        image_path     = Path(body["image_path"])
        txt_path       = Path(body["txt_path"])
        captioner_type = body.get("captioner_type", "llm_client")
        instruction    = body.get("instruction", "Describe this image in detail.")
        trigger_word   = body.get("trigger_word", "")
        clean          = bool(body.get("clean_caption", True))

        send_status_update(
            DATASET_CAPTION_STATUS_ID,
            f"Dataset Captioner: re-captioning via {captioner_type}",
            source="dataset_captioner",
        )
        caption = _run_vision_inference(
            str(image_path),
            instruction,
            captioner_type=captioner_type,
            clean=clean,
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
