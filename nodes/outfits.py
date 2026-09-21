"""Outfit registry REST routes (/fbtools/outfits/*) including SAM2 outfit extraction.

Moved out of extension.py (pure code motion)."""
from __future__ import annotations

from aiohttp import web
from ..utils.outfit_registry import load_outfit_registry as _load_outfit_registry, save_outfit_registry as _save_outfit_registry
from .llm_assistant import _route_llm
import asyncio
import folder_paths
import os
import uuid
from .shared import bump_reload, default_outfit_registry_path, routes
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@routes.post("/fbtools/outfits/reload")
async def _outfits_reload(request):
    """Increment reload counter so OutfitRegistryLoad nodes re-execute."""
    _outfit_reload_counter = bump_reload("outfit")
    logger.info("Outfit registry reload requested (counter=%d)", _outfit_reload_counter)
    return web.json_response({"success": True, "counter": _outfit_reload_counter})


@routes.get("/fbtools/outfits/registry")
async def _outfits_get_registry(request):
    """Return the outfit registry as JSON for the frontend."""
    try:
        registry = _load_outfit_registry(default_outfit_registry_path())
        return web.json_response(registry.to_dict())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/outfits/save")
async def _outfits_save(request):
    """Create or update one outfit entry.

    Body: { id, name, description, tags?: [], reference_images?: [{file, role}] }
    reference_images replaces the stored list when present; omit to preserve existing.
    """
    try:
        data = await request.json()
        outfit_id = data.get("id", "").strip()
        if not outfit_id:
            return web.json_response({"error": "id is required"}, status=400)
        path = default_outfit_registry_path()
        registry = _load_outfit_registry(path)
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        ref_images = data.get("reference_images")  # None → preserve existing
        updated = registry.define(
            outfit_id,
            data.get("name", ""),
            data.get("description", ""),
            tags,
            reference_images=ref_images,
        )
        _save_outfit_registry(updated, path, backup=True)
        return web.json_response({"success": True, "id": outfit_id})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/outfits/analyze_media")
async def _outfits_analyze_media(request):
    """Analyze an image or video for outfit description using the loaded LLM.

    For images: runs LLM directly on the image.
    For videos: extracts a frame at frame_time seconds (default 1.0) and runs LLM
    on that frame.  The extracted frame is saved permanently to the ComfyUI input
    directory as _outfit_ref_<uuid>.jpg so the caller can add it to reference_images.

    Body: { filename, query?, max_tokens?, frame_time? }
    Returns: { description, frame_file }  (frame_file is null for images)
    """
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
    try:
        body = await request.json()
        filename = (body.get("filename") or "").strip()
        if not filename:
            return web.json_response({"error": "filename is required"}, status=400)
        query = body.get("query") or (
            "Describe the outfit in this image for use in video generation prompts. "
            "Focus on garment types, colors, materials, textures, patterns, and accessories. "
            "Do not describe the person's face, hair, or pose."
        )
        max_tokens = int(body.get("max_tokens", 400))
        frame_time  = float(body.get("frame_time", 1.0))

        import folder_paths
        input_dir = folder_paths.get_input_directory()
        src_path = os.path.join(input_dir, filename)
        if not os.path.exists(src_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        ext = os.path.splitext(filename)[1].lower()
        frame_file: str | None = None
        pil_image = None

        def _prepare():
            nonlocal frame_file, pil_image
            from PIL import Image as _PILImage
            if ext in _VIDEO_EXTS:
                import cv2
                cap = cv2.VideoCapture(src_path)
                try:
                    fps         = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    target      = min(int(frame_time * fps), max(0, frame_count - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Could not read frame {target} from {filename}")
                    ref_name  = f"_outfit_ref_{uuid.uuid4().hex[:12]}.jpg"
                    ref_path  = os.path.join(input_dir, ref_name)
                    cv2.imwrite(ref_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    frame_file = ref_name
                    pil_image  = _PILImage.open(ref_path).convert("RGB")
                finally:
                    cap.release()
            else:
                pil_image = _PILImage.open(src_path).convert("RGB")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _prepare)

        result = await _route_llm(query, images=[pil_image], max_tokens=max_tokens, temperature=0.5)
        if not result.get("success"):
            return web.json_response(
                {"error": result.get("message", "LLM generate failed")}, status=503
            )
        return web.json_response({
            "description": result.get("text", "").strip(),
            "frame_file": frame_file,
        })
    except Exception as exc:
        logger.error("outfit analyze_media error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


_sam2_segmenter = None  # lazy singleton; reset if model_path changes


@routes.get("/fbtools/outfits/sam2_status")
async def _outfits_sam2_status(request):
    """Return SAM2 availability: packages present and model file found."""
    try:
        from ..utils.sam2_segmenter import check_dependencies, find_sam2_model
        def _gfp(name):
            try: return folder_paths.get_folder_paths(name)
            except KeyError: return []
        sams_dirs = (_gfp("sams") + _gfp("sam2")) or [
            os.path.join(folder_paths.models_dir, "sams"),
            os.path.join(folder_paths.models_dir, "sam2"),
        ]
        deps = check_dependencies()
        model_file = find_sam2_model(sams_dirs) if deps["available"] else None
        install_hint = (
            "pip install git+https://github.com/facebookresearch/sam2.git"
            if not deps["available"] else None
        )
        model_hint = (
            "Download sam2_hiera_tiny.safetensors from "
            "https://huggingface.co/Kijai/sam2-safetensors "
            f"and place in {sams_dirs[0]}"
            if not model_file else None
        )
        return web.json_response({
            "available":        deps["available"] and model_file is not None,
            "packages_ok":      deps["available"],
            "missing_packages": deps["missing"],
            "model_file":       model_file,
            "model_dirs":       sams_dirs,
            "install_hint":     install_hint,
            "model_hint":       model_hint,
        })
    except Exception as exc:
        logger.error("sam2_status error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/outfits/extract_outfit")
async def _outfits_extract_outfit(request):
    """Run SAM2 point-prompt segmentation on an input image.

    Body: {filename, point_x, point_y, point_label}
    Returns: {result_file} — basename of the saved RGBA PNG in the input dir.
    """
    global _sam2_segmenter
    try:
        body = await request.json()
        filename    = (body.get("filename") or "").strip()
        point_x     = float(body.get("point_x", 0.5))
        point_y     = float(body.get("point_y", 0.5))
        point_label = int(body.get("point_label", 1))

        if not filename:
            return web.json_response({"error": "filename required"}, status=400)

        # The picker offers files from input/ and output/ (including subfolders),
        # so resolve `filename` (a relative path) inside the requested folder.
        folder = (body.get("folder") or "input").strip()
        if folder == "output":
            base_dir = folder_paths.get_output_directory()
        elif folder == "input":
            base_dir = folder_paths.get_input_directory()
        else:
            return web.json_response({"error": f"Unknown folder: {folder}"}, status=400)
        base_real  = os.path.realpath(base_dir)
        image_path = os.path.realpath(os.path.join(base_real, filename))
        if os.path.commonpath([base_real, image_path]) != base_real:
            return web.json_response({"error": "Invalid path"}, status=400)
        if not os.path.isfile(image_path):
            return web.json_response({"error": f"File not found: {folder}/{filename}"}, status=404)
        # Result always lands in the input dir root so the UI can load it via /view?type=input.
        out_path_target = os.path.join(
            folder_paths.get_input_directory(), f"_outfit_seg_{uuid.uuid4().hex[:12]}.png"
        )

        from ..utils.sam2_segmenter import (
            SAM2Segmenter,
            check_dependencies,
            find_sam2_model,
        )

        deps = check_dependencies()
        if not deps["available"]:
            return web.json_response(
                {"error": f"SAM2 packages missing: {', '.join(deps['missing'])}"},
                status=503,
            )

        def _gfp(name):
            try: return folder_paths.get_folder_paths(name)
            except KeyError: return []
        sams_dirs = (_gfp("sams") + _gfp("sam2")) or [
            os.path.join(folder_paths.models_dir, "sams"),
            os.path.join(folder_paths.models_dir, "sam2"),
        ]
        model_file = find_sam2_model(sams_dirs)
        if not model_file:
            return web.json_response(
                {"error": "No SAM2 safetensors model found in models/sams/ or models/sam2/"},
                status=503,
            )

        if _sam2_segmenter is None or _sam2_segmenter._model_path != model_file:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _sam2_segmenter = SAM2Segmenter(model_file, device=device)

        import asyncio
        out_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _sam2_segmenter.segment(image_path, point_x, point_y, point_label, output_path=out_path_target),
        )
        return web.json_response({"result_file": os.path.basename(out_path)})

    except Exception as exc:
        logger.error("outfit extract_outfit error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/outfits/delete")
async def _outfits_delete(request):
    """Delete an outfit entry by ?id=<outfit_id>."""
    outfit_id = request.rel_url.query.get("id", "")
    if not outfit_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_outfit_registry_path()
        registry = _load_outfit_registry(path)
        updated = registry.remove(outfit_id)
        _save_outfit_registry(updated, path, backup=True)
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
