"""Background and camera/sound preset REST routes (/fbtools/backgrounds/*, /fbtools/presets/*).

Moved out of extension.py (pure code motion)."""
from __future__ import annotations

from ..utils.composition_resources import list_backgrounds as _list_backgrounds, get_background as _get_background, save_background as _save_background, delete_background as _delete_background, list_camera_presets as _list_camera_presets, save_camera_preset as _save_camera_preset, delete_camera_preset as _delete_camera_preset, list_sound_presets as _list_sound_presets, save_sound_preset as _save_sound_preset, delete_sound_preset as _delete_sound_preset
from aiohttp import web
from .llm_assistant import _route_llm
import asyncio
import json
import os
import uuid
from .shared import routes, user_data_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@routes.get("/fbtools/backgrounds/list")
async def _backgrounds_list(request):
    try:
        return web.json_response({"backgrounds": _list_backgrounds(user_data_dir())})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/backgrounds/get")
async def _backgrounds_get(request):
    bg_id = request.rel_url.query.get("id", "")
    if not bg_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        bg = _get_background(user_data_dir(), bg_id)
        if bg is None:
            return web.json_response({"error": f"Background '{bg_id}' not found"}, status=404)
        return web.json_response(bg)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/backgrounds/save")
async def _backgrounds_save(request):
    try:
        bg = await request.json()
        saved = _save_background(user_data_dir(), bg)
        return web.json_response({"success": True, "id": saved["id"]})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/backgrounds/delete")
async def _backgrounds_delete(request):
    bg_id = request.rel_url.query.get("id", "")
    if not bg_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        _delete_background(user_data_dir(), bg_id)
        return web.json_response({"success": True})
    except KeyError:
        return web.json_response({"error": f"Background '{bg_id}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/backgrounds/analyze_media")
async def _backgrounds_analyze_media(request):
    """Analyze an image or video for background scene description.

    Asks the loaded LLM to return structured JSON with three fields:
      description — environment / setting, no people
      lighting    — lighting conditions and quality
      soundscape  — expected ambient audio

    For videos, extracts a frame at frame_time seconds and saves it to the
    ComfyUI input directory as _bg_ref_<uuid>.jpg.

    Body: { filename, folder?, frame_time?, max_tokens? }
    Returns: { description, lighting, soundscape, frame_file }
    """
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
    _DEFAULT_QUERY = (
        "Analyze this scene for use in video production. "
        "Return ONLY a JSON object with exactly these three keys — no other text:\n"
        '{\n'
        '  "description": "2-3 sentences describing the environment, setting, and atmosphere. '
        'Do not mention any people or characters.",\n'
        '  "lighting": "1 sentence describing the lighting quality, direction, and mood.",\n'
        '  "soundscape": "1-2 sentences describing what you would expect to hear in this scene."\n'
        '}'
    )
    try:
        body = await request.json()
        filename = (body.get("filename") or "").strip()
        if not filename:
            return web.json_response({"error": "filename is required"}, status=400)
        folder     = (body.get("folder") or "input").strip()
        max_tokens = int(body.get("max_tokens", 500))
        frame_time = float(body.get("frame_time", 1.0))

        import folder_paths
        if folder == "output":
            src_dir = folder_paths.get_output_directory()
        else:
            src_dir = folder_paths.get_input_directory()
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        ext = os.path.splitext(filename)[1].lower()
        frame_file: str | None = None
        pil_image = None

        def _prepare():
            nonlocal frame_file, pil_image
            from PIL import Image as _PILImage
            input_dir = folder_paths.get_input_directory()
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
                    ref_name  = f"_bg_ref_{uuid.uuid4().hex[:12]}.jpg"
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

        result = await _route_llm(_DEFAULT_QUERY, images=[pil_image], max_tokens=max_tokens, temperature=0.4)
        if not result.get("success"):
            return web.json_response(
                {"error": result.get("message", "LLM generate failed")}, status=503
            )

        raw = result.get("text", "").strip()

        # Parse structured JSON from LLM response; strip markdown fences if present
        import re as _re
        json_text = raw
        fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if fence_match:
            json_text = fence_match.group(1).strip()
        try:
            parsed = json.loads(json_text)
            description = str(parsed.get("description", "")).strip()
            lighting    = str(parsed.get("lighting",    "")).strip()
            soundscape  = str(parsed.get("soundscape",  "")).strip()
        except (json.JSONDecodeError, AttributeError):
            # Fall back: use raw text as description
            description = raw
            lighting    = ""
            soundscape  = ""

        return web.json_response({
            "description": description,
            "lighting":    lighting,
            "soundscape":  soundscape,
            "frame_file":  frame_file,
        })
    except Exception as exc:
        logger.error("background analyze_media error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/presets/cameras")
async def _presets_cameras_list(request):
    try:
        return web.json_response({"camera_presets": _list_camera_presets(user_data_dir())})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/presets/cameras/save")
async def _presets_cameras_save(request):
    try:
        preset = await request.json()
        saved = _save_camera_preset(user_data_dir(), preset)
        return web.json_response({"success": True, "id": saved["id"]})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/presets/cameras/delete")
async def _presets_cameras_delete(request):
    pid = request.rel_url.query.get("id", "")
    if not pid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        _delete_camera_preset(user_data_dir(), pid)
        return web.json_response({"success": True})
    except KeyError:
        return web.json_response({"error": f"Camera preset '{pid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/presets/sounds")
async def _presets_sounds_list(request):
    try:
        return web.json_response({"sound_presets": _list_sound_presets(user_data_dir())})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/presets/sounds/save")
async def _presets_sounds_save(request):
    try:
        preset = await request.json()
        saved = _save_sound_preset(user_data_dir(), preset)
        return web.json_response({"success": True, "id": saved["id"]})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/presets/sounds/delete")
async def _presets_sounds_delete(request):
    pid = request.rel_url.query.get("id", "")
    if not pid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        _delete_sound_preset(user_data_dir(), pid)
        return web.json_response({"success": True})
    except KeyError:
        return web.json_response({"error": f"Sound preset '{pid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
