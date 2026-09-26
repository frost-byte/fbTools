"""Background and camera/sound preset REST routes (/fbtools/backgrounds/*, /fbtools/presets/*).

Moved out of extension.py (pure code motion)."""
from __future__ import annotations

from ..utils.composition_resources import list_backgrounds as _list_backgrounds, get_background as _get_background, save_background as _save_background, delete_background as _delete_background, list_camera_presets as _list_camera_presets, save_camera_preset as _save_camera_preset, delete_camera_preset as _delete_camera_preset, list_sound_presets as _list_sound_presets, save_sound_preset as _save_sound_preset, delete_sound_preset as _delete_sound_preset
from ..utils.source_profile_analysis import extract_frame_at_time
from ..utils.h3_template_runner import load_template, find_node_by_title, patch_prompt
from ..utils.h3_job_runner import submit_and_wait, H3JobError
from aiohttp import web
from .llm_assistant import _route_llm
from .composition_shared import _read_composition_settings
from .shared import routes, user_data_dir, PACKAGE_ROOT
import asyncio
import json
import os
import random
import shutil
import uuid
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_H3_BG_PLATE_TEMPLATE_PATH = os.path.join(PACKAGE_ROOT, "templates", "h3_background_plate.api.json")
_H3_BG_PLATE_DEFAULT_PROMPT = (
    "Edit <Picture 1>: remove people, remove animals, remove faces and body parts, and remove "
    "any watermarks from the image while recreating a photo-realistic version of the background."
)
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}


def _h3_bg_plate_overrides(settings: dict) -> dict[str, dict]:
    """Build the patch_prompt() `overrides` dict from the h3_bg_plate_* settings.

    Every value defaults to "" / 0 (unset) in _COMPOSITION_SETTINGS_DEFAULTS, meaning "use whatever
    the template itself specifies" — only non-empty/non-zero settings produce an override, and
    patch_prompt() itself silently skips any title the template doesn't happen to expose.
    """
    overrides: dict[str, dict] = {}
    if settings.get("h3_bg_plate_model"):
        overrides["IN:model"] = {"unet_name": settings["h3_bg_plate_model"]}
    if settings.get("h3_bg_plate_clip"):
        overrides["IN:clip"] = {"clip_name": settings["h3_bg_plate_clip"]}
    if settings.get("h3_bg_plate_lora"):
        # IN:lora targets fbTools' own LoraStackBuilder (row 0), not core LoraLoaderModelOnly —
        # its widgets are row-indexed (lora_0/strength_model_0/strength_clip_0/enabled_0), and
        # unlike core's loader it has a real "None"/enabled toggle rather than needing a
        # strength=0 workaround. strength_model and strength_clip both take the one configured
        # strength value — this repo doesn't expose them as separate knobs in Settings.
        strength = float(settings.get("h3_bg_plate_lora_strength", 0.38))
        overrides["IN:lora"] = {
            "lora_0": settings["h3_bg_plate_lora"],
            "strength_model_0": strength,
            "strength_clip_0": strength,
            "enabled_0": True,
        }
    if settings.get("h3_bg_plate_sampler"):
        overrides["IN:sampler"] = {"sampler_name": settings["h3_bg_plate_sampler"]}
    scheduler_fields = {}
    if settings.get("h3_bg_plate_scheduler"):
        scheduler_fields["scheduler"] = settings["h3_bg_plate_scheduler"]
    if settings.get("h3_bg_plate_steps"):
        scheduler_fields["steps"] = int(settings["h3_bg_plate_steps"])
    if scheduler_fields:
        overrides["IN:scheduler"] = scheduler_fields
    return overrides


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


@routes.post("/fbtools/backgrounds/remove_people")
async def _backgrounds_remove_people(request):
    """Run the H3 background-plate workflow to remove people from an image or video frame.

    Runs MiniMax H3 Reference-to-Video + Fizgig H3 Still (see templates/h3_background_plate.api.json)
    server-to-server via this same ComfyUI instance's own /prompt + /history endpoints — no browser
    tab is involved, so the user's own open canvas is untouched.

    Body: { filename, folder="input", frame_time=1.0, prompt? }
    Returns: { file, folder: "output" } — a path (with subfolder, if any) under output/, ready to
    pass straight to a Background's reference_images as {file, folder: "output"}.
    """
    temp_path: str | None = None
    try:
        body = await request.json()
        filename = (body.get("filename") or "").strip()
        if not filename:
            return web.json_response({"error": "filename is required"}, status=400)
        folder     = (body.get("folder") or "input").strip()
        frame_time = float(body.get("frame_time", 1.0))
        prompt_text = (body.get("prompt") or "").strip() or _H3_BG_PLATE_DEFAULT_PROMPT

        if not os.path.exists(_H3_BG_PLATE_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_H3_BG_PLATE_TEMPLATE_PATH}. Export the cleaned-up "
                          f"H3 background-plate workflow from ComfyUI (Workflow -> Export (API)) and "
                          f"save it there first."},
                status=500,
            )

        import folder_paths
        src_dir  = folder_paths.get_output_directory() if folder == "output" else folder_paths.get_input_directory()
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        input_dir = folder_paths.get_input_directory()
        ext = os.path.splitext(filename)[1].lower()
        is_video = ext in _VIDEO_EXTS

        def _prepare_input_image() -> tuple[str, str | None]:
            """Return (image filename relative to input/, temp file to clean up afterward or None).

            The temp file is only created for a video frame extraction or a copy pulled in from
            output/ — a plain image already sitting in input/ is used directly and must never be
            deleted (it's the user's actual source file, not a temp copy).
            """
            tmp_dir = os.path.join(input_dir, "fbtools_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            if is_video:
                out_name = f"h3_bg_src_{uuid.uuid4().hex[:12]}.jpg"
                out_path = os.path.join(tmp_dir, out_name)
                extract_frame_at_time(src_path, out_path, frame_time)
                return os.path.join("fbtools_tmp", out_name), out_path
            if folder == "output":
                # LoadImage can only read from input/ — copy a temp copy in.
                out_name = f"h3_bg_src_{uuid.uuid4().hex[:12]}{ext}"
                out_path = os.path.join(tmp_dir, out_name)
                shutil.copy2(src_path, out_path)
                return os.path.join("fbtools_tmp", out_name), out_path
            # Already a plain image sitting in input/ — use it directly, nothing to clean up.
            return filename, None

        loop = asyncio.get_event_loop()
        load_image_name, temp_path = await loop.run_in_executor(None, _prepare_input_image)

        template = load_template(_H3_BG_PLATE_TEMPLATE_PATH)
        save_node_id = find_node_by_title(template, "OUT:save")
        seed = random.randint(0, 2**32 - 1)
        filename_prefix = f"fbtools/h3_background_plates/{uuid.uuid4().hex[:12]}"
        overrides = _h3_bg_plate_overrides(_read_composition_settings())
        prompt = patch_prompt(
            template,
            image=load_image_name,
            prompt_text=prompt_text,
            seed=seed,
            filename_prefix=filename_prefix,
            overrides=overrides,
        )

        result = await submit_and_wait(request, prompt, save_node_id)
        subfolder = result.get("subfolder", "")
        out_filename = result.get("filename", "")
        out_file = f"{subfolder}/{out_filename}" if subfolder else out_filename
        return web.json_response({"file": out_file, "folder": "output"})
    except H3JobError as exc:
        logger.error("background remove_people job error: %s", exc)
        return web.json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.error("background remove_people error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        # Clean up the temp source-frame copy regardless of success/failure/early-return — never
        # leaked, even if the request body was invalid or the H3 job itself failed.
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as exc:
                logger.warning("background remove_people: could not clean up temp file %s: %s", temp_path, exc)


@routes.get("/fbtools/backgrounds/h3_settings_options")
async def _backgrounds_h3_settings_options(request):
    """Enumeration lists + capability flags for the Settings > H3 Background Plate section.

    Each has_*_override flag tells the UI whether templates/h3_background_plate.api.json currently
    exposes that optional override title at all — the corresponding Settings control should be
    disabled (not just left inert) when its flag is false, so a user can never configure an override
    that would silently do nothing.
    """
    try:
        if not os.path.exists(_H3_BG_PLATE_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_H3_BG_PLATE_TEMPLATE_PATH}"}, status=404
            )
        template = load_template(_H3_BG_PLATE_TEMPLATE_PATH)

        import folder_paths
        import comfy.samplers

        def _has(title: str) -> bool:
            return find_node_by_title(template, title, required=False) is not None

        def _field(title: str, field: str, default=None):
            """Current baked-in value of `field` on the node titled `title`, or `default` if the
            template doesn't have that title (or the field) at all. Lets the UI show what the
            template will actually do when nothing is overridden, instead of an arbitrary UI
            default disconnected from the template."""
            node_id = find_node_by_title(template, title, required=False)
            if node_id is None:
                return default
            return template[node_id].get("inputs", {}).get(field, default)

        return web.json_response({
            "models":     sorted(folder_paths.get_filename_list("diffusion_models")),
            "clips":      sorted(folder_paths.get_filename_list("text_encoders")),
            "samplers":   list(comfy.samplers.SAMPLER_NAMES),
            "schedulers": list(comfy.samplers.SCHEDULER_NAMES),
            "has_model_override":     _has("IN:model"),
            "has_clip_override":      _has("IN:clip"),
            "has_lora_override":      _has("IN:lora"),
            "has_sampler_override":   _has("IN:sampler"),
            "has_scheduler_override": _has("IN:scheduler"),
            "template_defaults": {
                "model":         _field("IN:model", "unet_name"),
                "clip":          _field("IN:clip", "clip_name"),
                "sampler":       _field("IN:sampler", "sampler_name"),
                "scheduler":     _field("IN:scheduler", "scheduler"),
                "steps":         _field("IN:scheduler", "steps"),
                "lora":          _field("IN:lora", "lora_0"),
                "lora_strength": _field("IN:lora", "strength_model_0"),
            },
        })
    except Exception as exc:
        logger.error("background h3_settings_options error: %s", exc)
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
