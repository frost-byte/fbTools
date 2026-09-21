"""LLM assistant, Modal/Unsloth vision backends, VLM activity: REST routes and the shared inference-routing helpers.

Moved out of extension.py (pure code motion). _route_llm / _run_vision_inference* route every inference call to the
active backend; extension.py re-imports the helpers its remaining nodes and routes still use."""
from __future__ import annotations

from ..utils import llm_client as _llm_client
from ..utils import modal_vision_client as _modal_client
from ..utils import unsloth_client as _unsloth_client
from ..utils import vlm_activity_log as _vlm_log
import os
from ..utils.source_profile_analysis import build_contact_sheet_image as _spa_build_contact_sheet
from ..utils.llm_scanner import scan_llm_dirs as _llm_scan_dirs, DEFAULT_MODEL as _LLM_DEFAULT_MODEL
import asyncio
from aiohttp import web
from folder_paths import get_input_directory, get_output_directory
import json
from ..utils import modal_vram_profiler as _vram_profiler
from ..utils import modal_deploy as _modal_deploy
from .shared import routes, send_status_update, user_data_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


_SPA_STATUS_ID = "fbt_SourceProfileAnalysis"


def _run_vision_inference(
    image_path: str,
    prompt: str,
    captioner_type: str = "auto",
    device: str = "auto",
    use_8bit: "bool | None" = None,
    clean: bool = False,
    profile_id: str = "",
    operation: str = "analyze",
    max_tokens: int = 512,
) -> str:
    """Run a single-image VLM call through the active backend.

    captioner_type:
      'gemini_flash' — Gemini API (GEMINI_API_KEY env var required).
      'unsloth'      — Unsloth Studio on Modal (must be active).
      'modal'        — Modal cloud VisionLLM (must be active).
      'local'        — local llm_client only.
      'auto'         — whatever _active_backend() returns.

    device / use_8bit are accepted for signature compatibility but ignored on
    remote and local llm_client paths.
    clean=True strips common VLM boilerplate from the response.
    Every call is recorded in the VLM activity log.
    """
    from pathlib import Path as _Path
    from ..captioner import caption_image_gemini as _cap_gemini, clean_caption_text as _cap_clean

    if captioner_type == "gemini_flash":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        text = _cap_gemini(_Path(image_path), prompt, api_key, clean=clean)
        _vlm_log.record(user_data_dir(), "gemini", "gemini-flash", operation, profile_id)
        return text

    backend = captioner_type if captioner_type in ("unsloth", "modal", "local") else _active_backend()

    from PIL import Image as _PIL_Image
    pil_image = _PIL_Image.open(image_path).convert("RGB")

    def _status_cb(msg: str) -> None:
        send_status_update(_SPA_STATUS_ID, msg, source="source_profile_analysis")

    if backend == "unsloth":
        if not _unsloth_client.is_active():
            raise RuntimeError("Unsloth backend is not active. Activate it in the LLM panel first.")
        if not _unsloth_client.active_endpoint_supports_vision():
            raise RuntimeError(
                f"The active Unsloth endpoint ({_unsloth_client.backend_status()['endpoint_label']}) "
                "is text-only. Switch to the 27B or Flash-Next endpoint for vision tasks."
            )
        result = _unsloth_client.generate(prompt, images=[pil_image], status_callback=_status_cb, max_tokens=max_tokens)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Unsloth generate returned no text")
        text = result.get("text", "")
        _vlm_log.record(user_data_dir(), "unsloth", _unsloth_client.backend_status()["model"], operation, profile_id)
        from ..captioner import clean_caption_text as _cc
        return _cc(text) if clean else text

    if backend == "modal":
        if not _modal_client.is_active():
            raise RuntimeError("Modal backend is not active. Activate it in the LLM panel first.")
        result = _modal_client.generate(prompt, images=[pil_image], status_callback=_status_cb, max_tokens=max_tokens)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Modal generate returned no text")
        text = result.get("text", "")
        _vlm_log.record(user_data_dir(), "modal", _modal_client.backend_status()["model_key"], operation, profile_id)
        from ..captioner import clean_caption_text as _cc
        return _cc(text) if clean else text

    # local
    st = _llm_client.backend_status()
    if not st.get("loaded_model"):
        raise RuntimeError("No model loaded. Load a vision-capable model in the Compose → LLM panel first.")
    if not st.get("supports_vision"):
        raise RuntimeError(
            f"The loaded model ({st['loaded_model']}) does not support vision inputs. "
            "Load a vision-capable model in the Compose → LLM panel."
        )
    result = _llm_client.generate(prompt, images=[pil_image], max_tokens=max_tokens)
    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("message") or "llm_client.generate returned no text")
    text = result.get("text", "")
    _vlm_log.record(user_data_dir(), "local", st.get("loaded_model", ""), operation, profile_id)
    return _cap_clean(text) if clean else text


def _run_text_inference(
    prompt: str,
    captioner_type: str = "auto",
    profile_id: str = "",
    operation: str = "text_inference",
    max_tokens: int = 1024,
) -> str:
    """Run a text-only LLM call (no image) through the active backend.

    captioner_type:
      'gemini_flash' — not supported for text-only; returns "".
      'unsloth'      — Unsloth Studio on Modal.
      'modal'        — Modal cloud VisionLLM.
      'local'        — local llm_client only.
      'auto'         — whatever _active_backend() returns.

    Returns "" if the chosen backend is unavailable.
    Every successful call is recorded in the VLM activity log.
    """
    if captioner_type == "gemini_flash":
        logger.debug("_run_text_inference: gemini_flash does not support text-only; skipping")
        return ""

    backend = captioner_type if captioner_type in ("unsloth", "modal", "local") else _active_backend()

    def _status_cb(msg: str) -> None:
        send_status_update(_SPA_STATUS_ID, msg, source="source_profile_analysis")

    if backend == "unsloth":
        if not _unsloth_client.is_active():
            return ""
        result = _unsloth_client.generate(prompt, max_tokens=max_tokens, status_callback=_status_cb)
        if not result.get("success"):
            logger.warning("Unsloth text inference failed: %s", result.get("message"))
            return ""
        _vlm_log.record(user_data_dir(), "unsloth", _unsloth_client.backend_status().get("model", ""), operation, profile_id)
        return result.get("text", "")

    if backend == "modal":
        if not _modal_client.is_active():
            return ""
        result = _modal_client.generate(prompt, images=None, status_callback=_status_cb, max_tokens=max_tokens)
        if not result.get("success"):
            logger.warning("Modal text inference failed: %s", result.get("message"))
            return ""
        _vlm_log.record(user_data_dir(), "modal", _modal_client.backend_status()["model_key"], operation, profile_id)
        return result.get("text", "")

    # local
    st = _llm_client.backend_status()
    if not st.get("loaded_model"):
        return ""
    result = _llm_client.generate(prompt, max_tokens=max_tokens)
    if not result.get("success"):
        logger.warning("Local text inference failed: %s", result.get("error") or result.get("message"))
        return ""
    _vlm_log.record(user_data_dir(), "local", st.get("loaded_model", ""), operation, profile_id)
    return result.get("text", "")


def _run_vision_inference_clip(
    frames: list,
    timestamps: list[float],
    sample_fps: float,
    raw_fps: float,
    prompt: str,
    captioner_type: str = "auto",
    profile_id: str = "",
    operation: str = "detect_segments",
) -> str:
    """Run a multi-frame VLM call, routing by captioner_type and model capabilities.

    captioner_type:
      'unsloth' — Unsloth Studio on Modal; native video for 27B/flash_next,
                  contact-sheet fallback for text-only endpoints.
      'modal'   — Modal cloud VisionLLM; native video if supported, else contact-sheet.
      'local'   — local llm_client; native video if supported, else contact-sheet.
      'auto'    — whatever _active_backend() returns.

    (gemini_flash is handled at the call site with a contact sheet + image path.)

    Raises RuntimeError if the required backend is unavailable.
    Every call is recorded in the VLM activity log.
    """
    backend = captioner_type if captioner_type in ("unsloth", "modal", "local") else _active_backend()

    def _status_cb(msg: str) -> None:
        send_status_update(_SPA_STATUS_ID, msg, source="source_profile_analysis")

    if backend == "unsloth":
        if not _unsloth_client.is_active():
            raise RuntimeError("Unsloth backend is not active. Activate it in the LLM panel first.")
        if not _unsloth_client.active_endpoint_supports_vision():
            raise RuntimeError(
                f"The active Unsloth endpoint ({_unsloth_client.backend_status()['endpoint_label']}) "
                "is text-only. Switch to the 27B or Flash-Next endpoint for vision tasks."
            )
        if _unsloth_client.active_endpoint_supports_native_video():
            result = _unsloth_client.generate(prompt, video_frames=frames, status_callback=_status_cb)
        else:
            sheet = _spa_build_contact_sheet(frames, timestamps)
            result = _unsloth_client.generate(prompt, images=[sheet], status_callback=_status_cb)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Unsloth generate returned no text")
        _vlm_log.record(user_data_dir(), "unsloth", _unsloth_client.backend_status()["model"], operation, profile_id)
        return result.get("text", "")

    if backend == "modal":
        if not _modal_client.is_active():
            raise RuntimeError("Modal backend is not active. Activate it in the LLM panel first.")
        st = _modal_client.backend_status()
        if st.get("native_video"):
            result = _modal_client.generate(
                prompt,
                video_frames=frames,
                video_meta={"sample_fps": sample_fps, "raw_fps": raw_fps},
                status_callback=_status_cb,
            )
        else:
            sheet = _spa_build_contact_sheet(frames, timestamps)
            result = _modal_client.generate(prompt, images=[sheet], status_callback=_status_cb)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Modal generate returned no text")
        _vlm_log.record(user_data_dir(), "modal", st["model_key"], operation, profile_id)
        return result.get("text", "")

    # local
    st = _llm_client.backend_status()
    if not st.get("loaded_model"):
        raise RuntimeError("No model loaded. Load a vision-capable model in the Compose → LLM panel first.")
    if not st.get("supports_vision"):
        raise RuntimeError(f"The loaded model ({st['loaded_model']}) does not support vision inputs.")
    if st.get("native_video"):
        result = _llm_client.generate(
            prompt,
            video_frames=frames,
            video_meta={"sample_fps": sample_fps, "raw_fps": raw_fps},
        )
    else:
        sheet = _spa_build_contact_sheet(frames, timestamps)
        result = _llm_client.generate(prompt, images=[sheet])
    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("message") or "llm_client returned no text")
    _vlm_log.record(user_data_dir(), "local", st.get("loaded_model", ""), operation, profile_id)
    return result.get("text", "")


@routes.get("/fbtools/llm/models")
async def _llm_models(request):
    """Scan LLM directories and return capability-annotated model list."""
    try:
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, _llm_scan_dirs)
        return web.json_response({
            "models":        models,
            "default_model": _LLM_DEFAULT_MODEL,
        })
    except Exception as exc:
        logger.error("LLM scan failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/llm/status")
async def _llm_status(request):
    """Return currently loaded model info and backend availability."""
    try:
        return web.json_response(_llm_client.backend_status())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/load")
async def _llm_load(request):
    """Load a model by its descriptor dict (from /fbtools/llm/models)."""
    try:
        body = await request.json()
        model_info = body.get("model_info")
        if not model_info:
            return web.json_response({"error": "model_info required"}, status=400)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llm_client.load_model, model_info)
        status = 200 if result["success"] else 503
        return web.json_response(result, status=status)
    except Exception as exc:
        logger.error("LLM load error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/unload")
async def _llm_unload(request):
    """Unload the current model and free VRAM."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llm_client.unload_model)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/llm/vram_analysis")
async def _llm_vram_analysis(request):
    """VRAM budget and KV-cache context-size table for the loaded GGUF model."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llm_client.vram_analysis)
        status = 200 if result.get("success") else 503
        return web.json_response(result, status=status)
    except Exception as exc:
        logger.error("LLM vram_analysis error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/context_estimate")
async def _llm_context_estimate(request):
    """Pre-load VRAM/context-size estimate for a candidate GGUF model.

    Reads only the GGUF header (no weights loaded), so the LLM panel can show
    a capacity guideline as soon as the user picks a model, before Load.
    """
    try:
        body = await request.json()
        model_info = body.get("model_info")
        if not model_info:
            return web.json_response({"error": "model_info required"}, status=400)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llm_client.estimate_context_table, model_info)
        status = 200 if result.get("success") else 503
        return web.json_response(result, status=status)
    except Exception as exc:
        logger.error("LLM context_estimate error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


def _active_backend() -> str:
    """Return the currently active inference backend: 'unsloth', 'modal', or 'local'."""
    if _unsloth_client.is_active():
        return "unsloth"
    if _modal_client.is_active():
        return "modal"
    return "local"


async def _route_llm(
    prompt: str,
    *,
    images: list | None = None,
    video_frames: list | None = None,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    video_meta: dict | None = None,
) -> dict:
    """Route an inference call to the currently active backend.

    Pass images or video_frames for vision tasks; omit both for text-only.
    Returns {success, text, message}.
    """
    backend = _active_backend()

    if backend == "unsloth":
        if (images or video_frames) and not _unsloth_client.active_endpoint_supports_vision():
            ep_label = _unsloth_client.backend_status().get("endpoint_label", "")
            return {
                "success": False,
                "text":    "",
                "message": (
                    f"Active Unsloth endpoint ({ep_label}) is text-only. "
                    "Switch to the 27B or Flash-Next endpoint for vision tasks."
                ),
            }
        try:
            return await asyncio.to_thread(
                _unsloth_client.generate,
                prompt,
                images=images,
                video_frames=video_frames,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            _unsloth_client.mark_container_gone()
            raise

    if backend == "modal":
        return await asyncio.to_thread(
            _modal_client.generate,
            prompt,
            images=images,
            video_frames=video_frames,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            video_meta=video_meta,
        )

    # local
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _llm_client.generate(
            prompt,
            images=images,
            video_frames=video_frames,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            video_meta=video_meta,
        ),
    )


@routes.post("/fbtools/llm/generate")
async def _llm_generate(request):
    """Generate text (optionally with image filenames from ComfyUI input dir).

    Routes through Unsloth when active; falls back to local llm_client otherwise.

    Body fields:
        prompt (str)           — user prompt
        system_prompt (str)    — optional system prompt
        images (list[str])     — filenames inside the ComfyUI input directory
        max_tokens (int)       — default 512
        temperature (float)    — default 0.7
    """
    try:
        body = await request.json()
        prompt        = body.get("prompt", "")
        system_prompt = body.get("system_prompt", "")
        max_tokens    = int(body.get("max_tokens", 512))
        temperature   = float(body.get("temperature", 0.7))
        image_filenames: list[str] = body.get("images", [])

        pil_images = []
        if image_filenames:
            try:
                from PIL import Image
                import folder_paths
                input_dir  = folder_paths.get_input_directory()
                output_dir = folder_paths.get_output_directory()
                for fname in image_filenames:
                    fpath = os.path.join(input_dir, fname)
                    if not os.path.exists(fpath):
                        alt = os.path.join(output_dir, fname)
                        if os.path.exists(alt):
                            fpath = alt
                    if os.path.exists(fpath):
                        pil_images.append(Image.open(fpath).convert("RGB"))
                    else:
                        logger.warning("LLM generate: image not found: %s", fpath)
            except Exception as img_err:
                logger.warning("LLM generate: image load error: %s", img_err)

        if pil_images:
            result = await _route_llm(
                prompt,
                images=pil_images,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            result = await _route_llm(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        logger.error("LLM generate error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/generate/shot_action")
async def _llm_gen_shot_action(request):
    """Generate a shot action description using template-aware prompt builder."""
    try:
        body        = await request.json()
        shot_number = int(body.get("shot_number", 1))
        subjects    = body.get("subjects", [])
        environment = body.get("environment", "")
        style       = body.get("style", "cinematic")
        existing    = body.get("existing", "")
        system, user = _llm_client.prompt_for_shot_action(
            shot_number, subjects, environment, style, existing
        )
        result = await _route_llm(user, system_prompt=system, max_tokens=256)
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/generate/dialogue")
async def _llm_gen_dialogue(request):
    """Generate a dialogue line for a named speaker."""
    try:
        body     = await request.json()
        speaker  = body.get("speaker", "Character")
        context  = body.get("context", "")
        tone     = body.get("tone", "")
        language = body.get("language", "en-us")
        system, user = _llm_client.prompt_for_shot_dialogue(speaker, context, tone, language)
        result = await _route_llm(user, system_prompt=system, max_tokens=128)
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/generate/polish")
async def _llm_gen_polish(request):
    """Polish existing text for clarity and vividness."""
    try:
        body    = await request.json()
        text    = body.get("text", "")
        context = body.get("context", "")
        system, user = _llm_client.prompt_for_polish(text, context)
        result = await _route_llm(user, system_prompt=system, max_tokens=512)
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/describe_video")
async def _llm_describe_video(request):
    """Describe character actions/expressions/appearance in a video clip.

    Body: { video_path, shot_number, subjects, environment, style, intent, max_frames }
    intent: "actions" | "expressions" | "appearance"
    video_path: filename relative to input or output dir, or an absolute path.
    """
    try:
        body = await request.json()
        video_path    = body.get("video_path", "")
        video_dir     = body.get("dir", "input")
        frame_indices = body.get("frame_indices", None)   # preferred: explicit list from UI
        shot_number   = int(body.get("shot_number", 1))
        subjects      = body.get("subjects", [])
        environment   = body.get("environment", "")
        style         = body.get("style", "")
        intent        = body.get("intent", "actions")

        if not video_path:
            return web.json_response({"error": "video_path required"}, status=400)

        base = get_output_directory() if video_dir == "output" else get_input_directory()
        resolved = None
        for candidate in [
            os.path.join(base, video_path),
            video_path,  # absolute path fallback
        ]:
            if os.path.isfile(candidate):
                resolved = candidate
                break
        if not resolved:
            return web.json_response({"error": f"Video not found: {video_path}"}, status=404)

        def _extract_frames(path: str, indices: list[int]):
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return [], {}
            raw_fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration  = total / raw_fps if raw_fps > 0 else 0.0
            frames = []
            for idx in indices:
                idx = max(0, min(idx, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, bgr = cap.read()
                if not ret:
                    continue
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            cap.release()
            # sample_fps: effective rate of the selected frames across the full clip.
            # This drives temporal RoPE position IDs in Qwen2.5-Omni/VL.
            sample_fps = len(frames) / duration if duration > 0 else raw_fps
            meta = {"raw_fps": raw_fps, "sample_fps": sample_fps, "duration": duration}
            return frames, meta

        if not frame_indices:
            return web.json_response({"error": "frame_indices required"}, status=400)

        loop = asyncio.get_event_loop()
        frames, video_meta = await loop.run_in_executor(None, _extract_frames, resolved, frame_indices)
        if not frames:
            return web.json_response({"error": "Could not extract frames from video"}, status=422)

        # Allow the caller to supply fully custom prompts (from the UI editor).
        # Fall back to auto-generated prompts when not provided.
        system_override = body.get("system_prompt", "").strip()
        user_override   = body.get("user_prompt",   "").strip()
        if system_override or user_override:
            system = system_override
            user   = user_override
        else:
            system, user = _llm_client.prompt_for_video_action(
                shot_number=shot_number,
                subjects=subjects,
                intent=intent,
                environment=environment,
                style=style,
            )

        logger.info(
            "describe_video: %d frames, sample_fps=%.2f, raw_fps=%.2f, duration=%.1fs",
            len(frames), video_meta["sample_fps"], video_meta["raw_fps"], video_meta["duration"],
        )
        result = await _route_llm(
            user,
            video_frames=frames,
            system_prompt=system,
            max_tokens=512,
            video_meta=video_meta,
        )
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        logger.error("LLM describe_video error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/video_prompt")
async def _llm_video_prompt(request):
    """Return the auto-generated system + user prompts for a video describe call.

    Body: { shot_number, subjects, intent, environment, style }
    Response: { system, user }
    """
    try:
        body = await request.json()
        system, user = _llm_client.prompt_for_video_action(
            shot_number=int(body.get("shot_number", 1)),
            subjects=body.get("subjects", []),
            intent=body.get("intent", "actions"),
            environment=body.get("environment", ""),
            style=body.get("style", ""),
        )
        return web.json_response({"system": system, "user": user})
    except Exception as exc:
        logger.error("LLM video_prompt error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


def _llm_history_path() -> str:
    return os.path.join(user_data_dir(), "llm_history.json")


def _llm_history_migrate(e: dict) -> dict:
    """Promote a legacy flat video_describe entry to the unified envelope schema."""
    if "params" in e:
        return e  # already migrated
    e.setdefault("kind", "video_describe")
    e["params"] = {k: e.pop(k) for k in (
        "videoPath", "videoDir", "vidInfo", "startTime", "duration",
        "everyNth", "cap", "frameIndices", "intent", "systemPrompt", "userPrompt",
    ) if k in e}
    raw_result = e.pop("result", "")
    e["result"] = {"text": raw_result} if isinstance(raw_result, str) else (raw_result or {})
    return e


def _llm_history_load() -> list:
    p = _llm_history_path()
    if not os.path.exists(p):
        # Migrate from the old per-feature file if present
        old_p = os.path.join(user_data_dir(), "video_describe_history.json")
        if os.path.exists(old_p):
            try:
                with open(old_p, encoding="utf-8") as f:
                    entries = json.load(f)
                entries = [_llm_history_migrate(e) for e in (entries if isinstance(entries, list) else [])]
                _llm_history_save(entries)
                return entries
            except Exception:
                pass
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _llm_history_save(entries: list) -> None:
    p = _llm_history_path()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


@routes.get("/fbtools/llm/history")
async def _llm_history_list(request):
    """Return LLM run history entries, newest first.

    Optional query param ?kind=shot_action,dialogue  — comma-separated kind filter.
    """
    try:
        kinds_param = request.rel_url.query.get("kind", "")
        kinds = {k.strip() for k in kinds_param.split(",") if k.strip()} if kinds_param else set()
        entries = _llm_history_load()
        if kinds:
            entries = [e for e in entries if e.get("kind") in kinds]
        return web.json_response(entries)
    except Exception as exc:
        logger.error("llm_history list error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/history")
async def _llm_history_add(request):
    """Append a new history entry.  Body: the full entry dict."""
    try:
        entry   = await request.json()
        entries = _llm_history_load()
        entries.insert(0, entry)
        if len(entries) > 50:
            entries = entries[:50]
        _llm_history_save(entries)
        return web.json_response({"success": True})
    except Exception as exc:
        logger.error("llm_history add error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/history/delete")
async def _llm_history_delete(request):
    """Remove a single entry.  Body: { id: <numeric id> }"""
    try:
        body     = await request.json()
        entry_id = int(body["id"])
        entries  = [e for e in _llm_history_load() if e.get("id") != entry_id]
        _llm_history_save(entries)
        return web.json_response({"success": True})
    except Exception as exc:
        logger.error("llm_history delete error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


# Legacy shims — keep old URL paths working during any in-flight requests
# (/fbtools/llm/describe_history/*) by delegating to the new unified handlers.
@routes.get("/fbtools/llm/describe_history")
async def _llm_describe_history_list(request):
    return await _llm_history_list(request)


@routes.post("/fbtools/llm/describe_history")
async def _llm_describe_history_add(request):
    return await _llm_history_add(request)


@routes.post("/fbtools/llm/describe_history/delete")
async def _llm_describe_history_delete(request):
    return await _llm_history_delete(request)


@routes.get("/fbtools/modal/status")
async def _modal_status(request):
    """Return Modal backend status (active model, availability, VRAM recommendation)."""
    try:
        st = _modal_client.backend_status()
        st["presets"] = _modal_client.PRESET_MODELS
        # Attach a quick recommendation for the currently-active model (non-blocking).
        active_key = st.get("model_key")
        if active_key:
            try:
                st["recommendation"] = _vram_profiler.get_recommendation(
                    active_key,
                    quantize=st.get("quantize", True),
                    data_dir=user_data_dir(),
                )
            except Exception:
                pass
        return web.json_response(st)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/modal/activate")
async def _modal_activate(request):
    """Activate the Modal backend.  Body: { model_key, quantize, gpu }."""
    try:
        body      = await request.json()
        model_key = str(body.get("model_key", "qwen2.5-vl-7b")).strip()
        quantize  = bool(body.get("quantize", True))
        gpu       = str(body.get("gpu", "L40S")).strip().upper()
        if _unsloth_client.is_active():
            _unsloth_client.deactivate()
        result    = _modal_client.activate(model_key, quantize, gpu=gpu)
        if result["success"]:
            _vlm_log.record(user_data_dir(), "modal", model_key, "activate")
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/modal/deactivate")
async def _modal_deactivate(request):
    """Deactivate the Modal backend (clears local state only)."""
    try:
        result = _modal_client.deactivate()
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/modal/recommend")
async def _modal_recommend(request):
    """Compute a VRAM recommendation for a given model configuration.

    Query params:
      model_key       (str)  preset key or HF repo ID [required]
      quantize        (bool) use 4-bit quant (default true)
      context_length  (int)  token budget (default 8192)
      frame_budget    (int)  frames passed per call (default 20)
      modality        (str)  image / video / text (default image)
      priority        (str)  cost / throughput / bandwidth (default cost)
    """
    try:
        params         = request.rel_url.query
        model_key      = params.get("model_key", "").strip()
        if not model_key:
            return web.json_response({"error": "model_key is required"}, status=400)
        quantize       = params.get("quantize", "true").lower() not in ("false", "0", "no")
        context_length = int(params.get("context_length", 8192))
        frame_budget   = int(params.get("frame_budget", 20))
        modality       = params.get("modality", "image")
        priority       = params.get("priority", "cost")

        rec = await asyncio.to_thread(
            _vram_profiler.get_recommendation,
            model_key,
            quantize=quantize,
            context_length=context_length,
            frame_budget=frame_budget,
            modality=modality,
            data_dir=user_data_dir(),
            priority=priority,
        )
        return web.json_response(rec)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/modal/profile_repo")
async def _modal_profile_repo(request):
    """Fetch and cache a VRAM profile for a custom HuggingFace repo.

    Body (JSON):
      repo_id    (str)  HF repo ID, e.g. "Qwen/Qwen3-VL-8B-Instruct" [required]
      hf_token   (str)  optional; falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars
      refresh    (bool) re-fetch even if a cached profile exists (default false)
    """
    try:
        body     = await request.json()
        repo_id  = str(body.get("repo_id", "")).strip()
        if not repo_id:
            return web.json_response({"error": "repo_id is required"}, status=400)

        hf_token = (
            body.get("hf_token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        refresh  = bool(body.get("refresh", False))

        data_dir = user_data_dir()
        if not refresh:
            cache = _vram_profiler.load_profile_cache(data_dir)
            if repo_id in cache:
                profile = cache[repo_id]
                rec     = _vram_profiler.get_recommendation(repo_id, data_dir=data_dir)
                return web.json_response({"profile": profile, "recommendation": rec, "cached": True})

        profile = await asyncio.to_thread(
            _vram_profiler.fetch_hf_profile, repo_id, hf_token
        )
        cache           = _vram_profiler.load_profile_cache(data_dir)
        cache[repo_id]  = profile
        _vram_profiler.save_profile_cache(data_dir, cache)

        rec = _vram_profiler.get_recommendation(repo_id, data_dir=data_dir)
        return web.json_response({"profile": profile, "recommendation": rec, "cached": False})
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=422)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/unsloth/status")
async def _unsloth_status(request):
    """Return Unsloth backend status (active endpoint, warmup state, endpoint list)."""
    try:
        return web.json_response(_unsloth_client.backend_status())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/activate")
async def _unsloth_activate(request):
    """Activate the Unsloth backend and start container warm-up.

    Body: { endpoint_key }  — one of "27b" (default), "8b", "flash_next".
    Returns immediately; warm-up runs in a background thread.
    """
    try:
        body         = await request.json()
        endpoint_key = str(body.get("endpoint_key", _unsloth_client.DEFAULT_ENDPOINT)).strip()
        if _modal_client.is_active():
            _modal_client.deactivate()
        result       = _unsloth_client.activate(endpoint_key)
        if result["success"]:
            st = _unsloth_client.backend_status()
            _vlm_log.record(user_data_dir(), "unsloth", st.get("model", endpoint_key), "activate")
        return web.json_response(result, status=200 if result["success"] else 400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/deactivate")
async def _unsloth_deactivate(request):
    """Deactivate the Unsloth backend (clears local state only)."""
    try:
        result = _unsloth_client.deactivate()
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/unsloth/health")
async def _unsloth_health(request):
    """Fast probe: check if the active endpoint's container is warm.

    Query param: endpoint_key (optional; defaults to currently active endpoint).
    Returns {status: warm|starting|down|error, message}.
    """
    try:
        params       = request.rel_url.query
        endpoint_key = params.get("endpoint_key", None)
        result = await asyncio.to_thread(_unsloth_client.health_check, endpoint_key)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/unsloth/setup_status")
async def _unsloth_setup_status(request):
    """Return setup readiness: workspace known, API key stored, app deployed.

    Used by the LLM panel to show the user what setup steps remain.
    Returns {workspace, workspace_set, api_key_set, app_deployed, ready}.
    """
    try:
        st         = _unsloth_client.backend_status()
        app_info   = await asyncio.to_thread(_modal_deploy.app_status)
        return web.json_response({
            "workspace":     st.get("workspace", ""),
            "workspace_set": st.get("workspace_set", False),
            "api_key_set":   st.get("api_key_set", False),
            "app_deployed":  app_info.get("deployed", False),
            "app_message":   app_info.get("message", ""),
            "ready":         st.get("workspace_set") and st.get("api_key_set") and app_info.get("deployed"),
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/deploy")
async def _unsloth_deploy(request):
    """Deploy the bundled Unsloth Studio app to the user's Modal workspace.

    Runs `modal deploy modal/unsloth_studio.py` as a subprocess.
    Typical duration: 30-90 seconds.
    Returns {success, message, output}.
    """
    try:
        result = await asyncio.to_thread(_modal_deploy.deploy, user_data_dir())
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/undeploy")
async def _unsloth_undeploy(request):
    """Stop (undeploy) the Unsloth Studio app — all endpoints go offline.

    Returns {success, message}.
    """
    try:
        result = await asyncio.to_thread(_modal_deploy.undeploy)
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/unsloth/serve_mode")
async def _unsloth_get_serve_mode(request):
    """Return current serve mode preference and last-deployed mode.

    Returns {api_only, last_deployed_api_only}.
    """
    try:
        cfg = _modal_deploy.load_serve_config(user_data_dir())
        return web.json_response({
            "api_only":              cfg.get("api_only", True),
            "last_deployed_api_only": cfg.get("last_deployed_api_only", None),
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/serve_mode")
async def _unsloth_set_serve_mode(request):
    """Update serve mode preference and write to Modal Volume.

    Body: {api_only: bool}
    Returns {success, message}.
    """
    try:
        body     = await request.json()
        api_only = bool(body.get("api_only", True))
        result   = await asyncio.to_thread(
            _modal_deploy.set_serve_mode, user_data_dir(), api_only
        )
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/unsloth/containers")
async def _unsloth_containers(request):
    """List running Unsloth Studio containers (warm = actively billing).

    Returns {success, containers: [{id, raw}]}.
    """
    try:
        result = await asyncio.to_thread(_modal_deploy.list_containers)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/containers/stop")
async def _unsloth_containers_stop(request):
    """Force-stop running containers immediately (don't wait for idle scaledown).

    Body: { container_id } to stop one, or omit to stop all.
    Returns {success, message} or {success, stopped, message}.
    """
    try:
        body         = await request.json()
        container_id = str(body.get("container_id", "")).strip()
        if container_id:
            result = await asyncio.to_thread(_modal_deploy.stop_container, container_id)
        else:
            result = await asyncio.to_thread(_modal_deploy.stop_all_containers)
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/bootstrap_key")
async def _unsloth_bootstrap_key(request):
    """Run Unsloth Studio setup: install Studio + capture API key.

    Long-running (~5-30 min).  Runs in a background thread; the response
    is returned when complete.  The captured key is stored in user data dir
    and the client is reconfigured immediately.

    Body: { force_reinstall: bool }  (default false)
    Returns {success, api_key, message}.
    """
    try:
        body            = await request.json()
        force_reinstall = bool(body.get("force_reinstall", False))
        result = await asyncio.to_thread(
            _modal_deploy.run_bootstrap, user_data_dir(), force_reinstall
        )
        if result["success"] and result.get("api_key"):
            # Reconfigure the live client with the new key
            _unsloth_client.configure(api_key=result["api_key"])
            _vlm_log.record(user_data_dir(), "unsloth", "", "bootstrap_key")
        return web.json_response(result, status=200 if result["success"] else 503)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/unsloth/inference_settings")
async def _unsloth_inference_settings(request):
    """Set thinking mode and reasoning depth for all Unsloth inference calls.

    Body: { thinking: bool, reasoning_effort: "low"|"medium"|"xhigh"|null }
    thinking=false → instruct mode (reasoning_effort ignored)
    thinking=true  → thinking mode with specified depth (default "xhigh")
    """
    try:
        body             = await request.json()
        thinking         = bool(body.get("thinking", True))
        reasoning_effort = body.get("reasoning_effort") or None
        result = _unsloth_client.set_inference_settings(thinking, reasoning_effort)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/generate/stream")
async def _llm_generate_stream(request):
    """Streaming text generation via SSE (text/event-stream).

    Routes through Unsloth when active; returns a 503 for local llm_client
    (streaming not supported for local models).

    Body: same fields as /fbtools/llm/generate (prompt, system_prompt, max_tokens).
    Response: SSE stream — data: {"text": "<chunk>"} per token, then data: [DONE].
    """
    import json as _json

    if not _unsloth_client.is_active():
        return web.json_response(
            {"error": "Streaming requires the Unsloth backend to be active."},
            status=503,
        )

    try:
        body          = await request.json()
        prompt        = body.get("prompt", "")
        system_prompt = body.get("system_prompt", "")
        max_tokens    = int(body.get("max_tokens", 2048))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    response = web.StreamResponse()
    response.headers["Content-Type"]  = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    await response.prepare(request)

    queue: asyncio.Queue = asyncio.Queue()
    loop  = asyncio.get_event_loop()

    def _stream_worker():
        try:
            for chunk in _unsloth_client.generate_stream(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    import threading as _threading
    _threading.Thread(target=_stream_worker, daemon=True).start()

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                await response.write(
                    f"data: {_json.dumps({'error': str(item)})}\n\n".encode()
                )
                break
            await response.write(
                f"data: {_json.dumps({'text': item})}\n\n".encode()
            )
        await response.write(b"data: [DONE]\n\n")
    except Exception:
        pass

    return response


@routes.get("/fbtools/vlm/activity")
async def _vlm_activity_list(request):
    """Return recent VLM activity entries.  Query params: n (int), backend (str)."""
    try:
        params  = request.rel_url.query
        n       = int(params.get("n", 100))
        backend = params.get("backend", "")
        entries = _vlm_log.recent(user_data_dir(), n=min(n, 500))
        if backend:
            entries = [e for e in entries if e.get("backend") == backend]
        return web.json_response({"entries": entries})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/vlm/model_history")
async def _vlm_model_history(request):
    """Return deduplicated model IDs for a backend, most recent first.
    Query param: backend (str, required).
    """
    try:
        backend = request.rel_url.query.get("backend", "")
        if not backend:
            return web.json_response({"error": "backend param required"}, status=400)
        history = _vlm_log.model_history(user_data_dir(), backend)
        return web.json_response({"model_ids": history})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/llm/download/default")
async def _llm_download_default(request):
    """Download the recommended default LLM model using huggingface_hub.

    Creates ComfyUI/models/LLMs/<model_name>/ and downloads the GGUF files.
    Returns streaming progress via JSON. Raises 503 if huggingface_hub absent.
    """
    try:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return web.json_response(
                {"error": "huggingface_hub not installed. pip install huggingface_hub"},
                status=503,
            )

        import folder_paths

        # Determine target directory
        llm_dirs = folder_paths.get_folder_paths("LLMs")
        if llm_dirs:
            base_llm_dir = llm_dirs[0]
        else:
            base_llm_dir = os.path.join(folder_paths.base_path, "models", "LLMs")
        os.makedirs(base_llm_dir, exist_ok=True)

        m = _LLM_DEFAULT_MODEL
        model_dir = os.path.join(base_llm_dir, m["name"])
        os.makedirs(model_dir, exist_ok=True)

        downloaded: list[str] = []

        def _do_download():
            files = [m["filename"]]
            if m.get("mmproj"):
                files.append(m["mmproj"])
            for fname in files:
                dest = os.path.join(model_dir, fname)
                if os.path.exists(dest):
                    downloaded.append(f"already present: {fname}")
                    continue
                logger.info("Downloading %s / %s → %s", m["repo_id"], fname, dest)
                hf_hub_download(
                    repo_id=m["repo_id"],
                    filename=fname,
                    local_dir=model_dir,
                )
                downloaded.append(f"downloaded: {fname}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_download)

        return web.json_response({
            "success":    True,
            "model_dir":  model_dir,
            "downloaded": downloaded,
            "model_name": m["name"],
        })
    except Exception as exc:
        logger.error("LLM download error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
