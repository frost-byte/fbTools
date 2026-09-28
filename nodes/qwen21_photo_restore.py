"""Qwen-Image-2.1 photo-restoration REST routes (/fbtools/tools/restore_photo,
/fbtools/tools/restore_photo_settings_options).

Runs templates/qwen21_photo_restore.api.json against a single input photo. Unlike the MiniMax H3
templates in nodes/backgrounds_presets.py / nodes/h3_character_sheet.py, Qwen-Image-2.1's dedicated
edit-encode node (`TextEncodeQwenImage21`) genuinely VAE-encodes the source photo as part of its own
conditioning rather than only conditioning on a reference — MiniMax H3's reference-to-video and
image-to-video nodes always build an EMPTY starting latent and only ever attach real images as
conditioning (confirmed against comfy_extras/nodes_minimax_h3.py, 2026-09-27), which made faithful
photo restoration impossible with H3 and is why that path (an earlier patch_photo_restore_prompt()
in utils/h3_template_runner.py) was abandoned in favor of this one. Same title-based-contract /
server-to-server-submission approach as the H3 features, reusing
utils/h3_template_runner.patch_qwen21_photo_restore_prompt() and utils/h3_job_runner.py unchanged.
"""
from __future__ import annotations

from ..utils.h3_template_runner import load_template, find_node_by_title, patch_qwen21_photo_restore_prompt
from ..utils.h3_job_runner import submit_and_wait, free_vram, H3JobError
from aiohttp import web
from .composition_shared import _read_composition_settings
from .shared import routes, PACKAGE_ROOT
import asyncio
import os
import random
import shutil
import uuid
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH = os.path.join(
    PACKAGE_ROOT, "templates", "qwen21_photo_restore.api.json"
)

# A caller-supplied restore_hint replaces this literal placeholder wherever it appears in the
# default prompt below; with no hint supplied, the placeholder is still replaced with "" so it
# never leaks into the actual model prompt as literal text. Same convention as
# nodes/h3_character_sheet.py's _OUTFIT_HINT_PLACEHOLDER, kept in sync with the same placeholder
# baked into templates/qwen21_photo_restore.api.json's IN:restore_prompt node purely so the text
# looks the same when opened in ComfyUI directly — the route never reads that baked-in value,
# always sending prompt_text explicitly (see patch_qwen21_photo_restore_prompt).
_RESTORE_HINT_PLACEHOLDER = "{{RESTORE_HINT}}"

_QWEN21_RESTORE_DEFAULT_PROMPT = f"""subject_definitions:
<image1> is a single photograph, scanned from an old slide, showing exactly one moment in
time. It is not a character reference and not a composite of multiple sources — everything in
this photograph is real and factual: the person or people shown, their appearance, pose,
clothing, expression, and the surrounding scene. Nothing about who or what is depicted is
invented, guessed, or restyled.

The output should be a restored photograph, The photograph shows real people and a real scene exactly as originally captured; treat every visible detail - faces, bodies, clothing, posture, objects, architecture,
foliage, light and shadow - as ground truth to be preserved, not material to reinterpret.

RESTORE the physical condition of the photograph:
- Remove dust, speckling, scratches and specks introduced by the scanning or the aging film itself.
- Resolve blur and softness introduced by the original optics, film grain, or the scan, recovering
  natural sharpness and fine detail without inventing texture that was not implicit in the source.
- Reduce excessive film grain and noise to a clean, natural photographic look, without smoothing
  away real texture such as skin pores, fabric weave, or foliage detail.
- Repair burn spots, light leaks, chemical staining, discoloration blotches, creases, and physical
  tears or missing fragments of the emulsion, reconstructing only the damaged area itself using the
  surrounding undamaged image as the only guide for what belongs there.
- Correct faded, shifted, or yellowed color balance back to a natural, accurate rendition
  consistent with the rest of the image's own visible tones. If the photograph is genuinely
  black-and-white, keep it black-and-white - do not invent color that is not inferable from the
  photograph itself.
- Even out exposure and contrast lost to age, without introducing artificial HDR, glow, or
  stylized contrast the original photograph would not have had.

{_RESTORE_HINT_PLACEHOLDER}

Do not alter anything else about the photograph. The people shown keep their exact identity,
proportions, expressions, pose and clothing. The background, framing and camera angle are
unchanged. Nothing is added, removed, relocated, or reimagined - this is a cleanup of the
physical print, not a reinterpretation of the scene.

This is a FROZEN FRAME. Nothing moves at all. The camera holds a static shot and does not move,
zoom, tilt, shake or drift. Every frame is identical to the first: the shot reads as a single
restored photograph, not as a video.

No text, letters, numbers, captions, labels, watermarks, borders or frames added anywhere in the
image, no new objects, people or animals introduced, no removal of any person or object present
in the original, no change of identity, age, body proportions or facial features of anyone shown,
no colorization of a genuinely black-and-white photograph, no invented background detail beyond
what the undamaged parts of the photograph already show, no stylized, painterly, cartoon, HDR or
AI-generated look, no added grain, vignette, lens flare, or artistic color grading, no cropping or
change of aspect ratio, no camera movement, no cuts, no fades, no motion blur."""


def _qwen21_photo_restore_overrides(settings: dict) -> dict[str, dict]:
    """Build the patch_qwen21_photo_restore_prompt() `overrides` dict from the
    qwen21_photo_restore_* settings.

    Every value defaults to "" / 0 (unset) in _COMPOSITION_SETTINGS_DEFAULTS, meaning "use whatever
    the template itself specifies" — only non-empty/non-zero settings produce an override, and
    patch_qwen21_photo_restore_prompt() itself silently skips any title the template doesn't expose.
    """
    overrides: dict[str, dict] = {}
    if settings.get("qwen21_photo_restore_model"):
        overrides["IN:model"] = {"unet_name": settings["qwen21_photo_restore_model"]}
    if settings.get("qwen21_photo_restore_clip"):
        overrides["IN:clip"] = {"clip_name": settings["qwen21_photo_restore_clip"]}
    if settings.get("qwen21_photo_restore_vae"):
        overrides["IN:vae"] = {"vae_name": settings["qwen21_photo_restore_vae"]}

    sampler_fields: dict = {}
    if settings.get("qwen21_photo_restore_sampler"):
        sampler_fields["sampler_name"] = settings["qwen21_photo_restore_sampler"]
    if settings.get("qwen21_photo_restore_scheduler"):
        sampler_fields["scheduler"] = settings["qwen21_photo_restore_scheduler"]
    if settings.get("qwen21_photo_restore_steps"):
        sampler_fields["steps"] = int(settings["qwen21_photo_restore_steps"])
    if settings.get("qwen21_photo_restore_cfg"):
        sampler_fields["cfg"] = float(settings["qwen21_photo_restore_cfg"])
    if settings.get("qwen21_photo_restore_denoise"):
        sampler_fields["denoise"] = float(settings["qwen21_photo_restore_denoise"])
    if sampler_fields:
        overrides["IN:seed"] = sampler_fields

    encode_fields: dict = {}
    if settings.get("qwen21_photo_restore_negative_prompt"):
        encode_fields["negative_prompt"] = settings["qwen21_photo_restore_negative_prompt"]
    if settings.get("qwen21_photo_restore_resolution"):
        encode_fields["resolution"] = int(settings["qwen21_photo_restore_resolution"])
    if encode_fields:
        overrides["IN:negative_prompt"] = encode_fields

    return overrides


@routes.post("/fbtools/tools/restore_photo")
async def _tools_restore_photo(request):
    """Run the Qwen-Image-2.1 photo-restoration workflow against a single image.

    Runs server-to-server via this same ComfyUI instance's own /prompt + /history endpoints — no
    browser canvas is touched (see utils/h3_job_runner.py's module docstring for why a global
    progress indicator can still show activity even though no canvas gets node-level rendering for
    this job).

    Body: { filename, folder="input", restore_hint?, prompt? }
      restore_hint — optional freeform text substituted into the default restoration prompt
                     wherever {{RESTORE_HINT}} appears (e.g. "focus on the tear across the middle").
                     Ignored if `prompt` is also given.
      prompt       — optional full prompt override, replacing the default restoration prompt
                     entirely (advanced use — restore_hint is not applied in this case).
    Returns: { file, folder: "output" } — same shape as /fbtools/backgrounds/remove_people.
    """
    temp_path: str | None = None
    try:
        body = await request.json()
        filename = (body.get("filename") or "").strip()
        if not filename:
            return web.json_response({"error": "filename is required"}, status=400)
        folder = (body.get("folder") or "input").strip()
        prompt_override = (body.get("prompt") or "").strip()
        restore_hint = (body.get("restore_hint") or "").strip()
        prompt_text = prompt_override or _QWEN21_RESTORE_DEFAULT_PROMPT.replace(
            _RESTORE_HINT_PLACEHOLDER, restore_hint
        )

        if not os.path.exists(_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH}. Export the "
                          f"qwen21_photo_restore workflow from ComfyUI (Workflow -> Export (API)) "
                          f"and save it there first."},
                status=500,
            )

        import folder_paths
        src_dir = folder_paths.get_output_directory() if folder == "output" else folder_paths.get_input_directory()
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        input_dir = folder_paths.get_input_directory()

        def _prepare_input_image() -> tuple[str, str | None]:
            """Return (image filename relative to input/, temp file to clean up afterward or None).

            Same reasoning as nodes/backgrounds_presets.py's own _prepare_input_image: the Deno
            loader can only browse input/, so a file already there is used directly (never touched/
            deleted) and a file pulled from output/ gets a throwaway copy instead.
            """
            if folder == "output":
                tmp_dir = os.path.join(input_dir, "fbtools_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                ext = os.path.splitext(filename)[1]
                out_name = f"qwen21_restore_src_{uuid.uuid4().hex[:12]}{ext}"
                out_path = os.path.join(tmp_dir, out_name)
                shutil.copy2(src_path, out_path)
                return os.path.join("fbtools_tmp", out_name), out_path
            return filename, None

        loop = asyncio.get_event_loop()
        load_image_name, temp_path = await loop.run_in_executor(None, _prepare_input_image)

        template = load_template(_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH)
        save_node_id = find_node_by_title(template, "OUT:save")
        seed = random.randint(0, 2**32 - 1)
        filename_prefix = f"fbtools/qwen21_photo_restore/{uuid.uuid4().hex[:12]}"
        settings = _read_composition_settings()
        overrides = _qwen21_photo_restore_overrides(settings)
        prompt = patch_qwen21_photo_restore_prompt(
            template,
            image=load_image_name,
            prompt_text=prompt_text,
            seed=seed,
            filename_prefix=filename_prefix,
            overrides=overrides,
        )

        try:
            result = await submit_and_wait(request, prompt, save_node_id)
        finally:
            # VRAM is a machine-wide resource shared with the H3 features — reuse their single
            # "Unload model after each run" setting rather than a duplicate per-template one.
            if settings.get("h3_bg_plate_unload_after_run"):
                await free_vram(request)

        subfolder = result.get("subfolder", "")
        out_filename = result.get("filename", "")
        out_file = f"{subfolder}/{out_filename}" if subfolder else out_filename
        return web.json_response({"file": out_file, "folder": "output"})
    except H3JobError as exc:
        logger.error("tools restore_photo job error: %s", exc)
        return web.json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.error("tools restore_photo error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as exc:
                logger.warning(
                    "restore_photo: could not clean up temp file %s: %s", temp_path, exc
                )


@routes.get("/fbtools/tools/restore_photo_settings_options")
async def _tools_restore_photo_settings_options(request):
    """Enumeration lists + capability flags for the Settings > Qwen Photo Restore section.

    Each has_*_override flag tells the UI whether templates/qwen21_photo_restore.api.json currently
    exposes that optional override title at all — the corresponding Settings control should be
    disabled (not just left inert) when its flag is false, so a user can never configure an override
    that would silently do nothing. Same pattern as
    nodes/backgrounds_presets.py::_backgrounds_h3_settings_options.
    """
    try:
        if not os.path.exists(_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH}"}, status=404
            )
        template = load_template(_QWEN21_PHOTO_RESTORE_TEMPLATE_PATH)

        import folder_paths
        import comfy.samplers

        def _has(title: str) -> bool:
            return find_node_by_title(template, title, required=False) is not None

        def _field(title: str, field: str, default=None):
            """Current baked-in value of `field` on the node titled `title`, or `default` if the
            template doesn't have that title (or the field) at all."""
            node_id = find_node_by_title(template, title, required=False)
            if node_id is None:
                return default
            return template[node_id].get("inputs", {}).get(field, default)

        return web.json_response({
            "models":     sorted(folder_paths.get_filename_list("diffusion_models")),
            "clips":      sorted(folder_paths.get_filename_list("text_encoders")),
            "vaes":       sorted(folder_paths.get_filename_list("vae")),
            "samplers":   list(comfy.samplers.SAMPLER_NAMES),
            "schedulers": list(comfy.samplers.SCHEDULER_NAMES),
            "has_model_override":           _has("IN:model"),
            "has_clip_override":            _has("IN:clip"),
            "has_vae_override":             _has("IN:vae"),
            "has_sampler_override":         _has("IN:seed"),
            "has_negative_prompt_override": _has("IN:negative_prompt"),
            "template_defaults": {
                "model":           _field("IN:model", "unet_name"),
                "clip":            _field("IN:clip", "clip_name"),
                "vae":             _field("IN:vae", "vae_name"),
                "sampler":         _field("IN:seed", "sampler_name"),
                "scheduler":       _field("IN:seed", "scheduler"),
                "steps":           _field("IN:seed", "steps"),
                "cfg":             _field("IN:seed", "cfg"),
                "denoise":         _field("IN:seed", "denoise"),
                "negative_prompt": _field("IN:negative_prompt", "negative_prompt"),
                "resolution":      _field("IN:negative_prompt", "resolution"),
            },
        })
    except Exception as exc:
        logger.error("tools restore_photo_settings_options error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
