"""H3 character/face-sheet generation REST routes (/fbtools/bundles/generate_character_sheet,
/fbtools/bundles/character_sheet_settings_options).

Runs templates/h3_character_sheet.api.json (see templates/README.md) against up to 9 of a Reference
Bundle's own images, producing either a character sheet or a head/face-shot sheet. Same
title-based-contract / server-to-server-submission approach as nodes/backgrounds_presets.py's H3
Background Plate feature — reuses utils/h3_template_runner.py and utils/h3_job_runner.py unchanged
except for the new patch_character_sheet_prompt() helper (this template's required-node contract
differs from the background-plate one: up to 9 reference images plus a mode switch, no single
IN:image/IN:prompt pair).
"""
from __future__ import annotations

from ..utils.h3_template_runner import load_template, find_node_by_title, patch_character_sheet_prompt, MAX_REF_IMAGES
from ..utils.h3_job_runner import submit_and_wait, free_vram, H3JobError
from ..utils.reference_bundles import load_registry as _load_bundle_registry
from aiohttp import web
from .composition_shared import _read_composition_settings
from .shared import routes, default_bundle_registry_path, PACKAGE_ROOT
import asyncio
import os
import random
import shutil
import uuid
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_H3_CHAR_SHEET_TEMPLATE_PATH = os.path.join(PACKAGE_ROOT, "templates", "h3_character_sheet.api.json")

# IN:mode is a KJNodes LazySwitchKJ node's boolean "switch" input (replaced the original 3-way
# ImpactSwitch — Video mode has no branch here at all, out of scope for this feature):
# on_false <- "Character Sheet Options" DictCreate, on_true <- "Face Sheet Options" DictCreate.
_MODE_SELECT = {"character": False, "face": True}

# Each mode's own prompt text lives in its own PrimitiveStringMultiline node (fed into that mode's
# DictCreate) — titling one of these is how a template author opts into the outfit-hint feature for
# that mode. A caller-supplied hint replaces this literal placeholder wherever the author placed it
# in their own prompt text; with no hint supplied (or no title set), any placeholder present is
# still replaced with "" so it never leaks into the actual model prompt as literal text.
_PROMPT_TITLE_BY_MODE = {"character": "IN:character_prompt", "face": "IN:face_prompt"}
_OUTFIT_HINT_PLACEHOLDER = "{{OUTFIT_HINT}}"


def _h3_char_sheet_prompt_override(template: dict, mode: str, outfit_hint: str) -> dict[str, dict]:
    """Build the single-title override that substitutes _OUTFIT_HINT_PLACEHOLDER in the active
    mode's own prompt text with `outfit_hint` (or removes the placeholder entirely if blank).

    Returns {} if the template doesn't title the active mode's prompt node, or if that node's
    current text doesn't contain the placeholder at all — same lenient "only offer what the
    template actually supports" convention as every other override in this feature.
    """
    title = _PROMPT_TITLE_BY_MODE.get(mode)
    if not title:
        return {}
    node_id = find_node_by_title(template, title, required=False)
    if node_id is None:
        return {}
    current = template[node_id].get("inputs", {}).get("value", "")
    if _OUTFIT_HINT_PLACEHOLDER not in current:
        return {}
    return {title: {"value": current.replace(_OUTFIT_HINT_PLACEHOLDER, outfit_hint or "")}}


def _h3_char_sheet_overrides(settings: dict) -> dict[str, dict]:
    """Build the patch_character_sheet_prompt() `overrides` dict from the h3_char_sheet_* settings.

    Every value defaults to "" / 0 (unset) in _COMPOSITION_SETTINGS_DEFAULTS, meaning "use whatever
    the template itself specifies" — only non-empty/non-zero settings produce an override, and
    patch_character_sheet_prompt() itself silently skips any title the template doesn't expose.
    """
    overrides: dict[str, dict] = {}
    if settings.get("h3_char_sheet_model"):
        overrides["IN:model"] = {"unet_name": settings["h3_char_sheet_model"]}
    if settings.get("h3_char_sheet_clip"):
        overrides["IN:clip"] = {"clip_name": settings["h3_char_sheet_clip"]}
    if settings.get("h3_char_sheet_lora"):
        # Same fbTools LoraStackBuilder row-0 contract as IN:lora on the background-plate template —
        # see templates/README.md. strength_model and strength_clip both take the one configured value.
        strength = float(settings.get("h3_char_sheet_lora_strength", 0.38))
        overrides["IN:lora"] = {
            "lora_0": settings["h3_char_sheet_lora"],
            "strength_model_0": strength,
            "strength_clip_0": strength,
            "enabled_0": True,
        }
    if settings.get("h3_char_sheet_sampler1"):
        overrides["IN:sampler1"] = {"sampler_name": settings["h3_char_sheet_sampler1"]}
    if settings.get("h3_char_sheet_scheduler1"):
        # Only "scheduler" — first-pass steps come from the active mode's own DictCreate, not from
        # Settings (see templates/README.md).
        overrides["IN:scheduler1"] = {"scheduler": settings["h3_char_sheet_scheduler1"]}
    if settings.get("h3_char_sheet_sampler2"):
        overrides["IN:sampler2"] = {"sampler_name": settings["h3_char_sheet_sampler2"]}
    if settings.get("h3_char_sheet_upscale_steps_select"):
        overrides["IN:upscale_steps_select"] = {"select": int(settings["h3_char_sheet_upscale_steps_select"])}
    if settings.get("h3_char_sheet_upscale_factor"):
        # The node's displayed label is "scale", but its real API-JSON field name is the dotted
        # "mode.scale" (a MinimaxH3LatentUpscaler3D compound widget) — confirmed 2026-09-27.
        overrides["IN:upscale_factor"] = {"mode.scale": float(settings["h3_char_sheet_upscale_factor"])}
    aspect_fields = {}
    if settings.get("h3_char_sheet_aspect_ratio"):
        aspect_fields["aspect_ratio"] = settings["h3_char_sheet_aspect_ratio"]
    if settings.get("h3_char_sheet_megapixels"):
        aspect_fields["megapixels"] = float(settings["h3_char_sheet_megapixels"])
    if aspect_fields:
        overrides["IN:aspect_ratio"] = aspect_fields
    return overrides


def _resolve_bundle_ref_images(files: list) -> tuple[list[str], list[str]]:
    """Resolve a bundle's visual.files entries (str or {file, role}) to filenames LoadImage-style
    widgets can read (input/ directory only).

    A file already sitting in input/ is used directly (never touched/copied). A file that only
    exists in output/ is copied into input/fbtools_tmp/ under a throwaway name, mirroring
    nodes/backgrounds_presets.py's own _prepare_input_image approach for the same reason (H3's
    reference-image widgets can only browse input/).

    Returns (load_names, temp_paths) — temp_paths lists only the copies this call made, for the
    caller to clean up afterward; entries not found on disk at all are skipped with a warning.
    """
    import folder_paths
    input_dir = folder_paths.get_input_directory()
    output_dir = folder_paths.get_output_directory()
    tmp_dir = os.path.join(input_dir, "fbtools_tmp")

    load_names: list[str] = []
    temp_paths: list[str] = []
    for entry in files:
        fname = entry["file"] if isinstance(entry, dict) else entry
        if not fname:
            continue
        in_path = os.path.join(input_dir, fname)
        if os.path.exists(in_path):
            load_names.append(fname)
            continue
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            os.makedirs(tmp_dir, exist_ok=True)
            ext = os.path.splitext(fname)[1]
            tmp_name = f"h3_char_sheet_src_{uuid.uuid4().hex[:12]}{ext}"
            tmp_path = os.path.join(tmp_dir, tmp_name)
            shutil.copy2(out_path, tmp_path)
            load_names.append(os.path.join("fbtools_tmp", tmp_name))
            temp_paths.append(tmp_path)
        else:
            logger.warning("h3_character_sheet: bundle reference image not found on disk: %s", fname)
    return load_names, temp_paths


@routes.post("/fbtools/bundles/generate_character_sheet")
async def _bundles_generate_character_sheet(request):
    """Run the H3 character/face-sheet workflow against up to 9 of a bundle's own reference images.

    Runs server-to-server via this same ComfyUI instance's own /prompt + /history endpoints — no
    browser canvas is touched (see utils/h3_job_runner.py's module docstring for why a global
    progress indicator can still show activity even though no canvas gets node-level rendering for
    this job). See templates/README.md's "h3_character_sheet.api.json" section for the full title
    contract.

    Body: { bundle_id, mode: "character"|"face", refs: [...], outfit_hint? }
      refs — an ORDERED list (capped at 9, the H3 reference-image hard limit) mixing two kinds:
        {kind:"image", index}  — index into the bundle's own SAVED visual.files (resolved here
                                  against the on-disk registry, not any in-memory client state).
        {kind:"frame", file}   — a plain filename already sitting in the ComfyUI input directory
                                  root, e.g. a video-reference frame pulled via
                                  POST /fbtools/media/extract_frame. Never written into the bundle.
      Order matters, not just membership: this template's own prompt treats position 0 as
      "Picture 1", the sole outfit reference — every other position only contributes identity
      (face/hair/build), never clothing. The caller controls order; this route preserves it.
      outfit_hint — optional freeform text substituted into the active mode's own prompt wherever
      its author placed the literal {{OUTFIT_HINT}} placeholder (see
      _h3_char_sheet_prompt_override) — no-op if that mode's prompt node isn't titled or has no
      placeholder in it.
    Returns: { file, folder: "output" } — same shape as /fbtools/backgrounds/remove_people.
    """
    temp_paths: list[str] = []
    try:
        body = await request.json()
        bundle_id = (body.get("bundle_id") or "").strip()
        if not bundle_id:
            return web.json_response({"error": "bundle_id is required"}, status=400)
        mode = (body.get("mode") or "").strip()
        if mode not in _MODE_SELECT:
            return web.json_response(
                {"error": f"mode must be one of {sorted(_MODE_SELECT)}, got {mode!r}"}, status=400
            )

        if not os.path.exists(_H3_CHAR_SHEET_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_H3_CHAR_SHEET_TEMPLATE_PATH}. Export the cleaned-up "
                          f"H3 character-sheet workflow from ComfyUI (Workflow -> Export (API)) and "
                          f"save it there first."},
                status=500,
            )

        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle = registry.get(bundle_id)
        if bundle is None:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)

        visual_files = list(bundle.get("visual", {}).get("files", []))
        raw_files: list = []
        for ref in (body.get("refs") or []):
            if not isinstance(ref, dict):
                continue
            kind = ref.get("kind")
            if kind == "image":
                idx = ref.get("index")
                if isinstance(idx, int) and 0 <= idx < len(visual_files):
                    raw_files.append(visual_files[idx])
            elif kind == "frame":
                fname = ref.get("file")
                if isinstance(fname, str) and fname:
                    raw_files.append(fname)
        raw_files = raw_files[:MAX_REF_IMAGES]
        if not raw_files:
            return web.json_response({"error": "no reference images selected"}, status=400)

        loop = asyncio.get_event_loop()
        load_names, temp_paths = await loop.run_in_executor(None, _resolve_bundle_ref_images, raw_files)
        if not load_names:
            return web.json_response(
                {"error": "none of the bundle's reference images could be found on disk"}, status=404
            )

        template = load_template(_H3_CHAR_SHEET_TEMPLATE_PATH)
        save_node_id = find_node_by_title(template, "OUT:save")
        seed = random.randint(0, 2**32 - 1)
        filename_prefix = f"fbtools/h3_character_sheets/{uuid.uuid4().hex[:12]}"
        settings = _read_composition_settings()
        overrides = _h3_char_sheet_overrides(settings)
        outfit_hint = (body.get("outfit_hint") or "").strip()
        overrides.update(_h3_char_sheet_prompt_override(template, mode, outfit_hint))
        prompt = patch_character_sheet_prompt(
            template,
            ref_images=load_names,
            mode_select=_MODE_SELECT[mode],
            seed=seed,
            filename_prefix=filename_prefix,
            overrides=overrides,
        )

        try:
            # Two-pass (base + upscale) generation plus cold model loading legitimately runs
            # 5-6+ minutes (confirmed via a live run, 2026-09-27) — well past submit_and_wait's
            # 240s default (tuned for the background-plate feature's single pass). A too-short
            # timeout here doesn't cancel the underlying ComfyUI job (it keeps running on the
            # queue regardless), it just makes this route return a premature 502 while the real
            # generation succeeds moments later — so this needs real headroom, not a quick retry.
            result = await submit_and_wait(request, prompt, save_node_id, timeout=900.0)
        finally:
            # VRAM is a machine-wide resource shared with the background-plate feature — reuse its
            # single "Unload model after each run" setting rather than a duplicate per-template one.
            if settings.get("h3_bg_plate_unload_after_run"):
                await free_vram(request)

        subfolder = result.get("subfolder", "")
        out_filename = result.get("filename", "")
        out_file = f"{subfolder}/{out_filename}" if subfolder else out_filename
        return web.json_response({"file": out_file, "folder": "output"})
    except H3JobError as exc:
        logger.error("bundles generate_character_sheet job error: %s", exc)
        return web.json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.error("bundles generate_character_sheet error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        for tp in temp_paths:
            if os.path.exists(tp):
                try:
                    os.remove(tp)
                except OSError as exc:
                    logger.warning(
                        "generate_character_sheet: could not clean up temp file %s: %s", tp, exc
                    )


@routes.get("/fbtools/bundles/character_sheet_settings_options")
async def _bundles_character_sheet_settings_options(request):
    """Enumeration lists + capability flags for the Settings > H3 Character Sheet section.

    Each has_*_override flag tells the UI whether templates/h3_character_sheet.api.json currently
    exposes that optional override title at all — the corresponding Settings control should be
    disabled (not just left inert) when its flag is false, so a user can never configure an override
    that would silently do nothing. Same pattern as
    nodes/backgrounds_presets.py::_backgrounds_h3_settings_options.
    """
    try:
        if not os.path.exists(_H3_CHAR_SHEET_TEMPLATE_PATH):
            return web.json_response(
                {"error": f"Template not found: {_H3_CHAR_SHEET_TEMPLATE_PATH}"}, status=404
            )
        template = load_template(_H3_CHAR_SHEET_TEMPLATE_PATH)

        import folder_paths
        import comfy.samplers
        from comfy_extras.nodes_resolution import AspectRatio

        def _has(title: str) -> bool:
            return find_node_by_title(template, title, required=False) is not None

        def _field(title: str, field: str, default=None):
            node_id = find_node_by_title(template, title, required=False)
            if node_id is None:
                return default
            return template[node_id].get("inputs", {}).get(field, default)

        return web.json_response({
            "models":     sorted(folder_paths.get_filename_list("diffusion_models")),
            "clips":      sorted(folder_paths.get_filename_list("text_encoders")),
            "samplers":       list(comfy.samplers.SAMPLER_NAMES),
            "schedulers":     list(comfy.samplers.SCHEDULER_NAMES),
            "aspect_ratios":  [v.value for v in AspectRatio],
            "has_model_override":                _has("IN:model"),
            "has_clip_override":                 _has("IN:clip"),
            "has_lora_override":                 _has("IN:lora"),
            "has_sampler1_override":             _has("IN:sampler1"),
            "has_scheduler1_override":           _has("IN:scheduler1"),
            "has_sampler2_override":             _has("IN:sampler2"),
            "has_upscale_steps_select_override": _has("IN:upscale_steps_select"),
            "has_upscale_factor_override":       _has("IN:upscale_factor"),
            "has_aspect_ratio_override":         _has("IN:aspect_ratio"),
            "has_character_prompt_override":     _has("IN:character_prompt"),
            "has_face_prompt_override":          _has("IN:face_prompt"),
            "template_defaults": {
                "model":                _field("IN:model", "unet_name"),
                "clip":                 _field("IN:clip", "clip_name"),
                "sampler1":             _field("IN:sampler1", "sampler_name"),
                "scheduler1":           _field("IN:scheduler1", "scheduler"),
                "sampler2":             _field("IN:sampler2", "sampler_name"),
                "upscale_steps_select": _field("IN:upscale_steps_select", "select"),
                "upscale_factor":       _field("IN:upscale_factor", "mode.scale"),
                "aspect_ratio":         _field("IN:aspect_ratio", "aspect_ratio"),
                "megapixels":           _field("IN:aspect_ratio", "megapixels"),
                "lora":                 _field("IN:lora", "lora_0"),
                "lora_strength":        _field("IN:lora", "strength_model_0"),
            },
        })
    except Exception as exc:
        logger.error("bundles character_sheet_settings_options error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
