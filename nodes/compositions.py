"""Composition-assembly nodes: CompositionLoad, PromptCompositionLoader,
CompositionToH3Conditioning, plus /fbtools/compositions/* REST routes and settings.

Moved out of extension.py (Plan 29, pure code motion) -- the last and largest piece of the
extension.py -> nodes/ package-split series. _bundle_proxy_eligible/_composition_settings_path
family/_h3_resolve_path/_h3_load_audio come from nodes/composition_shared.py (this file and
nodes/bundles.py each need something the other owns, so neither is a leaf relative to the other).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
from collections import deque

import numpy as np
import torch
from PIL import Image
from aiohttp import web
from comfy_api.latest import io
from folder_paths import get_input_directory, get_output_directory

from .shared import (
    routes, prefixed_node_id, send_status_update, user_data_dir, reload_counter,
    default_bundle_registry_path, default_subject_profiles_path, default_source_profiles_path,
    default_outfit_registry_path, bump_reload,
)
from .composition_types import CompositionIOType, CastIOType, H3RefplanType
from .composition_shared import (
    _bundle_proxy_eligible, _BUNDLE_PROXY_SHORT_EDGE, _composition_settings_path,
    _read_composition_settings, _write_composition_settings, _h3_resolve_path, _h3_load_audio,
)
from .subjects import _load_subject_images
from .libber import Libber, LibberStateManager
from .run_tracking import register_track_formatter
from .lora_stacks import LoraStackData
from ..utils.logging_utils import get_logger
from ..utils.subject_profiles import load_registry as _load_subject_registry
from ..utils.source_profiles import load_registry as _load_source_registry
from ..utils.reference_bundles import load_registry as _load_bundle_registry
from ..utils.outfit_registry import load_outfit_registry as _load_outfit_registry
from ..utils.proxy_cache import ensure_bundle_video_proxy as _ensure_bundle_proxy
from ..utils.h3_vram_estimator import tokens_for as h3_tokens_for, max_safe_scale as h3_max_safe_scale
from ..utils.composition_track_summary import summarize_scene_cast, summarize_loras, summarize_composition_meta
from ..utils.composition_resources import load_backgrounds as _load_backgrounds_dict
from ..utils.prompt_compositions import (
    list_compositions as _list_compositions,
    load_composition as _load_composition,
    save_composition as _save_composition,
    delete_composition as _delete_composition,
    resolve_subjects as _resolve_composition_subjects,
    resolve_background as _resolve_composition_background,
    validate_composition as _validate_composition,
    apply_cast_to_subjects as _apply_cast_to_subjects,
    apply_composition_overrides as _apply_composition_overrides,
)
from ..utils.prompt_assembler import (
    assemble_composition as _assemble_composition,
    _build_h3_refplan,
    estimate_speech_duration as _estimate_speech_duration,
    _PACE_CHARS_PER_SEC,
    MODEL_TYPES as _PROMPT_MODEL_TYPES,
    validate_h3_refs_pre as _validate_h3_refs_pre,
    validate_h3_audio_clip as _validate_h3_audio_clip,
    validate_h3_audio_total as _validate_h3_audio_total,
)

logger = get_logger(__name__)


# ── Prompt Composition routes ─────────────────────────────────────────────────

@routes.get("/fbtools/compositions/list")
async def _compositions_list(request):
    try:
        items = _list_compositions(user_data_dir())
        return web.json_response({"compositions": items})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/compositions/get")
async def _compositions_get(request):
    cid = request.rel_url.query.get("id", "")
    if not cid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        composition = _load_composition(user_data_dir(), cid)
        return web.json_response(composition)
    except FileNotFoundError:
        return web.json_response({"error": f"Composition '{cid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/save")
async def _compositions_save(request):
    try:
        composition = await request.json()
        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        saved = _save_composition(user_data_dir(), composition, registry, backgrounds)
        return web.json_response({"success": True, "id": saved["id"]})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/compositions/delete")
async def _compositions_delete(request):
    cid = request.rel_url.query.get("id", "")
    if not cid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        _delete_composition(user_data_dir(), cid)
        return web.json_response({"success": True})
    except FileNotFoundError:
        return web.json_response({"error": f"Composition '{cid}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/assemble")
async def _compositions_assemble(request):
    """Assemble a prompt from a composition.

    Body: {"scene_id": "...", "model_type": "..."} or
          {"composition": {...}, "model_type": "..."}  (inline, unsaved)
    """
    try:
        body = await request.json()
        model_type = body.get("model_type", "h3_ref2va")

        if "scene_id" in body:
            composition = _load_composition(user_data_dir(), body["scene_id"])
        elif "composition" in body:
            composition = body["composition"]
        else:
            return web.json_response({"error": "scene_id or composition required"}, status=400)

        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        outfit_reg = _load_outfit_registry(default_outfit_registry_path())

        resolved_subjects = _resolve_composition_subjects(composition, registry)
        resolved_background = _resolve_composition_background(composition, backgrounds)

        outfit_ids = composition.get("outfit_ids", {})
        resolved_outfits = {
            sk: outfit_reg.get_outfit(oid)
            for sk, oid in outfit_ids.items()
            if oid and outfit_reg.get_outfit(oid)
        }

        warnings = _validate_composition(composition)
        result = _assemble_composition(
            composition, resolved_subjects, resolved_background, model_type,
            resolved_outfits=resolved_outfits,
        )
        result["warnings"] = warnings
        return web.json_response(result)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Composition assembly failed")
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/compositions/reload")
async def _compositions_reload(request):
    """Increment reload counter so PromptCompositionLoader nodes re-execute."""
    _composition_reload_counter = bump_reload("composition")
    logger.info("Composition reload requested (counter=%d)", _composition_reload_counter)
    return web.json_response({"success": True, "counter": _composition_reload_counter})


# _composition_settings_path/_COMPOSITION_SETTINGS_DEFAULTS/_read_composition_settings/
# _write_composition_settings moved to nodes/composition_shared.py (Plan 29)


@routes.get("/fbtools/compositions/settings")
async def _compositions_settings_get(request):
    """Return global composition system settings."""
    return web.json_response(_read_composition_settings())


@routes.post("/fbtools/compositions/settings")
async def _compositions_settings_post(request):
    """Update global composition system settings."""
    try:
        body = await request.json()
        settings = _read_composition_settings()

        if "libber_delimiter" in body:
            d = str(body["libber_delimiter"])
            if len(d) == 1:
                settings["libber_delimiter"] = d

        if "libber_max_depth" in body:
            v = int(body["libber_max_depth"])
            settings["libber_max_depth"] = max(1, min(50, v))

        if "default_speech_pace" in body:
            pace = str(body["default_speech_pace"])
            if pace in ("slow", "normal", "fast"):
                settings["default_speech_pace"] = pace

        if "default_audio_noise_removal" in body:
            settings["default_audio_noise_removal"] = bool(body["default_audio_noise_removal"])

        if "default_audio_normalize_lufs" in body:
            settings["default_audio_normalize_lufs"] = bool(body["default_audio_normalize_lufs"])

        if "default_audio_target_lufs" in body:
            v = float(body["default_audio_target_lufs"])
            settings["default_audio_target_lufs"] = max(-36.0, min(-6.0, v))

        if "melband_model_path" in body:
            settings["melband_model_path"] = str(body["melband_model_path"]).strip()

        if "h3_max_frames" in body:
            v = int(body["h3_max_frames"])
            settings["h3_max_frames"] = max(0, min(9999, v))

        for _key in ("h3_bg_plate_model", "h3_bg_plate_clip", "h3_bg_plate_lora",
                     "h3_bg_plate_sampler", "h3_bg_plate_scheduler"):
            if _key in body:
                settings[_key] = str(body[_key]).strip()

        if "h3_bg_plate_lora_strength" in body:
            v = float(body["h3_bg_plate_lora_strength"])
            settings["h3_bg_plate_lora_strength"] = max(0.0, min(2.0, v))

        if "h3_bg_plate_steps" in body:
            v = int(body["h3_bg_plate_steps"])
            settings["h3_bg_plate_steps"] = 0 if v <= 0 else max(1, min(10000, v))

        _write_composition_settings(settings)
        return web.json_response(settings)
    except Exception as exc:
        logger.exception("Error saving composition settings")
        return web.json_response({"error": str(exc)}, status=500)


# LLM assistant / Unsloth-Modal backend routing left in extension.py (out of scope for
# Plan 29 -- not composition-assembly, and nodes/llm_assistant.py already owns this domain's
# routes; this stretch just happened to sit physically between two things this plan needed).


# ── Node: PromptCompositionLoader ─────────────────────────────────────────────

_COMP_MODEL_TYPE_OPTIONS = ["composition default"] + list(_PROMPT_MODEL_TYPES)


def _composition_get_names() -> list[str]:
    """Return saved composition names for the combo widget, sorted by name."""
    try:
        items = _list_compositions(user_data_dir())
        names = [c["name"] for c in items]
        return names if names else ["(none)"]
    except Exception:
        return ["(none)"]


# CompositionIOType moved to nodes/composition_types.py (Plan 27)
class CompositionLoad(io.ComfyNode):
    """Load a saved Prompt Composition and expose it for wiring into SceneCastBuild.

    Mirrors SourceProfileLoad: the composition's subjects become the cast pool
    in SceneCastBuild, and changing the selection here refreshes it downstream.
    Assembling a prompt is still PromptCompositionLoader's job.
    """
    node_id = prefixed_node_id("CompositionLoad")
    display_name = "Composition Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "composition_name",
                    options=_composition_get_names(),
                    display_name="Composition",
                    tooltip="Saved Prompt Composition to load. Press R to refresh the list after adding new compositions.",
                ),
            ],
            outputs=[
                CompositionIOType.Output(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip="Full composition dict for wiring into SceneCastBuild.",
                ),
                io.String.Output(
                    "subject_info",
                    display_name="Subject Info",
                    tooltip="Slot letter, subject and pronoun style for each subject in this composition.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, composition_name: str = "", **_):
        comps_dir = os.path.join(user_data_dir(), "prompt_compositions")
        try:
            dir_mtime = os.path.getmtime(comps_dir)
        except OSError:
            dir_mtime = 0
        file_mtime = 0
        if composition_name:
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == composition_name), None)
            if matched:
                try:
                    file_mtime = os.path.getmtime(os.path.join(comps_dir, f"{matched['id']}.json"))
                except OSError:
                    pass
        return (comps_dir, composition_name, dir_mtime, file_mtime, reload_counter("composition"))

    @classmethod
    def execute(cls, composition_name: str = "") -> io.NodeOutput:
        matched = None
        if composition_name and composition_name != "(none)":
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == composition_name), None)
        if matched is None:
            logger.warning("CompositionLoad: composition %r not found", composition_name)
            return io.NodeOutput(None, "")

        composition = _load_composition(user_data_dir(), matched["id"])
        registry = _load_subject_registry(default_subject_profiles_path())
        lines = []
        for slot, sid in composition.get("subjects", {}).items():
            subj = registry.get_subject(sid) if sid else None
            name = (subj or {}).get("name") or sid or "(empty)"
            pronoun = (subj or {}).get("pronoun_style", "")
            lines.append(f"{slot}: {name}" + (f" [{pronoun}]" if pronoun else ""))
        subject_info = "\n".join(lines) if lines else "(no subjects defined)"

        send_status_update(cls.node_id, f"Loaded: {composition_name} | {len(lines)} subject(s)")
        return io.NodeOutput(composition, subject_info)


def _resolve_cast_media(
    scene_cast: dict,
    bundle_registry,
) -> "dict":
    """Extract reference media and frame-sampling params from a scene cast dict.

    Iterates entries in cast order.  Returns a dict with:
      reference_video    – absolute path of first video-mode entry's file, or "".
      reference_images   – [N,H,W,3] tensor from all image-mode entries, or None.
      video_force_rate   – fps override for the visual Load Video node.
      video_frame_cap    – frame cap for the visual Load Video node.
      video_skip_first   – skip-first-frames for the visual Load Video node.
      video_every_nth    – select-every-nth for the visual Load Video node.
      audio_source       – "extract_from_visual" | "extract_from_video" | "file" | "none".
      audio_file         – absolute path to audio file (source=="file"), or "".
      audio_force_rate   – fps override for the audio Load Video node (extract_from_visual).
      audio_frame_cap    – frame cap for the audio Load Video node.
      audio_skip_first   – skip-first-frames for the audio Load Video node.
      audio_every_nth    – select-every-nth for the audio Load Video node.
      audio_start_time   – start time in seconds for Load Audio (source=="file").
      audio_duration     – duration in seconds for Load Audio; 0 = to end.
      video_entries_full – list of all video-mode entry descriptors (for refplan):
                           [{subject_id, video_file, load_params, audio_source,
                             audio_path, audio_start_time, audio_duration,
                             audio_retention, audio_role}, …]
    """
    entries = scene_cast.get("entries", [])
    reference_video = ""
    image_files: list[str] = []
    video_params: dict = {"force_rate": 24, "frame_load_cap": 96, "skip_first_frames": 0, "select_every_nth": 1}
    audio_source = "none"
    audio_file = ""
    audio_params: dict = {"force_rate": 0, "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}
    audio_time: dict = {"start_time": 0.0, "duration": 0.0}
    video_entries_full: list[dict] = []

    input_dir = get_input_directory()

    for entry in entries:
        bundle_id = entry.get("bundle_id", "")
        visual_mode = entry.get("visual_mode", "images")
        subject_id = entry.get("subject_id", "")
        if not bundle_id:
            continue
        bundle = bundle_registry.get(bundle_id)
        if bundle is None:
            logger.debug("_resolve_cast_media: bundle %r not found", bundle_id)
            continue
        visual = bundle.get("visual", {})
        audio = bundle.get("audio", {})

        want_video  = visual_mode in ("video", "both")
        want_images = visual_mode != "video"

        if want_video:
            vfile = visual.get("file", "")
            if vfile:
                vdir = visual.get("video_dir", "input")
                abs_vfile = os.path.join(
                    get_output_directory() if vdir == "output" else input_dir,
                    vfile,
                )
                entry_load_params = {
                    "start_time":        float(visual.get("start_time", 0.0)),
                    "duration":          float(visual.get("duration",   0.0)),
                    "force_rate":        visual.get("force_rate", 24),  # H3 requires 24fps
                    "frame_load_cap":    visual.get("frame_load_cap", 96),
                    "skip_first_frames": visual.get("skip_first_frames", 0),
                    "select_every_nth":  visual.get("select_every_nth", 1),
                }
                # Legacy flat output: first video only
                if not reference_video:
                    reference_video = abs_vfile
                    video_params = entry_load_params

                # Determine audio config for this video entry
                a_src = audio.get("source", "none")
                entry_audio_source = "none"
                entry_audio_path   = ""
                entry_audio_start  = 0.0
                entry_audio_dur    = 0.0
                if a_src == "extract_from_visual":
                    entry_audio_source = "extract_from_visual"
                    entry_audio_path   = abs_vfile
                    # Convert frame-based params to time for ffmpeg seeking.
                    # start_time/duration in the audio bundle are for "file" source only.
                    a_fps  = audio.get("force_rate", 0) or visual.get("force_rate", 0) or 0
                    a_skip = audio.get("skip_first_frames", 0)
                    a_cap  = audio.get("frame_load_cap", 0)
                    entry_audio_start = (a_skip / a_fps) if (a_fps > 0 and a_skip > 0) else 0.0
                    entry_audio_dur   = (a_cap  / a_fps) if (a_fps > 0 and a_cap  > 0) else 0.0
                elif a_src == "extract_from_video":
                    av_file = audio.get("video_file", "")
                    if av_file:
                        av_dir = audio.get("video_dir", "input")
                        entry_audio_source = "extract_from_visual"
                        entry_audio_path   = os.path.join(
                            get_output_directory() if av_dir == "output" else input_dir,
                            av_file,
                        )
                        entry_audio_start  = audio.get("start_time", 0.0)
                        entry_audio_dur    = audio.get("duration",   0.0)
                elif a_src == "file":
                    af = audio.get("file", "")
                    if af:
                        af_dir = audio.get("audio_dir", "input")
                        entry_audio_source = "file"
                        entry_audio_path   = os.path.join(
                            get_output_directory() if af_dir == "output" else input_dir,
                            af,
                        )
                        entry_audio_start  = audio.get("start_time", 0.0)
                        entry_audio_dur    = audio.get("duration", 0.0)

                # Generation-time proxy fallback (Plan 16): if a fresh proxy already exists this
                # is a near-instant filesystem check; if not, this builds one on the spot. Audio
                # extraction above always keeps using abs_vfile (the real original) — proxies are
                # silent (-an), never a valid audio source. Never blocks generation on failure.
                video_file_for_entry = vfile
                if _bundle_proxy_eligible(entry_load_params["force_rate"], entry_load_params["duration"]):
                    try:
                        proxy_path = _ensure_bundle_proxy(
                            source_path=abs_vfile,
                            bundle_id=bundle_id,
                            start_time=entry_load_params["start_time"],
                            end_time=entry_load_params["start_time"] + entry_load_params["duration"],
                            short_edge=_BUNDLE_PROXY_SHORT_EDGE,
                            base_dir=str(user_data_dir()),
                        )
                    except Exception as _proxy_exc:
                        proxy_path = None
                        logger.warning(
                            "_resolve_cast_media: proxy generation failed for bundle %r (%s): %s",
                            bundle_id, os.path.basename(abs_vfile), _proxy_exc,
                        )
                    if proxy_path:
                        video_file_for_entry = proxy_path
                        entry_load_params = dict(entry_load_params)
                        entry_load_params["start_time"] = 0.0  # proxy is already trimmed

                video_entries_full.append({
                    "subject_id":       subject_id,
                    "video_file":       video_file_for_entry,
                    "load_params":      entry_load_params,
                    "audio_source":     entry_audio_source,
                    "audio_path":       entry_audio_path,
                    "audio_start_time": entry_audio_start,
                    "audio_duration":   entry_audio_dur,
                    "audio_retention":  audio.get("retention", "timbre"),
                    "audio_role":       audio.get("role", ""),
                    "audio_cache":      audio.get("audio_cache", ""),
                    # Default excludes the reference video's own background/setting from the
                    # generated output — H3 was observed picking it up in some generations.
                    # Opt-in checkbox in the Scene Cast Build tab ("Keep BG").
                    "include_video_background": bool(entry.get("include_video_background", False)),
                })

        if want_images:
            raw_files = visual.get("files", [])
            # image_selection: None=all, list[int]=specific indices, int=legacy single
            img_sel = entry.get("image_selection")
            if img_sel is not None:
                if isinstance(img_sel, list):
                    raw_files = [raw_files[i] for i in img_sel if isinstance(i, int) and 0 <= i < len(raw_files)]
                else:
                    try:
                        idx = int(img_sel)
                        raw_files = [raw_files[idx]] if 0 <= idx < len(raw_files) else []
                    except (TypeError, ValueError):
                        pass
            image_files.extend(raw_files)

        # Legacy flat audio: first non-"none" source across all entries
        if audio_source == "none":
            a_src = audio.get("source", "none")
            if a_src == "extract_from_visual" and want_video and visual.get("file"):
                audio_source = a_src
                audio_params = {
                    "force_rate":        audio.get("force_rate", 0),
                    "frame_load_cap":    audio.get("frame_load_cap", 0),
                    "skip_first_frames": audio.get("skip_first_frames", 0),
                    "select_every_nth":  audio.get("select_every_nth", 1),
                }
                a_fps  = audio.get("force_rate", 0) or visual.get("force_rate", 0) or 0
                a_skip = audio.get("skip_first_frames", 0)
                a_cap  = audio.get("frame_load_cap", 0)
                audio_time = {
                    "start_time": (a_skip / a_fps) if (a_fps > 0 and a_skip > 0) else 0.0,
                    "duration":   (a_cap  / a_fps) if (a_fps > 0 and a_cap  > 0) else 0.0,
                }
            elif a_src == "extract_from_video":
                av_file = audio.get("video_file", "")
                if av_file:
                    audio_source = "extract_from_video"
                    audio_file   = os.path.join(input_dir, av_file)
                    audio_params = {
                        "force_rate":        audio.get("force_rate", 0),
                        "frame_load_cap":    audio.get("frame_load_cap", 0),
                        "skip_first_frames": audio.get("skip_first_frames", 0),
                        "select_every_nth":  audio.get("select_every_nth", 1),
                    }
                    audio_time = {
                        "start_time": audio.get("start_time", 0.0),
                        "duration":   audio.get("duration",   0.0),
                    }
            elif a_src == "file":
                a_file = audio.get("file", "")
                if a_file:
                    audio_source = a_src
                    audio_file = os.path.join(input_dir, a_file)
                    audio_time = {
                        "start_time": audio.get("start_time", 0.0),
                        "duration":   audio.get("duration", 0.0),
                    }

    # ── Source-derived entries ─────────────────────────────────────────────────
    # Source profiles are grouped by profile_id; subjects sharing the same
    # profile reference the same source media, so one video_entries_full entry
    # is emitted per unique profile (not per subject).
    source_profiles_data = scene_cast.get("source_profiles", {})
    seen_profile_ids: set = set()

    for entry in entries:
        sp_id = entry.get("source_profile_id", "")
        if not sp_id or sp_id in seen_profile_ids:
            continue
        seen_profile_ids.add(sp_id)

        profile = source_profiles_data.get(sp_id)
        if not profile:
            continue

        media_type = profile.get("media_type", "video")
        if media_type != "video":
            continue  # image-only sources: no video_entries_full entry

        media_file = profile.get("media_filename", "")
        if not media_file:
            continue

        media_dir = profile.get("media_dir", "input")
        abs_file = os.path.join(
            get_output_directory() if media_dir == "output" else input_dir,
            media_file,
        )

        # First source video fills the legacy flat reference_video output
        if not reference_video:
            reference_video = abs_file

        # Only include subject_ids from pure source-only entries (no bundle).
        # Hybrid entries (bundle + source) use the bundle for visual references;
        # including the source profile video would produce a spurious <Video N>
        # even when the user selected images mode for that slot.
        sp_subject_ids = [
            e.get("subject_id") or e.get("source_subject_id", "")
            for e in entries
            if e.get("source_profile_id") == sp_id and not e.get("bundle_id")
        ]
        if not sp_subject_ids:
            # All cast entries for this profile have bundles — skip source video
            continue

        # Resolve clip load_params if a clip_id was specified for this profile
        clip_ids_map = scene_cast.get("clip_ids", {})
        clip_id = clip_ids_map.get(sp_id, "")
        load_params: dict
        if clip_id:
            _sp_path = default_source_profiles_path()
            _sp_reg = _load_source_registry(_sp_path)
            _clip_lp = _sp_reg.clip_load_params(sp_id, clip_id)
            load_params = _clip_lp if _clip_lp else {
                "start_time": 0.0, "duration": 0.0,
                "force_rate": 24, "frame_load_cap": 96,  # H3 requires 24fps
                "skip_first_frames": 0, "select_every_nth": 1,
            }
        else:
            load_params = {
                "start_time": 0.0, "duration": 0.0,
                "force_rate": 24, "frame_load_cap": 96,  # H3 requires 24fps
                "skip_first_frames": 0, "select_every_nth": 1,
            }

        video_entries_full.append({
            "subject_id":        sp_subject_ids[0] if sp_subject_ids else "",
            "subject_ids":       sp_subject_ids,
            "source_profile_id": sp_id,
            "clip_id":           clip_id,
            "video_file":        media_file,
            "load_params":       load_params,
            "audio_source":     "none",
            "audio_path":       "",
            "audio_start_time": 0.0,
            "audio_duration":   0.0,
            "audio_retention":  "timbre",
            "audio_role":       "",
            "audio_cache":      "",
        })

    reference_images = _load_subject_images(image_files) if image_files else None
    return {
        "reference_video":    reference_video,
        "reference_images":   reference_images,
        "video_force_rate":   video_params["force_rate"],
        "video_frame_cap":    video_params["frame_load_cap"],
        "video_skip_first":   video_params["skip_first_frames"],
        "video_every_nth":    video_params["select_every_nth"],
        "audio_source":       audio_source,
        "audio_file":         audio_file,
        "audio_force_rate":   audio_params["force_rate"],
        "audio_frame_cap":    audio_params["frame_load_cap"],
        "audio_skip_first":   audio_params["skip_first_frames"],
        "audio_every_nth":    audio_params["select_every_nth"],
        "audio_start_time":   audio_time["start_time"],
        "audio_duration":     audio_time["duration"],
        "video_entries_full": video_entries_full,
    }


def _apply_composition_libbers(text: str, libbers: list, delimiter: str, manager) -> str:
    """Apply libbers in order to text, resolving references in three passes.

    Pass 0 — %*%        : random value drawn from the combined pool of all attached
                          libbers (sampling without replacement; wraps when exhausted).
    Pass 1 — %key:N%    : named key from the Nth libber (1-based).
             %*:N%      : random value from the Nth libber (sampling without replacement).
    Pass 2 — %key%      : plain ref chained through all libbers in order (first match wins).

    Within a single call every %*:N% (or %*%) draws a DIFFERENT key from that libber's
    shuffled queue.  When all keys have been drawn the queue refills with a new shuffle
    so no key repeats until every other key has been used at least once.
    """
    if not libbers or not text:
        return text

    # Load all referenced libbers; strip .json extension for the state manager key
    loaded: list = []
    for name in libbers:
        key = name[:-5] if name.endswith(".json") else name
        loaded.append(manager.ensure_libber(key))

    esc = re.escape(delimiter)

    # ── Per-libber shuffled queues (for %*:N%) ────────────────────────────────
    _per_queue: dict[int, deque] = {}

    def _libber_keys(idx: int) -> list[str]:
        lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
        return list(lb.libs.keys()) if lb else []

    def _pop_from_libber(idx: int) -> str | None:
        """Pop the next key from libber idx's shuffled deque; refill when empty."""
        if idx not in _per_queue:
            keys = _libber_keys(idx)
            if not keys:
                return None
            random.shuffle(keys)
            _per_queue[idx] = deque(keys)
        q = _per_queue[idx]
        if not q:
            keys = _libber_keys(idx)
            if not keys:
                return None
            random.shuffle(keys)
            q.extend(keys)
        return q.popleft()

    def _resolve_value(lb, key: str) -> str:
        val = lb.get_lib(key)
        if val is None:
            return ""
        temp = Libber(lib_dict=dict(lb.libs), delimiter=delimiter, max_depth=lb.max_depth)
        return temp.substitute(val)

    # ── Combined queue (for %*%) ───────────────────────────────────────────────
    # Each entry is (libber_index, key) — shuffle once across all attached libbers.
    _combined_queue: deque = deque()

    def _fill_combined() -> None:
        pairs = [
            (i, key)
            for i, lb in enumerate(loaded)
            if lb
            for key in lb.libs
        ]
        random.shuffle(pairs)
        _combined_queue.extend(pairs)

    def _pop_combined() -> str:
        if not _combined_queue:
            _fill_combined()
        if not _combined_queue:
            return ""
        idx, key = _combined_queue.popleft()
        lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
        return _resolve_value(lb, key) if lb else ""

    # ── Pass 0: %*% — combined-pool wildcard ──────────────────────────────────
    _fill_combined()
    text = re.sub(re.escape(delimiter) + r'\*' + re.escape(delimiter), lambda _: _pop_combined(), text)

    # ── Pass 1: %key:N% and %*:N% — indexed refs ─────────────────────────────
    def _replace_indexed(m: re.Match) -> str:
        inner = m.group(1)   # e.g. "key:2", "*:1" or "*.1"
        if inner.startswith("*."):
            inner = "*:" + inner[2:]
        raw_key, idx_str = inner.rsplit(":", 1)
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return m.group(0)

        if raw_key == "*":
            key = _pop_from_libber(idx)
            if key is None:
                return ""
            lb = loaded[idx] if 0 <= idx < len(loaded) and loaded[idx] else None
            return _resolve_value(lb, key) if lb else ""

        # Named key from specific libber slot
        if 0 <= idx < len(loaded) and loaded[idx]:
            return _resolve_value(loaded[idx], raw_key) or m.group(0)
        return m.group(0)

    # Accept "%*.N%" as a spelling of "%*:N%" (a common typo that otherwise never resolves).
    indexed_pat = esc + r'((?:[a-z0-9_]+:|\*[:.])[0-9]+)' + esc
    text = re.sub(indexed_pat, _replace_indexed, text)

    # ── Pass 2: plain unindexed refs — chain through libbers in order ─────────
    for lb in loaded:
        if lb:
            temp = Libber(lib_dict=dict(lb.libs), delimiter=delimiter, max_depth=lb.max_depth)
            text = temp.substitute(text)

    return text


class PromptCompositionLoader(io.ComfyNode):
    """Load a saved prompt composition and assemble it into a model-specific prompt.

    Select a composition by name from the dropdown. The assembled prompt and
    concept IDs are ready to wire into text-conditioning and ConceptResolve nodes.

    Optionally wire a SCENE_CAST from SceneCastLoad or SceneCastBuild to resolve
    reference media (video file path and/or image batch) from the cast's bundles.

    After saving a composition in the Prompt Compositions sidebar panel, any
    PromptCompositionLoader nodes on the canvas automatically re-execute to pick
    up the latest content (via the reload counter). A full page refresh is needed
    for brand-new compositions to appear in the name dropdown.
    """

    node_id = prefixed_node_id("PromptCompositionLoader")
    display_name = "Prompt Composition Loader"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        names = _composition_get_names()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.Combo.Input(
                    "composition_name",
                    options=names,
                    display_name="Composition",
                    tooltip="Select a saved composition. Press R to refresh the list after saving new ones.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_COMP_MODEL_TYPE_OPTIONS,
                    display_name="Model Type",
                    tooltip="'composition default' uses the model type stored inside the composition.",
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    default="",
                    tooltip=(
                        "Optional prefix prepended literally to the composition name for the filename_prefix output. "
                        "Recommended: wire this from SceneCastBuild's filename_prefix output for automatic "
                        "primary-subject/bundle/compositions foldering (e.g. 'video/alex/alex_salon_eyes/compositions/' "
                        "-> '.../compositions/bbc_ride'). "
                        "Or type a literal root here for standalone use without a cast "
                        "(e.g. 'video/' -> 'video/bbc_ride'). "
                        "Wire the output into a VHS_VideoCombine filename_prefix input."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Optional cast from SceneCastLoad or SceneCastBuild. Resolves reference media (video path and images) from bundles.",
                    optional=True,
                ),
                CompositionIOType.Input(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip=(
                        "Optional composition from CompositionLoad (or SceneCastBuild's pass-through). "
                        "When connected it drives this node and the Composition dropdown is ignored."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Prompt"),
                io.String.Output("composition_name", display_name="Composition Name"),
                io.String.Output(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="prefix + composition name (e.g. 'video/bbc_ride'). Wire into VHS_VideoCombine filename_prefix.",
                ),
                LoraStackData.Output(
                    "lora_stack_data",
                    display_name="LoRA Stack",
                    tooltip=(
                        "LoRAs attached to this composition as LORA_STACK_DATA. "
                        "Wire into LoraStackApply. Empty if no LoRAs are attached."
                    ),
                ),
                H3RefplanType.Output(
                    "h3_refplan",
                    display_name="H3 Ref Plan",
                    tooltip=(
                        "Ordered reference descriptor bundle (FBTOOLS_H3_REFPLAN). "
                        "Wire into CompositionToH3Conditioning to load media and build conditioning "
                        "without manual VHS/audio-loader wiring."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, composition_name: str = "", model_type: str = "composition default", filename_prefix: str = "", scene_cast=None, prompt_composition=None, **_):
        comps_dir = os.path.join(user_data_dir(), "prompt_compositions")
        try:
            dir_mtime = os.path.getmtime(comps_dir)
        except OSError:
            dir_mtime = 0
        # Also key off the matched composition file's own mtime so that out-of-band
        # JSON edits (editing the file on disk without touching the editor) invalidate
        # the node without needing the reload counter.
        file_mtime = 0
        if composition_name:
            items = _list_compositions(user_data_dir())
            matched = next((c for c in items if c["name"] == composition_name), None)
            if matched:
                cpath = os.path.join(comps_dir, f"{matched['id']}.json")
                try:
                    file_mtime = os.path.getmtime(cpath)
                except OSError:
                    pass
        # A wired composition replaces the dropdown; key off its full content so any edit re-runs.
        pc_digest = ""
        if isinstance(prompt_composition, dict) and prompt_composition:
            pc_digest = hashlib.md5(json.dumps(prompt_composition, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        cast_id = scene_cast.get("id", "") if scene_cast else ""
        cast_modified = scene_cast.get("modified", "") if scene_cast else ""
        try:
            bundle_mtime = os.path.getmtime(default_bundle_registry_path()) if scene_cast else 0
        except OSError:
            bundle_mtime = 0
        try:
            settings_mtime = os.path.getmtime(_composition_settings_path())
        except OSError:
            settings_mtime = 0
        return (comps_dir, composition_name, model_type, dir_mtime, file_mtime,
                reload_counter("composition"), cast_id, cast_modified, bundle_mtime, settings_mtime, pc_digest)

    @classmethod
    def execute(
        cls,
        composition_name: str = "",
        model_type: str = "composition default",
        filename_prefix: str = "",
        scene_cast=None,
        prompt_composition=None,
    ) -> io.NodeOutput:
        if isinstance(prompt_composition, dict) and prompt_composition:
            # Wired composition (CompositionLoad): drives everything; the dropdown is ignored.
            composition = copy.deepcopy(prompt_composition)
            composition_name = composition.get("name") or composition_name
            filename_prefix_out = f"{filename_prefix}{composition_name}"
        else:
            filename_prefix_out = f"{filename_prefix}{composition_name}"

            items = _list_compositions(user_data_dir())
            matched = next((c for c in items if c["name"] == composition_name), None)
            if matched is None:
                logger.warning("PromptCompositionLoader: composition %r not found", composition_name)
                return io.NodeOutput("", composition_name, filename_prefix_out, None, None)

            composition = _load_composition(user_data_dir(), matched["id"])

        model_type_used = (
            composition.get("model_type", "h3_ref2va")
            if model_type == "composition default"
            else model_type
        )

        registry = _load_subject_registry(default_subject_profiles_path())
        backgrounds = _load_backgrounds_dict(user_data_dir())
        outfit_reg = _load_outfit_registry(default_outfit_registry_path())
        # Per-run overrides from SceneCastBuild (background, background-as-reference)
        _cast_overrides = scene_cast.get("composition_overrides") if isinstance(scene_cast, dict) else None
        if _cast_overrides:
            composition, _ov_warnings = _apply_composition_overrides(composition, _cast_overrides, backgrounds)
            for _w in _ov_warnings:
                logger.warning("PromptCompositionLoader: %s", _w)
        resolved_subjects = _resolve_composition_subjects(composition, registry)
        resolved_background = _resolve_composition_background(composition, backgrounds)

        outfit_ids = composition.get("outfit_ids", {})
        resolved_outfits_node = {
            sk: outfit_reg.get_outfit(oid)
            for sk, oid in outfit_ids.items()
            if oid and outfit_reg.get_outfit(oid)
        }

        # Resolve cast: build video_entries for assembler + reference media tensors
        cast_media: dict = {
            "reference_video": "", "reference_images": None,
            "video_force_rate": 24, "video_frame_cap": 16, "video_skip_first": 0, "video_every_nth": 1,
            "audio_source": "none", "audio_file": "",
            "audio_force_rate": 0, "audio_frame_cap": 0, "audio_skip_first": 0, "audio_every_nth": 1,
            "audio_start_time": 0.0, "audio_duration": 0.0,
            "video_entries_full": [],
        }
        if scene_cast:
            bundle_registry = _load_bundle_registry(default_bundle_registry_path())
            cast_media = _resolve_cast_media(scene_cast, bundle_registry)

            # Replace/enrich subjects by cast position, then apply bundle media
            resolved_subjects = _apply_cast_to_subjects(
                resolved_subjects, composition, scene_cast, bundle_registry, registry
            )

        # video_entries_full carries full descriptors (paths, load params, audio config)
        # for all video-mode cast entries; used by both the prompt assembler (for
        # <Video N> labels) and _build_h3_refplan (for the terminal node).
        video_entries: list[dict] = cast_media["video_entries_full"]

        result = _assemble_composition(
            composition, resolved_subjects, resolved_background, model_type_used,
            video_entries, resolved_outfits=resolved_outfits_node,
        )

        prompt = result.get("prompt", "")

        # Apply attached libbers using the composition delimiter from global settings
        libbers_list = composition.get("libbers", [])
        if libbers_list:
            cs = _read_composition_settings()
            prompt = _apply_composition_libbers(
                prompt, libbers_list, cs.get("libber_delimiter", "%"), LibberStateManager.instance()
            )


        # Build LORA_STACK_DATA from composition's attached LoRAs
        lora_stack_data = [
            {
                "lora":           e["name"],
                "strength_model": float(e.get("weight", 1.0)),
                "strength_clip":  float(e.get("weight", 1.0)),
                "enabled":        True,
                "model_target":   e.get("target", "MiniMaxH3"),
                "audio_enabled":  False,
            }
            for e in composition.get("loras", [])
            if e.get("name")
        ] or None

        # Compute per-slot trim_to from shot dialogue durations.
        # Each shot's dialogue text is resolved through Libbers independently —
        # a different random pick than what landed in the assembled prompt, but
        # durations are close enough for trim purposes since Libber values for
        # the same key are authored in similar length ranges.
        # Composition subject-slot keys are already letters (A, B, ...), so the
        # dialogue speaker key IS the slot_trim_to key directly — no mapping needed.
        slot_trim_to: dict[str, float] = {}
        _comp_subjects = composition.get("subjects", {})
        _cs = _read_composition_settings()
        _libbers_l = composition.get("libbers", [])
        _delim = _cs.get("libber_delimiter", "%")
        for _shot in composition.get("shots", []):
            _dlg = _shot.get("dialogue")
            if not _dlg or not _dlg.get("text"):
                continue
            _raw = _dlg["text"]
            _resolved = (
                _apply_composition_libbers(_raw, _libbers_l, _delim, LibberStateManager.instance())
                if _libbers_l else _raw
            )
            _cps = _PACE_CHARS_PER_SEC.get(_dlg.get("speech_pace") or "normal", 13.0)
            _dur = _estimate_speech_duration(_resolved, _cps)
            _letter = _dlg.get("speaker", "")
            if _letter and _letter in _comp_subjects:
                slot_trim_to[_letter] = slot_trim_to.get(_letter, 0.0) + _dur

        # Build FBTOOLS_H3_REFPLAN from enriched subjects + full video descriptors
        # Use the scene_instance the prompt was assembled from: it carries the minted
        # background / outfit-reference / bundle slots, so ref-plan picture ordinals match
        # the <Subject N> labels in the prompt. (Fallback keeps older assemblers working.)
        scene_instance_for_plan = result.get("scene_instance") or {
            "slot_assignments": resolved_subjects,
            "outfit_overrides": composition.get("outfit_overrides", {}),
        }
        h3_refplan = _build_h3_refplan(scene_instance_for_plan, video_entries, slot_trim_to or None)
        h3_refplan["prompt"]          = prompt
        h3_refplan["model_type"]      = model_type_used
        h3_refplan["ref_image_size"]  = "match"
        h3_refplan["has_turbo_lora"]  = any(
            "turbo" in (e.get("name", "") or "").lower()
            for e in composition.get("loras", [])
            if e.get("name")
        )

        cast_note = f" | cast: {scene_cast.get('name', '?')}" if scene_cast else ""
        send_status_update(
            cls.node_id,
            f"Loaded: {composition_name} | {model_type_used} | {len(prompt)} chars{cast_note}",
        )

        comp_name_out = composition.get("name", composition_name)
        filename_prefix_out = f"{filename_prefix}{comp_name_out}"

        # Reference media, audio timing, model type and concept IDs are no longer separate
        # outputs: the media travels in h3_refplan, and Run History shows the rest.
        return io.NodeOutput(prompt, comp_name_out, filename_prefix_out, lora_stack_data, h3_refplan)


# =============================================================================
# CompositionToH3Conditioning — media-loading terminal node
# =============================================================================

# _h3_resolve_path moved to nodes/composition_shared.py (Plan 29)


def _h3_load_image(path: str):
    """Load an image file and return a [1,H,W,3] float32 tensor in [0,1]."""
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: image not found: %s", path)
        return None
    try:
        pil = Image.open(resolved).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0   # [H,W,3]
        return torch.from_numpy(arr).unsqueeze(0)        # [1,H,W,3]
    except Exception as exc:
        logger.warning("CompositionToH3: failed to load image %s: %s", path, exc)
        return None


def _h3_load_video_frames(path: str, load_params: dict):
    """Load video frames → [B,H,W,3] float32 in [0,1] (RGB) using cv2.

    VHS (VideoHelperSuite) is intentionally not used here.  VHS loads its own
    sub-package via relative imports, which caches modules under its parent
    package key in sys.modules.  An absolute ``from videohelpersuite.load_video_nodes``
    import looks for a different key and fails because ComfyUI adds
    ``custom_nodes/`` — not ``custom_nodes/ComfyUI-VideoHelperSuite/`` — to
    sys.path.  cv2 is always available (it is a direct dependency of both
    ComfyUI and VHS) and produces identical output.
    """
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: video not found: %s", path)
        return None

    start_time        = float(load_params.get("start_time",        0.0))
    duration          = float(load_params.get("duration",          0.0))
    force_rate        = int(load_params.get("force_rate",          0))
    frame_load_cap    = int(load_params.get("frame_load_cap",      0))
    skip_first_frames = int(load_params.get("skip_first_frames",   0))
    select_every_nth  = int(load_params.get("select_every_nth",    1)) or 1

    # start_time (time-based seek) and skip_first_frames (legacy frame-count seek)
    # are two alternative mechanisms for the same purpose.  When start_time is set,
    # suppress skip_first_frames to avoid double-offsetting into old bundles that
    # had skip_first_frames set before the trim slider was introduced.
    if start_time > 0.0 and skip_first_frames > 0:
        logger.warning(
            "CompositionToH3: both start_time=%.2fs and skip_first_frames=%d are set "
            "for %s — skip_first_frames ignored when start_time is used. "
            "Clear skip_first_frames in the bundle to suppress this warning.",
            start_time, skip_first_frames, os.path.basename(path),
        )
        skip_first_frames = 0

    try:
        import cv2
    except ImportError:
        logger.error("CompositionToH3: cv2 not available; cannot load video %s", path)
        return None
    try:
        cap = cv2.VideoCapture(resolved)
        if not cap.isOpened():
            logger.warning("CompositionToH3: cv2 cannot open %s", path)
            return None
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        target_fps = float(force_rate) if force_rate > 0 else native_fps
        base_frame_time   = 1.0 / native_fps
        target_frame_time = 1.0 / target_fps

        # Time-based start: seek directly to start_time using cv2.
        # This is preferred over skip_first_frames for large offsets and is
        # what users naturally specify (seconds, not frame counts).
        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

        # Time-based duration: convert to an output-frame cap so the accumulator
        # loop can stop without tracking wall-clock time on every frame.
        # `duration` wins over frame_load_cap when both are set. `frame_load_cap` is compared
        # against `sampled` below (17441-17443), which counts frames *after* the select_every_nth
        # filter — so the cap must be expressed in post-filter frames too, or it silently reads
        # select_every_nth times too much of the source (e.g. select_every_nth=2 → exactly 2x the
        # requested duration is decoded before the cap is hit).
        if duration > 0.0:
            duration_cap = max(1, int(duration * target_fps / select_every_nth))
            frame_load_cap = duration_cap if frame_load_cap == 0 else min(frame_load_cap, duration_cap)

        # Time-accumulator resampling — mirrors VHS cv_frame_generator logic:
        # read native frames until the virtual clock reaches the next output slot,
        # then emit the most recently decoded frame for that slot.  Starting
        # time_offset at target_frame_time means the first native frame is
        # immediately emitted without needing extra reads.
        frames: list = []
        current_bgr = None
        time_offset = target_frame_time
        total_count = 0   # virtual frames seen (drives skip_first_frames)
        evaluated   = -1  # frames after skip (drives select_every_nth)
        sampled     = 0   # frames actually appended

        # Pre-read the first native frame (VHS does an initial grab() before its loop).
        ret, current_bgr = cap.read()
        if not ret:
            cap.release()
            logger.warning("CompositionToH3: cv2 loaded no frames from %s", path)
            return None

        while cap.isOpened():
            # Advance native frames until the virtual clock reaches the next slot.
            if time_offset < target_frame_time:
                ret, bgr = cap.read()
                if not ret:
                    break
                current_bgr = bgr
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue

            time_offset -= target_frame_time
            total_count += 1
            if total_count <= skip_first_frames:
                continue
            evaluated += 1
            if evaluated % select_every_nth != 0:
                continue

            rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
            sampled += 1
            if frame_load_cap > 0 and sampled >= frame_load_cap:
                break

        cap.release()
        if not frames:
            logger.warning("CompositionToH3: cv2 loaded no frames from %s", path)
            return None

        # Ping-pong to the nearest valid H3 frame count (17k+5: 5, 22, 39, …).
        # MiniMaxH3ReferenceToVideo trims DOWN to 17k+5 and errors below 5, so
        # whenever the loaded count is not already a valid number we ping-pong up
        # to the next valid count — never further.  This avoids both the model
        # error and unnecessary frame trimming while keeping fabricated motion
        # to the minimum needed.
        n = len(frames)
        _k = max(1, (n - 5 + 16) // 17)   # ceiling: smallest k with 17k+5 >= n, floor k=1 (22 frames)
        target = 17 * _k + 5
        if n < target:
            logger.warning(
                "CompositionToH3: %d frame(s) from %s is not a valid H3 count "
                "(17k+5) — ping-pong looping to %d.",
                n, os.path.basename(path), target,
            )
            # Single frame: repeat.  Multi-frame: bounce (omit endpoints from
            # the reversed half so they are not duplicated at each turn).
            cycle = frames if n == 1 else frames + list(reversed(frames[1:-1]))
            result: list = []
            i = 0
            while len(result) < target:
                result.append(cycle[i % len(cycle)])
                i += 1
            frames = result

        logger.debug(
            "CompositionToH3: loaded %d frames from %s (start=%.2fs, dur=%.2fs, cap=%d, skip=%d)",
            len(frames), path, start_time, duration, frame_load_cap, skip_first_frames,
        )
        return torch.stack(frames, dim=0)  # [B,H,W,3]
    except Exception as exc:
        logger.warning("CompositionToH3: cv2 frame load failed for %s: %s", path, exc)
        return None


# _h3_load_audio moved to nodes/composition_shared.py (Plan 29)


def _track_format_prompt_composition_loader(kwargs: dict):
    """Run History rows for a tracked Prompt Composition Loader. Covers what the node no
    longer exposes as outputs (model type used, concept IDs, reference media and audio,
    via the cast rows) plus its LoRAs. See utils/composition_track_summary.py."""
    rows: dict = {}
    consumed: set = set()
    composition = None
    wired = kwargs.get("prompt_composition")
    if isinstance(wired, dict) and wired:
        # The wired composition drives the node; the dropdown value is stale/ignored.
        consumed.add("prompt_composition")
        composition = wired
        rows["Composition (wired)"] = str(wired.get("name") or wired.get("id") or "?")
    else:
        name = kwargs.get("composition_name")
        if name:
            matched = next((c for c in _list_compositions(user_data_dir()) if c["name"] == name), None)
            if matched:
                composition = _load_composition(user_data_dir(), matched["id"])

    if composition is not None:
        subject_lookup = _load_subject_registry(default_subject_profiles_path()).get_subject
        rows.update(summarize_composition_meta(composition, kwargs.get("model_type", ""), subject_lookup))

    scene_cast = kwargs.get("scene_cast")
    if composition is not None:
        # Effective background (after any SceneCastBuild override) so a run's setting is recoverable.
        overrides = scene_cast.get("composition_overrides") if isinstance(scene_cast, dict) else None
        bgs = _load_backgrounds_dict(user_data_dir())
        eff, _ = _apply_composition_overrides(composition, overrides, bgs)
        bg = _resolve_composition_background(eff, bgs)
        rows["Background"] = (
            (bg.get("name") or eff.get("background") or "?") if bg else "(none)"
        ) + (" (override)" if overrides and "background" in overrides else "")
        as_ref = bool(eff.get("background_as_reference"))
        has_refs = bool(bg and bg.get("reference_images"))
        rows["Background as reference"] = (
            ("yes" if as_ref else "no")
            + ("" if not as_ref or has_refs else " (no reference images, text only)")
            + (" (override)" if overrides and "background_as_reference" in overrides else "")
        )
        if overrides and overrides.get("background_soundscape"):
            rows["Soundscape"] = "from the background (override)"
    if isinstance(scene_cast, dict):
        bundle_reg = _load_bundle_registry(default_bundle_registry_path())
        rows.update(summarize_scene_cast(scene_cast, bundle_reg.get))
        consumed.add("scene_cast")

    if composition is not None:
        loras = summarize_loras(composition.get("loras", []))
        if loras:
            rows["LoRAs"] = loras
    return consumed, rows


register_track_formatter(prefixed_node_id("PromptCompositionLoader"), _track_format_prompt_composition_loader)


class CompositionToH3Conditioning(io.ComfyNode):
    """Terminal node: consume FBTOOLS_H3_REFPLAN, decode all media internally,
    and delegate to MiniMaxH3ReferenceToVideo to produce conditioning + latent.

    Common-case graph: PromptCompositionLoader → CompositionToH3Conditioning → sampler.
    """

    node_id = prefixed_node_id("CompositionToH3Conditioning")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name="Composition → H3 Conditioning",
            category="🧊 frost-byte/conditioning",
            description=(
                "Decodes all references in an FBTOOLS_H3_REFPLAN bundle "
                "(images, videos, audio) and calls MiniMaxH3ReferenceToVideo "
                "to produce conditioning + AV latent."
            ),
            inputs=[
                H3RefplanType.Input("h3_refplan"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("width",  default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768,  min=32, max=8192, step=32),
                io.Int.Input("length", default=124,  min=5,  max=3600, step=17,
                             tooltip="Frame count at 24 fps (124 = ~5s)"),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip=(
                        "'match' scales refs to the generation canvas area (faster). "
                        "'max' uses full 2048px short-edge fidelity (slower)."
                    ),
                ),
                io.Boolean.Input(
                    "estimate_vram",
                    display_name="Estimate VRAM",
                    default=True,
                    tooltip=(
                        "Estimate second-pass attention memory from this canvas + the "
                        "resolved references, and recommend a safe scale for "
                        "MinimaxH3LatentUpscaler3D's 'scale' input. A heuristic "
                        "calibrated from real OOM incidents on this machine — not a "
                        "guarantee. Disable to skip the computation entirely."
                    ),
                ),
                io.Float.Input(
                    "vram_safety_buffer",
                    display_name="VRAM Safety Buffer",
                    default=0.85, min=0.1, max=1.0, step=0.05,
                    optional=True,
                    tooltip=(
                        "Fraction of the estimated safe headroom to actually recommend "
                        "(0.85 = recommend 85% of the theoretical max scale). Lower this "
                        "if recommended scales still OOM in practice."
                    ),
                ),
                io.Float.Input(
                    "desired_scale",
                    display_name="Desired Scale",
                    default=0.0, min=0.0, max=4.0, step=0.05,
                    optional=True,
                    tooltip=(
                        "Target scale for MinimaxH3LatentUpscaler3D's 'scale' input. "
                        "0.0 = auto (use the calculated safe maximum). Any other value "
                        "is validated against the estimate: passed through unchanged if "
                        "it fits within budget, clamped down to the safe maximum (with "
                        "a warning) if it doesn't."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Float.Output(display_name="Recommended Scale",
                                tooltip="Wire into MinimaxH3LatentUpscaler3D's 'scale' input."),
                io.String.Output(display_name="VRAM Estimate",
                                  tooltip="Human-readable summary of the estimate — also logged."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, h3_refplan, width, height, length, ref_image_size,
                            estimate_vram=True, vram_safety_buffer=0.85, desired_scale=0.0, **_):
        bundle_hash = hashlib.md5(
            json.dumps(h3_refplan or {}, sort_keys=True).encode()
        ).hexdigest()
        file_mtimes = []
        for ref in (h3_refplan or {}).get("references", []):
            path = _h3_resolve_path(ref.get("path", ""))
            try:
                file_mtimes.append(f"{path}:{os.path.getmtime(path):.3f}")
            except OSError:
                file_mtimes.append(f"{path}:missing")
        return (bundle_hash, *file_mtimes, width, height, length, ref_image_size,
                estimate_vram, vram_safety_buffer, desired_scale)

    @classmethod
    def execute(cls, h3_refplan, clip, vae, audio_vae,
                width, height, length, ref_image_size="match",
                estimate_vram=True, vram_safety_buffer=0.85, desired_scale=0.0) -> io.NodeOutput:
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except ImportError as exc:
            raise RuntimeError(
                f"MiniMaxH3ReferenceToVideo not found in comfy_extras. "
                f"Ensure ComfyUI includes nodes_minimax_h3.py. ({exc})"
            ) from exc

        if not h3_refplan:
            raise ValueError("h3_refplan is required — wire PromptCompositionLoader's "
                             "'H3 Ref Plan' output into this node.")

        references = h3_refplan.get("references", [])
        prompt     = h3_refplan.get("prompt", "")

        # ── §1 pre-load validation (MiniMax H3 Ref2VA hard limits) ─────────────
        pre_errors = _validate_h3_refs_pre(references)
        if pre_errors:
            raise ValueError(
                "H3 Ref2VA validation failed:\n"
                + "\n".join(f"  • {e}" for e in pre_errors)
            )

        # ── Turbo LoRA warning ──────────────────────────────────────────────────
        # Any audio reference counts, whether it is a video's soundtrack or a standalone clip.
        audio_refs = [r for r in references if r.get("modality") in ("audio", "soundtrack_audio")]
        if h3_refplan.get("has_turbo_lora") and audio_refs:
            _turbo_msg = (
                "Turbo LoRA detected with audio references. "
                "Turbo LoRA is known to degrade audio quality badly — "
                "disable it for voice/dialogue generation."
            )
            logger.warning("CompositionToH3: %s", _turbo_msg)
            send_status_update(cls.node_id, f"⚠️ {_turbo_msg}")

        # ── Load references ─────────────────────────────────────────────────────
        ref_images       = {}
        ref_videos       = {}
        ref_video_audios = {}
        ref_audios       = {}
        standalone_idx   = 0
        loaded_audio_durations: list[float] = []

        # VRAM estimate: reference token count. With ref_image_size=="match",
        # MiniMaxH3ReferenceToVideo rescales every reference to the generation
        # canvas's own pixel area, so approximate each reference frame's cost
        # as the canvas's own per-frame token cost rather than the reference's
        # original (pre-rescale) resolution. With "max" (full 2048px-short-edge
        # fidelity), use the reference's actual loaded resolution instead.
        reference_tokens = 0.0
        canvas_tokens_per_frame = h3_tokens_for(width, height, 1) if estimate_vram else 0.0

        logger.info("CompositionToH3: loading %d reference item(s) — canvas %dx%d, %d frames",
                    len(references), width, height, length)

        for ref in references:
            modality = ref.get("modality", "")
            path     = ref.get("path", "")
            fname    = os.path.basename(path) if path else "(no path)"

            if modality == "image":
                frames = _h3_load_image(path)
                if frames is not None:
                    n = ref["picture_ordinal"] - 1
                    ref_images[f"ref_image_{n}"] = frames
                    h, w = frames.shape[1], frames.shape[2]
                    logger.info("  <Picture %d>  image  %s  %dx%d",
                                ref["picture_ordinal"], fname, w, h)
                    if estimate_vram:
                        reference_tokens += (
                            canvas_tokens_per_frame if ref_image_size == "match"
                            else h3_tokens_for(w, h, 1)
                        )
                else:
                    logger.warning("  <Picture %d>  image  %s  FAILED TO LOAD",
                                   ref["picture_ordinal"], fname)

            elif modality == "video":
                frames = _h3_load_video_frames(path, ref.get("load_params", {}))
                if frames is not None:
                    n = ref["video_ordinal"] - 1
                    ref_videos[f"ref_video_{n}"] = frames
                    lp = ref.get("load_params", {})
                    n_frames = frames.shape[0]
                    logger.info("  <Video %d>    video  %s  %d frames  start=%.1fs dur=%.1fs",
                                ref["video_ordinal"], fname, n_frames,
                                lp.get("start_time", 0.0), lp.get("duration", 0.0))
                    if estimate_vram:
                        reference_tokens += (
                            canvas_tokens_per_frame * n_frames if ref_image_size == "match"
                            else h3_tokens_for(frames.shape[2], frames.shape[1], n_frames)
                        )
                else:
                    logger.warning("  <Video %d>    video  %s  FAILED TO LOAD",
                                   ref["video_ordinal"], fname)

            elif modality == "soundtrack_audio":
                _cache = ref.get("audio_cache", "")
                if _cache and os.path.isfile(_cache):
                    audio = _h3_load_audio(_cache, 0.0, 0.0)
                    src_note = f"cache:{os.path.basename(_cache)}"
                else:
                    audio = _h3_load_audio(path, ref.get("start_time", 0.0),
                                           ref.get("duration", 0.0))
                    src_note = fname
                if audio is not None:
                    n = ref["video_ordinal"] - 1
                    ref_video_audios[f"ref_video_audio_{n}"] = audio
                    dur = audio["waveform"].shape[-1] / max(audio["sample_rate"], 1)
                    logger.info("  <Audio %d>    soundtrack  %s  %.2fs  retention=%s  (paired with <Video %d>)",
                                ref["audio_ordinal"], src_note, dur,
                                ref.get("retention", "timbre"), ref["video_ordinal"])
                else:
                    logger.warning("  <Audio %d>    soundtrack  %s  FAILED TO LOAD",
                                   ref["audio_ordinal"], src_note)

            elif modality == "audio":
                _cache = ref.get("audio_cache", "")
                if _cache and os.path.isfile(_cache):
                    audio = _h3_load_audio(_cache, 0.0, 0.0)
                    src_note = f"cache:{os.path.basename(_cache)}"
                else:
                    audio = _h3_load_audio(path, ref.get("start_time", 0.0),
                                           ref.get("duration", 0.0))
                    src_note = fname
                if audio is None:
                    logger.warning("  <Audio %d>    standalone  %s  FAILED TO LOAD",
                                   ref["audio_ordinal"], src_note)
                    continue

                # Apply trim_to: shorten waveform to estimated dialogue line duration
                trim_to = ref.get("trim_to")
                if trim_to is not None and trim_to > 0:
                    sr = audio["sample_rate"]
                    target_samples = int(trim_to * sr)
                    audio["waveform"] = audio["waveform"][:, :, :target_samples]

                # Per-clip duration validation (post-trim, actual samples)
                actual_samples = audio["waveform"].shape[-1]
                sr = audio["sample_rate"]
                actual_dur = actual_samples / sr if sr > 0 else 0.0
                clip_err = _validate_h3_audio_clip(
                    actual_dur, ref.get("audio_ordinal", "?"), os.path.basename(path)
                )
                if clip_err:
                    raise ValueError(clip_err)

                trim_note = f"  trimmed→{actual_dur:.2f}s" if trim_to else ""
                logger.info("  <Audio %d>    standalone  %s  %.2fs  retention=%s%s",
                            ref["audio_ordinal"], src_note, actual_dur,
                            ref.get("retention", "timbre"), trim_note)

                loaded_audio_durations.append(actual_dur)
                ref_audios[f"ref_audio_{standalone_idx}"] = audio
                standalone_idx += 1

        # Total audio duration check (uses actual loaded/trimmed durations)
        total_audio = sum(loaded_audio_durations)
        total_err = _validate_h3_audio_total(loaded_audio_durations)
        if total_err:
            raise ValueError(total_err)

        logger.info(
            "CompositionToH3: passing to MiniMaxH3ReferenceToVideo — "
            "%d image(s), %d video(s), %d soundtrack(s), %d standalone audio(s)%s",
            len(ref_images), len(ref_videos), len(ref_video_audios), len(ref_audios),
            f" | {total_audio:.1f}s audio total" if loaded_audio_durations else "",
        )

        audio_note = f" | audio {total_audio:.1f}s total" if loaded_audio_durations else ""
        send_status_update(
            cls.node_id,
            f"H3 conditioning: {len(ref_images)} image(s), "
            f"{len(ref_videos)} video(s), "
            f"{len(ref_video_audios)} soundtrack(s), "
            f"{len(ref_audios)} standalone audio(s){audio_note}",
        )

        mm_result = MiniMaxH3ReferenceToVideo.execute(
            clip, prompt, width, height, length,
            ref_image_size=ref_image_size,
            vae=vae,
            audio_vae=audio_vae,
            ref_images=ref_images or None,
            ref_videos=ref_videos or None,
            ref_video_audios=ref_video_audios or None,
            ref_audios=ref_audios or None,
        )
        positive, latent = mm_result.args[0], mm_result.args[1]

        if estimate_vram:
            recommended_scale, vram_summary = cls._vram_estimate_summary(
                width, height, length, reference_tokens, vram_safety_buffer, desired_scale,
            )
        else:
            recommended_scale = desired_scale if desired_scale > 0.0 else 1.0
            vram_summary = "VRAM estimate disabled (Estimate VRAM input is off)."

        return io.NodeOutput(positive, latent, recommended_scale, vram_summary)

    @classmethod
    def _vram_estimate_summary(cls, width, height, length, reference_tokens,
                                vram_safety_buffer, desired_scale=0.0):
        """Compute (output_scale, summary_text) for the second-pass upscale
        factor to feed MinimaxH3LatentUpscaler3D's 'scale' input. Heuristic —
        see utils/h3_vram_estimator.py for the calibration this is based on.

        desired_scale <= 0.0 means "auto": output the calculated safe maximum.
        Any other value is treated as a target: passed through unchanged if it
        fits within the estimate, clamped down to the safe maximum (with a
        warning) if it doesn't.
        """
        main_tokens = h3_tokens_for(width, height, length)
        try:
            budget_gib = torch.cuda.mem_get_info()[1] / (1024 ** 3) if torch.cuda.is_available() else None
        except Exception:
            budget_gib = None

        if budget_gib is None:
            fallback = desired_scale if desired_scale > 0.0 else 1.0
            summary = (
                f"VRAM estimate unavailable (no CUDA device detected) — "
                f"passing through scale {fallback:.2f}x unvalidated."
            )
            logger.warning("CompositionToH3: %s", summary)
            return fallback, summary

        max_scale, at_risk = h3_max_safe_scale(
            main_tokens, reference_tokens, budget_gib, safety_buffer=vram_safety_buffer,
        )

        auto_mode = desired_scale <= 0.0
        if auto_mode:
            output_scale = max_scale
            clamped = False
        else:
            clamped = desired_scale > max_scale
            output_scale = max_scale if clamped else desired_scale

        align = 32
        second_w = round(width * output_scale / align) * align
        second_h = round(height * output_scale / align) * align
        mode_str = "auto" if auto_mode else ("target, clamped" if clamped else "target, confirmed")
        summary = (
            f"VRAM estimate: pass 1 {width}x{height} ({main_tokens:,.0f} main + "
            f"{reference_tokens:,.0f} ref tokens) -> scale {output_scale:.2f}x [{mode_str}] "
            f"for pass 2 (~{second_w}x{second_h}) | safe max {max_scale:.2f}x | "
            f"budget {budget_gib:.1f} GiB, buffer {vram_safety_buffer:.0%}"
        )
        if clamped:
            summary += f" | WARNING: requested {desired_scale:.2f}x exceeds estimated safe max — clamped"
            logger.warning("CompositionToH3: %s", summary)
        elif at_risk:
            summary += " | WARNING: even scale=1.0 may be tight at this canvas/reference load"
            logger.warning("CompositionToH3: %s", summary)
        else:
            logger.info("CompositionToH3: %s", summary)
        return output_scale, summary


