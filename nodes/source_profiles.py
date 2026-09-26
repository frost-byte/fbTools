"""Source Profile nodes (SourceProfileLoad, SourceProfileDefine, SourceProfileList,
SourceProfileClipPrompt) and their 19 REST routes (/fbtools/source_profiles/*).

Moved out of extension.py (pure code motion, Plan 28). A source profile is a media-first
subject catalog: one video or image annotated with the identifiable subjects it contains.

Two of this module's dependencies were previously satisfied only by extension.py's own
top-to-bottom load order rather than a real import -- both had to become explicit imports
here since this is now its own module with its own independent load order (see
docs/session plan notes, Plan 28): `asyncio` (extension.py has no top-of-file import of it
at all, only one deep in an unrelated LLM-routes section) and `_build_h3_refplan` (imported
by extension.py itself only ~4000 lines below where SourceProfileClipPrompt used to live).

SourceProfileIOType/CastIOType/H3RefplanType come from nodes/composition_types.py (Plan 27)
-- that module exists specifically so this file and the not-yet-moved composition-assembly
layer can both import them without depending on each other.
"""
from __future__ import annotations

import asyncio
import math
import os
import json
import time

from aiohttp import web
from comfy_api.latest import io
from folder_paths import get_input_directory, get_output_directory

from .shared import (
    prefixed_node_id, routes, send_status_update, user_data_dir, bump_reload, reload_counter,
    default_source_profiles_path, default_subject_profiles_path, default_bundle_registry_path,
)
from .libber import LibberStateManager
from .lora_stacks import LoraStackData
from .llm_assistant import _SPA_STATUS_ID, _run_text_inference, _run_vision_inference, _run_vision_inference_clip
from .composition_types import SourceProfileIOType, CastIOType, H3RefplanType
from ..utils.subject_profiles import load_registry as _load_subject_registry
from ..utils.source_profiles import (
    load_registry as _load_source_registry,
    save_registry as _save_source_registry,
    MEDIA_TYPES as _SOURCE_MEDIA_TYPES,
    MEDIA_DIRS as _SOURCE_MEDIA_DIRS,
)
from ..utils.source_profile_analysis import (
    build_segment_detection_prompt as _spa_build_segment_prompt,
    build_clip_description_prompt as _spa_build_clip_desc_prompt,
    _parse_segments_response as _spa_parse_segments,
    build_subject_inference_prompt as _spa_build_subject_inference_prompt,
    parse_inferred_subjects_response as _spa_parse_inferred_subjects,
    parse_clip_description_response as _spa_parse_clip_desc,
    build_multi_prompt as _spa_build_multi_prompt,
    _parse_vlm_json_response as _spa_parse_response,
    append_history_entry as _spa_append_history,
    history_for_profile as _spa_history_for_profile,
    extract_video_frame as _spa_extract_frame,
    extract_frame_at_time as _spa_extract_frame_at_time,
    extract_clip_frames as _spa_extract_clip_frames,
    build_contact_sheet_image as _spa_build_contact_sheet,
    probe_video_fps as _spa_probe_fps,
    probe_video_resolution as _spa_probe_resolution,
    PASS_TYPES as _SPA_PASS_TYPES,
)
from ..utils.proxy_cache import ensure_source_profile_proxy as _ensure_proxy
from ..utils.reference_bundles import load_registry as _load_bundle_registry
from ..utils.composition_resources import get_background as _get_background
from ..utils.prompt_assembler import (
    assemble_prompt as _assemble_prompt,
    MODEL_TYPES as _PROMPT_MODEL_TYPES,
    _build_h3_refplan,
    _build_background_slot,
)
from ..utils.libber_resolve import extract_libber_names, apply_slot_dialogue
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Source Profile helpers ────────────────────────────────────────────────────

def _source_profile_get_names() -> list[str]:
    """Read source_profiles.json and return profile names for combo widgets."""
    try:
        reg = _load_source_registry(default_source_profiles_path())
        names = reg.profile_names()
        return names if names else ["(none)"]
    except Exception:
        return ["(none)"]


# SourceProfileIOType moved to nodes/composition_types.py (Plan 27)
# ── Node: SourceProfileLoad ───────────────────────────────────────────────────

class SourceProfileLoad(io.ComfyNode):
    """Load a source profile from disk and expose it for wiring into CastBuild.

    A source profile is a media-first subject catalog: one video or image
    annotated with the identifiable subjects it contains. Connect the
    source_profile output to a CastBuild node to make all subjects in this
    profile available as cast slot options.

    Use POST /fbtools/source_profiles/reload to force re-execution after
    editing source_profiles.json externally.
    """
    node_id = prefixed_node_id("SourceProfileLoad")
    display_name = "Source Profile Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        profile_names = _source_profile_get_names()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "profile_name",
                    options=profile_names,
                    display_name="Source Profile",
                    tooltip="Source profile to load. Press R to refresh the list after adding new profiles.",
                ),
            ],
            outputs=[
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Full source profile dict for wiring into CastBuild.",
                ),
                io.String.Output(
                    "subject_info",
                    display_name="Subject Info",
                    tooltip="Formatted summary of subjects in this profile.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, profile_name: str = "", **_):
        path = default_source_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, profile_name, mtime, reload_counter("source_profile"))

    @classmethod
    def execute(cls, profile_name: str = "") -> io.NodeOutput:
        path = default_source_profiles_path()
        registry = _load_source_registry(path)

        profile = None
        if profile_name and profile_name != "(none)":
            profile = registry.get_profile_by_name(profile_name)
            # Compat: old saved workflows stored the profile ID in this widget before
            # it was renamed from profile_id → profile_name.  Fall back to ID lookup
            # so those workflows continue to work without manual intervention.
            if profile is None:
                profile = registry.get_profile(profile_name)
        if profile is None:
            logger.warning("SourceProfileLoad: profile %r not found in %s", profile_name, path)
            return io.NodeOutput(None, "")

        subjects = profile.get("subjects", [])
        lines = [f"[{s.get('entity_type','?')}] {s.get('label', s.get('id',''))} — {s.get('role_description','')}"
                 for s in subjects]
        subject_info = "\n".join(lines) if lines else "(no subjects defined)"

        send_status_update(
            cls.node_id,
            f"Loaded: {profile_name} | {len(subjects)} subject(s)",
        )
        return io.NodeOutput(profile, subject_info)


# ── Node: SourceProfileDefine ─────────────────────────────────────────────────

class SourceProfileDefine(io.ComfyNode):
    """Create or update a source profile entry.

    Defines the profile metadata (name, media file) only. Use the Source
    Profiles UI panel or SourceProfileDefine nodes chained together to add
    subjects. When auto_save is enabled the registry is written to
    source_profiles.json immediately.
    """
    node_id = prefixed_node_id("SourceProfileDefine")
    display_name = "Source Profile Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "profile_id",
                    display_name="Profile ID",
                    default="",
                    multiline=False,
                    tooltip="Unique snake_case identifier, e.g. theater_scene_video1.",
                ),
                io.String.Input(
                    "name",
                    display_name="Name",
                    default="",
                    multiline=False,
                    tooltip="Human-readable name for this source profile.",
                ),
                io.String.Input(
                    "media_filename",
                    display_name="Media Filename",
                    default="",
                    multiline=False,
                    tooltip="Filename of the source video or image in the ComfyUI input or output directory.",
                ),
                io.Combo.Input(
                    "media_dir",
                    options=_SOURCE_MEDIA_DIRS,
                    display_name="Media Dir",
                    tooltip="Whether the source file lives in the input or output directory.",
                ),
                io.Combo.Input(
                    "media_type",
                    options=_SOURCE_MEDIA_TYPES,
                    display_name="Media Type",
                    tooltip="video or image.",
                ),
                io.String.Input(
                    "subjects_json",
                    display_name="Subjects JSON",
                    default="[]",
                    multiline=True,
                    tooltip='Optional JSON array of subject entries to add/update. Each entry: {"id","label","role_description","entity_type","notes"}.',
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=True,
                    tooltip="Write source_profiles.json immediately after defining this profile.",
                ),
            ],
            outputs=[
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="The created or updated source profile dict.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        profile_id: str,
        name: str = "",
        media_filename: str = "",
        media_dir: str = "input",
        media_type: str = "video",
        subjects_json: str = "[]",
        auto_save: bool = True,
    ) -> io.NodeOutput:
        if not profile_id.strip():
            raise ValueError("SourceProfileDefine: profile_id cannot be empty")

        pid = profile_id.strip()
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        registry = registry.define_profile(
            profile_id=pid,
            name=name,
            media_filename=media_filename,
            media_dir=media_dir,
            media_type=media_type,
        )

        try:
            subjects = json.loads(subjects_json) if subjects_json.strip() not in ("", "[]") else []
        except Exception:
            subjects = []
            logger.warning("SourceProfileDefine: invalid subjects_json, skipping subjects")

        for s in subjects:
            sid = s.get("id", "").strip()
            if sid:
                registry = registry.define_subject(
                    profile_id=pid,
                    subject_id=sid,
                    label=s.get("label", ""),
                    role_description=s.get("role_description", ""),
                    entity_type=s.get("entity_type", "person"),
                    notes=s.get("notes", ""),
                )

        if auto_save:
            _save_source_registry(registry, path, backup=True)
            logger.info("SourceProfileDefine: saved %r to %s", pid, path)
            send_status_update(cls.node_id, f"Saved source profile: {pid}")

        profile = registry.get_profile(pid)
        return io.NodeOutput(profile)


# ── Node: SourceProfileList ───────────────────────────────────────────────────

class SourceProfileList(io.ComfyNode):
    """Display all defined source profiles and their subjects.

    Useful for quickly reviewing the source catalog without opening the
    JSON file.  Filter by media type to narrow the listing.
    """
    node_id = prefixed_node_id("SourceProfileList")
    display_name = "Source Profile List"
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
                    "filter_type",
                    options=["all"] + _SOURCE_MEDIA_TYPES,
                    display_name="Filter by Type",
                    tooltip="Show all profiles, or only video/image sources.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "profile_list",
                    display_name="Profile List",
                    tooltip="Formatted listing of all source profiles and their subjects.",
                ),
                io.Int.Output("profile_count", display_name="Profile Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, filter_type: str = "all", **_):
        path = default_source_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, filter_type, mtime, reload_counter("source_profile"))

    @classmethod
    def execute(cls, filter_type: str = "all") -> io.NodeOutput:
        registry = _load_source_registry(default_source_profiles_path())
        listing = registry.list_profiles(filter_type=filter_type)
        count = len(registry.profiles)
        return io.NodeOutput(listing, count, ui={"profile_list": listing, "profile_count": count})


# ── Node: SourceProfileClipPrompt ─────────────────────────────────────────────

class SourceProfileClipPrompt(io.ComfyNode):
    """Build a model-specific prompt from a Source Profile clip.

    For H3 models: outputs a full 6-section H3 prompt with task_flags=["video editing"],
    so the model treats <Video 1> as the source material to edit. Preserves
    {A}/{B}/{C}/{D} → <Subject 1>/<Subject 2>/… mapping from clip.subjects order.

    clip_duration_frames = ceil(duration_s × 24) — the H3 output length in frames.
    LoRAs defined on the clip are emitted as LORA_STACK_DATA for direct connection
    to LoraStackApply.
    """
    node_id = prefixed_node_id("SourceProfileClipPrompt")
    display_name = "Source Profile Clip Prompt"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                SourceProfileIOType.Input(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Source profile from SourceProfileLoad. Provides subjects and clip library.",
                ),
                io.String.Input(
                    "clip_id",
                    display_name="Clip ID",
                    default="",
                    tooltip="ID of the clip to process (e.g. 'clip_1'). Leave empty to use the first clip.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_PROMPT_MODEL_TYPES,
                    display_name="Model Type",
                    tooltip="Target model. h3_ref2va emits a 6-section structured brief with video editing task flag.",
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    default="",
                    tooltip=(
                        "Optional prefix prepended to 'profile_name/clip_label' for the filename_prefix output. "
                        "Recommended: wire this from SceneCastBuild's filename_prefix output for automatic "
                        "primary-subject/source_profiles foldering (e.g. 'video/alex/source_profiles/office_work/' "
                        "-> '.../office_work/clip_1'). Or type a literal root here for standalone use without a cast "
                        "(e.g. 'video/' -> 'video/office_work/clip_1'). "
                        "Wire the output into a VHS_VideoCombine filename_prefix input."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    optional=True,
                    tooltip=(
                        "Optional cast from SceneCastBuild. When provided, bundle appearance data "
                        "and images replace raw source-profile role descriptions for matched subjects."
                    ),
                ),
                io.Boolean.Input(
                    "include_original_subject_tags",
                    display_name="Tag Replaced Subjects",
                    default=False,
                    tooltip=(
                        "When enabled, each source-profile subject being replaced by a SceneCastBuild bundle "
                        "gets its own <Subject N> tag in subject_definitions (minimal description) and an "
                        "attribute_transfer entry in retention_analysis that explicitly scopes motion transfer "
                        "to the replacement while stating that face, hair, and clothing are NOT copied. "
                        "Use to A/B test whether explicit original-subject tagging improves swap quality."
                    ),
                    optional=True,
                ),
                io.Int.Input(
                    "clip_duration_multiplier",
                    display_name="Duration Multiplier",
                    default=1,
                    min=1,
                    max=4,
                    tooltip=(
                        "Scale factor applied to native clip duration for the clip_duration_frames output. "
                        "1 = native duration, 2 = double, up to 4×. "
                        "Wire from SceneCastBuild.clip_duration_multiplier."
                    ),
                    optional=True,
                ),
                io.Int.Input(
                    "max_clip_frames",
                    display_name="Max Clip Frames",
                    default=360,
                    min=0,
                    max=9999,
                    tooltip=(
                        "Hard ceiling on clip_duration_frames (after the duration multiplier is applied). "
                        "0 = unclamped. Default 360 = 15 s × 24 fps — the practical H3 output limit. "
                        "Lower this on memory-constrained systems or at high output resolution."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Model-specific assembled prompt text.",
                ),
                H3RefplanType.Output(
                    "h3_refplan",
                    display_name="H3 Ref Plan",
                    tooltip=(
                        "Ordered reference descriptor bundle (FBTOOLS_H3_REFPLAN) for "
                        "CompositionToH3Conditioning. Contains the source video reference "
                        "with clip load params. Empty when model_type is not an H3 variant."
                    ),
                ),
                LoraStackData.Output(
                    "lora_stack_data",
                    display_name="LoRA Stack",
                    tooltip="LoRAs defined on this clip. Feed into LoraStackApply.",
                ),
                io.String.Output(
                    "concept_ids",
                    display_name="Concept IDs",
                    tooltip="Comma-separated concept IDs from assigned subjects. Wire into ConceptResolve.",
                ),
                io.Int.Output(
                    "clip_duration_frames",
                    display_name="Clip Frames",
                    tooltip=(
                        "Clip duration converted to H3 output frames at 24 fps: ceil(duration_s × 24). "
                        "Use as the 'length' input for CompositionToH3Conditioning."
                    ),
                ),
                io.Int.Output(
                    "width",
                    display_name="Width",
                    tooltip="Source video width in pixels (probed via ffprobe). 0 if probe failed.",
                ),
                io.Int.Output(
                    "height",
                    display_name="Height",
                    tooltip="Source video height in pixels (probed via ffprobe). 0 if probe failed.",
                ),
                io.String.Output(
                    "clip_summary",
                    display_name="Clip Summary",
                    tooltip="Human-readable description of the assembled clip and its subject assignments.",
                ),
                io.String.Output(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="prefix + profile_name/clip_label (e.g. 'video/office_work/segment_1'). Wire into VHS_VideoCombine filename_prefix.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_profile=None,
        clip_id: str = "",
        model_type: str = "h3_ref2va",
        filename_prefix: str = "",
        scene_cast=None,
        include_original_subject_tags: bool = False,
        clip_duration_multiplier: int = 1,
        max_clip_frames: int = 360,
    ) -> io.NodeOutput:
        if source_profile is None:
            return io.NodeOutput("", None, [], "", 0, 0, 0, "No source profile connected.", "")

        # ── Resolve clip ───────────────────────────────────────────────────────
        profile_id     = source_profile.get("id", "")
        profile_name   = source_profile.get("name", profile_id)
        clips          = source_profile.get("clips", [])
        subjects_all   = source_profile.get("subjects", [])
        media_filename = source_profile.get("media_filename", "")
        media_dir      = source_profile.get("media_dir", "input")

        # scene_cast.clip_ids takes priority — wiring scene_cast is the primary
        # path and must override the serialized widget value, which can be stale.
        # The widget/link value is only consulted when scene_cast is absent, to
        # support standalone use without a SceneCastBuild node.
        effective_clip_id = ""
        if isinstance(scene_cast, dict):
            effective_clip_id = scene_cast.get("clip_ids", {}).get(profile_id, "")
        if not effective_clip_id:
            effective_clip_id = clip_id.strip() if clip_id else ""

        clip = None
        if effective_clip_id:
            for c in clips:
                if c.get("id") == effective_clip_id:
                    clip = c
                    break
        if clip is None:
            clip = clips[0] if clips else None
        if clip is None:
            return io.NodeOutput("", None, [], "", 0, 0, 0, f"No clips found in profile '{profile_name}'.", "")

        clip_id_used = clip.get("id", "")
        clip_label   = clip.get("label", clip_id_used)

        # ── Resolve clip subjects ──────────────────────────────────────────────
        # ALL entity types (people, environments, objects) go into slots so that
        # {A}/{B}/{C}/{D} placeholders in the clip action text resolve correctly.
        # Stale IDs (removed from the profile) are silently skipped; cap at 4.
        # Fallback: if the clip has no subjects tagged (e.g. manually created
        # clips or clips auto-partitioned before subjects were added), use all
        # profile subjects in definition order so nothing is silently dropped.
        subject_index = {s["id"]: s for s in subjects_all}
        source_subject_ids: list[str] = [
            sid for sid in clip.get("subjects", [])
            if sid in subject_index
        ][:10]
        if not source_subject_ids:
            source_subject_ids = [s["id"] for s in subjects_all if s.get("id")][:10]

        SOURCE_SLOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]   # preserve placeholder mapping
        BUNDLE_SLOTS = ["K", "L", "M", "N"]   # replacement subjects added after source slots

        # ── Build cast lookup for bundle substitution ──────────────────────────
        # Hybrid cast entries (source_profile_id + source_subject_id + bundle_id)
        # mean a bundle is replacing a source subject. Source subject stays in its
        # original slot (so action placeholders resolve); bundle replacement gets an
        # additional slot from BUNDLE_SLOTS.
        cast_lookup: dict[str, dict] = {}   # source_subject_id → cast entry
        bundle_reg = None
        subject_reg = None
        if isinstance(scene_cast, dict):
            for entry in scene_cast.get("entries", []):
                if (entry.get("source_profile_id") == profile_id
                        and entry.get("source_subject_id")
                        and entry.get("bundle_id")):
                    cast_lookup[entry["source_subject_id"]] = entry
            if cast_lookup:
                try:
                    bundle_reg = _load_bundle_registry(default_bundle_registry_path())
                except Exception:
                    bundle_reg = None
                try:
                    subject_reg = _load_subject_registry(default_subject_profiles_path())
                except Exception:
                    subject_reg = None

        # ── Build slot_assignments ─────────────────────────────────────────────
        slot_assignments: dict = {}
        bundle_slot_pairs: list[tuple[str, str]] = []   # (bundle_slot, source_slot)
        bundle_slot_idx = 0
        bundle_video_entries: list[dict] = []           # video refs for bundle slots

        _ENTITY_PRONOUN_DEFAULTS = {
            "location":   "location",
            "object":     "object",
            "soundscape": "object",
        }  # anything else (person, animal, …) → "neutral"

        for i, sid in enumerate(source_subject_ids):
            src_slot = SOURCE_SLOTS[i]
            subj      = subject_index[sid]
            label     = subj.get("label", sid)
            etype     = subj.get("entity_type", "person").lower()
            role_desc = subj.get("role_description", "")

            # Pronoun style: prefer explicit field on the subject; fall back to
            # entity_type-based default so location/object subjects say "its"/"the X's"
            # without needing manual configuration.
            pronoun_style = (subj.get("pronoun_style")
                             or _ENTITY_PRONOUN_DEFAULTS.get(etype, "neutral"))
            short_name    = subj.get("short_name", "")

            # Source subject entry — always added; provides the video reference and
            # preserves {A}/{B}/… placeholder mapping in the clip action text.
            # _cast_retention and _transfer_to_slot are filled in below if a bundle
            # replacement is found for this subject.
            slot_assignments[src_slot] = {
                "subject_id":             sid,
                "name":                   label,
                "concept_id":             None,
                "character_sheet_images": [],
                "appearance": {
                    "summary":        role_desc or label,
                    "hair":           "",
                    "face":           "",
                    "body":           "",
                    "default_outfit": "",
                },
                "voice":           {},
                "_cast_retention": "fully_preserved",
                "_pronoun_style":  pronoun_style,
                "_short_name":     short_name,
                "entity_type":     etype,
            }

            # If a bundle is assigned to this source subject, add a replacement slot.
            cast_entry = cast_lookup.get(sid)
            if cast_entry and bundle_reg is not None and bundle_slot_idx < len(BUNDLE_SLOTS):
                bundle = bundle_reg.get(cast_entry["bundle_id"])
                if bundle:
                    bun_slot = BUNDLE_SLOTS[bundle_slot_idx]
                    bundle_slot_idx += 1
                    bundle_slot_pairs.append((bun_slot, src_slot))
                    # Mark the source slot as a motion donor for the assembler
                    slot_assignments[src_slot]["_cast_retention"] = "replaced"
                    slot_assignments[src_slot]["_transfer_to_slot"] = bun_slot

                    bun_name   = bundle.get("name") or label
                    # Look up the Subject linked to this bundle for pronoun/short_name/appearance.
                    bun_subject_id = bundle.get("subject_id", "")
                    bun_subj = (subject_reg.get_subject(bun_subject_id)
                                if subject_reg and bun_subject_id else None) or {}
                    # Merge bundle appearance over subject appearance (bundle wins if set).
                    bun_app_dict = bundle.get("appearance") or {}
                    if isinstance(bun_app_dict, str):   # legacy appearance_override string
                        bun_app_dict = {"summary": bun_app_dict}
                    subj_app_dict = bun_subj.get("appearance", {}) if bun_subj else {}
                    def _merge(key):
                        return bun_app_dict.get(key) or subj_app_dict.get(key, "")
                    bun_appear = (
                        bun_app_dict.get("summary")
                        or bundle.get("appearance_override", "")  # legacy fallback
                        or subj_app_dict.get("summary", "")
                    )
                    visual = bundle.get("visual", {})
                    # Resolve mode: cast entry overrides the bundle's default mode.
                    # "both" = use images AND video simultaneously.
                    bun_visual_mode = cast_entry.get("visual_mode") or visual.get("type", "images")

                    # Images path — character sheet images (images mode OR both mode)
                    bun_images: list = []
                    if bun_visual_mode != "video":
                        raw_files = visual.get("files", [])
                        bun_images = [
                            {"file": f.get("file", ""), "role": f.get("role", "character sheet")}
                            if isinstance(f, dict)
                            else {"file": f, "role": "character sheet"}
                            for f in raw_files
                            if (f.get("file", "") if isinstance(f, dict) else f)
                        ]
                        # image_selection: None=all, list[int]=specific indices, int=legacy single
                        img_sel = cast_entry.get("image_selection")
                        if img_sel is not None:
                            if isinstance(img_sel, list):
                                bun_images = [bun_images[i] for i in img_sel if isinstance(i, int) and 0 <= i < len(bun_images)]
                            else:
                                try:
                                    idx = int(img_sel)
                                    bun_images = [bun_images[idx]] if 0 <= idx < len(bun_images) else []
                                except (TypeError, ValueError):
                                    pass

                    audio        = bundle.get("audio", {})
                    audio_source = audio.get("source", "none")

                    # Video path — add a video_entry for the bundle slot so the
                    # assembler emits a <Video N> reference for this subject.
                    # Runs when mode is "video" OR "both".
                    if bun_visual_mode in ("video", "both"):
                        vfile = visual.get("file", "")
                        if vfile:
                            vdir = visual.get("video_dir", "input")
                            bun_base = (get_output_directory() if vdir == "output"
                                        else get_input_directory())
                            bun_load = {
                                "start_time":        float(visual.get("start_time", 0.0)),
                                "duration":          float(visual.get("duration", 0.0)),
                                "force_rate":        visual.get("force_rate", 0),
                                "frame_load_cap":    visual.get("frame_load_cap", 96),
                                "skip_first_frames": visual.get("skip_first_frames", 0),
                                "select_every_nth":  visual.get("select_every_nth", 1),
                            }
                            # extract_from_visual: audio extracted from this same video.
                            # extract_from_video / file: separate source — handled as
                            # bun_voice below; this entry has no audio.
                            # use_audio on the cast entry: treat bundle video's audio
                            # track as the voice-timbre reference for this subject.
                            # allows_dialogue=False on the clip means no audio
                            # involvement for this shot at all, regardless of where
                            # the bundle's audio would otherwise come from.
                            ve_audio_src = (
                                "extract_from_visual"
                                if (clip.get("allows_dialogue", True)
                                    and (audio_source == "extract_from_visual"
                                         or cast_entry.get("use_audio")))
                                else "none"
                            )
                            bundle_video_entries.append({
                                "subject_id":       cast_entry["bundle_id"],
                                "subject_ids":      [cast_entry["bundle_id"]],
                                "video_file":       os.path.join(bun_base, vfile),
                                "load_params":      bun_load,
                                "audio_source":     ve_audio_src,
                                "audio_path":       "",
                                "audio_start_time": 0.0,
                                "audio_duration":   0.0,
                                "audio_retention":  audio.get("retention", "timbre"),
                                "audio_role":       audio.get("role", ""),
                                "audio_cache":      audio.get("audio_cache", ""),
                                # Default excludes this reference video's own background/setting
                                # from the generated output; opt-in via the Scene Cast Build tab's
                                # "Keep BG" checkbox on this cast entry.
                                "include_video_background": bool(cast_entry.get("include_video_background", False)),
                            })

                    # extract_from_video: a SEPARATE video whose audio track is the
                    # voice reference.  Add an audio_only video_entry linked to the
                    # bundle slot so the assembler emits <Audio N> without also
                    # assigning a spurious <Video N> visual reference to the slot.
                    # Gated on allows_dialogue like the other audio paths above —
                    # this is an independent file, but still audio for this clip's
                    # shot, so the clip's "no audio" setting must still apply.
                    if audio_source == "extract_from_video" and clip.get("allows_dialogue", True):
                        aud_vfile = audio.get("video_file", "")
                        if aud_vfile:
                            aud_vdir = audio.get("video_dir", "input")
                            aud_base = (get_output_directory() if aud_vdir == "output"
                                        else get_input_directory())
                            aud_load = {
                                "start_time":        float(audio.get("start_time", 0.0)),
                                "duration":          float(audio.get("duration", 0.0)),
                                "force_rate":        audio.get("force_rate", 0),
                                "frame_load_cap":    audio.get("frame_load_cap", 0) or 4,
                                "skip_first_frames": audio.get("skip_first_frames", 0),
                                "select_every_nth":  audio.get("select_every_nth", 1),
                            }
                            # When visual mode is images, link to the bundle slot via
                            # bundle_id so the assembler can resolve soundtrack_num.
                            # audio_only=True tells the assembler to assign <Audio N>
                            # only (no <Video N> added to the slot's visual refs).
                            # When visual mode is video or both the visual entry already
                            # holds bundle_id; use a distinct id to avoid collisions.
                            has_vid_entry = bun_visual_mode in ("video", "both")
                            aud_sid = (cast_entry["bundle_id"] + "_audvid" if has_vid_entry
                                       else cast_entry["bundle_id"])
                            bundle_video_entries.append({
                                "subject_id":       aud_sid,
                                "subject_ids":      [aud_sid],
                                "video_file":       os.path.join(aud_base, aud_vfile),
                                "load_params":      aud_load,
                                "audio_source":     "extract_from_visual",
                                "audio_only":       not has_vid_entry,
                                "audio_path":       "",
                                "audio_start_time": 0.0,
                                "audio_duration":   0.0,
                                "audio_retention":  audio.get("retention", "timbre"),
                                "audio_role":       audio.get("role", ""),
                                "audio_cache":      audio.get("audio_cache", ""),
                            })

                    # Standalone audio voice reference (<Audio N>) — emitted when the
                    # bundle carries a separate audio file (source == "file").
                    # extract_from_visual / extract_from_video are handled via
                    # video_entries above. Same allows_dialogue gate: a bundle's own
                    # dedicated audio file is unrelated to the clip's footage, but
                    # it's still audio attached to this clip's shot.
                    bun_voice: dict = {}
                    if audio_source == "file" and audio.get("file") and clip.get("allows_dialogue", True):
                        bun_voice = {
                            "audio_reference_file": os.path.join(
                                get_input_directory(), audio["file"]
                            ),
                            "audio_start_time": float(audio.get("start_time", 0.0)),
                            "audio_duration":   float(audio.get("duration", 0.0)),
                            "audio_retention":  audio.get("retention", "timbre"),
                            "audio_role":       audio.get("role", ""),
                            "audio_cache":      audio.get("audio_cache", ""),
                            "description":      audio.get("description", ""),
                            "language":         audio.get("language", "en-us"),
                        }

                    bun_pronoun = (bun_subj.get("pronoun_style")
                                   or _ENTITY_PRONOUN_DEFAULTS.get(
                                       bundle.get("entity_type", "person"), "neutral"))
                    bun_short_name = bun_subj.get("short_name", "")
                    slot_assignments[bun_slot] = {
                        "subject_id":             cast_entry["bundle_id"],
                        "name":                   bun_name,
                        "concept_id":             None,
                        "character_sheet_images": bun_images,
                        "appearance": {
                            "summary":        bun_appear,
                            "hair":           _merge("hair"),
                            "face":           _merge("face"),
                            "body":           _merge("body"),
                            "default_outfit": _merge("default_outfit"),
                        },
                        "voice":             bun_voice,
                        "_cast_retention":   "attribute_transfer",
                        "_transfer_to_slot": src_slot,
                        "_pronoun_style":    bun_pronoun,
                        "_short_name":       bun_short_name,
                    }

        # ── Task flags and scene synopsis ──────────────────────────────────────
        if bundle_slot_pairs:
            task_flags = ["video editing", "reference generation"]
            # Auto-generate a replacement synopsis.  Bundle slot placeholders {b}
            # are resolved by _bare() to <Subject N>.  Source slot placeholders
            # {s} resolve the same way ONLY when include_original_subject_tags
            # is set — that's what gives a replaced source subject a Subject N
            # label in the first place (see _pre_subject_nums above); otherwise
            # {s} would resolve to an empty string, so the source is named
            # literally instead.
            repl_parts = []
            for _b, _s in bundle_slot_pairs:
                _s_idx = SOURCE_SLOTS.index(_s)
                _src_sid = source_subject_ids[_s_idx]
                if include_original_subject_tags:
                    repl_parts.append(
                        f"Replace {{{_s}}} in <Video 1> with {{{_b}}}, adopting "
                        f"{{{_b}}}'s full appearance and outfit while retaining "
                        f"{{{_s}}}'s original motion, pose and screen position "
                        f"throughout the shot"
                    )
                else:
                    _src_name = subject_index[_src_sid].get("label", _src_sid)
                    repl_parts.append(
                        f"{{{_b}}} takes the place of {_src_name}, "
                        f"replicating their pose, movement, and screen position"
                    )
            retained = [
                SOURCE_SLOTS[i] for i, sid in enumerate(source_subject_ids)
                if sid not in cast_lookup
            ]
            if retained:
                if len(retained) == 1:
                    ret_str = f"{{{retained[0]}}} performs the same actions from <Video 1>"
                else:
                    parts = [f"{{{s}}}" for s in retained]
                    ret_str = (", ".join(parts[:-1]) + " and " + parts[-1]
                               + " perform the same actions from <Video 1>")
                scene_synopsis = ". ".join(repl_parts) + ". " + ret_str + "."
            else:
                scene_synopsis = ". ".join(repl_parts) + "."
        else:
            task_flags    = ["video editing"]
            scene_synopsis = ""

        # ── Collect per-slot dialogue from cast entries ────────────────────────
        # slot_dialogue maps slot key (e.g. "E", "A") → raw dialogue string.
        # Hybrid entries: dialogue belongs to the bundle replacement slot.
        # Source-only entries: dialogue belongs to the original source slot.
        slot_dialogue: dict[str, str] = {}
        if isinstance(scene_cast, dict):
            _src_dlg_lookup: dict[str, str] = {}  # source_subject_id → dialogue
            for _ce in scene_cast.get("entries", []):
                if _ce.get("source_profile_id") != profile_id:
                    continue
                _d = str(_ce.get("dialogue", "") or "").strip()
                if not _d:
                    continue
                _ssid = _ce.get("source_subject_id", "")
                if _ssid and not _ce.get("bundle_id"):
                    _src_dlg_lookup[_ssid] = _d

            for _i2, _sid2 in enumerate(source_subject_ids):
                _d2 = _src_dlg_lookup.get(_sid2, "")
                if _d2:
                    slot_dialogue[SOURCE_SLOTS[_i2]] = _d2

            for _bslot, _sslot in bundle_slot_pairs:
                _s_idx2 = SOURCE_SLOTS.index(_sslot)
                _src_sid2 = source_subject_ids[_s_idx2]
                _ce2 = cast_lookup.get(_src_sid2)
                if _ce2:
                    _d3 = str(_ce2.get("dialogue", "") or "").strip()
                    if _d3:
                        slot_dialogue[_bslot] = _d3

        # ── Background as visual reference (per clip) ───────────────────────────
        # Mirrors assemble_composition()'s {BG}-shortcut slot exactly (same shared
        # _build_background_slot helper), minus the shortcut itself: a clip is always
        # exactly one synthetic shot (see _shot_id below), so there's no multi-shot
        # ambiguity for a {BG} token to resolve -- the slot is simply always included
        # when a background resolves, and naturally reads as "appears throughout" in
        # retention_analysis. "O" is the next free letter after SOURCE_SLOTS (A-J) and
        # BUNDLE_SLOTS (K-N) in this file's fixed reserved-letter scheme.
        BACKGROUND_SLOT = "O"
        effective_bg_id = clip.get("background_id", "") or source_profile.get("default_background_id", "")
        if effective_bg_id:
            try:
                resolved_background = _get_background(user_data_dir(), effective_bg_id)
            except Exception:
                resolved_background = None
            if resolved_background:
                bg_slot = _build_background_slot(resolved_background)
                if bg_slot is not None:
                    slot_assignments[BACKGROUND_SLOT] = bg_slot

        # ── Build scene_instance ────────────────────────────────────────────────
        _shot_id = f"{clip_id_used}_shot_1"
        scene_instance = {
            "template_id":                  profile_id,
            "template_name":                profile_name,
            "task_flags":                   task_flags,
            "scene_synopsis":               scene_synopsis,
            "slot_assignments":             slot_assignments,
            "dialogue":                     {},
            "outfit_overrides":             {},
            "include_original_subject_tags": include_original_subject_tags,
            "template": {
                "shots": [
                    {
                        "id":       _shot_id,
                        "action":   clip.get("action", ""),
                        "camera":   "",
                        "dialogue": None,
                    }
                ],
                "environment":         {},
                "style":               clip.get("style", ""),
                "overall_soundscape":  clip.get("overall_soundscape", ""),
                "non_diegetic_music":  clip.get("non_diegetic_music", ""),
            },
        }

        # ── Resolve dialogue and populate shot ─────────────────────────────────
        if not clip.get("allows_dialogue", True):
            slot_dialogue = {}
        if slot_dialogue:
            _libber_mgr = LibberStateManager.instance()
            _all_raw = " ".join(slot_dialogue.values())
            _lib_registry: dict = {}
            for _lname in extract_libber_names(_all_raw):
                _lb = _libber_mgr.ensure_libber(_lname)
                if _lb:
                    _lib_registry[_lname] = _lb.libs

            _dlg_map, _shot_patch = apply_slot_dialogue(slot_dialogue, _lib_registry, _shot_id)
            scene_instance["dialogue"].update(_dlg_map)
            _shot = scene_instance["template"]["shots"][0]
            _shot.update(_shot_patch)

        # ── Build video_entries ────────────────────────────────────────────────
        # ONE entry shared by all clip subjects: `subject_ids` (list) causes
        # _build_ref_map to assign the same <Video 1> ordinal to every subject,
        # while `subject_id` (primary) feeds _build_h3_refplan's singular lookup.
        base_dir  = get_output_directory() if media_dir == "output" else get_input_directory()
        video_abs = os.path.join(base_dir, media_filename) if media_filename else ""

        clip_start  = clip.get("start_time", 0.0)
        clip_end    = clip.get("end_time", 0.0)
        clip_dur    = max(0.0, clip_end - clip_start)

        load_params = {
            "start_time":        clip_start,
            "duration":          clip_dur,
            "force_rate":        24,  # H3 requires 24fps reference video
            "frame_load_cap":    clip.get("frame_load_cap", 120),
            "skip_first_frames": 0,
            "select_every_nth":  clip.get("select_every_nth", 2),
        }

        # ── Per-segment proxy ──────────────────────────────────────────────────
        # Trim + downscale once so CompositionToH3Conditioning doesn't need to
        # seek inside a large source file on every generation run.
        video_file_for_entry = video_abs
        if video_abs:
            try:
                proxy_path = _ensure_proxy(
                    source_path=video_abs,
                    profile_id=profile_id,
                    clip_id=clip_id_used,
                    start_time=clip_start,
                    end_time=clip_end,
                    short_edge=source_profile.get("proxy_short_edge", 768),
                    base_dir=str(user_data_dir()),
                )
                if proxy_path:
                    video_file_for_entry = proxy_path
                    load_params = dict(load_params)
                    load_params["start_time"] = 0.0
                    load_params["duration"]   = clip_dur
            except Exception as _proxy_exc:
                logger.warning(
                    "SourceProfileClipPrompt: proxy generation failed for %s: %s",
                    os.path.basename(video_abs), _proxy_exc,
                )

        video_entries = [{
            "subject_id":       source_subject_ids[0] if source_subject_ids else "",
            "subject_ids":      source_subject_ids,
            "video_file":       video_file_for_entry,
            "load_params":      load_params,
            "audio_source":     "none",
            "audio_path":       "",
            "audio_start_time": 0.0,
            "audio_duration":   0.0,
            "audio_retention":  "timbre",
            "audio_role":       "",
            "audio_cache":      "",
        }] if source_subject_ids else []

        # Bundle slots that have visual_mode="video" contribute their own video
        # reference entries (separate from the source profile's motion-donor video).
        video_entries.extend(bundle_video_entries)

        # ── use_audio: voice-timbre reference from the subject's video clip ──────
        # Cast entries with use_audio=True request an <Audio N> voice-timbre
        # reference for that subject.
        # • VIDEO mode bundles: audio_source is already set to "extract_from_visual"
        #   on the bundle video entry above — no extra entry needed.
        # • IMAGE mode bundles / source-only: add an audio_only entry pointing to
        #   the source profile video (the motion-donor clip).
        # allows_dialogue=False is this clip's "no audio involvement" setting —
        # it must also suppress use_audio extraction, not just dialogue text,
        # or a Scene Cast Audio checkbox silently pulls the clip's own audio
        # back in regardless of what's set here.
        if isinstance(scene_cast, dict) and video_file_for_entry and clip.get("allows_dialogue", True):
            _audio_load = dict(load_params, frame_load_cap=4, select_every_nth=1)
            for _ce in scene_cast.get("entries", []):
                if _ce.get("source_profile_id") != profile_id:
                    continue
                if not _ce.get("use_audio"):
                    continue
                _bid  = _ce.get("bundle_id", "")
                _ssid = _ce.get("source_subject_id", "")
                # VIDEO/BOTH mode bundle: audio handled by bundle video entry — skip.
                if _bid and _ce.get("visual_mode") in ("video", "both"):
                    continue
                _audio_sid = _bid if _bid else _ssid
                if not _audio_sid:
                    continue
                video_entries.append({
                    "subject_id":       _audio_sid,
                    "subject_ids":      [_audio_sid],
                    "video_file":       video_file_for_entry,
                    "load_params":      _audio_load,
                    "audio_source":     "extract_from_visual",
                    "audio_only":       True,
                    "audio_path":       "",
                    "audio_start_time": 0.0,
                    "audio_duration":   0.0,
                    "audio_retention":  "timbre",
                    "audio_role":       "",
                    "audio_cache":      "",
                })

        # ── Assemble prompt ────────────────────────────────────────────────────
        try:
            result = _assemble_prompt(scene_instance, model_type, video_entries)
        except Exception as exc:
            logger.error("SourceProfileClipPrompt: prompt assembly failed: %s", exc)
            return io.NodeOutput("", None, [], "", 0, 0, 0, f"Prompt assembly error: {exc}", "")

        prompt       = result["prompt"]
        concept_ids  = result.get("concept_ids", [])

        # ── Build h3_refplan (H3 models only) ─────────────────────────────────
        h3_refplan = None
        if model_type in ("h3_ref2va", "h3_fl2va"):
            rp = _build_h3_refplan(scene_instance, video_entries)
            rp["prompt"]         = prompt
            rp["model_type"]     = model_type
            rp["ref_image_size"] = "match"
            rp["has_turbo_lora"] = any(
                "turbo" in (e.get("name", "") or "").lower()
                for e in clip.get("loras", [])
            )
            h3_refplan = rp

        # ── Build LORA_STACK_DATA from clip loras ──────────────────────────────
        lora_stack_data: list = []
        for entry in clip.get("loras", []):
            name = entry.get("name", "")
            if name:
                lora_stack_data.append({
                    "lora":           name,
                    "strength_model": entry.get("strength_model", 1.0),
                    "strength_clip":  entry.get("strength_clip", 1.0),
                    "enabled":        True,
                    "model_target":   "both",
                    "audio_enabled":  False,
                })

        # ── Clip frame count + resolution ──────────────────────────────────────
        # clip_frames is the H3 output length: how many frames the model should
        # generate.  frame_load_cap and select_every_nth are VHS reference-loading
        # constraints and must NOT affect the output frame count.
        duration_s  = max(0.0, clip.get("end_time", 0.0) - clip.get("start_time", 0.0))
        clip_frames = math.ceil(duration_s * max(1, int(clip_duration_multiplier or 1)) * 24)
        _max = int(max_clip_frames or 0)
        if _max > 0:
            clip_frames = min(clip_frames, _max)
        vid_w, vid_h = _spa_probe_resolution(video_abs) if video_abs else (0, 0)

        # ── Summary ────────────────────────────────────────────────────────────
        slot_desc = ", ".join(
            f"{SOURCE_SLOTS[i]}={subject_index[sid].get('label', sid)}"
            for i, sid in enumerate(source_subject_ids)
        )
        concept_ids_str = ", ".join(c for c in concept_ids if c)

        # Bundle substitution summary — one line per matched entry
        bundle_lines: list[str] = []
        if cast_lookup and bundle_reg is None:
            bundle_lines.append("  WARNING: bundle registry failed to load — no substitutions applied")
        for src_sid, ce in cast_lookup.items():
            src_label = subject_index.get(src_sid, {}).get("label", src_sid)
            bun_id    = ce.get("bundle_id", "")
            bun       = bundle_reg.get(bun_id) if bundle_reg else None
            if bun is None:
                bundle_lines.append(f"  {src_label} → bundle '{bun_id}' NOT FOUND in registry")
            else:
                vm      = ce.get("visual_mode") or bun.get("visual", {}).get("type", "images")
                nimgs   = len(bun.get("visual", {}).get("files", []))
                has_vid = bool(bun.get("visual", {}).get("file", ""))
                aud     = bun.get("audio", {})
                aud_src = aud.get("source", "none")
                if aud_src == "extract_from_visual":
                    audio_str = f"audio=extract_from_visual:{'ok' if has_vid else 'NO VIDEO'}"
                elif aud_src == "extract_from_video":
                    aud_vf = aud.get("video_file", "")
                    audio_str = f"audio=extract_from_video:{aud_vf or 'NO FILE'}"
                elif aud_src == "file":
                    audio_str = f"audio=file:{aud.get('file', '') or 'NO FILE'}"
                else:
                    audio_str = "audio=none"
                if vm == "video":
                    media_str = f"video={'yes' if has_vid else 'NO'}"
                elif vm == "both":
                    media_str = f"images={nimgs}+video={'yes' if has_vid else 'NO'}"
                else:
                    media_str = f"images={nimgs}"
                bundle_lines.append(
                    f"  {src_label} → '{bun.get('name', bun_id)}' "
                    f"[mode={vm}, {media_str}, {audio_str}]"
                )

        # Slot map — shows which slots actually reached the assembler
        slot_map_parts = []
        for sl, sa in sorted(slot_assignments.items()):
            nimgs_sl = len(sa.get("character_sheet_images", []))
            has_aud  = bool(sa.get("voice", {}).get("audio_reference_file"))
            marker   = "📷" if nimgs_sl else ""
            marker  += "🔊" if has_aud else ""
            slot_map_parts.append(f"{sl}={sa.get('name', sl)}{marker}")
        slot_map_str = ", ".join(slot_map_parts)
        bundle_section = ("\nBundles:\n" + "\n".join(bundle_lines)) if bundle_lines else ""

        res_str = f"{vid_w}×{vid_h}" if vid_w and vid_h else "unknown"
        proxy_used = video_file_for_entry != video_abs and video_file_for_entry
        proxy_note = f"\nProxy: {os.path.basename(video_file_for_entry)}" if proxy_used else ""
        clip_summary = (
            f"Profile: {profile_name} | Clip: {clip_label} ({clip_id_used})\n"
            f"Duration: {duration_s:.1f}s → {clip_frames} frames | Resolution: {res_str}\n"
            f"Slots: {slot_map_str}\n"
            f"Model: {model_type} | LoRAs: {len(lora_stack_data)}\n"
            f"Concept IDs: {concept_ids_str or '(none)'}"
            f"{bundle_section}"
            f"{proxy_note}"
        )

        # ── Filename prefix ────────────────────────────────────────────────────
        def _slug(s: str) -> str:
            import re as _re
            return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

        filename_prefix_out = f"{filename_prefix}{_slug(profile_name)}/{_slug(clip_label)}"

        send_status_update(
            cls.node_id,
            f"Clip: {clip_label} | {duration_s:.0f}s → {clip_frames}fr | {model_type} | {len(prompt)} chars",
        )

        return io.NodeOutput(prompt, h3_refplan, lora_stack_data, concept_ids_str, clip_frames, vid_w, vid_h, clip_summary, filename_prefix_out)


# ── Source Profile REST API endpoints ─────────────────────────────────────────

@routes.post("/fbtools/source_profiles/reload")
async def _source_profiles_reload(request):
    """Increment reload counter so SourceProfileLoad/List nodes re-execute."""
    _source_profile_reload_counter = bump_reload("source_profile")
    logger.info("Source profiles reload requested (counter=%d)", _source_profile_reload_counter)
    return web.json_response({"success": True, "counter": _source_profile_reload_counter})


@routes.get("/fbtools/source_profiles/list")
async def _source_profiles_list(request):
    """Return [{id, name, media_type, media_filename, media_dir, subject_count}] sorted by name."""
    try:
        registry = _load_source_registry(default_source_profiles_path())
        items = []
        for pid, p in registry.profiles.items():
            items.append({
                "id":             pid,
                "name":           p.get("name", pid),
                "media_type":     p.get("media_type", "video"),
                "media_filename": p.get("media_filename", ""),
                "media_dir":      p.get("media_dir", "input"),
                "subject_count":  len(p.get("subjects", [])),
            })
        items.sort(key=lambda x: x["name"].lower())
        return web.json_response({"profiles": items})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/get")
async def _source_profiles_get_one(request):
    """Return a single source profile by ?id=<profile_id> or ?name=<profile_name>."""
    pid  = request.rel_url.query.get("id", "").strip()
    name = request.rel_url.query.get("name", "").strip()
    if not pid and not name:
        return web.json_response({"error": "id or name parameter required"}, status=400)
    try:
        registry = _load_source_registry(default_source_profiles_path())
        if pid:
            profile = registry.get_profile(pid)
        else:
            profile = registry.get_profile_by_name(name)
        if profile is None:
            key = pid or name
            return web.json_response({"error": f"Source profile '{key}' not found"}, status=404)
        return web.json_response(profile)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/save")
async def _source_profiles_save(request):
    """Create or update a source profile. Body: full profile dict with 'id'."""
    try:
        data = await request.json()
        pid = data.get("id", "").strip()
        if not pid:
            return web.json_response({"error": "Profile 'id' is required"}, status=400)
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        seg_dur_raw = data.get("default_segment_duration")
        registry = registry.define_profile(
            profile_id=pid,
            name=data.get("name", pid),
            media_filename=data.get("media_filename", ""),
            media_dir=data.get("media_dir", "input"),
            media_type=data.get("media_type", "video"),
            default_segment_duration=float(seg_dur_raw) if seg_dur_raw is not None else None,
            default_background_id=data.get("default_background_id", ""),
        )
        if "subjects" in data:
            registry = registry.set_subjects(pid, data["subjects"])
        if "clips" in data:
            registry = registry.set_clips(pid, data["clips"])
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"success": True, "id": pid})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/source_profiles/delete")
async def _source_profiles_delete(request):
    """Delete a source profile by ?id=<profile_id>."""
    pid = request.rel_url.query.get("id", "")
    if not pid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_source_profiles_path()
        registry = _load_source_registry(path)
        if registry.get_profile(pid) is None:
            return web.json_response({"error": f"Source profile '{pid}' not found"}, status=404)
        registry = registry.remove_profile(pid)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)










@routes.post("/fbtools/source_profiles/analyze")
async def _source_profiles_analyze(request: web.Request) -> web.Response:
    """Run a focused VLM pass on a source profile's media.

    JSON body:
        profile_id        str   — profile to analyze
        pass_type         str   — one of PASS_TYPES ("people", "setting", …)
        prompt_override   str   — optional; replaces the template body
        captioner_type    str   — "gemini_flash" uses Gemini API (reads GEMINI_API_KEY
                                  env var); everything else routes through the LLM panel model
        start_time        float — clip start in seconds (omit for single-frame mode)
        end_time          float — clip end in seconds   (omit for single-frame mode)
        select_every_nth  int   — frame sampling stride (default: clip's own value or 1)
        max_frames        int   — hard cap on frames sent to the VLM (default: 20)

    Returns:
        {
          "candidates": [ {label, role_description, entity_type, notes}, … ],
          "pass_type":  str,
          "prompt":     str,
          "frame_mode": "clip_multi" | "single",
          "frame_count": int,
        }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        prompt_override  = str(body.get("prompt_override", "")).strip()
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        api_key          = os.environ.get("GEMINI_API_KEY", "")
        _raw_start       = body.get("start_time")
        _raw_end         = body.get("end_time")
        start_time       = float(_raw_start) if _raw_start is not None else None
        end_time         = float(_raw_end)   if _raw_end   is not None else None
        select_every_nth     = int(body.get("select_every_nth", 1)) or 1
        max_frames           = int(body.get("max_frames", 20)) or 20
        video_duration       = float(body.get("video_duration", 0.0)) or 0.0
        batch_window_seconds = max(30.0, float(body.get("batch_window_seconds", 60.0)))
        # Accept either pass_types (list) or legacy pass_type (string)
        _raw_pass_types = body.get("pass_types")
        if isinstance(_raw_pass_types, list) and _raw_pass_types:
            pass_types = [str(pt).strip() for pt in _raw_pass_types if str(pt).strip() in _SPA_PASS_TYPES]
        else:
            _single = str(body.get("pass_type", "people")).strip()
            pass_types = [_single] if _single in _SPA_PASS_TYPES else ["people"]
        pass_type = pass_types[0]  # primary type for single-pass history label
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not pass_types:
        return web.json_response(
            {"error": f"pass_type must be one of {_SPA_PASS_TYPES}"}, status=400
        )

    use_clip_mode = (start_time is not None and end_time is not None and end_time > start_time)

    # Load the profile to get the media file
    try:
        registry = _load_source_registry(default_source_profiles_path())
    except Exception as exc:
        return web.json_response({"error": f"Could not load source profiles: {exc}"}, status=500)

    profile = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    media_type     = profile.get("media_type", "video")

    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    # Resolve absolute path
    base_dir = get_output_directory() if media_dir == "output" else get_input_directory()
    abs_media = os.path.join(base_dir, media_filename)
    if not os.path.exists(abs_media):
        return web.json_response({"error": f"Media file not found: {abs_media}"}, status=404)

    # Build the analysis prompt — combined for multiple pass types, single otherwise
    prompt = _spa_build_multi_prompt(pass_types, prompt_override)

    pass_label = "+".join(pass_types)
    send_status_update(
        _SPA_STATUS_ID,
        f"Source analysis: preparing {media_type} {'clip' if use_clip_mode else 'frame'} "
        f"for '{profile_id}' ({pass_label} pass)",
        source="source_profile_analysis",
    )

    frame_count = 1

    if use_clip_mode and media_type == "video":
        # Multi-frame clip path
        try:
            raw_fps = _spa_probe_fps(abs_media)
            pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                abs_media,
                start_time=start_time,
                end_time=end_time,
                max_frames=max_frames,
                select_every_nth=select_every_nth,
                raw_fps=raw_fps,
            )
            frame_count = len(pil_frames)
        except Exception as exc:
            return web.json_response({"error": f"Could not extract clip frames: {exc}"}, status=500)

        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: {frame_count} frames extracted, calling LLM ({pass_type} pass)",
            source="source_profile_analysis",
        )

        try:
            if captioner_type == "gemini_flash":
                # Gemini API doesn't support video_frames — use a contact sheet
                sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    sheet.save(tmp.name, quality=85)
                    tmp.close()
                    raw_response = await asyncio.to_thread(
                        _run_vision_inference,
                        tmp.name, prompt, "gemini_flash",
                        "auto", None, False, profile_id, pass_type,
                    )
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
            else:
                raw_response = await asyncio.to_thread(
                    _run_vision_inference_clip,
                    pil_frames, timestamps, sample_fps, raw_fps, prompt,
                    captioner_type, profile_id, pass_type,
                )

        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        frame_mode = "clip_multi"

    elif media_type == "video" and video_duration > 0:
        # Windowed whole-video analysis: split into batch_window_seconds windows,
        # run the clip-mode path on each, then deduplicate candidates by label.
        if video_duration <= batch_window_seconds:
            analyze_windows: list[tuple[float, float]] = [(0.0, video_duration)]
        else:
            analyze_windows = []
            w = 0.0
            while w < video_duration:
                analyze_windows.append((w, min(w + batch_window_seconds, video_duration)))
                w += batch_window_seconds

        n_windows = len(analyze_windows)
        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: {n_windows} window(s) × {batch_window_seconds:.0f}s "
            f"over {video_duration:.0f}s video ({pass_type} pass)",
            source="source_profile_analysis",
        )

        raw_fps = _spa_probe_fps(abs_media)
        all_candidates_raw: list[dict] = []
        seen_labels: set[str] = set()
        total_frames = 0

        for win_idx, (w_start, w_end) in enumerate(analyze_windows):
            if n_windows > 1:
                send_status_update(
                    _SPA_STATUS_ID,
                    f"Window {win_idx + 1}/{n_windows}: {w_start:.0f}s – {w_end:.0f}s ({pass_type}) ...",
                    source="source_profile_analysis",
                )
            try:
                pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                    abs_media,
                    start_time=w_start,
                    end_time=w_end,
                    max_frames=max_frames,
                    select_every_nth=select_every_nth,
                    raw_fps=raw_fps,
                )
            except Exception as exc:
                logger.warning("analyze: frame extraction failed for window %s–%s: %s",
                               w_start, w_end, exc)
                continue

            total_frames += len(pil_frames)
            send_status_update(
                _SPA_STATUS_ID,
                f"Window {win_idx + 1}/{n_windows}: {len(pil_frames)} frame(s) extracted, calling VLM...",
                source="source_profile_analysis",
            )

            try:
                if captioner_type == "gemini_flash":
                    sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                    import tempfile as _tf2
                    tmp2 = _tf2.NamedTemporaryFile(suffix=".jpg", delete=False)
                    try:
                        sheet.save(tmp2.name, quality=85)
                        tmp2.close()
                        win_raw = await asyncio.to_thread(
                            _run_vision_inference,
                            tmp2.name, prompt, "gemini_flash",
                            "auto", None, False, profile_id, pass_type,
                        )
                    finally:
                        try: os.unlink(tmp2.name)
                        except Exception: pass
                else:
                    win_raw = await asyncio.to_thread(
                        _run_vision_inference_clip,
                        pil_frames, timestamps, sample_fps, raw_fps, prompt,
                        captioner_type, profile_id, pass_type,
                    )
            except Exception as exc:
                logger.warning("analyze: VLM call failed for window %s–%s: %s",
                               w_start, w_end, exc)
                continue

            for cand in _spa_parse_response(win_raw, pass_type):
                label_key = cand.get("label", "").strip().lower()
                if label_key and label_key not in seen_labels:
                    seen_labels.add(label_key)
                    all_candidates_raw.append(cand)

            if n_windows > 1:
                send_status_update(
                    _SPA_STATUS_ID,
                    f"Window {win_idx + 1}/{n_windows}: {len(all_candidates_raw)} unique candidate(s) so far",
                    source="source_profile_analysis",
                )

        candidates = all_candidates_raw
        frame_count = total_frames
        raw_response = ""
        frame_mode = "windowed_multi"

    else:
        # Single-frame path (image media, or video with no duration provided)
        frame_path = abs_media
        _tmp_frame = None
        if media_type == "video":
            try:
                _tmp_frame = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                _tmp_frame.close()
                frame_path = _spa_extract_frame(abs_media, _tmp_frame.name, position_frac=0.10)
            except Exception as exc:
                if _tmp_frame and os.path.exists(_tmp_frame.name):
                    os.unlink(_tmp_frame.name)
                return web.json_response(
                    {"error": f"Could not extract video frame: {exc}"}, status=500
                )

        send_status_update(
            _SPA_STATUS_ID,
            f"Source analysis: calling LLM for {pass_type} pass",
            source="source_profile_analysis",
        )

        try:
            raw_response = await asyncio.to_thread(
                _run_vision_inference,
                str(frame_path), prompt, captioner_type,
                "auto", None, False, profile_id, pass_type,
            )
        finally:
            if _tmp_frame and os.path.exists(_tmp_frame.name):
                try: os.unlink(_tmp_frame.name)
                except Exception: pass

        frame_mode = "single"

    # Windowed mode builds candidates directly; other paths produce raw_response to parse.
    if frame_mode != "windowed_multi":
        candidates = _spa_parse_response(raw_response, pass_type)

    # Persist to history — use joined label for multi-pass runs, store pass_types list
    history_pass_type = "+".join(pass_types) if len(pass_types) > 1 else pass_type
    _spa_append_history(
        data_dir   = user_data_dir(),
        profile_id = profile_id,
        media_file = media_filename,
        pass_type  = history_pass_type,
        prompt     = prompt,
        candidates = candidates,
        pass_types = pass_types,
    )

    send_status_update(
        _SPA_STATUS_ID,
        f"Source analysis: {len(candidates)} candidate(s) found ({history_pass_type} pass, "
        f"{frame_count} frame(s))",
        source="source_profile_analysis",
    )

    return web.json_response({
        "candidates":  candidates,
        "pass_type":   history_pass_type,
        "pass_types":  pass_types,
        "prompt":      prompt,
        "frame_mode":  frame_mode,
        "frame_count": frame_count,
    })


@routes.get("/fbtools/source_profiles/frame_at")
async def _source_profiles_frame_at(request: web.Request) -> web.Response:
    """Return a single JPEG frame at an exact timestamp — used by the clip
    editor's Start/End boundary thumbnails. Returns raw image bytes (not JSON)
    so it can be used directly as an <img> src.

    Query params:
        profile_id  str   — required
        t           float — timestamp in seconds — required
        w           int   — thumbnail width in pixels (default 160, clamped 32-640)
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    t_raw      = request.rel_url.query.get("t", "")
    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    try:
        t = float(t_raw)
    except ValueError:
        return web.json_response({"error": "t must be a number"}, status=400)
    try:
        w = max(32, min(640, int(request.rel_url.query.get("w", "160"))))
    except ValueError:
        w = 160

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir   = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _spa_extract_frame_at_time, video_path, tmp.name, t, w)
        with open(tmp.name, "rb") as f:
            data = f.read()
        return web.Response(
            body=data, content_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except Exception as exc:
        logger.warning("frame_at extraction failed for profile %r @ %.2fs: %s", profile_id, t, exc)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


@routes.get("/fbtools/source_profiles/analysis_history")
async def _source_profiles_analysis_history(request: web.Request) -> web.Response:
    """Return analysis history entries for a source profile.

    Query params:
        profile_id  str   — required
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    if not profile_id:
        return web.json_response({"error": "profile_id parameter required"}, status=400)
    try:
        entries = _spa_history_for_profile(user_data_dir(), profile_id)
        return web.json_response({"entries": entries})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/segment_prompt_preview")
async def _source_profiles_segment_prompt_preview(request: web.Request) -> web.Response:
    """Return the exact VLM prompt Detect boundaries would send for the given
    flags/override, without running any detection. Lets the editor UI show a
    live preview as the user toggles flags or edits the override — sourced
    from the same builder the real request uses, so it can never drift.

    JSON body:
        prompt_override  str   — optional; when non-empty, flags are ignored
        flags             dict  — camera_cuts / subject_changes / lower_threshold

    Returns:
        { "prompt": str }
    """
    try:
        body            = await request.json()
        prompt_override = str(body.get("prompt_override", "")).strip()
        flags           = body.get("flags") if isinstance(body.get("flags"), dict) else None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    prompt = _spa_build_segment_prompt(prompt_override, flags)
    return web.json_response({"prompt": prompt})


@routes.post("/fbtools/source_profiles/detect_segments")
async def _source_profiles_detect_segments(request: web.Request) -> web.Response:
    """Run VLM boundary detection on a source profile's video.

    Extracts one frame per interval_seconds across the full duration,
    composites them into a labelled contact sheet, then asks the VLM
    to identify meaningful action transitions.

    JSON body:
        profile_id          str   — required
        video_duration      float — required (caller probes via /fbtools/media/info)
        interval_seconds      float — seconds between sampled frames — this is the
                                      single "precision" knob the UI exposes
                                      (default: 3.0)
        batch_window_seconds  float — split video into windows of this many seconds;
                                      each window is a separate VLM call. Normally
                                      OMITTED and auto-derived as interval_seconds * 20
                                      so every window uses the full 20-frame-per-call
                                      budget evenly (no wasted capacity, no silent
                                      truncation). Pass explicitly only to override.
        prompt_override       str   — optional full prompt replacement (bypasses flags)
        flags               dict  — optional flag overrides for prompt construction:
                                    camera_cuts (bool, default True)
                                    subject_changes (bool, default False)
                                    lower_threshold (bool, default False)
        captioner_type      str   — "qwen_vl" | "qwen_omni" | "gemini_flash"
        device              str   — "auto" | "cpu" | "cuda"
        use_8bit            bool

    Returns:
        { "segments": [{start_time, end_time, label, action, overall_soundscape,
                         setting_label}, …],
          "raw_response": str }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        video_duration   = float(body.get("video_duration", 0.0))
        interval_seconds = float(body.get("interval_seconds", 0.0)) or 0.0
        prompt_override  = str(body.get("prompt_override", "")).strip()
        flags            = body.get("flags") if isinstance(body.get("flags"), dict) else None
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        device           = str(body.get("device", "auto")).strip()
        _use_8bit_raw    = body.get("use_8bit")
        use_8bit         = bool(_use_8bit_raw) if _use_8bit_raw is not None else None
        api_key          = os.environ.get("GEMINI_API_KEY", "")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if video_duration <= 0:
        return web.json_response({"error": "video_duration must be > 0"}, status=400)

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir    = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path  = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    if not interval_seconds:
        interval_seconds = 3.0

    # Split video into fixed-size windows; each window is processed separately so
    # the VLM sees a focused contact sheet rather than one frame per many seconds.
    # Window length is derived from interval_seconds so every window lands on
    # exactly 20 frames (the per-call cap below) — full utilization of the frame
    # budget at the requested precision, with no silent truncation.
    _bw_raw = body.get("batch_window_seconds")
    batch_window_seconds = max(20.0, float(_bw_raw)) if _bw_raw else max(20.0, interval_seconds * 20.0)

    _tmp_frames: list[str] = []

    async def _run_window(win_start: float, win_end: float) -> tuple[list[dict], str]:
        """Build a contact sheet for [win_start, win_end) and run the VLM.

        Frames carry absolute timestamps, so the VLM returns absolute
        start_time/end_time values that can be concatenated across windows
        without any offset adjustment.
        """
        win_ts: list[float] = []
        t = win_start + interval_seconds / 2.0
        while t < win_end:
            win_ts.append(t)
            t += interval_seconds
        if not win_ts:
            win_ts = [(win_start + win_end) / 2.0]

        frame_paths: list[str] = []
        for ts in win_ts[:20]:  # cap at 20 frames per window
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            _tmp_frames.append(tmp.name)
            try:
                _spa_extract_frame(video_path, tmp.name, position_frac=ts / video_duration)
                frame_paths.append(tmp.name)
            except Exception:
                pass

        if not frame_paths:
            return [], ""

        contact_path = tempfile.mktemp(suffix=".jpg")
        _tmp_frames.append(contact_path)
        try:
            from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw
            imgs = [_PIL_Image.open(p).convert("RGB") for p in frame_paths]
            # Smaller thumbnails = fewer input tokens; readable enough for boundary detection
            thumb_w, thumb_h = 200, 112
            imgs = [img.resize((thumb_w, thumb_h)) for img in imgs]
            n_cols = min(len(imgs), 6)
            n_rows = (len(imgs) + n_cols - 1) // n_cols
            sheet = _PIL_Image.new("RGB", (thumb_w * n_cols, (thumb_h + 20) * n_rows), (30, 30, 30))
            draw  = _PIL_Draw.Draw(sheet)
            for i, (img, ts) in enumerate(zip(imgs, win_ts[:len(imgs)])):
                row, col = divmod(i, n_cols)
                x, y = col * thumb_w, row * (thumb_h + 20)
                sheet.paste(img, (x, y))
                draw.text((x + 4, y + thumb_h + 2), f"{ts:.1f}s", fill=(200, 200, 200))
            sheet.save(contact_path, quality=85)
        except Exception:
            import shutil as _shutil
            _shutil.copy2(frame_paths[0], contact_path)

        send_status_update(
            _SPA_STATUS_ID,
            f"Extracted {len(frame_paths)} frame(s) from {win_start:.0f}s–{win_end:.0f}s — querying VLM...",
            source="source_profile_analysis",
        )

        window_note = (
            f"\n\nNOTE: These frames cover {win_start:.1f}s – {win_end:.1f}s of a "
            f"{video_duration:.1f}s video. Return all start_time/end_time values as "
            f"absolute timestamps (seconds from the start of the full video) within "
            f"this range."
        )
        win_prompt = _spa_build_segment_prompt(prompt_override, flags) + window_note

        raw = await asyncio.to_thread(
            _run_vision_inference,
            contact_path, win_prompt, captioner_type, device, use_8bit,
            False, profile_id, "detect_segments",
            2048,  # max_tokens: segment lists can be long; 512 truncates JSON mid-object
        )
        # video_duration=0 suppresses the auto-fallback — caller handles empty windows
        segs = _spa_parse_segments(raw, 0.0)
        return segs, raw or ""

    # Build non-overlapping windows covering the full video
    if video_duration <= batch_window_seconds:
        windows: list[tuple[float, float]] = [(0.0, video_duration)]
    else:
        windows = []
        w = 0.0
        while w < video_duration:
            windows.append((w, min(w + batch_window_seconds, video_duration)))
            w += batch_window_seconds

    try:
        n_windows = len(windows)
        run_started_at = time.time()
        send_status_update(
            _SPA_STATUS_ID,
            (f"Detect segments: {n_windows} window(s) × {batch_window_seconds:.0f}s "
             f"over {video_duration:.0f}s video — sending to VLM..."),
            source="source_profile_analysis",
            extra={
                "phase":       "start",
                "profile_id":  profile_id,
                "n_windows":   n_windows,
                "windows":     [[round(s, 2), round(e, 2)] for s, e in windows],
                "video_duration": round(video_duration, 2),
            },
        )

        all_segments: list[dict] = []
        all_raw:      list[str]  = []

        for idx, (w_start, w_end) in enumerate(windows):
            win_started_at = time.time()
            start_text = (
                f"Window {idx + 1}/{n_windows}: {w_start:.0f}s – {w_end:.0f}s ..."
                if n_windows > 1 else "Analyzing video for boundaries..."
            )
            send_status_update(
                _SPA_STATUS_ID, start_text, source="source_profile_analysis",
                extra={
                    "phase": "window_start", "profile_id": profile_id,
                    "window_idx": idx, "n_windows": n_windows,
                },
            )
            segs, raw_text = await _run_window(w_start, w_end)
            all_segments.extend(segs)
            if raw_text:
                prefix = f"[{w_start:.0f}s–{w_end:.0f}s]\n" if n_windows > 1 else ""
                all_raw.append(f"{prefix}{raw_text}")
            win_elapsed = time.time() - win_started_at
            done_text = (
                f"Window {idx + 1}/{n_windows}: {len(segs)} segment(s) found"
                if n_windows > 1 else f"{len(segs)} segment(s) found"
            )
            _win_frames = min(20, math.ceil((w_end - w_start) / interval_seconds)) if interval_seconds > 0 else 0
            send_status_update(
                _SPA_STATUS_ID, done_text, source="source_profile_analysis",
                extra={
                    "phase": "window_done", "profile_id": profile_id,
                    "window_idx": idx, "n_windows": n_windows,
                    "segments_found": len(segs),
                    "elapsed_s": round(win_elapsed, 2),
                    "frames": _win_frames,
                },
            )

        total_elapsed = time.time() - run_started_at
        send_status_update(
            _SPA_STATUS_ID,
            (f"Detect segments complete: {len(all_segments)} segment(s) "
             f"from {n_windows} window(s)"),
            source="source_profile_analysis",
            extra={
                "phase": "complete", "profile_id": profile_id,
                "n_windows": n_windows, "total_segments": len(all_segments),
                "elapsed_s": round(total_elapsed, 2),
            },
        )

        if not all_segments:
            all_segments.append({
                "start_time": 0.0,
                "end_time":   round(video_duration, 3),
                "label":      "Full video",
                "action":     "",
            })

        # --- Infer subjects from action descriptions via text-only LLM ---
        inferred_subjects: list[dict] = []
        action_texts = [s.get("action", "") for s in all_segments if s.get("action", "").strip()]
        if len(action_texts) >= 2:
            send_status_update(
                _SPA_STATUS_ID,
                f"Inferring subjects from {len(action_texts)} action description(s)...",
                source="source_profile_analysis",
            )
            try:
                existing_labels = [
                    s.get("label", "") for s in profile.get("subjects", [])
                    if s.get("label", "").strip()
                ]
                subj_prompt = _spa_build_subject_inference_prompt(action_texts, existing_labels)
                raw_subjects = await asyncio.to_thread(
                    _run_text_inference,
                    # 1024 was too tight for thinking-mode models (e.g. reasoning_effort=xhigh):
                    # the <think> block alone can exhaust the budget before any JSON is emitted,
                    # silently yielding an empty response with zero error surfaced. Match the
                    # 2048 already used for segment detection for the same reason.
                    subj_prompt, captioner_type, profile_id, "subject_inference", 2048,
                )
                if not raw_subjects:
                    logger.warning(
                        "Subject inference returned empty text for profile %r "
                        "(model may be thinking-only-truncated or backend unavailable)",
                        profile_id,
                    )
                if raw_subjects:
                    inferred_subjects = _spa_parse_inferred_subjects(raw_subjects)
            except Exception as _subj_exc:
                logger.warning("Subject inference from detect segments failed: %s", _subj_exc)
            send_status_update(
                _SPA_STATUS_ID,
                f"Detect segments done: {len(all_segments)} segment(s), "
                f"{len(inferred_subjects)} inferred subject(s)",
                source="source_profile_analysis",
            )

        return web.json_response({
            "segments":          all_segments,
            "raw_response":      "\n\n".join(all_raw),
            "inferred_subjects": inferred_subjects,
        })

    except Exception as exc:
        logger.exception("detect_segments failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        for p in _tmp_frames:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


@routes.post("/fbtools/source_profiles/describe_clip")
async def _source_profiles_describe_clip(request: web.Request) -> web.Response:
    """Run VLM to generate an action description for a clip.

    Extracts up to max_frames frames spread across [start_time, end_time] and
    sends them to the VLM as a contact sheet (or natively for video-capable
    backends).  Falls back to a single midpoint frame for non-video media.

    JSON body:
        profile_id        str
        start_time        float   — clip start (seconds)
        end_time          float   — clip end (seconds)
        prompt_override   str     — optional
        existing_action   str     — optional; VLM refines rather than generating fresh
        captioner_type    str
        device            str
        use_8bit          bool
        max_frames        int     — max frames to sample (default 5)
        select_every_nth  int     — frame stride before applying max_frames cap (default 1)

    Returns:
        { "action": "1-2 sentence action description", "frame_count": int }
    """
    import tempfile

    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        start_time       = float(body.get("start_time", 0.0))
        end_time         = float(body.get("end_time", 0.0))
        prompt_override  = str(body.get("prompt_override", "")).strip()
        captioner_type   = str(body.get("captioner_type", "auto")).strip()
        device           = str(body.get("device", "auto")).strip()
        _use_8bit_raw    = body.get("use_8bit")
        use_8bit         = bool(_use_8bit_raw) if _use_8bit_raw is not None else None
        existing_action  = str(body.get("existing_action", "")).strip()
        max_frames       = int(body.get("max_frames", 5)) or 5
        select_every_nth = int(body.get("select_every_nth", 1)) or 1
        raw_subjects     = body.get("subjects") or []
        subjects: list[tuple[str, str, str]] = [
            (str(s.get("slot", "")).strip(),
             str(s.get("name", "")).strip(),
             str(s.get("appearance", "")).strip())
            for s in raw_subjects
            if isinstance(s, dict) and s.get("slot") and s.get("name")
        ]
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    registry = _load_source_registry(default_source_profiles_path())
    profile  = registry.get_profile(profile_id)
    if profile is None:
        return web.json_response({"error": f"Source profile '{profile_id}' not found"}, status=404)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    media_type     = profile.get("media_type", "video")
    if not media_filename:
        return web.json_response({"error": "Profile has no media_filename set"}, status=400)

    base_dir   = get_output_directory() if media_dir == "output" else get_input_directory()
    video_path = os.path.join(base_dir, media_filename)
    if not os.path.exists(video_path):
        return web.json_response({"error": f"Video not found: {media_filename}"}, status=404)

    prompt = _spa_build_clip_desc_prompt(
        prompt_override,
        subjects=subjects or None,
        existing_action=existing_action,
    )

    send_status_update(
        _SPA_STATUS_ID,
        f"Describe clip: extracting frames for '{profile_id}' ({start_time:.1f}–{end_time:.1f}s)",
        source="source_profile_analysis",
    )

    frame_count = 1
    try:
        if media_type == "video" and end_time > start_time:
            # Multi-frame path — mirrors the analyze endpoint
            raw_fps = _spa_probe_fps(video_path)
            pil_frames, timestamps, sample_fps = _spa_extract_clip_frames(
                video_path,
                start_time=start_time,
                end_time=end_time,
                max_frames=max_frames,
                select_every_nth=select_every_nth,
                raw_fps=raw_fps,
            )
            frame_count = len(pil_frames)
            send_status_update(
                _SPA_STATUS_ID,
                f"Describe clip: {frame_count} frames extracted, calling LLM",
                source="source_profile_analysis",
            )
            if captioner_type == "gemini_flash":
                sheet = _spa_build_contact_sheet(pil_frames, timestamps)
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    sheet.save(tmp.name, quality=85)
                    tmp.close()
                    raw = await asyncio.to_thread(
                        _run_vision_inference,
                        tmp.name, prompt, "gemini_flash",
                        device, use_8bit, False, profile_id, "describe_clip",
                    )
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
            else:
                raw = await asyncio.to_thread(
                    _run_vision_inference_clip,
                    pil_frames, timestamps, sample_fps, raw_fps,
                    prompt, captioner_type, profile_id, "describe_clip",
                )
        else:
            # Non-video or zero-duration: single midpoint frame fallback
            midpoint = (start_time + end_time) / 2.0 if end_time > start_time else start_time
            _dur: float = 0.0
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True, timeout=10,
                )
                _dur = float(result.stdout.strip())
            except Exception:
                pass
            frac = (midpoint / _dur) if _dur > 0 else 0.1
            _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            _tmp.close()
            try:
                _spa_extract_frame(video_path, _tmp.name, position_frac=max(0.0, min(1.0, frac)))
                raw = await asyncio.to_thread(
                    _run_vision_inference,
                    _tmp.name, prompt, captioner_type, device, use_8bit,
                    False, profile_id, "describe_clip",
                )
            finally:
                try: os.unlink(_tmp.name)
                except Exception: pass

        action = _spa_parse_clip_desc(raw)

        _spa_append_history(
            data_dir    = user_data_dir(),
            profile_id  = profile_id,
            media_file  = media_filename,
            pass_type   = "describe_clip",
            prompt      = prompt,
            candidates  = [],
            clip_start  = start_time,
            clip_end    = end_time,
            action      = action,
            frame_count = frame_count,
        )

        return web.json_response({"action": action, "frame_count": frame_count})

    except Exception as exc:
        logger.exception("describe_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        pass  # temp files cleaned up inside each branch above


@routes.post("/fbtools/source_profiles/auto_partition")
async def _source_profiles_auto_partition(request: web.Request) -> web.Response:
    """Auto-partition a source profile's video into equal-duration clips.

    JSON body:
        profile_id       str
        video_duration   float   — total video length in seconds
        segment_duration float   — segment length (0 = use profile default or 10s)

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body             = await request.json()
        profile_id       = str(body.get("profile_id", "")).strip()
        video_duration   = float(body.get("video_duration", 0.0))
        segment_duration = float(body.get("segment_duration", 0.0)) or None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if video_duration <= 0:
        return web.json_response({"error": "video_duration must be > 0"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.auto_partition(
            profile_id, video_duration, segment_duration=segment_duration
        )
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("auto_partition failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/set_clips")
async def _source_profiles_set_clips(request: web.Request) -> web.Response:
    """Replace a profile's entire clip list.

    Used to apply VLM-detected segment suggestions as clips in one shot.

    JSON body:
        profile_id  str
        clips       [{id, label, start_time, end_time, action, subjects, ...}, ...]

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clips      = body.get("clips")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(clips, list):
        return web.json_response({"error": "clips must be a list"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.set_clips(profile_id, clips)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("set_clips failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/merge_subjects")
async def _source_profiles_merge_subjects(request: web.Request) -> web.Response:
    """Add inferred subjects to a profile without duplicating existing ones.

    Subjects are deduplicated by label (case-insensitive).  New subjects get
    an auto-generated ID of the form ``subj_<8hex>``.

    JSON body:
        profile_id  str
        subjects    [{label, role_description, entity_type}, ...]

    Returns: { "profile": <updated profile dict>, "added": N }
    """
    import uuid as _uuid

    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        subjects   = body.get("subjects")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(subjects, list):
        return web.json_response({"error": "subjects must be a list"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if profile is None:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)

        existing_labels = {
            s.get("label", "").strip().lower()
            for s in profile.get("subjects", [])
        }

        added = 0
        for entry in subjects:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            if not label or label.lower() in existing_labels:
                continue
            subject_id = f"subj_{_uuid.uuid4().hex[:8]}"
            registry = registry.define_subject(
                profile_id,
                subject_id,
                label=label,
                role_description=str(entry.get("role_description", "")).strip(),
                entity_type=str(entry.get("entity_type", "person")).strip(),
            )
            existing_labels.add(label.lower())
            added += 1

        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id), "added": added})
    except Exception as exc:
        logger.exception("merge_subjects failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/upsert_clip")
async def _source_profiles_upsert_clip(request: web.Request) -> web.Response:
    """Add or update a single clip within a source profile.

    JSON body:
        profile_id  str
        clip        { id, label, start_time, end_time, select_every_nth,
                      frame_load_cap, subjects, action }

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clip       = body.get("clip")
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not isinstance(clip, dict) or not clip.get("id"):
        return web.json_response({"error": "clip must be an object with an id field"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.upsert_clip(profile_id, clip)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("upsert_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/source_profiles/remove_clip")
async def _source_profiles_remove_clip(request: web.Request) -> web.Response:
    """Remove a clip from a source profile.

    JSON body:
        profile_id  str
        clip_id     str

    Returns: { "profile": <updated profile dict> }
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        clip_id    = str(body.get("clip_id", "")).strip()
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)
    if not clip_id:
        return web.json_response({"error": "clip_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        if not registry.get_profile(profile_id):
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
        registry = registry.remove_clip(profile_id, clip_id)
        _save_source_registry(registry, path)
        bump_reload("source_profile")
        return web.json_response({"profile": registry.get_profile(profile_id)})
    except Exception as exc:
        logger.exception("remove_clip failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/proxy_status")
async def _source_profiles_proxy_status(request: web.Request) -> web.Response:
    """Return proxy freshness for every clip in a profile.

    Query params:
        profile_id  str

    Returns: { "clips": [ { "clip_id", "fresh": bool, "proxy_path": str|null } ] }
    """
    profile_id = request.rel_url.query.get("profile_id", "").strip()
    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if not profile:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)

        media_filename = profile.get("media_filename", "")
        media_dir      = profile.get("media_dir", "input")
        base_dir       = get_input_directory() if media_dir != "output" else get_output_directory()
        video_abs      = os.path.join(base_dir, media_filename) if media_filename else ""
        short_edge     = profile.get("proxy_short_edge", 768)

        results = []
        for clip in profile.get("clips", []):
            clip_id    = clip.get("id", "")
            start_time = float(clip.get("start_time", 0.0))
            end_time   = float(clip.get("end_time", 0.0))

            # Re-derive the expected proxy path without generating it
            from ..utils.proxy_cache import _proxy_dir, _proxy_stem, _is_fresh
            stem       = _proxy_stem(profile_id, clip_id, start_time, end_time, short_edge)
            proxy_path = _proxy_dir(str(user_data_dir())) / f"{stem}.mp4"
            fresh      = _is_fresh(proxy_path, video_abs) if video_abs else False

            results.append({
                "clip_id":    clip_id,
                "fresh":      fresh,
                "proxy_path": str(proxy_path) if fresh else None,
            })

        return web.json_response({"clips": results})
    except Exception as exc:
        logger.exception("proxy_status failed for profile %r", profile_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/source_profiles/proxy_stream")
async def _source_profiles_proxy_stream(request: web.Request) -> web.Response:
    """Stream a cached source-profile proxy file for playback.

    ?path=<absolute_path> — must be inside user_data_dir()/proxies/source_profiles/.
    Supports HTTP Range requests so browsers can seek. Mirrors
    /fbtools/bundles/audio_cache/stream's allow-listed-root pattern. Never generates a proxy
    (that's prebuild_proxies' job, as a background job — ffmpeg here can take minutes) — this
    only ever serves one that already exists.
    """
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="path required")
    allowed_root = os.path.realpath(os.path.join(str(user_data_dir()), "proxies", "source_profiles"))
    real_path = os.path.realpath(path)
    if not real_path.startswith(allowed_root + os.sep):
        return web.Response(status=403, text="Forbidden")
    if not os.path.isfile(real_path):
        return web.Response(status=404, text="Not found")
    return web.FileResponse(real_path)


@routes.post("/fbtools/source_profiles/prebuild_proxies")
async def _source_profiles_prebuild_proxies(request: web.Request) -> web.Response:
    """Pre-generate proxies for all clips in a profile.

    JSON body:
        profile_id  str
        clip_id     str|null  (optional — build only this one clip)

    Progress is broadcast via websocket as fbtools.status events with
    source="proxy_build".  Returns immediately with { "started": true };
    the caller polls proxy_status or listens to websocket events.
    """
    try:
        body       = await request.json()
        profile_id = str(body.get("profile_id", "")).strip()
        only_clip  = (body.get("clip_id") or "").strip() or None
    except Exception as exc:
        return web.json_response({"error": f"Invalid request body: {exc}"}, status=400)

    if not profile_id:
        return web.json_response({"error": "profile_id is required"}, status=400)

    try:
        path     = default_source_profiles_path()
        registry = _load_source_registry(path)
        profile  = registry.get_profile(profile_id)
        if not profile:
            return web.json_response({"error": f"Profile '{profile_id}' not found"}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    media_filename = profile.get("media_filename", "")
    media_dir      = profile.get("media_dir", "input")
    base_dir       = get_input_directory() if media_dir != "output" else get_output_directory()
    video_abs      = os.path.join(base_dir, media_filename) if media_filename else ""
    short_edge     = profile.get("proxy_short_edge", 768)
    clips          = profile.get("clips", [])
    if only_clip:
        clips = [c for c in clips if c.get("id") == only_clip]

    if not video_abs or not os.path.exists(video_abs):
        return web.json_response({"error": "Source video file not found"}, status=404)

    def _build_all():
        total = len(clips)
        for i, clip in enumerate(clips):
            clip_id    = clip.get("id", "")
            start_time = float(clip.get("start_time", 0.0))
            end_time   = float(clip.get("end_time", 0.0))
            label      = clip.get("label", clip_id)
            send_status_update(
                "proxy_build",
                f"Building proxy {i + 1}/{total}: {label} ({start_time:.1f}–{end_time:.1f}s)",
                source="proxy_build",
            )
            try:
                result = _ensure_proxy(
                    source_path=video_abs,
                    profile_id=profile_id,
                    clip_id=clip_id,
                    start_time=start_time,
                    end_time=end_time,
                    short_edge=short_edge,
                    base_dir=str(user_data_dir()),
                )
                status = "ready" if result else "failed"
            except Exception as exc:
                status = f"error: {exc}"
                logger.warning("prebuild_proxies: clip %r failed: %s", clip_id, exc)
            send_status_update(
                "proxy_build",
                f"Proxy {i + 1}/{total} {status}: {label}",
                source="proxy_build",
            )
        send_status_update(
            "proxy_build",
            f"Proxy build complete ({total} clip{'s' if total != 1 else ''})",
            source="proxy_build",
        )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build_all)
    return web.json_response({"started": True, "clip_count": len(clips)})
