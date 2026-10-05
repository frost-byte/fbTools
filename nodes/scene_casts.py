"""Scene Cast nodes: SceneCastLoad, SceneCastBuild.

Moved out of extension.py (Plan 29, pure code motion). composition_ordinal_roster used to be
available here only by extension.py's own top-to-bottom load order (imported ~500 lines below
where SceneCastBuild used to live) -- same class of landmine Plan 28 found for
_build_h3_refplan -- so this is a genuine new top-of-file import, not just a re-shuffle.
"""
from __future__ import annotations

import json
import os

from comfy_api.latest import io

from .shared import (
    prefixed_node_id, default_cast_registry_path, default_bundle_registry_path,
    default_subject_profiles_path, default_source_profiles_path, user_data_dir,
    reload_counter, send_status_update,
)
from .composition_types import CastIOType, SourceProfileIOType, CompositionIOType
from ..utils.scene_casts import (
    load_registry as _load_cast_registry,
    resolve_primary_subject as _resolve_primary_subject,
    resolve_primary_bundle as _resolve_primary_bundle,
    build_cast_filename_prefix as _build_cast_filename_prefix,
)
from ..utils.reference_bundles import load_registry as _load_bundle_registry
from ..utils.subject_profiles import load_registry as _load_subject_registry
from ..utils.source_profiles import (
    resolved_pronoun_style as _sp_resolved_pronoun_style,
    resolve_ordinal_subject as _sp_resolve_ordinal_subject,
    resolve_ordinal_from_list as _sp_resolve_ordinal_from_list,
)
from ..utils.prompt_compositions import (
    composition_ordinal_roster as _composition_ordinal_roster,
    _slugify,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Scene Cast nodes  (Reference Bundle & Scene Cast system)
# ══════════════════════════════════════════════════════════════════════════════

# CastIOType/H3RefplanType moved to nodes/composition_types.py (Plan 27)
# ── Scene Cast helpers ────────────────────────────────────────────────────────

def _cast_get_ids() -> list[str]:
    """Return available cast IDs for combo population at schema time."""
    try:
        registry = _load_cast_registry(default_cast_registry_path())
        ids = registry.cast_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: SceneCastLoad ───────────────────────────────────────────────────────

class SceneCastLoad(io.ComfyNode):
    """Load a Scene Cast from disk.

    A Scene Cast assigns a reference bundle to each subject in a composition,
    with per-entry visual mode and audio toggles.  Wire the SCENE_CAST output
    into PromptCompositionLoader to resolve reference media during assembly.
    The combo is populated at extension load time; press R to refresh after
    saving a new cast in the Scene Casts sidebar panel.
    """
    node_id = prefixed_node_id("SceneCastLoad")
    display_name = "Scene Cast Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        cast_ids = _cast_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "cast_id",
                    options=cast_ids,
                    display_name="Cast ID",
                    tooltip="Scene cast to load. Press R to refresh after saving a new cast.",
                ),
            ],
            outputs=[
                CastIOType.Output(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Cast dict for wiring into PromptCompositionLoader.",
                ),
                io.String.Output(
                    "cast_summary",
                    display_name="Cast Summary",
                    tooltip="Human-readable summary of the cast entries.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cast_id: str = "", **_):
        path = default_cast_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, cast_id, mtime, reload_counter("cast"))

    @classmethod
    def execute(cls, cast_id: str = "") -> io.NodeOutput:
        if not cast_id or cast_id == "(none)":
            logger.warning("SceneCastLoad: no cast_id selected")
            return io.NodeOutput(None, "")

        path = default_cast_registry_path()
        registry = _load_cast_registry(path)
        cast = registry.get(cast_id)

        if cast is None:
            logger.warning("SceneCastLoad: cast_id %r not found in %s", cast_id, path)
            return io.NodeOutput(None, f"Cast not found: {cast_id}")

        entries = cast.get("entries", [])
        n = len(entries)
        lines = [f"Cast: {cast.get('name', cast_id)}  ({n} {'subject' if n == 1 else 'subjects'})"]
        for e in entries:
            mode = e.get("visual_mode", "images")
            audio_flag = " + audio" if e.get("use_audio") else ""
            lines.append(
                f"  • {e.get('subject_id', '?')} → {e.get('bundle_id', '?')} [{mode}{audio_flag}]"
            )

        summary = "\n".join(lines)
        send_status_update(
            cls.node_id,
            f"Loaded cast: {cast.get('name', cast_id)} | {n} {'entry' if n == 1 else 'entries'}",
        )
        return io.NodeOutput(cast, summary, ui={"cast_summary": summary})


# ── Node: SceneCastBuild ──────────────────────────────────────────────────────

class SceneCastBuild(io.ComfyNode):
    """Build a Scene Cast inline, with an optional Source Profile input.

    Accepts standalone subject → bundle assignments alongside source-derived
    subjects from a connected SourceProfileLoad node.  Assignments and retention
    modes are encoded in cast_entries_json (managed by the Scene Casts sidebar).

    Each entry may be:
      • Bundle-backed  — subject_id + bundle_id
      • Source-derived — subject_id + source_profile_id + source_subject_id
      • Hybrid         — bundle + source (bundle provides media, source provides video ref)

    Outputs the same SCENE_CAST type as SceneCastLoad, plus source_profile and
    clip_id pass-throughs so SourceProfileClipPrompt can be wired downstream
    without re-connecting the profile.
    """

    node_id = prefixed_node_id("SceneCastBuild")
    display_name = "Scene Cast Build"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "cast_entries_json",
                    display_name="Cast Entries",
                    default="[]",
                    tooltip=(
                        "JSON array of cast entries. "
                        "Managed via the Scene Casts sidebar — do not edit by hand."
                    ),
                ),
                SourceProfileIOType.Input(
                    "source_profile",
                    display_name="Source Profile",
                    optional=True,
                    tooltip="Source profile whose subjects are available as cast pool options.",
                ),
                CompositionIOType.Input(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    optional=True,
                    tooltip=(
                        "Prompt Composition whose subjects are the cast pool. Use this "
                        "instead of a Source Profile; if both are connected the Source "
                        "Profile drives the node and this input is ignored."
                    ),
                ),
                io.String.Input(
                    "clip_id",
                    display_name="Clip ID",
                    default="",
                    tooltip=(
                        "Optional clip ID from the Source Profile. "
                        "When set, only frames within that clip's time window are loaded."
                    ),
                ),
                io.Int.Input(
                    "clip_duration_multiplier",
                    display_name="Clip Duration Multiplier",
                    default=1,
                    min=1,
                    max=4,
                    tooltip=(
                        "Duration multiplier (1x–4x) managed by the timeline UI. "
                        "Wire to SourceProfileClipPrompt to scale the output frame count."
                    ),
                ),
                io.String.Input(
                    "composition_overrides_json",
                    display_name="Composition Overrides",
                    default="{}",
                    optional=True,
                    tooltip=(
                        "JSON object managed by the on-node Composition options block: "
                        "background id and background-as-reference overrides applied on top "
                        "of the connected Prompt Composition. Empty = use the composition's "
                        "own values. Do not edit by hand. Ignored in Source Profile mode — "
                        "use background_override_id instead."
                    ),
                ),
                io.String.Input(
                    "background_override_id",
                    display_name="Background Override",
                    default="",
                    optional=True,
                    tooltip=(
                        "Source Profile mode only: override the background used for this "
                        "generation without editing the clip's own background_id. Empty = use "
                        "the clip's background_id (falling back to the profile's "
                        "default_background_id). 'none' = explicitly no background for this "
                        "run. Managed by the on-node Background Override dropdown — do not "
                        "edit by hand. Ignored when a Prompt Composition drives the node "
                        "instead (use composition_overrides_json there)."
                    ),
                ),
                io.String.Input(
                    "action_preview",
                    display_name="Action Preview",
                    default="",
                    optional=True,
                    multiline=True,
                    tooltip=(
                        "Read-only. The active clip's action text with {A}/{B}/… "
                        "placeholders resolved to bundle names — computed and written "
                        "by the on-node preview widget so it rides along in the "
                        "submitted prompt for Run History tracking. Not read by "
                        "execute(); do not edit by hand."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional literal root (e.g. 'video/'). The primary-subject and "
                        "comps/source-profile segments are added automatically from "
                        "whichever cast entry's tab is marked primary (★) and whichever "
                        "of Source Profile / Prompt Composition is connected."
                    ),
                ),
            ],
            outputs=[
                CastIOType.Output(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip="Inline cast dict for wiring into PromptCompositionLoader.",
                ),
                io.String.Output(
                    "cast_summary",
                    display_name="Cast Summary",
                    tooltip="Human-readable summary of configured entries.",
                ),
                SourceProfileIOType.Output(
                    "source_profile",
                    display_name="Source Profile",
                    tooltip="Pass-through of the connected Source Profile (for SourceProfileClipPrompt).",
                ),
                io.String.Output(
                    "clip_id",
                    display_name="Clip ID",
                    tooltip="Pass-through of the selected Clip ID (for SourceProfileClipPrompt).",
                ),
                io.Int.Output(
                    "clip_duration_multiplier",
                    display_name="Duration Multiplier",
                    tooltip=(
                        "Pass-through of the duration multiplier set in the timeline UI (1–4). "
                        "Wire to SourceProfileClipPrompt clip_duration_multiplier input."
                    ),
                ),
                CompositionIOType.Output(
                    "prompt_composition",
                    display_name="Prompt Composition",
                    tooltip="Pass-through of the connected Prompt Composition.",
                ),
                io.String.Output(
                    "filename_prefix",
                    display_name="Filename Prefix",
                    tooltip=(
                        "prefix + primary_subject_id + bundle_id + compositions|source_profiles/<name> "
                        "(e.g. 'video/alex/alex_salon_eyes/compositions/'). Wire into "
                        "SourceProfileClipPrompt or PromptCompositionLoader's own "
                        "filename_prefix input, which appends the clip/composition name. "
                        "Empty if no cast entry is marked primary; the bundle_id segment is "
                        "omitted if the primary entry has none (a source-only entry)."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        cast_entries_json: str = "[]",
        source_profile=None,
        clip_id: str = "",
        clip_duration_multiplier: int = 1,
        prompt_composition=None,
        composition_overrides_json: str = "{}",
        background_override_id: str = "",
        filename_prefix: str = "",
        **_,
    ):
        bundle_mtime = subject_mtime = source_mtime = 0
        try:
            bundle_mtime = os.path.getmtime(default_bundle_registry_path())
        except OSError:
            pass
        try:
            subject_mtime = os.path.getmtime(
                os.path.join(user_data_dir(), "subject_profiles.json")
            )
        except OSError:
            pass
        try:
            source_mtime = os.path.getmtime(default_source_profiles_path())
        except OSError:
            pass
        sp_id = source_profile.get("id", "") if isinstance(source_profile, dict) else ""
        pc_id = prompt_composition.get("id", "") if isinstance(prompt_composition, dict) else ""
        pc_subjects = json.dumps(prompt_composition.get("subjects", {}), sort_keys=True) if isinstance(prompt_composition, dict) else ""
        return (bundle_mtime, subject_mtime, source_mtime, cast_entries_json, clip_id, sp_id,
                clip_duration_multiplier, pc_id, pc_subjects, composition_overrides_json,
                background_override_id, filename_prefix)

    @classmethod
    def execute(
        cls,
        cast_entries_json: str = "[]",
        source_profile=None,
        clip_id: str = "",
        clip_duration_multiplier: int = 1,
        prompt_composition=None,
        composition_overrides_json: str = "{}",
        background_override_id: str = "",
        filename_prefix: str = "",
        **_,
    ) -> io.NodeOutput:
        try:
            raw = json.loads(cast_entries_json or "[]")
            if not isinstance(raw, list):
                raw = []
        except Exception:
            raw = []

        # Build connected source profiles dict: profile_id → profile dict
        connected_profiles: dict = {}
        if isinstance(source_profile, dict):
            pid = source_profile.get("id", "")
            if pid:
                connected_profiles[pid] = source_profile

        # Composition-driven cast: the composition's own subjects are the pool
        # and ordinal entries resolve against them. A Source Profile, when also
        # connected, keeps driving the node (the UI hides the composition then).
        comp_roster = None
        if isinstance(prompt_composition, dict):
            if isinstance(source_profile, dict):
                logger.warning(
                    "SceneCastBuild: both source_profile and prompt_composition are connected; "
                    "using the Source Profile and ignoring the Prompt Composition."
                )
            else:
                comp_roster = _composition_ordinal_roster(
                    prompt_composition,
                    _load_subject_registry(default_subject_profiles_path()).get_subject,
                )

        _RETENTION_BUNDLE = "fully_preserved"
        _RETENTION_SOURCE = "partially_preserved"

        # Bundle/subject registries — only needed to resolve a bundle's own
        # pronoun_style for ordinal-match entries; loaded once up front.
        _ordinal_entries_present = any(str(e.get("match_mode", "")).strip() == "ordinal" for e in raw)
        bundle_reg = subject_reg = None
        if _ordinal_entries_present:
            try:
                bundle_reg = _load_bundle_registry(default_bundle_registry_path())
            except Exception:
                bundle_reg = None
            try:
                subject_reg = _load_subject_registry(default_subject_profiles_path())
            except Exception:
                subject_reg = None

        entries = []
        for e in raw:
            subject_id        = str(e.get("subject_id",        "")).strip()
            source_profile_id = str(e.get("source_profile_id", "")).strip()
            source_subject_id = str(e.get("source_subject_id", "")).strip()
            bundle_id         = str(e.get("bundle_id",         "")).strip()
            retention         = str(e.get("retention",         "")).strip()

            # Ordinal match: resolve source_subject_id fresh against whichever
            # profile/clip is *currently connected* — this node only ever has
            # one source_profile input, so there is nothing to disambiguate
            # and no reason to gate on a previously-stored source_profile_id,
            # which goes stale the moment the upstream Source Profile is
            # swapped for a different one (unlike an explicit subject pick
            # below, which legitimately should invalidate if its specific
            # profile disconnects). No match (or nothing to resolve against)
            # → clear source linkage entirely, falling through to the
            # bundle-only branch below, exactly as if no source had ever been
            # assigned for this subject.
            if (comp_roster is not None
                    and str(e.get("match_mode", "")).strip() == "ordinal" and bundle_id):
                # Composition ordinal: "the Nth subject in this composition sharing
                # my bundle's pronoun_style" becomes the entry's subject_id, and the
                # entry stays a plain bundle-backed one (no source linkage).
                _bundle = bundle_reg.get(bundle_id) if bundle_reg else None
                _resolved = ""
                if _bundle is not None:
                    _bun_subj = (subject_reg.get_subject(_bundle.get("subject_id", ""))
                                 if subject_reg and _bundle.get("subject_id") else None)
                    _pronoun = _sp_resolved_pronoun_style(
                        _bundle.get("entity_type", "person"),
                        (_bun_subj.get("pronoun_style") if _bun_subj else "") or _bundle.get("pronoun_style", ""),
                    )
                    try:
                        _n = int(e.get("ordinal", 0))
                    except (TypeError, ValueError):
                        _n = 0
                    _resolved = _sp_resolve_ordinal_from_list(comp_roster, _pronoun, _n, id_key="subject_id")
                if not _resolved:
                    logger.warning(
                        "SceneCastBuild: ordinal entry for bundle %r matched no subject in the composition, skipping",
                        bundle_id,
                    )
                    continue
                subject_id = _resolved
                source_profile_id = source_subject_id = ""
            elif (str(e.get("match_mode", "")).strip() == "ordinal"
                    and bundle_id and not source_subject_id):
                profile = source_profile if isinstance(source_profile, dict) else None
                bundle  = bundle_reg.get(bundle_id) if bundle_reg else None
                if profile is not None and bundle is not None and clip_id:
                    bun_subj = (subject_reg.get_subject(bundle.get("subject_id", ""))
                                if subject_reg and bundle.get("subject_id") else None)
                    bun_pronoun = _sp_resolved_pronoun_style(
                        bundle.get("entity_type", "person"),
                        (bun_subj.get("pronoun_style") if bun_subj else "") or bundle.get("pronoun_style", ""),
                    )
                    try:
                        ordinal = int(e.get("ordinal", 0))
                    except (TypeError, ValueError):
                        ordinal = 0
                    source_subject_id = _sp_resolve_ordinal_subject(profile, clip_id, bun_pronoun, ordinal)
                    if source_subject_id:
                        source_profile_id = profile.get("id", "")  # always the live connected profile
                if not source_subject_id:
                    source_profile_id = ""  # no match — treat as if never assigned

            if source_profile_id and source_subject_id:
                # Resolve source subject (shared by source-only and hybrid paths)
                profile = connected_profiles.get(source_profile_id)
                if profile is None:
                    logger.warning(
                        "SceneCastBuild: source_profile_id %r not connected, skipping",
                        source_profile_id,
                    )
                    continue
                subject_entry = next(
                    (s for s in profile.get("subjects", [])
                     if s.get("id") == source_subject_id),
                    None,
                )
                if subject_entry is None:
                    logger.warning(
                        "SceneCastBuild: subject %r not found in profile %r, skipping",
                        source_subject_id, source_profile_id,
                    )
                    continue
                src_fields = {
                    "source_profile_id": source_profile_id,
                    "source_subject_id": source_subject_id,
                    "role_description":  subject_entry.get("role_description", ""),
                    "entity_type":       subject_entry.get("entity_type", "person"),
                    "source_media_file": profile.get("media_filename", ""),
                    "source_media_dir":  profile.get("media_dir", "input"),
                    "source_media_type": profile.get("media_type", "video"),
                }

                if bundle_id:
                    # Hybrid: bundle provides appearance/media, source provides video reference
                    visual_mode    = e.get("visual_mode", "images")
                    use_audio      = bool(e.get("use_audio", False))
                    image_selection = e.get("image_selection")
                    entries.append({
                        "subject_id":      subject_id or source_subject_id,
                        "bundle_id":       bundle_id,
                        "visual_mode":     visual_mode if visual_mode in ("images", "video", "both") else "images",
                        "use_audio":       use_audio,
                        "image_selection": image_selection,
                        "retention":       retention or _RETENTION_SOURCE,
                        "dialogue":        str(e.get("dialogue", "") or "").strip(),
                        "primary":         bool(e.get("primary", False)),
                        "include_video_background": bool(e.get("include_video_background", False)),
                        **src_fields,
                    })
                else:
                    # Source-only: no bundle
                    entries.append({
                        "subject_id": subject_id or source_subject_id,
                        "retention":  retention or _RETENTION_SOURCE,
                        "dialogue":   str(e.get("dialogue", "") or "").strip(),
                        "primary":    bool(e.get("primary", False)),
                        **src_fields,
                    })

            elif subject_id and bundle_id:
                # Bundle-only: no source video reference
                visual_mode     = e.get("visual_mode", "images")
                use_audio       = bool(e.get("use_audio", False))
                image_selection = e.get("image_selection")
                entries.append({
                    "subject_id":      subject_id,
                    "bundle_id":       bundle_id,
                    "visual_mode":     visual_mode if visual_mode in ("images", "video", "both") else "images",
                    "use_audio":       use_audio,
                    "image_selection": image_selection,
                    "retention":       retention or _RETENTION_BUNDLE,
                    "dialogue":        str(e.get("dialogue", "") or "").strip(),
                    "primary":         bool(e.get("primary", False)),
                    "include_video_background": bool(e.get("include_video_background", False)),
                })

        # Flag (never silently resolve) two or more entries landing on the same
        # source subject — most likely two ordinal entries whose bundles share
        # a pronoun_style classification and whose ordinals don't actually pick
        # out distinct subjects in this clip, or an ordinal entry colliding
        # with an explicit one. Only one bundle can really replace a given
        # subject; whichever entry appears last in cast_entries_json is the one
        # whose bundle_id/visual settings "win" for that subject downstream,
        # but neither claim is intentional here, so surface it loudly.
        _claims: dict[str, list[str]] = {}
        for entry in entries:
            sid = entry.get("source_subject_id")
            if sid:
                _claims.setdefault(sid, []).append(entry.get("subject_id", "?"))
        for sid, claimants in _claims.items():
            if len(claimants) > 1:
                logger.warning(
                    "SceneCastBuild: %d cast entries (%s) all resolved to the same source "
                    "subject %r for clip %r — check for an ordinal/explicit assignment "
                    "conflict (e.g. two bundles sharing a pronoun_style whose ordinals "
                    "don't pick out distinct subjects in this clip).",
                    len(claimants), ", ".join(claimants), sid, clip_id,
                )

        # Map profile_id → clip_id
        clip_ids: dict = {}
        if isinstance(source_profile, dict) and clip_id and clip_id.strip():
            pid = source_profile.get("id", "")
            if pid:
                clip_ids[pid] = clip_id.strip()

        cast = {
            "id":              "_inline",
            "name":            "_inline",
            "entries":         entries,
            "source_profiles": connected_profiles,
            "clip_ids":        clip_ids,
        }
        # Per-run overrides for the connected composition (background etc.).
        # Ignored in Source Profile mode, where the composition is not the driver.
        if isinstance(prompt_composition, dict) and prompt_composition and not connected_profiles:
            try:
                ov = json.loads(composition_overrides_json or "{}")
            except Exception:
                ov = {}
            if isinstance(ov, dict) and ov:
                cast["composition_overrides"] = ov

        # Per-run background override for Source Profile mode — a separate mechanism
        # from composition_overrides above (that one is explicitly ignored here).
        # Read by SourceProfileClipPrompt ahead of the clip's own background_id.
        if connected_profiles and str(background_override_id or "").strip():
            cast["background_override"] = str(background_override_id).strip()

        n = len(entries)
        has_sp = bool(connected_profiles)
        lines = [f"Inline cast  ({n} {'subject' if n == 1 else 'subjects'}"
                 + (" + source profile" if has_sp else "") + ")"]
        for e in entries:
            ret_str = f" [{e.get('retention', '')}]" if e.get("retention") else ""
            bullet = "★" if e.get("primary") else "•"
            if e.get("source_profile_id") and e.get("bundle_id"):
                # Hybrid
                audio_flag = " + audio" if e.get("use_audio") else ""
                sp_subj = e.get("source_subject_id", "?")
                lines.append(
                    f"  {bullet} {e['subject_id']} [{e['bundle_id']},"
                    f" {e['visual_mode']}{audio_flag}] + src:{sp_subj}{ret_str}"
                )
            elif e.get("source_profile_id"):
                # Source-only
                etype = e.get("entity_type", "?")
                role  = e.get("role_description", e.get("subject_id", "?"))
                lines.append(f"  {bullet} [{etype}] {role}{ret_str}")
            else:
                # Bundle-only
                audio_flag = " + audio" if e.get("use_audio") else ""
                lines.append(
                    f"  {bullet} {e['subject_id']} → {e['bundle_id']} [{e['visual_mode']}{audio_flag}]{ret_str}"
                )
        summary = "\n".join(lines)

        # ── filename_prefix: {prefix}{primary_subject_id}/{bundle_id}/{compositions/<name>|source_profiles/<name>}/ ──
        # No fallback guessing here when nothing is tagged primary — that inference stays
        # in utils.generation_metadata.extract_cast_info() for archive-time reprocessing
        # of clips generated before this existed. See the design notes in
        # ~/.claude/plans/scene-cast-primary-subject-path.md.
        primary_subject_id = _resolve_primary_subject(entries)
        primary_bundle_id = _resolve_primary_bundle(entries)
        if has_sp:
            kind = f"source_profiles/{_slugify(source_profile.get('name', ''))}"
        elif isinstance(prompt_composition, dict) and prompt_composition:
            comp_label = prompt_composition.get("name") or prompt_composition.get("id") or "composition"
            kind = f"compositions/{_slugify(comp_label)}"
        else:
            kind = ""
        filename_prefix_out = _build_cast_filename_prefix(
            filename_prefix, primary_subject_id, primary_bundle_id, kind)

        send_status_update(
            cls.hidden.unique_id,
            f"Inline cast: {n} {'entry' if n == 1 else 'entries'}"
            + (" | source profile" if has_sp else ""),
        )
        mult = max(1, int(clip_duration_multiplier or 1))
        return io.NodeOutput(cast, summary, source_profile or {}, clip_id or "", mult,
                              prompt_composition or {}, filename_prefix_out)
