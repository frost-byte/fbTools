"""Recover cast/composition info from a generated clip's own embedded metadata.

Pure stdlib (plus an ffprobe shell-out, same pattern as utils/kdenlive_clips.py; no sibling utils
imports — see docs/GOTCHAS.md). Every clip ComfyUI saves carries the full API-format prompt graph
as container metadata (the same JSON utils/kdenlive_archive.py / utils/kdenlive_clips.py strip out
for Kdenlive) — read before stripping, it already has everything needed to know which bundles were
used in a clip and which subject is the "primary" one, with no new tracking required.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

# Node class_type strings (EXTENSION_PREFIX + display name, see nodes/shared.py::prefixed_node_id).
_SCENE_CAST_BUILD = "fbt_SceneCastBuild"
_COMPOSITION_LOAD = "fbt_CompositionLoad"
_PROMPT_COMPOSITION_LOADER = "fbt_PromptCompositionLoader"

# Custom mp4/mov metadata key a cleaned clip can carry a small extract_cast_info() summary under —
# written via utils.kdenlive_clips.strip_copy_video(extra_metadata=...), read back here. Outside
# ffmpeg's mov/mp4 "classic" key whitelist, so writing it requires -movflags use_metadata_tags (see
# strip_copy_video's own note — confirmed empirically to be the same mechanism ComfyUI's own
# "workflow"/"prompt" tags rely on).
CAST_SUMMARY_TAG = "fbtools_cast"


def _find_ffprobe() -> str | None:
    """Duplicated from utils/kdenlive_clips.py::find_ffprobe — pure utils modules in this repo
    don't import each other (see docs/GOTCHAS.md); keep the two in lockstep if this changes."""
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


def read_embedded_prompt(video_path: str, ffprobe: str | None = None) -> dict | None:
    """The API-format prompt graph embedded in a saved clip's metadata, or None if there
    isn't one (an older clip, or one already stripped)."""
    exe = ffprobe or _find_ffprobe()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format_tags=prompt",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        raw = out.stdout.strip()
        if not raw:
            return None
        graph = json.loads(raw)
        return graph if isinstance(graph, dict) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def generated_at_iso(src_path: str) -> str:
    """src_path's own mtime as an ISO 8601 UTC string — used as the "generated at" timestamp for a
    clip, since that's the closest we have to real generation time (not re-derived from anything
    already lossy, but not authoritative either: it's filesystem mtime at the moment this was
    computed, which is itself only as reliable as whatever copies produced the file up to here)."""
    return datetime.fromtimestamp(os.path.getmtime(src_path), tz=timezone.utc).isoformat()


def build_cast_summary_tag(info: dict, generated_at: str) -> str:
    """Compact JSON string for CAST_SUMMARY_TAG, from an extract_cast_info() result plus a
    generated_at timestamp (see generated_at_iso). Pure — no I/O."""
    return json.dumps({
        "composition": info.get("composition_name"),
        "primary_subject": info.get("primary_subject"),
        "primary_bundle": info.get("primary_bundle"),
        "tags": info.get("tags", []),
        "generated_at": generated_at,
    }, separators=(",", ":"))


def read_cast_summary_tag(video_path: str, ffprobe: str | None = None) -> dict | None:
    """Read a previously-embedded CAST_SUMMARY_TAG back, e.g. from a clip whose original, richer
    embedded prompt (read_embedded_prompt) is already gone because it was cleaned earlier. None if
    the tag is absent, unreadable, or not a JSON object."""
    exe = ffprobe or _find_ffprobe()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", f"format_tags={CAST_SUMMARY_TAG}",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        raw = out.stdout.strip()
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _find_nodes(graph: dict, class_type: str) -> list[dict]:
    return [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == class_type]


def _is_link(value) -> bool:
    """API-format prompt inputs represent a wired connection as [node_id, output_slot]."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def _composition_name(graph: dict, cast_node: dict) -> str | None:
    """The composition actually used by `cast_node` (an fbt_SceneCastBuild), mirroring
    PromptCompositionLoader.execute()'s own precedence: a wired prompt_composition always wins
    over a loader's own composition_name combo, which can be stale once that input is wired
    (this session hit exactly that bug once already)."""
    inputs = cast_node.get("inputs", {})
    pc_input = inputs.get("prompt_composition")
    if _is_link(pc_input):
        origin_id = str(pc_input[0])
        origin = graph.get(origin_id)
        if isinstance(origin, dict) and origin.get("class_type") == _COMPOSITION_LOAD:
            name = origin.get("inputs", {}).get("composition_name")
            if isinstance(name, str) and name and name != "(none)":
                return name
        return None  # wired to something else / unresolvable — don't guess

    # Not wired: fall back to whichever PromptCompositionLoader consumed this cast (if any).
    for loader in _find_nodes(graph, _PROMPT_COMPOSITION_LOADER):
        li = loader.get("inputs", {})
        if li.get("scene_cast") is None:
            continue
        name = li.get("composition_name")
        if isinstance(name, str) and name and name != "(none)":
            return name
    return None


def extract_cast_info(prompt_graph: dict, load_composition=None) -> dict:
    """Everything Kdenlive organizing needs, read straight out of an embedded prompt graph.

    load_composition(name) -> composition dict | None, injected so this module stays free of any
    fbTools-specific composition-loading/path code (the caller — nodes/kdenlive_archive.py — wires
    in utils.prompt_compositions.load_composition()).

    Returns {"tags": [bundle_id, ...], "primary_subject": subject_id | None,
             "primary_bundle": bundle_id | None, "composition_name": str | None, "note": str | None}.
    tags is every distinct bundle_id from the SceneCastBuild cast entries, in first-seen order.

    primary_subject prefers an explicit tag: a cast entry with "primary": true (set via the ★
    toggle in the Scene Cast Build tab UI — see nodes/scene_casts.py::SceneCastBuild.execute()'s
    own filename_prefix computation, which this mirrors) always wins, for both Composition- and
    Source-Profile-driven clips alike. Only when no entry is tagged does this fall back to the
    composition's first slot (dict insertion order — never sorted: slot_letter()'s A..Z, AA..
    scheme sorts wrong past Z as plain strings) — which, as before, only works for a
    Composition-driven clip; a Source-Profile-driven clip with nothing tagged still can't be
    resolved. primary_bundle is the specific bundle_id the cast used for that subject in this clip
    (None if that subject has no cast entry / no bundle here, distinct from having no
    primary_subject at all). note explains a None primary (no cast node, no composition,
    composition not found, or no slot assigned) — it is always None when primary_subject came from
    an explicit tag.
    """
    cast_nodes = _find_nodes(prompt_graph, _SCENE_CAST_BUILD)
    if not cast_nodes:
        return {"tags": [], "primary_subject": None, "primary_bundle": None, "composition_name": None,
                "note": "no Scene Cast Build node found — not a cast-driven generation"}

    cast_node = cast_nodes[0]
    tags: list[str] = []
    bundle_by_subject: dict[str, str] = {}
    try:
        entries = json.loads(cast_node.get("inputs", {}).get("cast_entries_json") or "[]")
    except ValueError:
        entries = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("bundle_id")
        if bid and bid not in tags:
            tags.append(bid)
        sid = entry.get("subject_id")
        if sid and bid and sid not in bundle_by_subject:
            bundle_by_subject[sid] = bid

    comp_name = _composition_name(prompt_graph, cast_node)

    # Explicit tag wins over everything below, for both Composition- and Source-Profile-driven
    # clips — mirrors subject_id-or-source_subject_id fallback SceneCastBuild.execute() itself
    # uses when resolving a source-derived entry's subject_id.
    explicit_primary = next(
        (
            (entry.get("subject_id") or entry.get("source_subject_id"))
            for entry in (entries if isinstance(entries, list) else [])
            if isinstance(entry, dict) and entry.get("primary")
            and (entry.get("subject_id") or entry.get("source_subject_id"))
        ),
        None,
    )
    if explicit_primary:
        return {"tags": tags, "primary_subject": explicit_primary,
                "primary_bundle": bundle_by_subject.get(explicit_primary),
                "composition_name": comp_name, "note": None}

    if comp_name is None:
        return {"tags": tags, "primary_subject": None, "primary_bundle": None, "composition_name": None,
                "note": "no Prompt Composition resolved — likely Source-Profile-driven; "
                        "primary subject can't be determined yet for that path"}

    if load_composition is None:
        return {"tags": tags, "primary_subject": None, "primary_bundle": None, "composition_name": comp_name,
                "note": "composition loader not supplied — primary subject not resolved"}

    composition = load_composition(comp_name)
    if not isinstance(composition, dict):
        return {"tags": tags, "primary_subject": None, "primary_bundle": None, "composition_name": comp_name,
                "note": f"composition {comp_name!r} could not be loaded"}

    subjects = composition.get("subjects", {})
    primary = next((sid for sid in subjects.values() if sid), None)
    if primary is None:
        return {"tags": tags, "primary_subject": None, "primary_bundle": None, "composition_name": comp_name,
                "note": f"composition {comp_name!r} has no subject assigned to any slot"}

    primary_bundle = bundle_by_subject.get(primary)
    note = None if primary_bundle else f"primary subject {primary!r} has no bundle in this clip's cast"
    return {"tags": tags, "primary_subject": primary, "primary_bundle": primary_bundle,
            "composition_name": comp_name, "note": note}
