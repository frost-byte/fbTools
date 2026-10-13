"""Plan a Scene/Story -> Source Profile import (pure; no ComfyUI deps, stdlib only).

Convention (see docs/scene_story_import.md):
    story            -> one image-driven Source Profile (a loose set of scenes -> a profile you name)
    scene_order      -> clip order
    scene base image -> a profile frame (referenced in place under the output dir), chosen by the clip
    scene mask       -> a profile subject (the scene's ``mask_name``, minus any ``_no_bg`` suffix)
    resolved video prompt -> the clip's action (see utils/scene_prompts.py), with provenance in
                        ``origin.action_source`` so clips that still need an action are visible

Disk and libber access are injected as two callables so this module is unit-testable:
    scene_info(scene_name)    -> {"exists": bool, "base_image": "base.png" | "", "masks": [str]}
    scene_prompts(scene_name) -> (prompt_dict, composition_dict)   # libbers already applied

``plan_import`` never writes anything; applying the plan is the caller's job.
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable

DEFAULT_CLIP_LENGTH = 5.0
_NO_BG_SUFFIXES = ("_no_bg", "_no_bkgd")
_TOKEN = re.compile(r"%[a-z0-9_]+%")

# resolve_video_prompt() sources that mean an explicit video prompt was written for the scene.
_EXPLICIT_SOURCES = ("custom", "prompt", "composition")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def profile_id_for(name: str) -> str:
    return f"sp_story_{slugify(name) or 'import'}"


def subject_name_from_mask(mask_name: str) -> str:
    """``girl_no_bg`` -> ``girl``: the background variants of a mask are the same subject."""
    name = str(mask_name or "").strip()
    for suffix in _NO_BG_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def action_source_for(video_source: str) -> str:
    """Map a resolve_video_prompt() source to a clip ``origin.action_source``."""
    if video_source in _EXPLICIT_SOURCES:
        return "resolved"
    if video_source == "image_prompt":
        return "image_prompt"
    return "empty"


def loose_scene_entries(scene_names: list[str]) -> list[dict]:
    """Story-entry-shaped dicts for scenes that are not part of any story (order = list order).

    No prompt source is chosen, so they resolve to an empty action (vision LLM later).
    """
    return [
        {
            "scene_name": name, "scene_order": i, "mask_name": "",
            "prompt_source": "", "prompt_key": "", "custom_prompt": "",
            "video_prompt_source": "auto", "video_prompt_key": "", "video_custom_prompt": "",
        }
        for i, name in enumerate(scene_names)
    ]


# ── Cleaning image-model prompts for video ──────────────────────────────────────────
# Scene prompts were written for image-edit models that take several numbered input images
# ("Picture 1" is the layout, "Picture 2" the character, the rest control images). A Source Profile
# clip already says that structurally, so those lines are dropped and the character is a slot.

_IMG = r"(?:image|picture)\s*\d+"
_IMG_LIST = rf"{_IMG}(?:\s*(?:,|and|&)\s*(?:and\s+)?{_IMG})*"
# "image3 provides no meaningful information and should be ignored." (and the "further depth" variant)
_RE_CONTROL_NOTE = re.compile(rf"{_IMG_LIST}\s+provides?\b[^.]*\.?", re.I)
# "Replace the masked area from image1 with the {male|female|girl} from image2."
_RE_MASK_REPLACE = re.compile(rf"\bReplace\s+the\s+masked\s+area\b[^.]*?{_IMG}[^.]*\.?", re.I)
# "Place <name>, the girl from Picture 2, in the position and orientation of the girl from Picture 1, ..."
_RE_PLACE = re.compile(rf"\b(?-i:Place)\b[^.]*?{_IMG}[^.]*\.?", re.I)
# "<name>, the girl from image2," at the start of a sentence, or a bare "the girl from image2".
_REF_NOUN = r"(?:\{[^}]*\}|[A-Za-z][\w-]*)"
_RE_NAMED_REF = re.compile(
    rf"(?P<lead>^|[.!?][\"')]?\s+)(?P<name>[A-Za-z0-9][\w-]*),\s*the\s+{_REF_NOUN}\s+from\s+(?:image|picture)\s*2\b,?", re.I)
_RE_PLAIN_REF = re.compile(rf"\bthe\s+{_REF_NOUN}\s+from\s+(?:image|picture)\s*2\b", re.I)
_RE_ANY_IMG_REF = re.compile(_IMG, re.I)


def clean_image_model_prompt(text: str, slot: str = "A") -> str:
    """Strip image-model scaffolding from a scene prompt and point its character at ``{slot}``.

    Removes the notes about what each control image does and the "put the girl from image2 into
    image1" instructions; "<trigger>, the girl from image2," (and later bare uses of that trigger
    word) become ``{A}``. Anything else is left alone, so prose describing the action survives.
    """
    out = str(text or "")
    out = _RE_CONTROL_NOTE.sub(" ", out)
    out = _RE_MASK_REPLACE.sub(" ", out)
    out = _RE_PLACE.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()

    token = f"{{{slot}}}"
    triggers: list[str] = []

    def _named(m: re.Match) -> str:
        triggers.append(m.group("name"))
        return f"{m.group('lead')}{token}"

    out = _RE_NAMED_REF.sub(_named, out)
    out = _RE_PLAIN_REF.sub(token, out)
    for trig in dict.fromkeys(triggers):
        out = re.sub(rf"(?<![\w{{]){re.escape(trig)}(?![\w}}])", token, out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


_RE_PREAMBLE = re.compile(r"^\s*(A\s+photo-?realistic\s+scene\.)\s*", re.I)


def split_preamble(text: str) -> tuple[str, str]:
    """``("A photo-realistic scene.", rest)``: that opener is a style line, not part of the action.

    It belongs in the clip's preamble (``style``), which opens detailed_description before Shot 1.
    """
    m = _RE_PREAMBLE.match(str(text or ""))
    return (m.group(1), text[m.end():].strip()) if m else ("", str(text or ""))


def has_action_text(text: str) -> bool:
    """False when nothing but placeholders and punctuation is left (the prompt was all scaffolding)."""
    return bool(re.search(r"[A-Za-z]", re.sub(r"\{[A-Za-z]\}", "", str(text or ""))))


def leftover_image_refs(text: str) -> list[str]:
    """``image2`` / ``Picture 3`` style references still in ``text`` after cleaning."""
    return sorted({m.group(0).lower().replace(" ", "") for m in _RE_ANY_IMG_REF.finditer(text or "")})


def _warn(warnings: list, scene: str, level: str, kind: str, message: str) -> None:
    warnings.append({"scene": scene, "level": level, "kind": kind, "message": message})


def _subject(name: str) -> dict:
    return {
        "id": f"subj_{slugify(name)}", "label": name, "role_description": "",
        "entity_type": "person", "pronoun_style": "neutral", "short_name": "", "notes": "",
    }


def _keep_existing_action(existing_clip: dict) -> bool:
    """True when the clip's action was written/edited by hand and must survive a re-import."""
    origin = existing_clip.get("origin") or {}
    action = str(existing_clip.get("action", "")).strip()
    source = origin.get("action_source")
    return source == "manual" or (source is None and bool(action))


def plan_import(
    entries: list[dict],
    *,
    scene_info: Callable[[str], dict],
    scene_prompts: Callable[[str], tuple[dict, dict]],
    resolve_video_prompt: Callable[[Any, dict, dict], tuple[str, str]],
    story_name: str = "",
    profile_name: str = "",
    profile_id: str = "",
    existing_profile: dict | None = None,
    clip_length: float = DEFAULT_CLIP_LENGTH,
    scenes_rel: str = "scenes",
) -> dict:
    """Build the profile (and a per-scene preview) that importing ``entries`` would produce.

    entries: story-entry dicts (see ``loose_scene_entries``), imported in ``scene_order``.
    existing_profile: the stored profile with this id, if any; the plan then merges into it
        (idempotent: clips are matched by ``origin.scene``; hand-written actions are kept).
    Returns ``{"profile", "scenes", "warnings", "stats"}``.
    """
    pid = profile_id or profile_id_for(story_name or profile_name)
    name = profile_name or story_name or pid
    existing = copy.deepcopy(existing_profile) if existing_profile else None
    existing_clips = {
        (c.get("origin") or {}).get("scene"): c
        for c in (existing or {}).get("clips", []) if (c.get("origin") or {}).get("scene")
    }

    warnings: list[dict] = []
    frames: list[dict] = []
    subjects: dict[str, dict] = {}
    clips: list[dict] = []
    rows: list[dict] = []
    stats = {"scenes": 0, "imported": 0, "skipped": 0, "kept_manual": 0,
             "resolved": 0, "image_prompt": 0, "empty": 0, "cleaned": 0}

    seen: set[str] = set()
    for entry in sorted(entries, key=lambda e: e.get("scene_order", 0)):
        scene = str(entry.get("scene_name", "")).strip()
        stats["scenes"] += 1
        row = {"scene": scene, "scene_order": entry.get("scene_order", 0), "status": "skipped",
               "clip_id": "", "frame_id": "", "action_source": "", "action_preview": "", "cleaned": False}
        rows.append(row)
        if not scene:
            stats["skipped"] += 1
            continue
        if scene in seen:
            _warn(warnings, scene, "error", "duplicate_scene", "Scene appears more than once; later copy skipped.")
            stats["skipped"] += 1
            continue
        seen.add(scene)

        info = scene_info(scene) or {}
        if not info.get("exists"):
            _warn(warnings, scene, "error", "scene_missing", "Scene folder not found; skipped.")
            stats["skipped"] += 1
            continue
        base = info.get("base_image") or ""
        if not base:
            _warn(warnings, scene, "error", "base_missing", "Scene has no base image; skipped.")
            stats["skipped"] += 1
            continue

        # Subject: the story entry's mask, else the scene's first mask.
        masks = info.get("masks") or []
        subj_name = subject_name_from_mask(entry.get("mask_name") or (masks[0] if masks else ""))
        clip_subjects: list[str] = []
        if subj_name:
            subj = _subject(subj_name)
            subjects.setdefault(subj["id"], subj)
            clip_subjects = [subj["id"]]
        else:
            _warn(warnings, scene, "warn", "no_mask", "Scene has no mask to name a subject.")

        # Action: the same prompt the Story nodes would send for this scene's video.
        prompt_dict, composition_dict = scene_prompts(scene)
        text, video_source = resolve_video_prompt(entry, prompt_dict, composition_dict)
        text = (text or "").strip()
        cleaned = clean_image_model_prompt(text)
        row["cleaned"] = cleaned != text
        if row["cleaned"]:
            stats["cleaned"] += 1
        preamble, text = split_preamble(cleaned)
        source = action_source_for(video_source)
        if source != "empty" and not has_action_text(text):
            source = "empty"
            text = ""
            preamble = ""
        leftover = leftover_image_refs(text)
        if leftover:
            _warn(warnings, scene, "warn", "image_reference",
                  f"Still refers to input image(s) ({', '.join(leftover)}); edit the action.")
        left = sorted(set(_TOKEN.findall(text)))
        if left:
            _warn(warnings, scene, "warn", "unresolved_libber",
                  f"Unresolved libber reference(s) left in the prompt: {', '.join(left)}.")
        if source == "image_prompt":
            _warn(warnings, scene, "info", "image_prompt",
                  "No video prompt; used the scene's image prompt as a draft. Review before use.")
        elif source == "empty":
            _warn(warnings, scene, "info", "no_prompt", "No prompt; needs a described action (vision LLM).")

        frame = {"id": f"frame_{slugify(scene)}", "filename": f"{scenes_rel}/{scene}/{base}",
                 "dir": "output", "label": scene}
        frames.append(frame)

        clip = {
            "id": f"clip_{slugify(scene)}", "label": scene,
            "start_time": 0.0, "end_time": float(clip_length),
            "subjects": clip_subjects, "frame_id": frame["id"],
            "action": text,
            **({"style": preamble} if preamble else {}),
            "origin": {"scene": scene, "scene_order": entry.get("scene_order", 0),
                       "action_source": source, **({"story": story_name} if story_name else {})},
        }
        prior = existing_clips.get(scene)
        if prior is not None:
            # Re-import: keep everything the user may have changed, refresh what the scene owns.
            merged = {**prior, "frame_id": frame["id"], "origin": {**(prior.get("origin") or {}), **clip["origin"]}}
            if _keep_existing_action(prior):
                merged["action"] = prior.get("action", "")
                merged["origin"]["action_source"] = (prior.get("origin") or {}).get("action_source") or "manual"
                stats["kept_manual"] += 1
                source = merged["origin"]["action_source"]
            else:
                merged["action"] = text
                if preamble and not str(prior.get("style", "")).strip():
                    merged["style"] = preamble
            clip = merged

        stats["imported"] += 1
        stats[source if source in stats else "empty"] += 1
        row.update(status="updated" if prior is not None else "new", clip_id=clip["id"], frame_id=frame["id"],
                   action_source=source, action_preview=str(clip.get("action", ""))[:120])
        clips.append(clip)

    profile = _merge_profile(existing, pid, name, story_name, frames, subjects, clips, clip_length)
    return {"profile": profile, "scenes": rows, "warnings": warnings, "stats": stats}


def _lay_out(clips: list[dict], fallback: float) -> list[dict]:
    """Clips end to end in list order, each keeping its own length (a still has no footage timeline).

    Giving every clip 0..N stacks them on top of each other in the timeline views.
    """
    out: list[dict] = []
    t = 0.0
    for c in clips:
        length = float(c.get("end_time") or 0.0) - float(c.get("start_time") or 0.0)
        if length <= 0:
            length = fallback
        out.append({**c, "start_time": round(t, 3), "end_time": round(t + length, 3)})
        t += length
    return out


def _merge_profile(existing, pid, name, story_name, frames, subjects, clips, clip_length) -> dict:
    """The full profile dict: planned frames/subjects/clips first, anything else already there kept."""
    planned_scenes = {(c.get("origin") or {}).get("scene") for c in clips}
    base = existing or {}
    kept_frames = [f for f in base.get("frames", []) if f.get("id") not in {f2["id"] for f2 in frames}]
    merged_subjects = list(base.get("subjects", []))
    for sid, subj in subjects.items():
        if not any(s.get("id") == sid for s in merged_subjects):
            merged_subjects.append(subj)
    # Existing clips that this import does not touch stay, after the imported ones.
    kept_clips = [c for c in base.get("clips", []) if (c.get("origin") or {}).get("scene") not in planned_scenes]
    return {
        **{k: v for k, v in base.items() if k not in ("frames", "subjects", "clips")},
        "id": pid,
        "name": base.get("name") or name,
        "media_type": "image",
        "media_dir": "output",
        "media_filename": (frames[0]["filename"] if frames else base.get("media_filename", "")),
        "frames": frames + kept_frames,
        "subjects": merged_subjects,
        "clips": _lay_out(clips + kept_clips, float(clip_length)),
        "default_segment_duration": base.get("default_segment_duration") or float(clip_length),
        "origin": {**(base.get("origin") or {}), **({"story": story_name} if story_name else {})},
    }


# ── Disk readers (stdlib only; the REST layer passes the directories) ──────────────

_BASE_IMAGE_NAMES = ("base.png", "base.jpg", "base.jpeg", "base.webp")


def safe_name(name: str) -> bool:
    """A scene/story name that is one path component (no separators, no traversal)."""
    n = str(name or "")
    return bool(n) and n == os.path.basename(n) and n not in (".", "..")


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def make_scene_info(scenes_dir: str) -> Callable[[str], dict]:
    """A ``scene_info`` callable reading ``<scenes_dir>/<scene>/`` (base image + mask names)."""
    def scene_info(scene: str) -> dict:
        folder = os.path.join(scenes_dir, scene)
        if not safe_name(scene) or not os.path.isdir(folder):
            return {"exists": False}
        base = next((n for n in _BASE_IMAGE_NAMES if os.path.isfile(os.path.join(folder, n))), "")
        masks_data = _read_json(os.path.join(folder, "masks.json")) or {}
        masks = [m.get("name", "") for m in masks_data.get("masks", []) if isinstance(m, dict) and m.get("name")]
        return {"exists": True, "base_image": base, "masks": masks}
    return scene_info


def read_story_entries(stories_dir: str, story: str) -> list[dict]:
    """The scene entries of ``<stories_dir>/<story>/story.json`` (ordered by scene_order); [] if unreadable."""
    if not safe_name(story):
        return []
    data = _read_json(os.path.join(stories_dir, story, "story.json")) or {}
    entries = [e for e in data.get("scenes", []) if isinstance(e, dict) and e.get("scene_name")]
    return sorted(entries, key=lambda e: e.get("scene_order", 0))


def list_stories(stories_dir: str) -> list[dict]:
    """``[{name, scenes: [scene names in order]}]`` for every story folder that has a story.json."""
    if not os.path.isdir(stories_dir):
        return []
    out = []
    for name in sorted(os.listdir(stories_dir)):
        entries = read_story_entries(stories_dir, name)
        if entries:
            out.append({"name": name, "scenes": [e["scene_name"] for e in entries]})
    return out


def list_loose_scenes(scenes_dir: str, stories: list[dict]) -> list[dict]:
    """Scene folders that belong to no story: ``[{name, has_base}]`` sorted by name."""
    if not os.path.isdir(scenes_dir):
        return []
    in_stories = {s for story in stories for s in story["scenes"]}
    info = make_scene_info(scenes_dir)
    out = []
    for name in sorted(os.listdir(scenes_dir)):
        if name in in_stories or not os.path.isdir(os.path.join(scenes_dir, name)):
            continue
        out.append({"name": name, "has_base": bool(info(name).get("base_image"))})
    return out
