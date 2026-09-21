"""Persistence layer for the Prompt Composition Editor.

Each composition is stored as an individual JSON file under:
  user_data_dir()/prompt_compositions/<id>.json

No ComfyUI dependencies — all folder_paths / torch access lives in extension.py.

Schema
------
{
    "id":   str,                     # slug used as filename key
    "name": str,                     # human display name
    "model_type": str,               # h3_ref2va | h3_fl2va | wan22 | bernini | ...
    "style": str,                    # overall visual style
    "subjects": {                    # slot key → subject_id reference; keys are
                                      # spreadsheet-column-style letters (A, B,
                                      # ..., Z, AA, ...) — see utils/slot_letters.py
        "A": "character_a",
        "B": "character_b"
    },
    "_subject_snapshots": {          # cached at save time; fallback if subject deleted
        "A": { ...full subject dict... },
        "B": { ...full subject dict... }
    },
    "outfit_overrides": {            # slot key → override string
        "A": "grey sweater"
    },
    "background": "cafe_interior",   # background_id reference (or "" / null)
    "_background_snapshot": { ... }, # cached background at save time
    "shots": [
        {
            "id": "shot_1",
            "timestamp": null | "MM:SS.mmm",
            "camera": str,           # may reference subjects via {A}/{B}/... placeholders
            "action": str,           # may reference subjects via {A}/{B}/... placeholders
            "dialogue": {
                "speaker": "A",
                "language": "English",
                "text": str
            } | null,
            "sound_events": str | null
        }
    ],
    "overall_soundscape": str,
    "non_diegetic_music": str
}
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:64] or "composition"


def _compositions_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "prompt_compositions")
    os.makedirs(path, exist_ok=True)
    return path


def _composition_path(data_dir: str, composition_id: str) -> str:
    return os.path.join(_compositions_dir(data_dir), f"{composition_id}.json")


# ── CRUD ───────────────────────────────────────────────────────────────────────

def list_compositions(data_dir: str) -> list[dict]:
    """Return summary dicts [{id, name, model_type, updated_at}] sorted by name."""
    d = _compositions_dir(data_dir)
    results = []
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(d, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            results.append({
                "id":          data.get("id", fname[:-5]),
                "name":        data.get("name", ""),
                "model_type":  data.get("model_type", ""),
                "updated_at":  data.get("updated_at", ""),
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["name"].lower())
    return results


def load_composition(data_dir: str, composition_id: str) -> dict:
    """Load and return a composition dict.  Raises FileNotFoundError if absent."""
    path = _composition_path(data_dir, composition_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Composition '{composition_id}' not found")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_composition(
    data_dir: str,
    composition: dict,
    subject_registry=None,
    background_store=None,
) -> dict:
    """Save a composition, refreshing subject and background snapshots.

    Args:
        data_dir:         package user-data directory
        composition:      composition dict (mutated in-place with id/timestamps)
        subject_registry: optional SubjectRegistry for refreshing snapshots
        background_store: optional backgrounds dict for refreshing snapshot

    Returns the saved composition dict.
    """
    # Ensure an id
    if not composition.get("id"):
        name = composition.get("name", "composition")
        composition["id"] = _slugify(name)

    # Refresh subject snapshots from live registry where available
    subjects = composition.get("subjects", {})
    snapshots = dict(composition.get("_subject_snapshots", {}))
    if subject_registry is not None:
        for slot, sid in subjects.items():
            subject = subject_registry.get_subject(sid) if hasattr(subject_registry, "get_subject") else None
            if subject is not None:
                snapshots[slot] = subject
    composition["_subject_snapshots"] = snapshots

    # Refresh background snapshot
    bg_id = composition.get("background") or ""
    if bg_id and background_store is not None:
        bg = background_store.get(bg_id)
        if bg is not None:
            composition["_background_snapshot"] = bg

    composition["updated_at"] = datetime.now(timezone.utc).isoformat()

    path = _composition_path(data_dir, composition["id"])
    # .bak backup on overwrite
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(composition, fh, indent=2, ensure_ascii=False)

    return composition


def delete_composition(data_dir: str, composition_id: str) -> None:
    """Delete a composition file.  Raises FileNotFoundError if absent."""
    path = _composition_path(data_dir, composition_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Composition '{composition_id}' not found")
    os.remove(path)
    bak = path + ".bak"
    if os.path.exists(bak):
        os.remove(bak)


def composition_ordinal_roster(composition: dict, subject_lookup) -> list[dict]:
    """Ordered [{subject_id, entity_type, pronoun_style}] for a composition's
    subjects, in slot order (empty slots skipped), for ordinal matching.

    subject_lookup(subject_id) -> subject dict | None supplies each subject's
    pronoun_style / entity_type (defaults: "person" / "" when absent).
    """
    roster = []
    for sid in (composition.get("subjects") or {}).values():
        if not sid:
            continue
        subj = subject_lookup(sid) or {}
        roster.append({
            "subject_id":    sid,
            "entity_type":   subj.get("entity_type", "person"),
            "pronoun_style": subj.get("pronoun_style", ""),
        })
    return roster


# ── Subject resolution ─────────────────────────────────────────────────────────

def resolve_subjects(composition: dict, subject_registry=None) -> dict[str, dict]:
    """Return {slot_key: subject_dict} with live registry preferred over snapshots.

    Falls back to the cached _subject_snapshots if a subject has been deleted.
    Returns empty dict for slots where neither source has data.
    """
    subjects = composition.get("subjects", {})
    snapshots = composition.get("_subject_snapshots", {})
    resolved = {}
    for slot, sid in subjects.items():
        live = None
        if subject_registry is not None and hasattr(subject_registry, "get_subject"):
            live = subject_registry.get_subject(sid)
        if live is not None:
            d = dict(live)
            d.setdefault("subject_id", sid)
            resolved[slot] = d
        elif slot in snapshots:
            d = dict(snapshots[slot])
            d.setdefault("subject_id", sid)
            resolved[slot] = d
    return resolved


def resolve_background(composition: dict, backgrounds: dict | None = None) -> dict | None:
    """Return the background dict, preferring live store over snapshot."""
    bg_id = composition.get("background") or ""
    if not bg_id:
        return None
    if backgrounds and bg_id in backgrounds:
        return backgrounds[bg_id]
    snap = composition.get("_background_snapshot")
    return snap if snap else None


def apply_composition_overrides(
    composition: dict, overrides: dict | None, backgrounds: dict | None = None
) -> tuple[dict, list[str]]:
    """Return (modified copy, warnings) with per-run overrides applied.

    Supported keys (each present only when it differs from the composition):
      background               - background id, or "none" to drop the background
      background_as_reference  - bool
    An unknown background id keeps the composition's own and adds a warning.
    The input dict is never mutated.
    """
    import copy

    out = copy.deepcopy(composition)
    warnings: list[str] = []
    if not isinstance(overrides, dict) or not overrides:
        return out, warnings

    if "background" in overrides:
        bg_id = str(overrides.get("background") or "").strip()
        if bg_id in ("", "none"):
            out["background"] = ""
            out.pop("_background_snapshot", None)
        elif backgrounds and bg_id in backgrounds:
            if bg_id != out.get("background"):
                out["background"] = bg_id
                out["_background_snapshot"] = backgrounds[bg_id]
        else:
            warnings.append(f"background override '{bg_id}' not found; using the composition's own")

    if "background_as_reference" in overrides:
        out["background_as_reference"] = bool(overrides["background_as_reference"])

    return out, warnings


def validate_composition(composition: dict) -> list[str]:
    """Return a list of warning strings.  Empty list means no issues."""
    warnings = []
    if not composition.get("id"):
        warnings.append("Composition has no id")
    if not composition.get("name"):
        warnings.append("Composition has no name")
    if not composition.get("subjects"):
        warnings.append("No subjects assigned")
    if not composition.get("shots"):
        warnings.append("No shots defined")
    return warnings


# ── Cast enrichment ────────────────────────────────────────────────────────────

def _enrich_subject_with_bundle(subj: dict, bundle: dict, entry: dict) -> None:
    """Apply bundle images/audio/appearance onto subj in place.

    image-mode visual.files      → appended to character_sheet_images
    use_audio=True + source=file → bundle audio → voice.audio_reference_file
    appearance_override          → replaces appearance.summary
    """
    visual_mode = entry.get("visual_mode", "images")
    visual      = bundle.get("visual", {})
    audio       = bundle.get("audio", {})

    # 1. Image-mode files → character_sheet_images (deduplicated, appended)
    if visual_mode == "images":
        files = [f for f in visual.get("files", []) if f]
        if files:
            sheets: list = subj.setdefault("character_sheet_images", [])
            for f in files:
                if f not in sheets:
                    sheets.append(f)

    # 2. Audio → voice fields
    # extract_from_visual means the audio is a VIDEO SOUNDTRACK — it is handled
    # at the video-entry level (video_entries_full / soundtrack_audio in the
    # refplan) and must NOT also appear as a standalone voice.audio_reference_file,
    # which would create a duplicate reference.  Only source="file" (a separate
    # standalone audio asset) populates the subject's voice fields.
    if entry.get("use_audio", False) and audio.get("source") == "file":
        audio_file = audio.get("file", "")
        if audio_file:
            v = subj.setdefault("voice", {})
            v["audio_reference_file"] = audio_file
            v["audio_start_time"]     = audio.get("start_time", 0.0)
            v["audio_duration"]       = audio.get("duration", 0.0)
            v["audio_retention"]      = audio.get("retention", "timbre")
            v["audio_role"]           = audio.get("role", "")
            v["audio_cache"]          = audio.get("audio_cache", "")

    # 3. Appearance — bundle fields override subject fields (bundle wins if non-empty).
    bun_app = bundle.get("appearance") or {}
    if isinstance(bun_app, str):
        bun_app = {"summary": bun_app}
    legacy = bundle.get("appearance_override", "").strip()
    app = subj.setdefault("appearance", {})
    bun_summary = bun_app.get("summary", "").strip() or legacy
    if bun_summary:
        app["summary"] = bun_summary
    for _k in ("hair", "face", "body", "default_outfit"):
        _v = bun_app.get(_k, "").strip()
        if _v:
            app[_k] = _v
    if bundle.get("pronoun_style"):
        subj["pronoun_style"] = bundle["pronoun_style"]
    if bundle.get("short_name"):
        subj["short_name"] = bundle["short_name"]


def apply_cast_to_subjects(
    resolved_subjects: dict,
    composition: dict,
    scene_cast: dict,
    bundle_registry,
    subject_registry=None,
) -> dict:
    """Apply a scene cast to resolved_subjects, matching cast entries to
    composition slots by subject identity (entry["subject_id"] against each
    slot's current subject_id) — array order in scene_cast["entries"] carries
    no meaning. This mirrors how SourceProfileClipPrompt matches SceneCastBuild
    cast entries to a clip's own subjects by source_subject_id, so a cast built
    once against a shared roster binds consistently whichever node consumes it.
    A blank entry (no subject_id), or one whose subject_id isn't currently in
    resolved_subjects, is a no-op.

    For each matched entry:
      - Pure source-derived (source_profile_id + source_subject_id, no
        bundle_id): the slot becomes a synthetic subject built from the
        source-profile annotation, retention from entry["retention"].
      - Pure bundle-backed (bundle_id, no source_profile_id/source_subject_id):
        bundle enrichment (images/audio/appearance) is applied in place on
        top of whoever currently occupies the slot.
      - Hybrid (source_profile_id + source_subject_id + bundle_id — a bundle
        replacing a subject that also has a source-video reference): the
        matched slot keeps the source-derived subject as the motion donor,
        marked "replaced", and a NEW slot ("<slot>_bundle") is minted for the
        bundle, marked "attribute_transfer", each pointing at the other via
        _transfer_to_slot. This mirrors SourceProfileClipPrompt's
        SOURCE_SLOTS/BUNDLE_SLOTS pairing so the same prompt_assembler.py
        features gated on retention_marker (the motion-transfer sentence, the
        sharpened discard-identity wording, the specific-video-naming edit
        description) fire identically for both the Source Profile and
        Composition paths — previously this branch never inspected bundle_id
        at all, so a hybrid entry's bundle data (appearance/images/audio) was
        silently dropped from the prompt text even though _resolve_cast_media
        still wired its media into the reference plan.

    bundle_registry must support .get(bundle_id) -> dict | None.
    subject_registry, if provided, must support .get_subject(subject_id) -> dict | None.

    Returns a new dict of deep-copied subject dicts so originals are never mutated.
    """
    import copy as _copy

    enriched = {slot: _copy.deepcopy(subj) for slot, subj in resolved_subjects.items()}

    # Identity lookup: which slot does a given subject_id currently occupy?
    # This — not array position — is what a cast entry binds against.
    slot_by_subject_id = {
        subj.get("subject_id", ""): slot
        for slot, subj in enriched.items()
        if subj.get("subject_id")
    }

    for entry in scene_cast.get("entries", []):
        subject_id        = entry.get("subject_id", "")
        bundle_id         = entry.get("bundle_id", "")
        source_profile_id = entry.get("source_profile_id", "")
        source_subject_id = entry.get("source_subject_id", "")

        if not subject_id:
            continue  # blank row — no-op

        slot = slot_by_subject_id.get(subject_id)
        if slot is None:
            continue  # entry references a subject not in this composition — no-op

        # ── Source-derived / hybrid entry ──────────────────────────────────────
        if source_profile_id and source_subject_id:
            role_description = entry.get("role_description", "")
            entity_type      = entry.get("entity_type", "person")
            retention        = entry.get("retention", "partially_preserved")
            donor = {
                "subject_id":        subject_id,
                "name":              role_description or subject_id,
                "source_profile_id": source_profile_id,
                "source_subject_id": source_subject_id,
                "entity_type":       entity_type,
                "appearance": {
                    "summary": role_description,
                },
                "voice": {},
                "character_sheet_images": [],
                "concept_id": "",
                "_cast_retention": retention,
            }

            if bundle_id:
                bundle = bundle_registry.get(bundle_id)
                if bundle is not None:
                    bundle_slot = f"{slot}_bundle"
                    donor["_cast_retention"]   = "replaced"
                    donor["_transfer_to_slot"] = bundle_slot

                    bun_subject_id = bundle.get("subject_id", "")
                    bun_subj = (subject_registry.get_subject(bun_subject_id)
                                if subject_registry and bun_subject_id else None) or {}
                    replacement = {
                        "subject_id":             bundle_id,
                        "name":                   bundle.get("name") or role_description or bundle_id,
                        "concept_id":             None,
                        "character_sheet_images": [],
                        "appearance": {"summary": "", "hair": "", "face": "", "body": "", "default_outfit": ""},
                        "voice": {},
                        "_cast_retention":   "attribute_transfer",
                        "_transfer_to_slot": slot,
                        "_pronoun_style":    bun_subj.get("pronoun_style", ""),
                        "_short_name":       bun_subj.get("short_name", ""),
                        "entity_type":       bundle.get("entity_type", entity_type),
                    }
                    _enrich_subject_with_bundle(replacement, bundle, entry)
                    enriched[bundle_slot] = replacement

            enriched[slot] = donor
            continue

        # ── Pure bundle-backed entry ─────────────────────────────────────────────
        if not bundle_id:
            continue
        bundle = bundle_registry.get(bundle_id)
        if bundle is None:
            continue

        _enrich_subject_with_bundle(enriched[slot], bundle, entry)

    return enriched
