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
    "subjects": {                    # slot key → subject_id reference
        "S1": "character_a",
        "S2": "character_b"
    },
    "_subject_snapshots": {          # cached at save time; fallback if subject deleted
        "S1": { ...full subject dict... },
        "S2": { ...full subject dict... }
    },
    "outfit_overrides": {            # slot key → override string
        "S1": "grey sweater"
    },
    "background": "cafe_interior",   # background_id reference (or "" / null)
    "_background_snapshot": { ... }, # cached background at save time
    "shots": [
        {
            "id": "shot_1",
            "timestamp": null | "MM:SS.mmm",
            "camera": str,
            "action": str,
            "dialogue": {
                "speaker": "S1",
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
            resolved[slot] = dict(live)
        elif slot in snapshots:
            resolved[slot] = dict(snapshots[slot])
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
