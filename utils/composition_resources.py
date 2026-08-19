"""Persistence for Prompt Composition Editor supporting resources.

Three JSON files, all stored in user_data_dir():
  backgrounds.json      — environment / setting definitions
  camera_presets.json   — reusable camera description snippets
  sound_presets.json    — reusable sound / ambience snippets

No ComfyUI dependencies.

Schemas
-------
Background:
{
    "id":          str,
    "name":        str,
    "description": str,   # environment description for prompt
    "lighting":    str,
    "soundscape":  str    # default soundscape for overall_soundscape section
}

CameraPreset:
{
    "id":          str,
    "name":        str,
    "description": str    # camera direction text to insert
}

SoundPreset:
{
    "id":          str,
    "name":        str,
    "description": str    # sound/ambience text to insert
}
"""
from __future__ import annotations

import json
import os
import re
import shutil


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:64] or "item"


def _load_json(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return dict(default)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: str, data: dict, backup: bool = True) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ── Path helpers ───────────────────────────────────────────────────────────────

def backgrounds_path(data_dir: str) -> str:
    return os.path.join(data_dir, "backgrounds.json")


def camera_presets_path(data_dir: str) -> str:
    return os.path.join(data_dir, "camera_presets.json")


def sound_presets_path(data_dir: str) -> str:
    return os.path.join(data_dir, "sound_presets.json")


# ── Backgrounds ────────────────────────────────────────────────────────────────

def load_backgrounds(data_dir: str) -> dict[str, dict]:
    """Return {id: background_dict}."""
    return _load_json(backgrounds_path(data_dir), {}).get("backgrounds", {})


def list_backgrounds(data_dir: str) -> list[dict]:
    """Return sorted full background dicts (id injected)."""
    bgs = load_backgrounds(data_dir)
    items = [{"id": k, **v} for k, v in bgs.items()]
    items.sort(key=lambda x: x.get("name", x["id"]).lower())
    return items


def get_background(data_dir: str, bg_id: str) -> dict | None:
    return load_backgrounds(data_dir).get(bg_id)


def save_background(data_dir: str, background: dict) -> dict:
    """Upsert a background.  Assigns id from name if absent."""
    if not background.get("id"):
        background["id"] = _slugify(background.get("name", "background"))
    path = backgrounds_path(data_dir)
    data = _load_json(path, {"version": 1, "backgrounds": {}})
    data.setdefault("backgrounds", {})[background["id"]] = background
    _save_json(path, data)
    return background


def delete_background(data_dir: str, bg_id: str) -> None:
    path = backgrounds_path(data_dir)
    data = _load_json(path, {"version": 1, "backgrounds": {}})
    bgs = data.setdefault("backgrounds", {})
    if bg_id not in bgs:
        raise KeyError(f"Background '{bg_id}' not found")
    del bgs[bg_id]
    _save_json(path, data)


# ── Camera presets ─────────────────────────────────────────────────────────────

def load_camera_presets(data_dir: str) -> dict[str, dict]:
    return _load_json(camera_presets_path(data_dir), {}).get("camera_presets", {})


def list_camera_presets(data_dir: str) -> list[dict]:
    presets = load_camera_presets(data_dir)
    items = [{"id": k, "name": v.get("name", k), "description": v.get("description", "")}
             for k, v in presets.items()]
    items.sort(key=lambda x: x["name"].lower())
    return items


def save_camera_preset(data_dir: str, preset: dict) -> dict:
    if not preset.get("id"):
        preset["id"] = _slugify(preset.get("name", "camera"))
    path = camera_presets_path(data_dir)
    data = _load_json(path, {"version": 1, "camera_presets": {}})
    data.setdefault("camera_presets", {})[preset["id"]] = preset
    _save_json(path, data)
    return preset


def delete_camera_preset(data_dir: str, preset_id: str) -> None:
    path = camera_presets_path(data_dir)
    data = _load_json(path, {"version": 1, "camera_presets": {}})
    presets = data.setdefault("camera_presets", {})
    if preset_id not in presets:
        raise KeyError(f"Camera preset '{preset_id}' not found")
    del presets[preset_id]
    _save_json(path, data)


# ── Sound presets ──────────────────────────────────────────────────────────────

def load_sound_presets(data_dir: str) -> dict[str, dict]:
    return _load_json(sound_presets_path(data_dir), {}).get("sound_presets", {})


def list_sound_presets(data_dir: str) -> list[dict]:
    presets = load_sound_presets(data_dir)
    items = [{"id": k, "name": v.get("name", k), "description": v.get("description", "")}
             for k, v in presets.items()]
    items.sort(key=lambda x: x["name"].lower())
    return items


def save_sound_preset(data_dir: str, preset: dict) -> dict:
    if not preset.get("id"):
        preset["id"] = _slugify(preset.get("name", "sound"))
    path = sound_presets_path(data_dir)
    data = _load_json(path, {"version": 1, "sound_presets": {}})
    data.setdefault("sound_presets", {})[preset["id"]] = preset
    _save_json(path, data)
    return preset


def delete_sound_preset(data_dir: str, preset_id: str) -> None:
    path = sound_presets_path(data_dir)
    data = _load_json(path, {"version": 1, "sound_presets": {}})
    presets = data.setdefault("sound_presets", {})
    if preset_id not in presets:
        raise KeyError(f"Sound preset '{preset_id}' not found")
    del presets[preset_id]
    _save_json(path, data)
