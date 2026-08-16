"""Pure logic for the Reference Bundle system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from datetime import datetime, timezone


# ── Constants ──────────────────────────────────────────────────────────────────

VISUAL_TYPES = ("video", "images")
AUDIO_SOURCES = ("extract_from_visual", "file", "none")

_EMPTY_DATA: dict = {"version": 1, "bundles": {}}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── BundleRegistry ─────────────────────────────────────────────────────────────

class BundleRegistry:
    """In-memory registry of reference bundles."""

    def __init__(
        self,
        bundles: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.bundles: dict = bundles or {}
        self.file_path: str | None = file_path
        self.version: int = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "BundleRegistry":
        return cls(
            bundles=copy.deepcopy(data.get("bundles", {})),
            file_path=file_path,
            version=data.get("version", 1),
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "bundles": copy.deepcopy(self.bundles)}

    # ── queries ────────────────────────────────────────────────────────────────

    def get(self, bundle_id: str) -> dict | None:
        b = self.bundles.get(bundle_id)
        return copy.deepcopy(b) if b is not None else None

    def bundle_ids(self) -> list[str]:
        return list(self.bundles.keys())

    def list_bundles(self, subject_id: str | None = None) -> list[dict]:
        """Return deep copies of all bundles, optionally filtered by subject_id."""
        return [
            copy.deepcopy(b)
            for b in self.bundles.values()
            if subject_id is None or b.get("subject_id") == subject_id
        ]

    # ── mutation (returns new instance — safe for multi-branch wiring) ─────────

    def upsert(self, bundle: dict) -> "BundleRegistry":
        """Return a NEW registry with the bundle added or updated.

        'created' is preserved from the existing record; 'modified' is refreshed.
        Required nested fields are defaulted if absent.
        """
        new_reg = copy.deepcopy(self)
        bundle_id = bundle["id"]
        existing = new_reg.bundles.get(bundle_id, {})

        updated = copy.deepcopy(bundle)
        updated["created"] = existing.get("created") or updated.get("created") or _now_iso()
        updated["modified"] = _now_iso()
        updated.setdefault("name", bundle_id)
        updated.setdefault("subject_id", "")
        updated.setdefault("appearance_override", "")
        updated.setdefault("tags", [])

        visual = updated.setdefault("visual", {})
        visual.setdefault("type", "images")
        visual.setdefault("file", "")
        visual.setdefault("files", [])
        visual.setdefault("start_time", 0.0)
        visual.setdefault("duration", 0.0)
        visual.setdefault("force_rate", 0)
        visual.setdefault("frame_load_cap", 96)
        visual.setdefault("skip_first_frames", 0)
        visual.setdefault("select_every_nth", 1)

        audio = updated.setdefault("audio", {})
        audio.setdefault("source", "none")
        audio.setdefault("file", "")
        # Frame-sampling params for extract_from_visual (second Load Video node)
        audio.setdefault("force_rate", 0)
        audio.setdefault("frame_load_cap", 0)
        audio.setdefault("skip_first_frames", 0)
        audio.setdefault("select_every_nth", 1)
        # Time-based params for file source (Load Audio node)
        audio.setdefault("start_time", 0.0)
        audio.setdefault("duration", 0.0)

        new_reg.bundles[bundle_id] = updated
        return new_reg

    def delete(self, bundle_id: str) -> "BundleRegistry":
        """Return a NEW registry with the bundle removed. No-op if not found."""
        new_reg = copy.deepcopy(self)
        new_reg.bundles.pop(bundle_id, None)
        return new_reg

    def save(self, path: str | None = None, backup: bool = True) -> str:
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for bundle registry save")
        save_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: str) -> BundleRegistry:
    """Load registry from JSON. Returns empty registry if file absent."""
    if not os.path.exists(path):
        return BundleRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return BundleRegistry.from_dict(data, file_path=path)


def save_registry(registry: BundleRegistry, path: str, backup: bool = True) -> None:
    """Write registry to JSON, optionally creating a .bak first."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry.to_dict(), fh, indent=2, ensure_ascii=False)
    registry.file_path = path


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_bundle(bundle: dict) -> list[str]:
    """Return warning strings for a bundle dict. Empty list means valid."""
    warnings: list[str] = []
    visual = bundle.get("visual", {})
    audio = bundle.get("audio", {})

    visual_type = visual.get("type", "images")
    audio_source = audio.get("source", "none")

    if visual_type not in VISUAL_TYPES:
        warnings.append(f"visual.type must be one of {VISUAL_TYPES!r}, got {visual_type!r}")
    if audio_source not in AUDIO_SOURCES:
        warnings.append(f"audio.source must be one of {AUDIO_SOURCES!r}, got {audio_source!r}")
    if visual_type == "video" and not visual.get("file"):
        warnings.append("visual.type is 'video' but visual.file is empty")
    if visual_type == "images" and not visual.get("files"):
        warnings.append("visual.type is 'images' but visual.files is empty")
    if audio_source == "extract_from_visual" and visual_type != "video":
        warnings.append("audio.source 'extract_from_visual' requires visual.type 'video'")
    if audio_source == "file" and not audio.get("file"):
        warnings.append("audio.source is 'file' but audio.file is empty")

    return warnings
