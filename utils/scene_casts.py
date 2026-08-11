"""Pure logic for the Scene Cast system.

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

VISUAL_MODES = ("video", "images")

_EMPTY_DATA: dict = {"version": 1, "casts": {}}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(
    subject_id: str,
    bundle_id: str,
    visual_mode: str = "images",
    use_audio: bool = False,
) -> dict:
    """Build a cast entry dict with all required fields."""
    return {
        "subject_id": subject_id,
        "bundle_id": bundle_id,
        "visual_mode": visual_mode,
        "use_audio": use_audio,
    }


# ── CastRegistry ───────────────────────────────────────────────────────────────

class CastRegistry:
    """In-memory registry of scene casts."""

    def __init__(
        self,
        casts: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.casts: dict = casts or {}
        self.file_path: str | None = file_path
        self.version: int = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "CastRegistry":
        return cls(
            casts=copy.deepcopy(data.get("casts", {})),
            file_path=file_path,
            version=data.get("version", 1),
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "casts": copy.deepcopy(self.casts)}

    # ── queries ────────────────────────────────────────────────────────────────

    def get(self, cast_id: str) -> dict | None:
        c = self.casts.get(cast_id)
        return copy.deepcopy(c) if c is not None else None

    def cast_ids(self) -> list[str]:
        return list(self.casts.keys())

    def list_casts(self) -> list[dict]:
        return [copy.deepcopy(c) for c in self.casts.values()]

    def find_entry(self, cast_id: str, subject_id: str) -> dict | None:
        """Return the entry for subject_id within cast_id, or None."""
        cast = self.casts.get(cast_id)
        if not cast:
            return None
        for entry in cast.get("entries", []):
            if entry.get("subject_id") == subject_id:
                return copy.deepcopy(entry)
        return None

    # ── mutation (returns new instance — safe for multi-branch wiring) ─────────

    def upsert(self, cast: dict) -> "CastRegistry":
        """Return a NEW registry with the cast added or updated.

        'created' is preserved from the existing record; 'modified' is refreshed.
        """
        new_reg = copy.deepcopy(self)
        cast_id = cast["id"]
        existing = new_reg.casts.get(cast_id, {})

        updated = copy.deepcopy(cast)
        updated["created"] = existing.get("created") or updated.get("created") or _now_iso()
        updated["modified"] = _now_iso()
        updated.setdefault("name", cast_id)
        updated.setdefault("entries", [])

        new_reg.casts[cast_id] = updated
        return new_reg

    def delete(self, cast_id: str) -> "CastRegistry":
        """Return a NEW registry with the cast removed. No-op if not found."""
        new_reg = copy.deepcopy(self)
        new_reg.casts.pop(cast_id, None)
        return new_reg

    def update_entry(
        self,
        cast_id: str,
        subject_id: str,
        bundle_id: str | None = None,
        visual_mode: str | None = None,
        use_audio: bool | None = None,
    ) -> "CastRegistry":
        """Return a NEW registry with one entry updated within an existing cast.

        If subject_id is not yet in the cast it is appended as a new entry.
        Fields passed as None are left unchanged on existing entries.
        """
        new_reg = copy.deepcopy(self)
        cast = new_reg.casts.get(cast_id)
        if cast is None:
            raise KeyError(f"Cast {cast_id!r} not found")

        for entry in cast["entries"]:
            if entry["subject_id"] == subject_id:
                if bundle_id is not None:
                    entry["bundle_id"] = bundle_id
                if visual_mode is not None:
                    entry["visual_mode"] = visual_mode
                if use_audio is not None:
                    entry["use_audio"] = use_audio
                cast["modified"] = _now_iso()
                return new_reg

        # Subject not yet in cast — append
        cast["entries"].append(make_entry(
            subject_id=subject_id,
            bundle_id=bundle_id or "",
            visual_mode=visual_mode or "images",
            use_audio=use_audio if use_audio is not None else False,
        ))
        cast["modified"] = _now_iso()
        return new_reg

    def remove_entry(self, cast_id: str, subject_id: str) -> "CastRegistry":
        """Return a NEW registry with subject_id removed from cast_id."""
        new_reg = copy.deepcopy(self)
        cast = new_reg.casts.get(cast_id)
        if cast is None:
            raise KeyError(f"Cast {cast_id!r} not found")
        cast["entries"] = [e for e in cast["entries"] if e["subject_id"] != subject_id]
        cast["modified"] = _now_iso()
        return new_reg

    def save(self, path: str | None = None, backup: bool = True) -> str:
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for cast registry save")
        save_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: str) -> CastRegistry:
    """Load registry from JSON. Returns empty registry if file absent."""
    if not os.path.exists(path):
        return CastRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CastRegistry.from_dict(data, file_path=path)


def save_registry(registry: CastRegistry, path: str, backup: bool = True) -> None:
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

def validate_cast(cast: dict) -> list[str]:
    """Return warning strings for a cast dict. Empty list means valid."""
    warnings: list[str] = []
    entries = cast.get("entries", [])

    seen: set[str] = set()
    for i, entry in enumerate(entries):
        sid = entry.get("subject_id", "")
        if not sid:
            warnings.append(f"entry[{i}]: subject_id is empty")
        elif sid in seen:
            warnings.append(f"entry[{i}]: subject {sid!r} appears more than once")
        else:
            seen.add(sid)

        if not entry.get("bundle_id"):
            warnings.append(f"entry[{i}] ({sid}): bundle_id is empty")

        vm = entry.get("visual_mode", "images")
        if vm not in VISUAL_MODES:
            warnings.append(f"entry[{i}] ({sid}): visual_mode must be one of {VISUAL_MODES!r}, got {vm!r}")

    return warnings


# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_cast_for_subject(cast: dict, subject_id: str) -> dict | None:
    """Return the cast entry for subject_id, or None if not in this cast."""
    for entry in cast.get("entries", []):
        if entry.get("subject_id") == subject_id:
            return copy.deepcopy(entry)
    return None
