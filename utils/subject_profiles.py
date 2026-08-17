"""Pure logic for the Subject Profile system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.
"""
from __future__ import annotations

import copy
import json
import os
import shutil

# ── Constants ──────────────────────────────────────────────────────────────────

# Valid role labels for character_sheet_images entries.
# Free-form strings are also accepted; these are the canonical values.
SHEET_ROLES: list[str] = [
    "character sheet",
    "portrait",
    "side profile",
    "full body",
    "costume detail",
    "reference",
]

SUPPORTED_LANGUAGES = [
    "en-us", "en-gb", "ja", "ko", "zh", "zh-tw",
    "es", "fr", "de", "it", "pt", "ru", "ar", "hi",
]

_EMPTY_DATA: dict = {"version": 1, "subjects": {}}


def _normalize_sheet_entry(entry: str | dict) -> dict:
    """Coerce a character_sheet_images entry to {file, role} form.

    Accepts either a bare filename string (legacy) or a dict.
    Falls back to role "character sheet" for any entry that lacks one.
    """
    if isinstance(entry, str):
        return {"file": entry, "role": "character sheet"}
    return {"file": str(entry.get("file", "")), "role": str(entry.get("role", "character sheet")) or "character sheet"}


def normalize_sheet_images(entries: list) -> list[dict]:
    """Normalize a character_sheet_images list to list[{file, role}]."""
    return [_normalize_sheet_entry(e) for e in (entries or []) if e]


# ── SubjectRegistry ────────────────────────────────────────────────────────────

class SubjectRegistry:
    """In-memory subject profile registry.  Wire between Load → Define nodes."""

    def __init__(
        self,
        subjects: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.subjects: dict = subjects or {}
        self.file_path: str | None = file_path
        self.version: int = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "SubjectRegistry":
        subjects = copy.deepcopy(data.get("subjects", {}))
        for subj in subjects.values():
            if "character_sheet_images" in subj:
                subj["character_sheet_images"] = normalize_sheet_images(
                    subj["character_sheet_images"]
                )
        return cls(
            subjects=subjects,
            file_path=file_path,
            version=data.get("version", 1),
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "subjects": copy.deepcopy(self.subjects)}

    # ── mutation (returns new instance — safe for multi-branch wiring) ─────────

    def define(
        self,
        subject_id: str,
        name: str = "",
        appearance_summary: str = "",
        face: str = "",
        hair: str = "",
        body: str = "",
        default_outfit: str = "",
        voice_description: str = "",
        audio_reference_file: str = "",
        language: str = "en-us",
        concept_id: str = "",
    ) -> "SubjectRegistry":
        """Return a NEW registry with the subject added or updated.

        Existing fields not provided here are preserved (e.g. character_sheet_images).
        Same subject_id → overwrites all text fields; character_sheet_images kept.
        """
        new_reg = copy.deepcopy(self)
        existing = new_reg.subjects.get(subject_id, {})
        new_reg.subjects[subject_id] = {
            "name": name or subject_id,
            "appearance": {
                "summary": appearance_summary,
                "face": face,
                "hair": hair,
                "body": body,
                "default_outfit": default_outfit,
            },
            "voice": {
                "description": voice_description,
                "audio_reference_file": audio_reference_file,
                "language": language or "en-us",
            },
            # Preserve existing character_sheet_images; normalize to {file, role} dicts
            "character_sheet_images": normalize_sheet_images(
                existing.get("character_sheet_images", [])
            ),
            "concept_id": concept_id,
        }
        return new_reg

    # ── queries ────────────────────────────────────────────────────────────────

    def get_subject(self, subject_id: str) -> dict | None:
        return self.subjects.get(subject_id)

    def subject_ids(self) -> list[str]:
        return list(self.subjects.keys())

    def list_subjects(self) -> str:
        """Return a formatted human-readable subject list."""
        if not self.subjects:
            return "No subjects defined."
        lines = []
        for sid in sorted(self.subjects):
            s = self.subjects[sid]
            name = s.get("name", sid)
            summary = s.get("appearance", {}).get("summary", "")
            concept = s.get("concept_id", "")
            concept_str = f"  [{concept}]" if concept else ""
            if len(summary) > 60:
                summary_str = f" - {summary[:60]}..."
            elif summary:
                summary_str = f" - {summary}"
            else:
                summary_str = ""
            lines.append(f"[{sid}] {name}{summary_str}{concept_str}")
        return "\n".join(lines)

    def save(self, path: str | None = None, backup: bool = True) -> str:
        """Save to file, return the path written."""
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for subject registry save")
        save_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: str) -> SubjectRegistry:
    """Load registry from JSON.  Returns empty registry if file absent."""
    if not os.path.exists(path):
        return SubjectRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SubjectRegistry.from_dict(data, file_path=path)


def save_registry(registry: SubjectRegistry, path: str, backup: bool = True) -> None:
    """Write registry to JSON, optionally creating a .bak first."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry.to_dict(), fh, indent=2, ensure_ascii=False)
    registry.file_path = path
