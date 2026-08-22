"""Pure logic for the Source Profile system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.

A SourceProfile is a media-first subject catalog: one video or image
annotated with the identifiable subjects it contains (people, objects,
locations, animals, soundscapes). Multiple subjects can reference the
same source media, which is loaded and conditioned once rather than once
per subject.
"""
from __future__ import annotations

import copy
import json
import os
import shutil

# ── Constants ──────────────────────────────────────────────────────────────────

ENTITY_TYPES: list[str] = ["person", "object", "location", "animal", "soundscape"]
MEDIA_TYPES: list[str]  = ["video", "image"]
MEDIA_DIRS: list[str]   = ["input", "output"]

_EMPTY_DATA: dict = {"version": 1, "profiles": {}}


# ── Internal normalisation ─────────────────────────────────────────────────────

def _normalize_subject(entry: dict) -> dict:
    return {
        "id":               str(entry.get("id", "")),
        "label":            str(entry.get("label", "")),
        "role_description": str(entry.get("role_description", "")),
        "entity_type":      str(entry.get("entity_type", "person")),
        "notes":            str(entry.get("notes", "")),
    }


def _normalize_profile(pid: str, entry: dict) -> dict:
    return {
        "id":             pid,
        "name":           str(entry.get("name", pid)),
        "media_filename": str(entry.get("media_filename", "")),
        "media_dir":      str(entry.get("media_dir", "input")),
        "media_type":     str(entry.get("media_type", "video")),
        "subjects":       [_normalize_subject(s) for s in entry.get("subjects", []) if s],
    }


# ── SourceProfileRegistry ──────────────────────────────────────────────────────

class SourceProfileRegistry:
    """In-memory source profile registry.

    All mutation methods return a NEW instance; the original is never modified.
    This makes it safe for multi-branch wiring in ComfyUI graphs.
    """

    def __init__(
        self,
        profiles: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.profiles:   dict       = profiles or {}
        self.file_path:  str | None = file_path
        self.version:    int        = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "SourceProfileRegistry":
        profiles: dict = {}
        for pid, pdata in data.get("profiles", {}).items():
            profiles[pid] = _normalize_profile(pid, pdata)
        return cls(profiles=profiles, file_path=file_path, version=data.get("version", 1))

    def to_dict(self) -> dict:
        return {"version": self.version, "profiles": copy.deepcopy(self.profiles)}

    # ── profile mutations ──────────────────────────────────────────────────────

    def define_profile(
        self,
        profile_id: str,
        name: str = "",
        media_filename: str = "",
        media_dir: str = "input",
        media_type: str = "video",
    ) -> "SourceProfileRegistry":
        """Return a NEW registry with the profile created or updated.

        Existing subjects within the profile are preserved on update.
        """
        new_reg = copy.deepcopy(self)
        existing = new_reg.profiles.get(profile_id, {})
        new_reg.profiles[profile_id] = {
            "id":             profile_id,
            "name":           name or profile_id,
            "media_filename": media_filename,
            "media_dir":      media_dir if media_dir in MEDIA_DIRS else "input",
            "media_type":     media_type if media_type in MEDIA_TYPES else "video",
            "subjects":       existing.get("subjects", []),
        }
        return new_reg

    def define_subject(
        self,
        profile_id: str,
        subject_id: str,
        label: str = "",
        role_description: str = "",
        entity_type: str = "person",
        notes: str = "",
    ) -> "SourceProfileRegistry":
        """Return a NEW registry with the subject added or updated within the profile.

        Creates a profile shell if the profile_id does not yet exist.
        Same subject_id within the same profile → overwrites all fields.
        Subject order is preserved; new subjects append at the end.
        """
        new_reg = copy.deepcopy(self)
        if profile_id not in new_reg.profiles:
            new_reg.profiles[profile_id] = _normalize_profile(profile_id, {})

        entry = {
            "id":               subject_id,
            "label":            label or subject_id,
            "role_description": role_description,
            "entity_type":      entity_type if entity_type in ENTITY_TYPES else "person",
            "notes":            notes,
        }

        subjects = new_reg.profiles[profile_id]["subjects"]
        for i, s in enumerate(subjects):
            if s.get("id") == subject_id:
                subjects[i] = entry
                return new_reg
        subjects.append(entry)
        return new_reg

    def remove_subject(self, profile_id: str, subject_id: str) -> "SourceProfileRegistry":
        """Return a NEW registry with the subject removed. No-op if not found."""
        if profile_id not in self.profiles:
            return self
        new_reg = copy.deepcopy(self)
        p = new_reg.profiles[profile_id]
        p["subjects"] = [s for s in p["subjects"] if s.get("id") != subject_id]
        return new_reg

    def remove_profile(self, profile_id: str) -> "SourceProfileRegistry":
        """Return a NEW registry with the profile removed. No-op if not found."""
        if profile_id not in self.profiles:
            return self
        new_reg = copy.deepcopy(self)
        del new_reg.profiles[profile_id]
        return new_reg

    # ── queries ────────────────────────────────────────────────────────────────

    def get_profile(self, profile_id: str) -> dict | None:
        return self.profiles.get(profile_id)

    def get_subject(self, profile_id: str, subject_id: str) -> dict | None:
        profile = self.profiles.get(profile_id)
        if not profile:
            return None
        for s in profile.get("subjects", []):
            if s.get("id") == subject_id:
                return s
        return None

    def profile_ids(self) -> list[str]:
        return list(self.profiles.keys())

    def subject_ids(self, profile_id: str) -> list[str]:
        profile = self.profiles.get(profile_id)
        if not profile:
            return []
        return [s.get("id", "") for s in profile.get("subjects", [])]

    def build_subject_wire_dict(self, profile_id: str, subject_id: str) -> dict | None:
        """Return a SUBJECT_PROFILE-compatible wire dict for a source-derived subject.

        The returned dict carries source_profile_id so PromptAssemble can choose
        the compact rendering path and deduplicate the media reference.
        Returns None if the profile or subject is not found.
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return None
        subject = self.get_subject(profile_id, subject_id)
        if not subject:
            return None
        return {
            "subject_id":        subject_id,
            "name":              subject.get("label", subject_id),
            "concept_id":        None,
            # Source-derived marker — absent on standalone SubjectProfile dicts
            "source_profile_id": profile_id,
            "source_media_type": profile.get("media_type", "video"),
            "source_media_file": profile.get("media_filename", ""),
            "source_media_dir":  profile.get("media_dir", "input"),
            "role_description":  subject.get("role_description", ""),
            "entity_type":       subject.get("entity_type", "person"),
        }

    def list_profiles(self, filter_type: str = "all") -> str:
        """Return a formatted human-readable listing of all profiles and their subjects."""
        if not self.profiles:
            return "No source profiles defined."
        lines: list[str] = []
        for pid in sorted(self.profiles):
            p = self.profiles[pid]
            mtype = p.get("media_type", "video")
            if filter_type != "all" and mtype != filter_type:
                continue
            name     = p.get("name", pid)
            filename = p.get("media_filename", "")
            lines.append(f"[{pid}] {name}  ({mtype}: {filename})")
            for s in p.get("subjects", []):
                etype = s.get("entity_type", "person")
                label = s.get("label", s.get("id", ""))
                role  = s.get("role_description", "")
                role_str = f" — {role}" if role else ""
                lines.append(f"  [{etype}] {label}{role_str}")
        return "\n".join(lines) if lines else "No source profiles match the filter."

    def save(self, path: str | None = None, backup: bool = True) -> str:
        """Save to file and return the path written."""
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for source profile registry save")
        save_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: str) -> SourceProfileRegistry:
    """Load registry from JSON. Returns an empty registry if the file is absent."""
    if not os.path.exists(path):
        return SourceProfileRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SourceProfileRegistry.from_dict(data, file_path=path)


def save_registry(registry: SourceProfileRegistry, path: str, backup: bool = True) -> None:
    """Write registry to JSON, optionally creating a .bak backup first."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry.to_dict(), fh, indent=2, ensure_ascii=False)
    registry.file_path = path
