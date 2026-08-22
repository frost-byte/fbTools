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

DEFAULT_SEGMENT_DURATION: float = 10.0
DEFAULT_SELECT_EVERY_NTH: int   = 2
DEFAULT_FRAME_LOAD_CAP:   int   = 120


def _normalize_subject(entry: dict) -> dict:
    return {
        "id":               str(entry.get("id", "")),
        "label":            str(entry.get("label", "")),
        "role_description": str(entry.get("role_description", "")),
        "entity_type":      str(entry.get("entity_type", "person")),
        "notes":            str(entry.get("notes", "")),
    }


def _normalize_clip(entry: dict) -> dict:
    return {
        "id":               str(entry.get("id", "")),
        "label":            str(entry.get("label", "")),
        "start_time":       float(entry.get("start_time", 0.0)),
        "end_time":         float(entry.get("end_time", 0.0)),
        "select_every_nth": int(entry.get("select_every_nth", DEFAULT_SELECT_EVERY_NTH)),
        "frame_load_cap":   int(entry.get("frame_load_cap", DEFAULT_FRAME_LOAD_CAP)),
        "subjects":         [str(s) for s in entry.get("subjects", []) if s],
        "action":           str(entry.get("action", "")),
    }


def _normalize_profile(pid: str, entry: dict) -> dict:
    seg_dur = entry.get("default_segment_duration")
    return {
        "id":                       pid,
        "name":                     str(entry.get("name", pid)),
        "media_filename":           str(entry.get("media_filename", "")),
        "media_dir":                str(entry.get("media_dir", "input")),
        "media_type":               str(entry.get("media_type", "video")),
        "subjects":                 [_normalize_subject(s) for s in entry.get("subjects", []) if s],
        "clips":                    [_normalize_clip(c) for c in entry.get("clips", []) if c],
        "default_segment_duration": float(seg_dur) if seg_dur is not None else None,
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
        default_segment_duration: float | None = None,
    ) -> "SourceProfileRegistry":
        """Return a NEW registry with the profile created or updated.

        Existing subjects and clips within the profile are preserved on update.
        """
        new_reg = copy.deepcopy(self)
        existing = new_reg.profiles.get(profile_id, {})
        new_reg.profiles[profile_id] = {
            "id":                       profile_id,
            "name":                     name or profile_id,
            "media_filename":           media_filename,
            "media_dir":                media_dir if media_dir in MEDIA_DIRS else "input",
            "media_type":               media_type if media_type in MEDIA_TYPES else "video",
            "subjects":                 existing.get("subjects", []),
            "clips":                    existing.get("clips", []),
            "default_segment_duration": default_segment_duration,
        }
        return new_reg

    def set_clips(
        self,
        profile_id: str,
        clips: list[dict],
    ) -> "SourceProfileRegistry":
        """Return a NEW registry with the profile's clips replaced entirely."""
        new_reg = copy.deepcopy(self)
        if profile_id not in new_reg.profiles:
            new_reg.profiles[profile_id] = _normalize_profile(profile_id, {})
        new_reg.profiles[profile_id]["clips"] = [_normalize_clip(c) for c in clips if c]
        return new_reg

    def upsert_clip(
        self,
        profile_id: str,
        clip: dict,
    ) -> "SourceProfileRegistry":
        """Return a NEW registry with a single clip added or updated (matched by id)."""
        new_reg = copy.deepcopy(self)
        if profile_id not in new_reg.profiles:
            new_reg.profiles[profile_id] = _normalize_profile(profile_id, {})
        norm = _normalize_clip(clip)
        clips: list = new_reg.profiles[profile_id].setdefault("clips", [])
        for i, c in enumerate(clips):
            if c.get("id") == norm["id"]:
                clips[i] = norm
                return new_reg
        clips.append(norm)
        return new_reg

    def remove_clip(self, profile_id: str, clip_id: str) -> "SourceProfileRegistry":
        """Return a NEW registry with the clip removed. No-op if not found."""
        if profile_id not in self.profiles:
            return self
        new_reg = copy.deepcopy(self)
        p = new_reg.profiles[profile_id]
        p["clips"] = [c for c in p.get("clips", []) if c.get("id") != clip_id]
        return new_reg

    def auto_partition(
        self,
        profile_id: str,
        video_duration: float,
        segment_duration: float | None = None,
        select_every_nth: int = DEFAULT_SELECT_EVERY_NTH,
        frame_load_cap: int = DEFAULT_FRAME_LOAD_CAP,
        subject_ids: list[str] | None = None,
    ) -> "SourceProfileRegistry":
        """Auto-generate equal-duration clips and replace the profile's clip list.

        Clips are numbered from 1.  The last clip extends to video_duration
        even if it is shorter than segment_duration.  Does nothing if
        video_duration <= 0.
        """
        if video_duration <= 0:
            return self
        dur = (
            segment_duration
            if segment_duration and segment_duration > 0
            else (self.profiles.get(profile_id, {}).get("default_segment_duration") or DEFAULT_SEGMENT_DURATION)
        )
        subs = subject_ids or [s["id"] for s in self.profiles.get(profile_id, {}).get("subjects", [])]
        clips: list[dict] = []
        t = 0.0
        idx = 1
        while t < video_duration:
            end = min(t + dur, video_duration)
            clips.append({
                "id":               f"clip_{idx}",
                "label":            f"Segment {idx}",
                "start_time":       round(t, 3),
                "end_time":         round(end, 3),
                "select_every_nth": select_every_nth,
                "frame_load_cap":   frame_load_cap,
                "subjects":         list(subs),
                "action":           "",
            })
            t += dur
            idx += 1
        return self.set_clips(profile_id, clips)

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

    def get_clip(self, profile_id: str, clip_id: str) -> dict | None:
        """Return the clip dict for the given profile + clip_id, or None."""
        profile = self.profiles.get(profile_id)
        if not profile:
            return None
        for c in profile.get("clips", []):
            if c.get("id") == clip_id:
                return c
        return None

    def clip_load_params(self, profile_id: str, clip_id: str) -> dict | None:
        """Return load_params dict for the given clip, ready for _h3_load_video_frames.

        Returns None if the clip_id is empty or not found.
        """
        if not clip_id:
            return None
        clip = self.get_clip(profile_id, clip_id)
        if not clip:
            return None
        duration = max(0.0, clip["end_time"] - clip["start_time"])
        return {
            "start_time":       clip["start_time"],
            "duration":         duration,
            "force_rate":       0,
            "frame_load_cap":   clip.get("frame_load_cap", DEFAULT_FRAME_LOAD_CAP),
            "skip_first_frames": 0,
            "select_every_nth": clip.get("select_every_nth", DEFAULT_SELECT_EVERY_NTH),
        }

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
