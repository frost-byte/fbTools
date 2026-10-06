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
AUDIO_SOURCES = ("extract_from_visual", "extract_from_video", "file", "none")

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
        updated.setdefault("pronoun_style", "")
        updated.setdefault("short_name", "")
        # Migrate legacy appearance_override → appearance.summary when upgrading.
        legacy_override = updated.pop("appearance_override", "") or ""
        appearance = updated.setdefault("appearance", {})
        appearance.setdefault("summary", legacy_override)
        appearance.setdefault("hair", "")
        appearance.setdefault("face", "")
        appearance.setdefault("body", "")
        appearance.setdefault("default_outfit", "")
        updated.setdefault("tags", [])

        visual = updated.setdefault("visual", {})
        # visual.type is the *default mode* preference ("images" | "video").
        # Both visual.file (video) and visual.files (images) may be set
        # simultaneously so the user can switch modes per cast entry without
        # maintaining separate bundles.
        visual.setdefault("type", "images")
        visual.setdefault("file", "")
        visual.setdefault("video_dir", "input")
        visual.setdefault("files", [])
        visual.setdefault("start_time", 0.0)
        visual.setdefault("duration", 0.0)
        visual.setdefault("force_rate", 24)  # H3 requires 24fps reference video
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

    visual_type = visual.get("type", "images")   # default-mode preference
    audio_source = audio.get("source", "none")
    has_video  = bool(visual.get("file", ""))
    has_images = bool(visual.get("files", []))

    if visual_type not in VISUAL_TYPES:
        warnings.append(f"visual.type must be one of {VISUAL_TYPES!r}, got {visual_type!r}")
    if audio_source not in AUDIO_SOURCES:
        warnings.append(f"audio.source must be one of {AUDIO_SOURCES!r}, got {audio_source!r}")
    # Warn only when the *default* mode lacks media; having both is valid.
    if visual_type == "video" and not has_video:
        warnings.append("default mode is 'video' but visual.file is empty")
    if visual_type == "images" and not has_images:
        warnings.append("default mode is 'images' but visual.files is empty")
    if audio_source == "extract_from_visual" and not has_video:
        warnings.append("audio.source 'extract_from_visual' requires a video file (visual.file)")
    if audio_source == "extract_from_video" and not audio.get("video_file", ""):
        warnings.append("audio.source 'extract_from_video' requires a video file (audio.video_file)")
    if audio_source == "file" and not audio.get("file"):
        warnings.append("audio.source is 'file' but audio.file is empty")

    return warnings


def bundle_audio_wanted(cast_entry: dict, clip: dict) -> bool:
    """Whether a Source Profile clip generation should include a cast entry's bundle audio
    reference, for ANY of the bundle's audio sources (file / extract_from_video /
    extract_from_visual): the entry's audio checkbox (use_audio) must be on AND the clip
    segment must allow dialogue (allows_dialogue, default True). allows_dialogue=False exists
    to keep the model from inventing speech or gibberish for that shot, so it suppresses the
    reference regardless of where the audio would otherwise come from."""
    return bool((cast_entry or {}).get("use_audio")) and bool((clip or {}).get("allows_dialogue", True))


BUNDLE_AUDIO_SWITCH_BUNDLE = 1
BUNDLE_AUDIO_SWITCH_FALLBACK = 2


def bundle_audio_switch_select(audio_loaded: bool) -> int:
    """1-based `select` value for an ImpactSwitch choosing between a Reference Bundle's own
    audio (input1) and a fallback such as the footage's own audio (input2): 1 when the bundle's
    audio actually loaded, 2 otherwise (no bundle picked, bundle not found, no audio configured,
    or the load failed). Computed by BundleAudioReferenceLoad itself because a workflow-side
    math node can't derive this -- ComfyMathExpression only accepts numeric inputs."""
    return BUNDLE_AUDIO_SWITCH_BUNDLE if audio_loaded else BUNDLE_AUDIO_SWITCH_FALLBACK


def resolve_bundle_audio_source(bundle: dict) -> dict | None:
    """Resolve a bundle's audio.source into {"file", "dir", "start_time", "duration"} ready
    for a caller to load (e.g. via ffmpeg), or None if the bundle has no audio configured
    (source == "none"/unset, or the field its source needs is empty).

    Mirrors the exact per-source field resolution already used server-side when building a
    bundle's audio reference for a Scene Cast generation (nodes/source_profiles.py's
    bundle_video_entries construction in SourceProfileClipPrompt.execute()):
      "file"                -> audio.file / audio.start_time / audio.duration (own
                                standalone clip, its own trim window), in audio.audio_dir
                                ("input" by default, or "output" -- same field the
                                Composition path in nodes/compositions.py honors)
      "extract_from_visual" -> visual.file / visual.start_time / visual.duration (the SAME
                                clip the bundle's own visual reference uses)
      "extract_from_video"  -> audio.video_file / audio.start_time / audio.duration (a
                                separate, audio-only reference video)
      "none" (or missing)   -> None
    """
    if not isinstance(bundle, dict):
        return None
    audio = bundle.get("audio", {}) or {}
    source = audio.get("source", "none")

    if source == "file":
        fname = audio.get("file", "")
        if not fname:
            return None
        return {
            "file": fname, "dir": audio.get("audio_dir", "input") or "input",
            "start_time": float(audio.get("start_time", 0.0) or 0.0),
            "duration": float(audio.get("duration", 0.0) or 0.0),
        }
    if source == "extract_from_visual":
        visual = bundle.get("visual", {}) or {}
        fname = visual.get("file", "")
        if not fname:
            return None
        return {
            "file": fname, "dir": visual.get("video_dir", "input") or "input",
            "start_time": float(visual.get("start_time", 0.0) or 0.0),
            "duration": float(visual.get("duration", 0.0) or 0.0),
        }
    if source == "extract_from_video":
        fname = audio.get("video_file", "")
        if not fname:
            return None
        return {
            "file": fname, "dir": audio.get("video_dir", "input") or "input",
            "start_time": float(audio.get("start_time", 0.0) or 0.0),
            "duration": float(audio.get("duration", 0.0) or 0.0),
        }
    return None
