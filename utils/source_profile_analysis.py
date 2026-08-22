"""LLM-assisted subject decomposition for Source Profiles.

Provides focused, additive analysis passes that ask a VLM to identify
subjects of a specific entity type within source media.  Each pass
produces a structured JSON list of candidate subjects that the user
can accept, edit, or discard before they are committed to the profile.

No ComfyUI dependencies — all ComfyUI I/O (frame extraction, captioner
calls, directory helpers) stays in extension.py so this module is fully
testable in CI.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
from datetime import datetime, timezone

# ── Pass types ─────────────────────────────────────────────────────────────────

PASS_TYPES: list[str] = [
    "people",
    "setting",
    "soundscape",
    "objects",
    "animals",
    "custom",
]

# entity_type emitted by each pass (used to filter VLM output and as a default
# when the model omits the field)
PASS_ENTITY_DEFAULTS: dict[str, str] = {
    "people":     "person",
    "setting":    "location",
    "soundscape": "soundscape",
    "objects":    "object",
    "animals":    "animal",
    "custom":     "object",  # safest generic default for unknown custom passes
}

# ── Prompt templates ───────────────────────────────────────────────────────────

_JSON_SCHEMA_INSTRUCTION = """
Return ONLY valid JSON matching this exact schema — no markdown, no prose, no code fences:
{
  "subjects": [
    {
      "label":            "short identifying phrase (≤10 words)",
      "role_description": "1-2 sentence description of how this subject appears in the scene",
      "entity_type":      "person|object|location|animal|soundscape",
      "notes":            "optional additional context or caveats (may be empty string)"
    }
  ]
}
If no subjects of the requested type are found, return: {"subjects": []}
""".strip()

_PASS_PROMPTS: dict[str, str] = {
    "people": (
        "Examine this image carefully. Identify every distinct person visible.\n\n"
        "For each person:\n"
        "- Disambiguate them by position, clothing colour, or prominent feature "
        "(e.g. 'woman in blue top, stage left', 'man seated at table, background right').\n"
        "- role_description: 1-2 sentences describing their appearance and position in the scene.\n"
        "- entity_type must be 'person'.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    ),
    "setting": (
        "Examine the environment and setting of this image. Identify the location and its "
        "distinct visual elements — room type, architecture, furniture layout, lighting quality, "
        "colour palette, time of day, weather if visible.\n\n"
        "Treat the overall setting as one subject labelled by the most specific location name "
        "you can give ('corner booth in a dimly-lit diner', not just 'restaurant').\n"
        "If distinct sub-zones are visible (foreground vs background, stage vs audience), "
        "list them as separate subjects.\n"
        "entity_type must be 'location'.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    ),
    "soundscape": (
        "This image is a frame from a video. Based on the visual cues visible — "
        "instruments, speakers, crowds, machinery, natural environment, signage — "
        "infer what audio layers are likely present in the original video.\n\n"
        "List each distinct audio layer as a separate subject "
        "(e.g. 'ambient cafe chatter', 'acoustic guitar performance', 'traffic from open window').\n"
        "entity_type must be 'soundscape'.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    ),
    "objects": (
        "Examine this image for significant objects and props. Focus on items that are "
        "visually prominent, narratively important, or that a director would specifically "
        "reference when describing the scene (e.g. 'the red suitcase', 'the vintage typewriter', "
        "'the chess board on the table').\n\n"
        "Exclude generic furniture unless it is a featured prop. "
        "entity_type must be 'object'.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    ),
    "animals": (
        "Examine this image for any animals. Identify each distinct animal visible, "
        "including pets, wildlife, birds, fish in tanks, insects if prominent.\n\n"
        "entity_type must be 'animal'.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    ),
    "custom": "",  # user supplies full prompt; _JSON_SCHEMA_INSTRUCTION appended at call time
}


def build_prompt(pass_type: str, prompt_override: str = "") -> str:
    """Return the final prompt string for the given pass type.

    If prompt_override is non-empty it replaces the template body; the
    JSON schema instruction is always appended so output remains parseable.
    For the 'custom' pass type, prompt_override is required.
    """
    if prompt_override.strip():
        body = prompt_override.strip()
        if _JSON_SCHEMA_INSTRUCTION not in body:
            body = body.rstrip() + "\n\n" + _JSON_SCHEMA_INSTRUCTION
        return body

    template = _PASS_PROMPTS.get(pass_type, "")
    if not template:
        return (
            f"Identify all subjects of interest in this image.\n\n"
            + _JSON_SCHEMA_INSTRUCTION
        )
    return template


# ── VLM response parsing ───────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_VALID_ENTITY_TYPES = {"person", "object", "location", "animal", "soundscape"}


def _parse_vlm_json_response(
    raw: str,
    pass_type: str = "custom",
) -> list[dict]:
    """Parse a VLM response string into a list of validated candidate dicts.

    Strips markdown code fences if present.  Returns an empty list if the
    response cannot be parsed or contains no valid subjects.

    Each returned dict is guaranteed to have:
        label            str  (non-empty)
        role_description str
        entity_type      str  (one of VALID_ENTITY_TYPES)
        notes            str
    """
    text = raw.strip()

    # Strip code fences
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try finding the first { … } block if the model added leading prose
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    subjects_raw = data.get("subjects") if isinstance(data, dict) else None
    if not isinstance(subjects_raw, list):
        return []

    default_entity = PASS_ENTITY_DEFAULTS.get(pass_type, "object")
    candidates: list[dict] = []
    for item in subjects_raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        entity_type = str(item.get("entity_type", "")).strip().lower()
        if entity_type not in _VALID_ENTITY_TYPES:
            entity_type = default_entity
        candidates.append({
            "label":            label,
            "role_description": str(item.get("role_description", "")).strip(),
            "entity_type":      entity_type,
            "notes":            str(item.get("notes", "")).strip(),
        })

    return candidates


# ── History ────────────────────────────────────────────────────────────────────

_HISTORY_FILENAME = "source_profile_analysis_history.json"


def _history_path(data_dir: str) -> str:
    return os.path.join(data_dir, _HISTORY_FILENAME)


def load_history(data_dir: str) -> list[dict]:
    """Load all analysis history entries. Returns [] if file is absent."""
    path = _history_path(data_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("entries", [])
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def history_for_profile(data_dir: str, profile_id: str) -> list[dict]:
    """Return history entries for a specific profile, newest first."""
    all_entries = load_history(data_dir)
    return [e for e in reversed(all_entries) if e.get("profile_id") == profile_id]


def append_history_entry(
    data_dir: str,
    profile_id: str,
    media_file: str,
    pass_type: str,
    prompt: str,
    candidates: list[dict],
    backup: bool = True,
) -> dict:
    """Append a new history entry and save.  Returns the new entry dict."""
    path = _history_path(data_dir)
    entries = load_history(data_dir)

    entry: dict = {
        "profile_id": profile_id,
        "media_file":  media_file,
        "pass_type":   pass_type,
        "prompt":      prompt,
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates":  copy.deepcopy(candidates),
    }
    entries.append(entry)

    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")

    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, indent=2, ensure_ascii=False)

    return entry


# ── Frame extraction (pure Python / PIL / subprocess — no ComfyUI) ─────────────

def extract_video_frame(
    video_path: str,
    out_path: str,
    position_frac: float = 0.10,
) -> str:
    """Extract a single frame from a video at position_frac (0–1) into out_path.

    Tries ffmpeg first (fast, no Python deps); falls back to imageio if
    ffmpeg is not on PATH.  Returns out_path on success, raises RuntimeError
    if neither method works.
    """
    import subprocess

    # Use ffmpeg if available
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 video_path]
    duration: float | None = None
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
    except Exception:
        pass

    seek_time = max(0.0, (duration or 10.0) * position_frac) if duration else 1.0

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True, timeout=30)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception:
        pass

    # Fallback: imageio
    try:
        import imageio  # type: ignore
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 24)
        n_frames = meta.get("nframes") or int((duration or 10) * fps)
        target_idx = min(int(n_frames * position_frac), max(n_frames - 1, 0))
        frame = reader.get_data(target_idx)
        from PIL import Image as _PIL
        _PIL.fromarray(frame).save(out_path)
        reader.close()
        return out_path
    except Exception:
        pass

    raise RuntimeError(
        f"Could not extract frame from {video_path!r}. "
        "Ensure ffmpeg is on PATH or imageio is installed."
    )
