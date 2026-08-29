"""LLM-assisted subject decomposition and segment detection for Source Profiles.

Two analysis families:

1. **Subject passes** (existing) — ask a VLM to identify subjects of a
   specific entity type (people, setting, objects …).  Returns a list of
   candidate subject dicts that the user can accept into the profile.

2. **Segment analysis** (new) — ask a VLM to detect meaningful action/scene
   transitions in a video, or describe the action visible in a single clip
   frame.  Returns boundary timestamps or a one-sentence action description.

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

    if isinstance(data, list):
        subjects_raw = data
    elif isinstance(data, dict):
        subjects_raw = data.get("subjects")
    else:
        subjects_raw = None
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


def probe_video_resolution(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe, or (0, 0)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def probe_video_fps(video_path: str) -> float:
    """Return the video's native FPS via ffprobe, defaulting to 24.0."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        num, _, den = result.stdout.strip().partition("/")
        if den and float(den):
            return float(num) / float(den)
        if num:
            return float(num)
    except Exception:
        pass
    return 24.0


def extract_clip_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames: int = 20,
    select_every_nth: int = 1,
    raw_fps: float | None = None,
) -> tuple[list, list[float], float]:
    """Extract frames from a video clip for multi-frame VLM input.

    Returns ``(pil_frames, timestamps, sample_fps)`` where:
    - ``pil_frames``  — list of ``PIL.Image.Image`` objects (RGB)
    - ``timestamps``  — corresponding absolute timestamps in seconds
    - ``sample_fps``  — effective frames-per-second of the returned sequence,
                         for use as ``video_meta["sample_fps"]``

    Args:
        video_path:      absolute path to the video file
        start_time:      clip start in seconds
        end_time:        clip end in seconds
        max_frames:      hard cap on the number of frames returned
        select_every_nth: treat the video as if sampled at raw_fps/select_every_nth;
                          controls the spacing between extracted timestamps
        raw_fps:         native video FPS (probed via ffprobe if None)
    """
    import subprocess
    import tempfile
    from PIL import Image as _PIL

    duration = max(end_time - start_time, 0.01)
    if raw_fps is None:
        raw_fps = probe_video_fps(video_path)

    effective_fps = raw_fps / max(select_every_nth, 1)
    n_ideal = int(duration * effective_fps)
    n_frames = max(1, min(n_ideal, max_frames))

    if n_frames == 1:
        timestamps = [start_time + duration / 2.0]
    else:
        step = duration / (n_frames - 1) if n_frames > 1 else duration
        timestamps = [start_time + i * step for i in range(n_frames)]

    sample_fps = len(timestamps) / duration

    pil_frames: list = []
    actual_timestamps: list[float] = []
    tmp_files: list[str] = []

    try:
        for ts in timestamps:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            tmp_files.append(tmp.name)
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ts:.4f}", "-i", video_path,
                "-vframes", "1", "-q:v", "2", tmp.name,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    pil_frames.append(_PIL.open(tmp.name).convert("RGB").copy())
                    actual_timestamps.append(ts)
            except Exception:
                pass
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

    if not pil_frames:
        raise RuntimeError(f"Could not extract any frames from {video_path!r} "
                           f"between {start_time:.1f}s–{end_time:.1f}s")

    actual_fps = len(pil_frames) / duration
    return pil_frames, actual_timestamps, actual_fps


def build_contact_sheet_image(
    pil_frames: list,
    timestamps: list[float],
    thumb_w: int = 320,
    thumb_h: int = 180,
    n_cols: int = 6,
):
    """Compose a list of PIL frames into a labelled contact-sheet image.

    Returns a single ``PIL.Image.Image``.  Falls back to the first frame if
    PIL is unavailable.
    """
    from PIL import Image as _PIL, ImageDraw as _Draw

    imgs = [img.resize((thumb_w, thumb_h)) for img in pil_frames]
    n_cols = min(len(imgs), n_cols)
    n_rows = (len(imgs) + n_cols - 1) // n_cols
    sheet = _PIL.new("RGB", (thumb_w * n_cols, (thumb_h + 20) * n_rows), (30, 30, 30))
    draw  = _Draw.Draw(sheet)
    for i, (img, ts) in enumerate(zip(imgs, timestamps)):
        row, col = divmod(i, n_cols)
        x = col * thumb_w
        y = row * (thumb_h + 20)
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h + 2), f"{ts:.2f}s", fill=(200, 200, 200))
    return sheet


# ── Segment / boundary detection ───────────────────────────────────────────────

_SEGMENT_SCHEMA_INSTRUCTION = """
Return ONLY valid JSON — no markdown, no prose, no code fences:
{
  "segments": [
    {
      "start_time": <float, seconds>,
      "end_time":   <float, seconds>,
      "label":      "short title for this segment (≤8 words)",
      "action":     "1-2 sentence description of the action in this segment"
    }
  ]
}
If you cannot identify distinct transitions, return a single segment covering
the full time range.  Times must be in ascending order with no gaps.
""".strip()

_CLIP_DESCRIPTION_SCHEMA = """
Return ONLY valid JSON — no markdown, no prose, no code fences:
{
  "action": "1-2 sentences describing what is happening in this frame"
}
""".strip()

_CLIP_DESCRIPTION_SCHEMA_WITH_SLOTS = """
Return ONLY valid JSON — no markdown, no prose, no code fences:
{
  "action": "1-2 sentences using the placeholder labels (e.g. {A}, {B}) for identified subjects"
}
""".strip()

_SUBJECT_CONTEXT_HEADER = (
    "\n\nThe following subjects may appear in this frame. "
    "Use their placeholder labels instead of describing their appearance:\n"
)

_DETECT_BOUNDARY_BASE = """\
You are analyzing a series of video frames, each labelled with its timestamp.
The frames are sampled at regular intervals from a single video clip.

Your task: identify where MEANINGFUL scene or action transitions occur —
moments where what is happening changes significantly.\
"""

# Flag-controlled sentence snippets appended to the base prompt.
_FLAG_CAMERA_CUTS = (
    "Hard camera cuts, lens angle changes, and scene edits are transition "
    "boundaries even when the subject or setting remains the same."
)
_FLAG_SUBJECT_CHANGES = (
    "A key subject entering or leaving the frame counts as a transition "
    "when it represents a notable shift in the scene's cast."
)
_FLAG_LOWER_THRESHOLD = (
    "When in doubt, err on the side of marking more boundaries rather than fewer."
)

_DETECT_BOUNDARY_TAIL = """\
Minor continuous motion (walking, talking) within the same scene is NOT a
transition unless there is a clear shift in what is happening.

For each segment between transitions, provide:
- start_time / end_time  (seconds)
- label  (very short title, ≤8 words)
- action  (1-2 sentence summary of the action in that segment)\
"""

_DESCRIBE_CLIP_PROMPT = (
    "Examine this video frame carefully.  Describe in 1-2 sentences "
    "what action or situation is visually depicted — focus on what the "
    "subjects are doing or what is occurring in the scene at this moment.\n\n"
    + _CLIP_DESCRIPTION_SCHEMA
)


def build_segment_detection_prompt(
    prompt_override: str = "",
    flags: dict | None = None,
) -> str:
    """Return the VLM prompt for boundary detection from a frame strip.

    Args:
        prompt_override: When non-empty, used verbatim (schema appended if absent).
        flags: Optional dict controlling optional sentence injections:
            camera_cuts (bool, default True)  — include camera cut / angle change language
            subject_changes (bool, default False) — include subject entry/exit language
            lower_threshold (bool, default False) — encourage more boundaries when unsure
    """
    if prompt_override.strip():
        body = prompt_override.strip()
        if _SEGMENT_SCHEMA_INSTRUCTION not in body:
            body = body.rstrip() + "\n\n" + _SEGMENT_SCHEMA_INSTRUCTION
        return body

    flags = flags or {}
    extras: list[str] = []
    if flags.get("camera_cuts", True):
        extras.append(_FLAG_CAMERA_CUTS)
    if flags.get("subject_changes", False):
        extras.append(_FLAG_SUBJECT_CHANGES)
    if flags.get("lower_threshold", False):
        extras.append(_FLAG_LOWER_THRESHOLD)

    parts = [_DETECT_BOUNDARY_BASE]
    if extras:
        parts.append("\nBoundary criteria:\n" + "\n".join(f"- {e}" for e in extras))
    parts.append("\n" + _DETECT_BOUNDARY_TAIL)
    parts.append("\n\n" + _SEGMENT_SCHEMA_INSTRUCTION)
    return "\n".join(parts)


def build_clip_description_prompt(
    prompt_override: str = "",
    subjects: list[tuple[str, str, str]] | None = None,
) -> str:
    """Return the VLM prompt for describing the action in a single clip frame.

    Args:
        prompt_override: If non-empty, used as the instruction body (schema appended).
        subjects: Optional list of (slot, name, appearance) tuples, e.g.
            [("A", "Elena", "woman with auburn hair"), ("B", "Marcus", "tall man")].
            When provided, the prompt instructs the VLM to use {A}, {B} placeholders
            instead of describing subjects' appearance inline.
    """
    if prompt_override.strip():
        body = prompt_override.strip()
        if _CLIP_DESCRIPTION_SCHEMA not in body:
            body = body.rstrip() + "\n\n" + _CLIP_DESCRIPTION_SCHEMA
        return body

    base = (
        "Examine this video frame carefully.  Describe in 1-2 sentences "
        "what action or situation is visually depicted — focus on what the "
        "subjects are doing or what is occurring in the scene at this moment."
    )

    if not subjects:
        return base + "\n\n" + _CLIP_DESCRIPTION_SCHEMA

    subject_lines = "\n".join(
        f"  {{{slot}}} — {name}: {appearance}" if appearance else f"  {{{slot}}} — {name}"
        for slot, name, appearance in subjects
    )
    placeholders = " / ".join(f"{{{slot}}}" for slot, _, _ in subjects)
    subject_block = (
        _SUBJECT_CONTEXT_HEADER
        + subject_lines
        + f"\n\nUse {placeholders} to refer to these subjects. "
        + "Describe only the action, framing, and emotional register — "
        + "not the subjects' appearance."
    )

    return base + subject_block + "\n\n" + _CLIP_DESCRIPTION_SCHEMA_WITH_SLOTS


def _parse_segments_response(raw: str, video_duration: float = 0.0) -> list[dict]:
    """Parse a VLM segment-detection response into a list of clip dicts.

    Each returned dict has: start_time, end_time, label, action.
    Invalid / out-of-order entries are discarded.  If the result is empty
    and video_duration > 0, returns a single segment covering [0, video_duration].
    """
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if isinstance(data, list):
        raw_segs = data
    elif isinstance(data, dict):
        raw_segs = data.get("segments", [])
    else:
        raw_segs = []
    if not isinstance(raw_segs, list):
        raw_segs = []

    segments: list[dict] = []
    prev_end: float = 0.0
    for item in raw_segs:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start_time", prev_end))
            end   = float(item.get("end_time", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if start < prev_end - 0.01:  # allow tiny float drift
            continue
        segments.append({
            "start_time": round(start, 3),
            "end_time":   round(end, 3),
            "label":      str(item.get("label", f"Segment {len(segments) + 1}")),
            "action":     str(item.get("action", "")),
        })
        prev_end = end

    if not segments and video_duration > 0:
        segments.append({
            "start_time": 0.0,
            "end_time":   round(video_duration, 3),
            "label":      "Full video",
            "action":     "",
        })

    return segments


def parse_clip_description_response(raw: str) -> str:
    """Parse a VLM clip description response into a plain action string."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get("action", "")).strip()
    except json.JSONDecodeError:
        pass
    # Fallback: return raw stripped text
    return text[:300]
