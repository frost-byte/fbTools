"""Tiny shared pieces that would otherwise create a circular import between
nodes/bundles.py and nodes/compositions.py (Plan 29): bundle-proxy eligibility/short-edge
(needed by bundle routes AND by compositions.py's _resolve_cast_media generation-time
fallback), composition system settings (needed by a bundle route AND by compositions.py's
own settings routes/PromptCompositionLoader), and the two H3 path/audio loaders bundles.py's
preprocess_audio route borrows from what is otherwise compositions.py's own territory.

Same pattern as nodes/composition_types.py (Plan 27): neither layer is a leaf relative to
the other here, so this is the neutral home both import from instead of each other.
"""
from __future__ import annotations

import json
import os

import torch
from folder_paths import get_input_directory, get_output_directory

from .shared import user_data_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


_BUNDLE_PROXY_SHORT_EDGE = 768


def _bundle_proxy_eligible(force_rate, duration: float) -> bool:
    try:
        force_rate = int(force_rate or 0)
    except (TypeError, ValueError):
        force_rate = 0
    return duration > 0.0 and force_rate in (0, 24)


# ── Composition system settings ───────────────────────────────────────────────

def _composition_settings_path() -> str:
    return os.path.join(user_data_dir(), "composition_settings.json")


_COMPOSITION_SETTINGS_DEFAULTS: dict = {
    "libber_delimiter":           "%",
    "libber_max_depth":           10,
    "default_speech_pace":        "normal",
    "default_audio_noise_removal":  False,
    "default_audio_normalize_lufs": True,
    "default_audio_target_lufs":   -14.0,
    "melband_model_path":          "",  # Kijai/MelBandRoFormer_comfy — fp16 or fp32 .safetensors
    # H3 model output limits
    "h3_max_frames":               360,  # 15 s × 24 fps — 0 = unclamped
    # H3 background-plate ("Remove People") generation overrides — each "" / 0 means "use whatever
    # templates/h3_background_plate.api.json itself specifies", i.e. unset. Only applied when the
    # template exposes the matching optional IN:* title (see utils/h3_template_runner.patch_prompt).
    "h3_bg_plate_model":            "",  # diffusion_models filename
    "h3_bg_plate_clip":             "",  # text_encoders filename
    "h3_bg_plate_lora":             "",  # loras filename
    "h3_bg_plate_lora_strength":    0.38,  # only applied when h3_bg_plate_lora is set
    "h3_bg_plate_sampler":          "",  # comfy.samplers.SAMPLER_NAMES entry
    "h3_bg_plate_scheduler":        "",  # comfy.samplers.SCHEDULER_NAMES entry
    "h3_bg_plate_steps":            0,   # 0 = unset/use template default
    # Default False: leave the model resident after a "Remove People" run, since chaining several
    # passes back-to-back (the auto-reselect flow) is the common case and reloading each time is
    # slow. Settings exposes this as "Unload model after each run" for the opposite preference.
    # Shared with the H3 Character Sheet feature below — VRAM is a machine-wide resource, not
    # per-template, so there's no separate h3_char_sheet_unload_after_run setting.
    "h3_bg_plate_unload_after_run": False,
    # H3 character/face-sheet generation overrides (Bundle editor) — same "" / 0 = unset convention
    # as h3_bg_plate_* above. See templates/README.md's "h3_character_sheet.api.json" section and
    # nodes/h3_character_sheet.py.
    "h3_char_sheet_model":                "",  # diffusion_models filename
    "h3_char_sheet_clip":                 "",  # text_encoders filename
    "h3_char_sheet_lora":                 "",  # loras filename
    "h3_char_sheet_lora_strength":        0.38,  # only applied when h3_char_sheet_lora is set
    "h3_char_sheet_sampler1":             "",  # comfy.samplers.SAMPLER_NAMES entry (first pass)
    "h3_char_sheet_scheduler1":           "",  # comfy.samplers.SCHEDULER_NAMES entry (first pass)
    "h3_char_sheet_sampler2":             "",  # comfy.samplers.SAMPLER_NAMES entry (upscale pass)
    "h3_char_sheet_upscale_steps_select": 0,   # ImpactStringSelector "select" (3/4/5-step preset); 0 = unset
    "h3_char_sheet_upscale_factor":       0.0,  # MinimaxH3LatentUpscaler3D scale multiplier; 0 = unset
    "h3_char_sheet_aspect_ratio":         "",  # ResolutionSelector aspect_ratio entry
    "h3_char_sheet_megapixels":           0.0,  # ResolutionSelector megapixels; 0 = unset
}


def _read_composition_settings() -> dict:
    defaults = dict(_COMPOSITION_SETTINGS_DEFAULTS)
    path = _composition_settings_path()
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {**defaults, **data}
    except Exception:
        return defaults


def _write_composition_settings(settings: dict) -> None:
    path = _composition_settings_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def _h3_resolve_path(path: str) -> str:
    """Return an absolute path, checking input then output directory."""
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(get_input_directory(), path)
    if os.path.exists(candidate):
        return candidate
    candidate_out = os.path.join(get_output_directory(), path)
    if os.path.exists(candidate_out):
        return candidate_out
    return path  # let callers decide what to do with a missing path


def _h3_load_audio(path: str, start_time: float = 0.0, duration: float = 0.0):
    """Load audio → {'waveform': [1,C,L] float32, 'sample_rate': int}.

    Uses ffmpeg directly (same approach as VHS get_audio) so video files and all
    audio codecs are handled correctly with accurate start_time/duration seeking.
    VHS is intentionally not imported — see nodes/compositions.py's _h3_load_video_frames
    for the reason.
    """
    resolved = _h3_resolve_path(path)
    if not os.path.exists(resolved):
        logger.warning("CompositionToH3: audio file not found: %s", path)
        return None

    # Prefer imageio_ffmpeg (ships with ComfyUI); fall back to system ffmpeg.
    ffmpeg_exe = None
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
    except Exception:
        pass
    if not ffmpeg_exe:
        import shutil as _shutil
        ffmpeg_exe = _shutil.which("ffmpeg")
    if not ffmpeg_exe:
        logger.error(
            "CompositionToH3: ffmpeg not found; cannot extract audio from %s", path
        )
        return None

    import re
    import subprocess
    args = [ffmpeg_exe, "-i", resolved]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    args += ["-f", "f32le", "-"]

    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "CompositionToH3: ffmpeg failed for %s: %s",
            path,
            exc.stderr.decode("utf-8", "backslashreplace")[:300],
        )
        return None

    stderr_text = res.stderr.decode("utf-8", "backslashreplace")
    match = re.search(r", (\d+) Hz, (\w+),", stderr_text)
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
    else:
        ar = 44100
        ac = 2
        logger.warning(
            "CompositionToH3: could not parse audio format from ffmpeg stderr for %s; "
            "assuming 44100 Hz stereo",
            path,
        )

    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}
