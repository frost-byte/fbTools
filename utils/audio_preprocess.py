"""Pure audio preprocessing utilities — no ComfyUI dependencies.

Pipeline (each step is optional):
  1. Spectral denoising   (scipy)
  2. LUFS normalization   (pyloudnorm, ITU BS.1770)
  3. Loop to minimum duration (2 s default)
  4. Truncate to maximum duration (15 s default)
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def cache_fingerprint(file_path: str, start_time: float, duration: float, config: dict) -> str:
    """Return an 8-hex-char fingerprint for a (source, trim, config) combination."""
    mtime = os.path.getmtime(file_path) if os.path.isfile(file_path) else 0.0
    key = f"{mtime}|{start_time}|{duration}|{json.dumps(config, sort_keys=True)}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ── Loudness measurement ───────────────────────────────────────────────────────

def measure_lufs(audio_np: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness (ITU BS.1770). Returns -inf on error/silence."""
    try:
        import pyloudnorm as pyln
        data = (audio_np.T if audio_np.ndim > 1 else audio_np.reshape(-1, 1)).astype(np.float64)
        meter = pyln.Meter(sample_rate)
        return float(meter.integrated_loudness(data))
    except Exception:
        return float("-inf")


# ── Processing steps ───────────────────────────────────────────────────────────

def spectral_denoise(audio_np: np.ndarray, sample_rate: int, strength: float = 1.5) -> np.ndarray:
    """Remove hiss/background noise via spectral subtraction. [C, L] → [C, L]."""
    from scipy import signal as ss

    nperseg  = 2048
    noverlap = nperseg // 2

    def _denoise_ch(ch: np.ndarray) -> np.ndarray:
        _, _, stft = ss.stft(ch.astype(np.float64), fs=sample_rate,
                             nperseg=nperseg, noverlap=noverlap, window="hann")
        mag, phase = np.abs(stft), np.angle(stft)
        n_noise = max(1, int(0.5 * sample_rate / (nperseg - noverlap)))
        n_noise = min(n_noise, mag.shape[1] // 4)
        noise_prof = np.mean(mag[:, :n_noise], axis=1, keepdims=True)
        cleaned = np.maximum(mag - strength * noise_prof, 0.2 * noise_prof)
        _, result = ss.istft(cleaned * np.exp(1j * phase), fs=sample_rate,
                             nperseg=nperseg, noverlap=noverlap, window="hann")
        return result[: ch.shape[0]].astype(np.float32)

    if audio_np.ndim == 1:
        return _denoise_ch(audio_np)
    return np.stack([_denoise_ch(audio_np[c]) for c in range(audio_np.shape[0])])


def lufs_normalize(audio_np: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize to target integrated loudness (ITU BS.1770). [C, L] → [C, L]."""
    current = measure_lufs(audio_np, sample_rate)
    if not np.isfinite(current):
        return audio_np  # silence or too short for BS.1770 block
    try:
        import pyloudnorm as pyln
        data = (audio_np.T if audio_np.ndim > 1 else audio_np.reshape(-1, 1)).astype(np.float64)
        normalized = pyln.normalize.loudness(data, current, target_lufs)
        out = (normalized.T if audio_np.ndim > 1 else normalized.ravel()).astype(np.float32)
    except Exception:
        # Fallback: simple gain adjustment
        gain = 10 ** ((target_lufs - current) / 20)
        out = (audio_np * gain).astype(np.float32)
    peak = np.abs(out).max()
    if peak > 1.0:
        out = out / peak
    return out


def loop_to_min(audio_np: np.ndarray, sample_rate: int, min_secs: float = 2.0) -> np.ndarray:
    """Tile waveform until at least min_secs long. Works on [L] or [C, L]."""
    min_samples = int(min_secs * sample_rate)
    axis = 1 if audio_np.ndim > 1 else 0
    while audio_np.shape[axis] < min_samples:
        audio_np = np.concatenate([audio_np, audio_np], axis=axis)
    return audio_np


def truncate_to_max(audio_np: np.ndarray, sample_rate: int, max_secs: float = 15.0) -> np.ndarray:
    """Trim waveform to at most max_secs. Works on [L] or [C, L]."""
    max_samples = int(max_secs * sample_rate)
    if audio_np.ndim > 1:
        return audio_np[:, :max_samples]
    return audio_np[:max_samples]


# ── Full pipeline ──────────────────────────────────────────────────────────────

def preprocess_audio(
    waveform,
    sample_rate: int,
    *,
    noise_removal: bool = False,
    normalize_lufs: bool = True,
    target_lufs: float = -14.0,
    min_secs: float = 2.0,
    max_secs: float = 15.0,
) -> tuple:
    """Apply the audio preprocessing pipeline.

    waveform: [1, C, L] torch float32 (as returned by _h3_load_audio)
    Returns: (waveform_out [1, C, L] torch float32, sample_rate, metrics dict)
    metrics: {duration, lufs_before, lufs_after}
    """
    import torch

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)
    lufs_before = measure_lufs(audio_np, sample_rate)

    if noise_removal:
        audio_np = spectral_denoise(audio_np, sample_rate)

    if normalize_lufs:
        audio_np = lufs_normalize(audio_np, sample_rate, target_lufs)

    audio_np = loop_to_min(audio_np, sample_rate, min_secs)
    audio_np = truncate_to_max(audio_np, sample_rate, max_secs)

    lufs_after = measure_lufs(audio_np, sample_rate)
    samples = audio_np.shape[1] if audio_np.ndim > 1 else audio_np.shape[0]
    duration = samples / sample_rate if sample_rate > 0 else 0.0

    waveform_out = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
    return waveform_out, sample_rate, {
        "duration":    round(duration, 2),
        "lufs_before": round(lufs_before, 1) if np.isfinite(lufs_before) else None,
        "lufs_after":  round(lufs_after, 1) if np.isfinite(lufs_after) else None,
    }
