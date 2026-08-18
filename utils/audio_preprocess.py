"""Pure audio preprocessing utilities — no ComfyUI dependencies.

Pipeline (each step is optional):
  1. Vocal extraction via MelBand Roformer (preferred) or spectral denoising fallback
  2. LUFS normalization   (pyloudnorm, ITU BS.1770)
  3. Loop to minimum duration (2 s default)
  4. Truncate to maximum duration (15 s default)

MelBand Roformer vocal extraction requires the ComfyUI-MelBandRoFormer custom node and a
safetensors checkpoint (Kijai/MelBandRoFormer_comfy on HuggingFace).  Set the path in
Settings → MelBand model path.  When not configured, the old scipy spectral subtraction
is used as a fallback (lower quality).
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


def _find_melband_class():
    """Locate MelBandRoformer class from the ComfyUI-MelBandRoFormer node pack."""
    import sys
    import os

    # Already importable (e.g. installed as a package)?
    try:
        from model.mel_band_roformer import MelBandRoformer  # noqa: PLC0415
        return MelBandRoformer
    except ImportError:
        pass

    # Find it relative to this utils/ dir: ../../ComfyUI-MelBandRoFormer/
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(os.path.dirname(utils_dir))
    for name in ("ComfyUI-MelBandRoFormer", "ComfyUI_MelBandRoFormer"):
        mel_dir = os.path.join(custom_nodes_dir, name)
        if os.path.isdir(os.path.join(mel_dir, "model")):
            if mel_dir not in sys.path:
                sys.path.insert(0, mel_dir)
            try:
                from model.mel_band_roformer import MelBandRoformer  # noqa: PLC0415
                return MelBandRoformer
            except ImportError:
                continue
    return None


_melband_cache: dict = {}  # model_path → loaded model


def melband_vocal_extract(waveform, sample_rate: int, model_path: str):
    """Extract vocals from *waveform* using MelBand Roformer source separation.

    waveform: [1, C, L] torch float32 (as returned by _h3_load_audio)
    Returns:  [1, C, L] torch float32 — vocal stem only, resampled back to sample_rate.
    """
    import torch
    import torch.nn.functional as F
    import torchaudio.functional as TAF

    MelBandRoformer = _find_melband_class()
    if MelBandRoformer is None:
        raise RuntimeError(
            "MelBandRoformer class not found. "
            "Install the ComfyUI-MelBandRoFormer custom node pack."
        )

    global _melband_cache
    if model_path not in _melband_cache:
        from safetensors.torch import load_file
        cfg = {
            "dim": 384, "depth": 6, "stereo": True, "num_stems": 1,
            "time_transformer_depth": 1, "freq_transformer_depth": 1,
            "num_bands": 60, "dim_head": 64, "heads": 8,
            "attn_dropout": 0.0, "ff_dropout": 0.0, "flash_attn": True,
            "dim_freqs_in": 1025, "sample_rate": 44100,
            "stft_n_fft": 2048, "stft_hop_length": 441, "stft_win_length": 2048,
            "stft_normalized": False, "mask_estimator_depth": 2,
            "multi_stft_resolution_loss_weight": 1.0,
            "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
            "multi_stft_hop_size": 147, "multi_stft_normalized": False,
        }
        model = MelBandRoformer(**cfg).eval()
        model.load_state_dict(load_file(model_path), strict=True)
        _melband_cache[model_path] = model

    model = _melband_cache[model_path]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    TARGET_SR = 44100
    audio = waveform  # [1, C, L]
    orig_channels = audio.shape[1]
    if orig_channels == 1:
        audio = audio.repeat(1, 2, 1)
    if sample_rate != TARGET_SR:
        audio = TAF.resample(audio, orig_freq=sample_rate, new_freq=TARGET_SR)

    audio_2d = audio[0]  # [2, L]
    audio_length = audio_2d.shape[1]

    C, N = 352800, 2
    step = C // N
    fade_size = C // 10
    border = C - step

    if audio_length > 2 * border and border > 0:
        audio_2d = F.pad(audio_2d, (border, border), mode="reflect")

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    win_base = torch.ones(C)
    win_base[:fade_size] *= fadein
    win_base[-fade_size:] *= fadeout

    audio_2d = audio_2d.to(device)
    vocals  = torch.zeros_like(audio_2d)
    counter = torch.zeros_like(audio_2d)
    total   = audio_2d.shape[1]

    with torch.inference_mode():
        for i in range(0, total, step):
            part = audio_2d[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                mode = "reflect" if length > C // 2 + 1 else "constant"
                part = F.pad(part, (0, C - length), mode=mode, value=0.0)
            x = model(part.unsqueeze(0))[0]
            win = win_base.clone().to(device)
            if i == 0:
                win[:fade_size] = 1.0
            if i + C >= total:
                win[-fade_size:] = 1.0
            vocals[..., i:i + length]  += x[..., :length] * win[..., :length]
            counter[..., i:i + length] += win[..., :length]

    vocals = (vocals / counter.clamp(min=1e-8))[..., :audio_length]

    if audio_length > 2 * border and border > 0:
        vocals = vocals[..., border:-border]

    vocals = vocals.unsqueeze(0).cpu()  # [1, 2, L]

    if sample_rate != TARGET_SR:
        vocals = TAF.resample(vocals, orig_freq=TARGET_SR, new_freq=sample_rate)

    # Restore to original channel count
    if orig_channels == 1:
        vocals = vocals.mean(dim=1, keepdim=True)

    return vocals


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
    melband_model_path: str | None = None,
) -> tuple:
    """Apply the audio preprocessing pipeline.

    waveform: [1, C, L] torch float32 (as returned by _h3_load_audio)
    melband_model_path: absolute path to a MelBandRoformer safetensors checkpoint.
        When set and noise_removal=True, MelBand vocal extraction is used instead
        of spectral subtraction.
    Returns: (waveform_out [1, C, L] torch float32, sample_rate, metrics dict)
    metrics: {duration, lufs_before, lufs_after}
    """
    import torch

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)
    lufs_before = measure_lufs(audio_np, sample_rate)

    if noise_removal:
        if melband_model_path:
            waveform = melband_vocal_extract(waveform, sample_rate, melband_model_path)
            audio_np = waveform.squeeze(0).numpy().astype(np.float32)
        else:
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
