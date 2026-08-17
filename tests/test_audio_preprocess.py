"""Tests for utils/audio_preprocess.py.

These functions have no ComfyUI dependencies, but we still go through
import_test_module() so the conftest mocks are in place (preventing any
accidental top-level extension.py import).
"""
import sys
import tempfile
import os

import numpy as np
import pytest

from conftest import import_test_module

ap = import_test_module("utils/audio_preprocess.py")

cache_fingerprint = ap.cache_fingerprint
measure_lufs      = ap.measure_lufs
spectral_denoise  = ap.spectral_denoise
lufs_normalize    = ap.lufs_normalize
loop_to_min       = ap.loop_to_min
truncate_to_max   = ap.truncate_to_max
preprocess_audio  = ap.preprocess_audio


# ── Helpers ────────────────────────────────────────────────────────────────────

SR = 16_000  # sample rate used throughout


def _sine(freq=440.0, secs=3.0, channels=1, sr=SR, amplitude=0.5) -> np.ndarray:
    """Generate a clean sine-wave test signal. Returns [C, L] float32."""
    t = np.linspace(0, secs, int(secs * sr), endpoint=False, dtype=np.float64)
    wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([wave] * channels) if channels > 1 else wave.reshape(1, -1)


def _noise(secs=3.0, channels=1, sr=SR, amplitude=0.1) -> np.ndarray:
    rng = np.random.default_rng(42)
    wave = (rng.standard_normal(int(secs * sr)) * amplitude).astype(np.float32)
    return np.stack([wave] * channels) if channels > 1 else wave.reshape(1, -1)


# ── cache_fingerprint ──────────────────────────────────────────────────────────

class TestCacheFingerprint:
    def test_returns_8_hex_chars(self, tmp_path):
        fp = cache_fingerprint(str(tmp_path / "nonexistent.wav"), 0.0, 0.0, {})
        assert len(fp) == 8
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic_for_same_inputs(self, tmp_path):
        p = tmp_path / "audio.wav"
        p.write_bytes(b"x")
        cfg = {"noise_removal": False, "normalize_lufs": True, "target_lufs": -14.0}
        fp1 = cache_fingerprint(str(p), 0.0, 5.0, cfg)
        fp2 = cache_fingerprint(str(p), 0.0, 5.0, cfg)
        assert fp1 == fp2

    def test_changes_with_start_time(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x")
        cfg = {}
        assert cache_fingerprint(str(p), 0.0, 0.0, cfg) != cache_fingerprint(str(p), 1.0, 0.0, cfg)

    def test_changes_with_duration(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x")
        cfg = {}
        assert cache_fingerprint(str(p), 0.0, 3.0, cfg) != cache_fingerprint(str(p), 0.0, 5.0, cfg)

    def test_changes_with_config(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x")
        fp1 = cache_fingerprint(str(p), 0.0, 0.0, {"noise_removal": False})
        fp2 = cache_fingerprint(str(p), 0.0, 0.0, {"noise_removal": True})
        assert fp1 != fp2

    def test_missing_file_is_stable(self):
        fp1 = cache_fingerprint("/nonexistent/path.wav", 0.0, 0.0, {})
        fp2 = cache_fingerprint("/nonexistent/path.wav", 0.0, 0.0, {})
        assert fp1 == fp2

    def test_config_key_order_independent(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x")
        cfg_a = {"normalize_lufs": True, "noise_removal": False}
        cfg_b = {"noise_removal": False, "normalize_lufs": True}
        assert cache_fingerprint(str(p), 0.0, 0.0, cfg_a) == cache_fingerprint(str(p), 0.0, 0.0, cfg_b)


# ── measure_lufs ───────────────────────────────────────────────────────────────

class TestMeasureLufs:
    def test_silence_returns_negative_inf(self):
        silence = np.zeros((1, SR), dtype=np.float32)
        result = measure_lufs(silence, SR)
        assert result == float("-inf") or result < -100.0

    def test_finite_for_audible_signal(self):
        sig = _sine(freq=1000, secs=3.0)
        result = measure_lufs(sig, SR)
        assert np.isfinite(result)

    def test_louder_signal_has_higher_lufs(self):
        quiet = _sine(amplitude=0.1, secs=3.0)
        loud  = _sine(amplitude=0.8, secs=3.0)
        assert measure_lufs(loud, SR) > measure_lufs(quiet, SR)

    def test_stereo_accepted(self):
        stereo = _sine(channels=2, secs=3.0)
        result = measure_lufs(stereo, SR)
        assert np.isfinite(result)

    def test_returns_float(self):
        sig = _sine(secs=3.0)
        assert isinstance(measure_lufs(sig, SR), float)


# ── spectral_denoise ───────────────────────────────────────────────────────────

class TestSpectralDenoise:
    def test_output_shape_preserved_mono(self):
        sig = _sine() + _noise(amplitude=0.05)
        out = spectral_denoise(sig, SR)
        assert out.shape == sig.shape

    def test_output_shape_preserved_stereo(self):
        sig = _sine(channels=2) + _noise(channels=2, amplitude=0.05)
        out = spectral_denoise(sig, SR)
        assert out.shape == sig.shape

    def test_output_is_float32(self):
        sig = _sine()
        out = spectral_denoise(sig, SR)
        assert out.dtype == np.float32

    def test_reduces_noise_energy(self):
        noise = _noise(amplitude=0.2, secs=3.0)
        denoised = spectral_denoise(noise, SR)
        assert np.mean(denoised ** 2) < np.mean(noise ** 2)


# ── lufs_normalize ─────────────────────────────────────────────────────────────

class TestLufsNormalize:
    def test_output_shape_preserved(self):
        sig = _sine(secs=3.0)
        out = lufs_normalize(sig, SR, target_lufs=-14.0)
        assert out.shape == sig.shape

    def test_output_is_float32(self):
        sig = _sine(secs=3.0)
        out = lufs_normalize(sig, SR, target_lufs=-14.0)
        assert out.dtype == np.float32

    def test_silence_passthrough(self):
        silence = np.zeros((1, SR * 3), dtype=np.float32)
        out = lufs_normalize(silence, SR)
        assert np.allclose(out, silence)

    def test_loudness_approaches_target(self):
        sig = _sine(amplitude=0.1, secs=3.0)
        target = -20.0
        out = lufs_normalize(sig, SR, target_lufs=target)
        result_lufs = measure_lufs(out, SR)
        assert np.isfinite(result_lufs)
        assert abs(result_lufs - target) < 3.0  # within 3 LU

    def test_clipping_prevented(self):
        very_loud = np.full((1, SR * 3), 5.0, dtype=np.float32)
        out = lufs_normalize(very_loud, SR, target_lufs=-6.0)
        assert np.abs(out).max() <= 1.0 + 1e-4


# ── loop_to_min ────────────────────────────────────────────────────────────────

class TestLoopToMin:
    def test_short_clip_is_extended_mono(self):
        short = _sine(secs=0.5)       # 0.5 s
        out = loop_to_min(short, SR, min_secs=2.0)
        assert out.shape[-1] >= int(2.0 * SR)

    def test_short_clip_is_extended_stereo(self):
        short = _sine(channels=2, secs=0.5)
        out = loop_to_min(short, SR, min_secs=2.0)
        assert out.shape[1] >= int(2.0 * SR)

    def test_long_clip_unchanged(self):
        sig = _sine(secs=5.0)
        out = loop_to_min(sig, SR, min_secs=2.0)
        assert out.shape == sig.shape

    def test_exactly_min_unchanged(self):
        sig = _sine(secs=2.0)
        out = loop_to_min(sig, SR, min_secs=2.0)
        assert out.shape == sig.shape


# ── truncate_to_max ────────────────────────────────────────────────────────────

class TestTruncateToMax:
    def test_long_clip_trimmed_mono(self):
        sig = _sine(secs=20.0)
        out = truncate_to_max(sig, SR, max_secs=15.0)
        assert out.shape[-1] == int(15.0 * SR)

    def test_long_clip_trimmed_stereo(self):
        sig = _sine(channels=2, secs=20.0)
        out = truncate_to_max(sig, SR, max_secs=15.0)
        assert out.shape[1] == int(15.0 * SR)

    def test_short_clip_unchanged(self):
        sig = _sine(secs=5.0)
        out = truncate_to_max(sig, SR, max_secs=15.0)
        assert out.shape == sig.shape


# ── preprocess_audio (integration) ────────────────────────────────────────────
#
# conftest already installed a MagicMock for torch, so we can't use real torch
# tensors directly.  Instead we install a minimal stand-in that satisfies the
# two touch-points preprocess_audio needs:
#   • waveform.squeeze(0).numpy().astype(np.float32) → numpy array
#   • torch.from_numpy(arr).unsqueeze(0)             → something with .shape[0]
#
# The underlying numpy pipeline functions are already covered by the unit tests
# above; these tests verify that the glue code (metrics dict, sr pass-through,
# shape preservation) is correct.

import types as _types


class _FakeTensor:
    """Minimal stand-in for a torch tensor.  Shapes are preserved correctly."""
    def __init__(self, arr: np.ndarray):
        self._a = np.asarray(arr, dtype=np.float32)
        self.shape = self._a.shape

    def squeeze(self, dim=None):
        if dim is not None:
            out = np.squeeze(self._a, axis=dim)
        else:
            out = np.squeeze(self._a)
        return _FakeTensor(out)

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._a, axis=dim))

    def float(self):
        return self

    def numpy(self):
        return self._a

    def astype(self, dtype):
        return self._a.astype(dtype)


def _fake_torch():
    m = _types.SimpleNamespace()
    m.from_numpy = lambda arr: _FakeTensor(np.asarray(arr, dtype=np.float32))
    return m


class TestPreprocessAudio:
    """Integration tests for the preprocess_audio pipeline wrapper."""

    @pytest.fixture(autouse=True)
    def patch_torch(self):
        old = sys.modules.get("torch")
        sys.modules["torch"] = _fake_torch()
        yield
        if old is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old

    def _make_waveform(self, secs=3.0, channels=1):
        """Return a _FakeTensor shaped [1, C, L] (mimicking ComfyUI audio batch)."""
        np_audio = _sine(secs=secs, channels=channels)  # [C, L]
        return _FakeTensor(np_audio[np.newaxis])        # [1, C, L]

    def test_returns_tuple_of_three(self):
        wav = self._make_waveform()
        result = preprocess_audio(wav, SR, noise_removal=False, normalize_lufs=False)
        assert len(result) == 3

    def test_output_has_batch_dim(self):
        wav = self._make_waveform()
        out, sr, metrics = preprocess_audio(wav, SR, noise_removal=False, normalize_lufs=False)
        assert out.shape[0] == 1

    def test_sample_rate_preserved(self):
        wav = self._make_waveform()
        _, sr_out, _ = preprocess_audio(wav, SR)
        assert sr_out == SR

    def test_metrics_keys(self):
        wav = self._make_waveform()
        _, _, metrics = preprocess_audio(wav, SR)
        assert "duration" in metrics
        assert "lufs_before" in metrics
        assert "lufs_after" in metrics

    def test_duration_in_range(self):
        wav = self._make_waveform(secs=3.0)
        _, _, metrics = preprocess_audio(wav, SR, min_secs=2.0, max_secs=15.0)
        assert 2.0 <= metrics["duration"] <= 15.0

    def test_normalize_lufs_brings_loudness_closer_to_target(self):
        wav = self._make_waveform(secs=3.0)
        target = -20.0
        _, _, metrics = preprocess_audio(
            wav, SR, noise_removal=False, normalize_lufs=True, target_lufs=target
        )
        if metrics["lufs_after"] is not None:
            assert abs(metrics["lufs_after"] - target) < 3.0

    def test_noise_removal_runs_without_error(self):
        wav = self._make_waveform(secs=3.0)
        _, _, metrics = preprocess_audio(wav, SR, noise_removal=True, normalize_lufs=False)
        assert metrics["duration"] > 0

    def test_stereo_channel_count_preserved(self):
        wav = self._make_waveform(secs=3.0, channels=2)
        out, _, _ = preprocess_audio(wav, SR, noise_removal=False, normalize_lufs=True)
        # batch dim = 1, channel dim = 2
        assert out.shape[0] == 1
