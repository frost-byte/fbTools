"""Tests for utils/marker_frame_split.py — pure logic, no ComfyUI dependencies."""
import numpy as np
import pytest

from conftest import import_test_module

mfs = import_test_module("utils/marker_frame_split.py")
find_marker_run = mfs.find_marker_run
hex_to_rgb01 = mfs.hex_to_rgb01

MAGENTA = (1.0, 0.0, 1.0)


def _solid_frames(n, color, h=4, w=4):
    """[n, h, w, 3] batch where every pixel of every frame is exactly `color`."""
    return np.tile(np.array(color, dtype=np.float64), (n, h, w, 1))


def _clip(n, seed):
    """Deterministic pseudo-random "real footage" frames, far from MAGENTA."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.4, size=(n, 4, 4, 3))  # kept well below magenta's (1,0,1)


# ── find_marker_run: core behavior ──────────────────────────────────────────────

def test_marker_in_the_middle_splits_correctly():
    clip_a = _clip(5, seed=1)
    marker = _solid_frames(3, MAGENTA)
    clip_b = _clip(4, seed=2)
    frames = np.concatenate([clip_a, marker, clip_b], axis=0)

    result = find_marker_run(frames, MAGENTA, tolerance=0.05)
    assert result["marker_start_idx"] == 5
    assert result["marker_end_idx"] == 7
    assert result["marker_frame_count"] == 3
    assert result["clip_a_end_idx"] == 4
    assert result["clip_b_start_idx"] == 8


def test_marker_at_start_has_no_clip_a():
    marker = _solid_frames(2, MAGENTA)
    clip_b = _clip(3, seed=3)
    frames = np.concatenate([marker, clip_b], axis=0)

    result = find_marker_run(frames, MAGENTA, tolerance=0.05)
    assert result["clip_a_end_idx"] == -1
    assert result["clip_b_start_idx"] == 2


def test_marker_at_end_has_no_clip_b():
    clip_a = _clip(3, seed=4)
    marker = _solid_frames(2, MAGENTA)
    frames = np.concatenate([clip_a, marker], axis=0)

    result = find_marker_run(frames, MAGENTA, tolerance=0.05)
    assert result["clip_a_end_idx"] == 2
    assert result["clip_b_start_idx"] == 5  # == len(frames)


def test_no_marker_returns_none():
    frames = _clip(6, seed=5)
    assert find_marker_run(frames, MAGENTA, tolerance=0.05) is None


def test_empty_frames_returns_none():
    frames = np.zeros((0, 4, 4, 3))
    assert find_marker_run(frames, MAGENTA, tolerance=0.05) is None


def test_min_marker_frames_rejects_short_incidental_match():
    # A single stray frame that happens to match the marker color shouldn't count as "the" marker.
    clip_a = _clip(3, seed=6)
    stray = _solid_frames(1, MAGENTA)
    clip_b = _clip(3, seed=7)
    frames = np.concatenate([clip_a, stray, clip_b], axis=0)

    assert find_marker_run(frames, MAGENTA, tolerance=0.05, min_marker_frames=3) is None


def test_longest_run_wins_over_a_shorter_incidental_match():
    stray = _solid_frames(1, MAGENTA)
    clip_a = _clip(3, seed=8)
    real_marker = _solid_frames(4, MAGENTA)
    clip_b = _clip(3, seed=9)
    frames = np.concatenate([stray, clip_a, real_marker, clip_b], axis=0)

    result = find_marker_run(frames, MAGENTA, tolerance=0.05, min_marker_frames=1)
    assert result["marker_frame_count"] == 4
    assert result["marker_start_idx"] == 4  # the real_marker run, not the leading stray frame


def test_tolerance_excludes_frames_too_far_from_marker_color():
    clip_a = _clip(3, seed=10)
    near_miss = np.tile(np.array([0.8, 0.0, 0.8]), (2, 4, 4, 1))  # noticeably off from pure magenta
    clip_b = _clip(3, seed=11)
    frames = np.concatenate([clip_a, near_miss, clip_b], axis=0)

    assert find_marker_run(frames, MAGENTA, tolerance=0.05) is None


def test_tolerance_includes_close_but_not_exact_marker_color():
    clip_a = _clip(3, seed=12)
    close_enough = np.tile(np.array([0.97, 0.02, 0.97]), (2, 4, 4, 1))
    clip_b = _clip(3, seed=13)
    frames = np.concatenate([clip_a, close_enough, clip_b], axis=0)

    result = find_marker_run(frames, MAGENTA, tolerance=0.1)
    assert result is not None
    assert result["marker_frame_count"] == 2


def test_accepts_already_reduced_per_frame_colors():
    # [N, C] input (caller already reduced H,W) instead of [N, H, W, C].
    colors = np.array([[0.1, 0.1, 0.1], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.2, 0.2, 0.2]])
    result = find_marker_run(colors, MAGENTA, tolerance=0.05)
    assert result["marker_start_idx"] == 1
    assert result["marker_end_idx"] == 2
    assert result["clip_a_end_idx"] == 0
    assert result["clip_b_start_idx"] == 3


def test_invalid_ndim_raises():
    with pytest.raises(ValueError, match="must be"):
        find_marker_run(np.zeros((5,)), MAGENTA, tolerance=0.05)


# ── hex_to_rgb01 ─────────────────────────────────────────────────────────────────

def test_hex_to_rgb01_with_hash():
    assert hex_to_rgb01("#FF00FF") == (1.0, 0.0, 1.0)


def test_hex_to_rgb01_without_hash():
    assert hex_to_rgb01("00FF00") == (0.0, 1.0, 0.0)


def test_hex_to_rgb01_lowercase():
    assert hex_to_rgb01("#ffffff") == (1.0, 1.0, 1.0)


def test_hex_to_rgb01_wrong_length_raises():
    with pytest.raises(ValueError, match="6-digit hex"):
        hex_to_rgb01("#FFF")


def test_hex_to_rgb01_non_hex_raises():
    with pytest.raises(ValueError, match="6-digit hex"):
        hex_to_rgb01("#GGGGGG")


def test_hex_to_rgb01_empty_raises():
    with pytest.raises(ValueError, match="6-digit hex"):
        hex_to_rgb01("")
