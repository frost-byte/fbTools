"""Pure logic for locating a deliberately-inserted marker-color frame run inside a video frame
batch, and splitting it into the "before" and "after" clip segments.

No ComfyUI dependencies (see docs/GOTCHAS.md convention) — operates on a plain numpy array, not
ComfyUI's torch IMAGE tensor; the node wrapper converts. Built for a clip-bridging pipeline: a hard
cut between two real clips is usually easy for an ML scene-detector to flag, but it's still a
probabilistic call on where exactly the boundary sits. A deliberately inserted marker-color segment
(e.g. solid magenta, chosen to never occur in real footage) turns "where's the real boundary" into a
deterministic per-frame color-distance threshold instead.
"""
from __future__ import annotations

import numpy as np


def find_marker_run(
    frames: np.ndarray,
    marker_color: tuple[float, float, float],
    tolerance: float,
    min_marker_frames: int = 1,
) -> dict | None:
    """Find the longest run of >= min_marker_frames consecutive frames whose mean color is within
    `tolerance` (Euclidean distance in normalized [0, 1] RGB) of marker_color.

    `frames` is either raw frames [N, H, W, C] or already-reduced per-frame mean colors [N, C] —
    both accepted, since the caller may prefer to do the reduction itself. Values are expected in
    [0, 1] to match ComfyUI's own IMAGE tensor convention.

    The *longest* qualifying run is picked (not just the first) so an incidental few-frame partial
    match elsewhere in the footage doesn't get mistaken for the real, deliberately-inserted marker
    segment.

    Returns None if no qualifying run exists. Otherwise a dict:
      marker_start_idx / marker_end_idx — inclusive bounds of the marker run
      marker_frame_count               — length of that run
      clip_a_end_idx                   — last real clip-A frame index, -1 if the marker starts at
                                          frame 0 (no clip-A frames at all)
      clip_b_start_idx                 — first real clip-B frame index, len(frames) if the marker
                                          runs to the last frame (no clip-B frames at all)
    """
    if frames.ndim not in (2, 4):
        raise ValueError(f"frames must be [N,H,W,C] or [N,C], got shape {frames.shape}")
    if frames.shape[0] == 0:
        return None

    if frames.ndim == 4:
        mean_colors = frames.reshape(frames.shape[0], -1, frames.shape[-1]).mean(axis=1)
    else:
        mean_colors = frames

    marker = np.asarray(marker_color, dtype=mean_colors.dtype)
    distances = np.linalg.norm(mean_colors - marker, axis=1)
    is_marker = distances <= tolerance

    def _run_length(run):
        return run[1] - run[0] + 1

    run_start = None
    best_run = None
    for i, flag in enumerate(is_marker):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            candidate = (run_start, i - 1)
            if _run_length(candidate) >= min_marker_frames and (
                best_run is None or _run_length(candidate) > _run_length(best_run)
            ):
                best_run = candidate
            run_start = None
    if run_start is not None:
        candidate = (run_start, len(is_marker) - 1)
        if _run_length(candidate) >= min_marker_frames and (
            best_run is None or _run_length(candidate) > _run_length(best_run)
        ):
            best_run = candidate

    if best_run is None:
        return None

    marker_start_idx, marker_end_idx = best_run
    return {
        "marker_start_idx": marker_start_idx,
        "marker_end_idx": marker_end_idx,
        "marker_frame_count": marker_end_idx - marker_start_idx + 1,
        "clip_a_end_idx": marker_start_idx - 1,
        "clip_b_start_idx": marker_end_idx + 1,
    }


def hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    """'#FF00FF' -> (1.0, 0.0, 1.0). Raises ValueError on anything not a 6-digit hex string."""
    stripped = (hex_str or "").strip().lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"marker_color must be a 6-digit hex string like '#FF00FF', got {hex_str!r}")
    try:
        r, g, b = (int(stripped[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"marker_color must be a 6-digit hex string like '#FF00FF', got {hex_str!r}")
    return (r / 255.0, g / 255.0, b / 255.0)
