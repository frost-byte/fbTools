"""Pure planning for H3 Source Guides -- which source frames to anchor at which output frames.

MiniMaxH3AddGuide places an image (or a short clip) at a chosen frame of the generated video, on
the target's own time axis. This module decides *where*: given the generated length and the
number of source frames available, it returns the list of guides to add. No ComfyUI imports.
"""
from __future__ import annotations

_MIN_CLIP_GUIDE = 5  # AddGuide treats batches below 5 frames as a single image


def valid_guide_frames(n: int) -> bool:
    """1 (single image) or a valid H3 clip length: 5, 22, 39, ... (17k + 5)."""
    return n == 1 or (n >= _MIN_CLIP_GUIDE and n % 17 == 5)


def plan_guides(
    frame_count: int,
    source_frames: int,
    interval: int,
    guide_frames: int = 1,
    max_guides: int = 24,
    include_last: bool = True,
    stretch: bool = False,
) -> list[dict]:
    """Return ``[{"frame_idx", "src_start"}]`` -- guides to add, in output-frame order.

    frame_count   frames in the generated video (from the latent)
    source_frames real frames available in the source clip at 24 fps
    interval      output frames between guides
    guide_frames  frames per guide: 1 (a single image) or a clip length 5 / 22 / 39 ...
    max_guides    cap; when exceeded the guides are thinned evenly (first and last kept)
    include_last  also anchor the end of the clip
    stretch       False: output frame f takes source frame f (real time), and guides past the
                  end of the source are dropped. True: the source is spread over the whole output.
    """
    if not valid_guide_frames(guide_frames):
        raise ValueError(f"guide_frames must be 1 or 17k+5 (5, 22, 39, ...), got {guide_frames}")
    if interval < 1:
        raise ValueError(f"interval must be at least 1 frame, got {interval}")
    if max_guides < 1:
        raise ValueError(f"max_guides must be at least 1, got {max_guides}")

    last_out = frame_count - guide_frames          # last output frame a guide may start at
    last_src = source_frames - guide_frames        # last source frame a guide may start at
    if last_out < 0 or last_src < 0:
        return []

    def src_for(frame_idx: int) -> int:
        if stretch:
            return round(frame_idx * last_src / last_out) if last_out > 0 else 0
        return frame_idx

    positions = list(range(0, last_out + 1, interval))
    if include_last:
        tail = last_out if stretch else min(last_out, last_src)
        if tail not in positions:
            positions.append(tail)
    positions = [p for p in positions if src_for(p) <= last_src]
    if not positions:
        return []

    if len(positions) > max_guides:
        if max_guides == 1:
            positions = positions[:1]
        else:
            step = (len(positions) - 1) / (max_guides - 1)
            positions = sorted({positions[round(i * step)] for i in range(max_guides)})

    return [{"frame_idx": p, "src_start": src_for(p)} for p in positions]


def describe_plan(plan: list[dict], frame_count: int, guide_frames: int, fps: int = 24) -> str:
    """One-line-per-guide summary for the log and status line."""
    if not plan:
        return "H3 Source Guides: no guides added."
    lines = [
        f"H3 Source Guides: {len(plan)} guide(s) of {guide_frames} frame(s) in a {frame_count}-frame video"
    ]
    for g in plan:
        lines.append(
            f"  frame {g['frame_idx']:>4} ({g['frame_idx'] / fps:5.2f}s)  <- source frame {g['src_start']}"
        )
    return "\n".join(lines)
