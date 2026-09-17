"""Pure VRAM / attention-memory estimation for H3 two-pass workflows.

No ComfyUI dependencies — extension.py wires this into
CompositionToH3Conditioning, which has width/height/length and the full
resolved reference list already.

This is a heuristic, not a physical simulation. H3's actual attention
implementation, any sparse-attention patches applied to the model, and
ComfyUI's own dynamic model loading all affect real usage in ways this
doesn't model precisely — it exists to give a directionally-useful
upscale recommendation for MinimaxH3LatentUpscaler3D's `scale` input, not
a guarantee. Revise the calibration constants below as more real
out-of-memory incidents are observed and logged (see docs/GOTCHAS.md).

Token math, established from H3's actual VAE + DiT compression:
  - VAE:  space_down=(2,2,2,2,1,1) -> 16x spatial; time_down=(1,2,2,1,1,1) -> 4x temporal.
  - DiT:  patch_size=(1,2,2)       -> additional 2x spatial.
  Combined: 32x spatial, 4x temporal, pixel -> token.

Calibration, from a real OOM incident (2026-09-16, canvas 832x640, 300 frames,
second sampler pass with 4 reference items: 1 image + 2 videos totalling 180
frames + 1 audio -> ~62,900 estimated tokens): the model requested a single
9.40 GiB attention allocation and failed, with the log reporting only 245.94
MiB free against a 23.56 GiB device limit — i.e. it failed at ~17.51 GiB of
nominal usage (8.11 GiB "currently allocated" + the 9.40 GiB request), well
under the 23.56 GiB device limit. That gap (~6 GiB) is real, unaccounted
overhead — fragmentation, the allocator's own reserved-but-unallocated
segments, --reserve-vram, etc. — that a naive "allocated + requested vs.
device total" comparison misses. A same-canvas run with only 1 reference
item (~39,000 tokens) completed successfully.

BASE_OVERHEAD_GIB below is calibrated against the *device limit*, not the
"currently allocated" figure, specifically so that comparing against raw
device memory (e.g. torch.cuda.get_device_properties().total_memory) already
bakes in that real-world gap: device_limit - attention_requested_at_failure
= 23.56 - 9.40 = 14.16 GiB. This is a single-incident calibration — expect to
refine both constants as more real out-of-memory incidents accumulate.
"""
from __future__ import annotations

PIXELS_PER_SPATIAL_TOKEN = 1024   # 32 * 32
FRAMES_PER_TEMPORAL_TOKEN = 4

# k in: attention_gib ~= k * total_tokens^2
ATTENTION_GIB_PER_TOKEN_SQUARED = 9.40 / (62_900 ** 2)

# Everything else that eats into the *device's total* memory by the time the
# second-pass attention allocation happens: weights, text encoder, VAEs,
# prior latents, allocator fragmentation, reserve-vram, driver overhead.
# Calibrated against device-limit, not "currently allocated" — see module
# docstring for why that ~6 GiB gap matters.
BASE_OVERHEAD_GIB = 23.56 - 9.40


def tokens_for(width: int, height: int, frames: int) -> float:
    """Approximate H3 token count for a WxH, `frames`-long clip."""
    latent_frames = -(-max(int(frames), 1) // FRAMES_PER_TEMPORAL_TOKEN)  # ceil div
    return (max(width, 0) * max(height, 0) / PIXELS_PER_SPATIAL_TOKEN) * latent_frames


def estimate_attention_gib(total_tokens: float) -> float:
    """Estimate the peak attention-allocation size for a packed sequence this long."""
    return ATTENTION_GIB_PER_TOKEN_SQUARED * (total_tokens ** 2)


def max_safe_scale(
    main_tokens: float,
    reference_tokens: float,
    budget_gib: float,
    safety_buffer: float = 0.85,
    min_scale: float = 1.0,
    max_scale: float = 4.0,
) -> tuple[float, bool]:
    """Return (recommended_scale, at_risk) for MinimaxH3LatentUpscaler3D's `scale`.

    `main_tokens` is the pass-1 main-video token count — this is what scales
    as scale^2 on upscale, since the upscaler holds frame count fixed and
    only scales width/height. `reference_tokens` stays fixed across passes
    (the same conditioning/references are reused). `budget_gib` is the total
    VRAM to plan against (caller subtracts its own margin before calling, or
    passes raw device memory).

    `at_risk` is True when even `min_scale` is estimated to exceed the
    budget — the recommendation still never drops below `min_scale` (that's
    not a valid upscale anyway), but the caller should log a warning in that
    case rather than silently proceeding.
    """
    available_for_attention = max(0.0, (budget_gib - BASE_OVERHEAD_GIB) * safety_buffer)
    if ATTENTION_GIB_PER_TOKEN_SQUARED <= 0 or main_tokens <= 0:
        return min_scale, False

    max_total_tokens = (available_for_attention / ATTENTION_GIB_PER_TOKEN_SQUARED) ** 0.5
    at_risk = max_total_tokens < (main_tokens + reference_tokens)

    if max_total_tokens <= reference_tokens:
        raw_scale = 0.0
    else:
        raw_scale = ((max_total_tokens - reference_tokens) / main_tokens) ** 0.5

    scale = max(min_scale, min(max_scale, raw_scale))
    return scale, at_risk
