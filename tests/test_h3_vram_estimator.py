"""Tests for utils/h3_vram_estimator.py — pure token/VRAM heuristic math."""
from conftest import import_test_module

est = import_test_module("utils/h3_vram_estimator.py")


def test_tokens_for_matches_calibration_incident_main_video():
    # canvas 832x640, 300 frames -> ~39,000 tokens (the calibration anchor).
    tokens = est.tokens_for(832, 640, 300)
    assert 38_000 < tokens < 40_000


def test_tokens_for_ceils_partial_temporal_groups():
    # 5 frames -> ceil(5/4) = 2 latent frames, not 1.
    assert est.tokens_for(32, 32, 5) == est.tokens_for(32, 32, 8)


def test_tokens_for_single_frame_image():
    tokens = est.tokens_for(1024, 1024, 1)
    assert tokens == (1024 * 1024 / est.PIXELS_PER_SPATIAL_TOKEN) * 1


def test_tokens_for_zero_or_negative_dims_is_zero():
    assert est.tokens_for(0, 640, 300) == 0
    assert est.tokens_for(832, 0, 300) == 0


def test_estimate_attention_gib_matches_calibration_incident():
    # The incident's own total token count should reproduce ~9.40 GiB.
    gib = est.estimate_attention_gib(62_900)
    assert 9.3 < gib < 9.5


def test_max_safe_scale_returns_min_scale_when_tight_budget():
    # Budget barely above base overhead — no room to upscale at all.
    scale, at_risk = est.max_safe_scale(
        main_tokens=39_000, reference_tokens=0, budget_gib=est.BASE_OVERHEAD_GIB + 0.1,
    )
    assert scale == 1.0


def test_max_safe_scale_recommends_above_one_with_headroom():
    # Large budget, single small reference load (like the successful 1-ref run).
    scale, at_risk = est.max_safe_scale(
        main_tokens=39_000, reference_tokens=1_500, budget_gib=23.5,
    )
    assert scale > 1.0
    assert at_risk is False


def test_max_safe_scale_flags_at_risk_when_pass_one_itself_is_tight():
    # Reproduce the calibration incident's own token count and device limit —
    # with no safety buffer this sits exactly at the boundary; any buffer
    # below 1.0 should push it into at_risk with a scale=1.0 fallback.
    scale, at_risk = est.max_safe_scale(
        main_tokens=39_000, reference_tokens=23_900, budget_gib=23.56, safety_buffer=0.95,
    )
    assert scale == 1.0
    assert at_risk is True


def test_max_safe_scale_never_exceeds_max_scale():
    scale, _ = est.max_safe_scale(
        main_tokens=100, reference_tokens=0, budget_gib=1000.0, max_scale=4.0,
    )
    assert scale == 4.0


def test_max_safe_scale_safety_buffer_reduces_recommendation():
    loose, _ = est.max_safe_scale(main_tokens=39_000, reference_tokens=1_500, budget_gib=23.5, safety_buffer=1.0)
    tight, _ = est.max_safe_scale(main_tokens=39_000, reference_tokens=1_500, budget_gib=23.5, safety_buffer=0.5)
    assert tight < loose
