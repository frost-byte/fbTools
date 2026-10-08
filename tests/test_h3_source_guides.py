"""Tests for utils/h3_source_guides.py (guide placement for H3 Source Guides)."""
import pytest

from conftest import import_test_module

sg = import_test_module("utils/h3_source_guides.py")
plan_guides = sg.plan_guides


def _idx(plan):
    return [g["frame_idx"] for g in plan]


class TestValidGuideFrames:
    @pytest.mark.parametrize("n", [1, 5, 22, 39])
    def test_valid(self, n):
        assert sg.valid_guide_frames(n)

    @pytest.mark.parametrize("n", [0, 2, 4, 6, 21, 23])
    def test_invalid(self, n):
        assert not sg.valid_guide_frames(n)


class TestPlanGuides:
    def test_every_interval_realtime(self):
        plan = plan_guides(frame_count=124, source_frames=124, interval=12, include_last=False)
        assert _idx(plan) == list(range(0, 124, 12))
        assert all(g["src_start"] == g["frame_idx"] for g in plan)

    def test_include_last_adds_final_frame_once(self):
        plan = plan_guides(frame_count=124, source_frames=124, interval=12, include_last=True)
        assert _idx(plan)[-1] == 123
        assert len(set(_idx(plan))) == len(plan)

    def test_include_last_not_duplicated_when_on_grid(self):
        plan = plan_guides(frame_count=121, source_frames=121, interval=12, include_last=True)
        assert _idx(plan).count(120) == 1

    def test_guides_past_the_source_are_dropped_in_realtime(self):
        plan = plan_guides(frame_count=240, source_frames=100, interval=12, include_last=False)
        assert max(_idx(plan)) <= 99

    def test_include_last_uses_end_of_source_in_realtime(self):
        plan = plan_guides(frame_count=240, source_frames=100, interval=12, include_last=True)
        assert _idx(plan)[-1] == 99
        assert plan[-1]["src_start"] == 99

    def test_stretch_spreads_source_over_output(self):
        plan = plan_guides(frame_count=241, source_frames=121, interval=60, include_last=True, stretch=True)
        assert _idx(plan) == [0, 60, 120, 180, 240]
        assert [g["src_start"] for g in plan] == [0, 30, 60, 90, 120]

    def test_clip_guides_stay_inside_the_video(self):
        plan = plan_guides(frame_count=124, source_frames=124, interval=24, guide_frames=22)
        assert all(g["frame_idx"] + 22 <= 124 for g in plan)
        assert all(g["src_start"] + 22 <= 124 for g in plan)

    def test_max_guides_thins_evenly_keeping_ends(self):
        plan = plan_guides(frame_count=241, source_frames=241, interval=1, max_guides=5)
        assert len(plan) == 5
        assert _idx(plan)[0] == 0 and _idx(plan)[-1] == 240

    def test_max_guides_one_keeps_first(self):
        plan = plan_guides(frame_count=124, source_frames=124, interval=12, max_guides=1)
        assert _idx(plan) == [0]

    def test_no_source_or_video_shorter_than_guide(self):
        assert plan_guides(frame_count=10, source_frames=10, interval=12, guide_frames=22) == []
        assert plan_guides(frame_count=124, source_frames=3, interval=12, guide_frames=5) == []

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            plan_guides(124, 124, interval=0)
        with pytest.raises(ValueError):
            plan_guides(124, 124, interval=12, guide_frames=4)
        with pytest.raises(ValueError):
            plan_guides(124, 124, interval=12, max_guides=0)


def test_describe_plan():
    plan = plan_guides(frame_count=124, source_frames=124, interval=60, include_last=False)
    text = sg.describe_plan(plan, 124, 1)
    assert "3 guide(s)" in text
    assert "frame   60" in text
    assert sg.describe_plan([], 124, 1) == "H3 Source Guides: no guides added."
