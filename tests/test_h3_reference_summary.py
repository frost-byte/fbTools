"""Tests for the H3 reference summary logic (pure -- no ComfyUI/torch)."""

from conftest import import_test_module

hrs = import_test_module("utils/h3_reference_summary.py")
describe_references = hrs.describe_references
valid_clip_frames = hrs.valid_clip_frames


def _vid(name, frames, w=832, h=480):
    return {"name": name, "frames": frames, "width": w, "height": h}


def _aud(seconds, rate=44100, channels=2, name=""):
    return {"name": name, "seconds": seconds, "sample_rate": rate, "channels": channels}


# ── valid_clip_frames ─────────────────────────────────────────────────────────

def test_valid_clip_frames_keeps_already_valid_lengths():
    for n in (5, 22, 39, 56, 73, 90, 107, 124, 175, 192):
        assert valid_clip_frames(n) == n


def test_valid_clip_frames_rounds_down_to_17k_plus_5():
    assert valid_clip_frames(96) == 90
    assert valid_clip_frames(64) == 56
    assert valid_clip_frames(48) == 39
    assert valid_clip_frames(191) == 175
    assert valid_clip_frames(200) == 192


def test_valid_clip_frames_below_minimum_is_zero():
    assert valid_clip_frames(4) == 0
    assert valid_clip_frames(0) == 0


# ── tag numbering (matches MiniMaxH3ReferenceToVideo) ─────────────────────────

def test_soundtrack_gets_its_audio_label_before_its_video():
    text, warnings = describe_references(
        videos=[_vid("ref_video_0", 90), _vid("ref_video_1", 90)],
        video_audios={"0": _aud(3.75), "1": _aud(3.75)},
    )
    lines = text.splitlines()
    order = [l.strip().split()[0] for l in lines[1:]]
    assert order == ["<Audio", "<Video", "<Audio", "<Video"]
    assert "<Audio 1>" in lines[1] and "soundtrack of ref_video_0" in lines[1]
    assert "<Video 1>" in lines[2]
    assert "<Audio 2>" in lines[3] and "<Video 2>" in lines[4]
    assert warnings == []


def test_pictures_come_first_and_standalone_audio_continues_the_audio_count():
    text, _ = describe_references(
        images=[{"name": "ref_image_0", "width": 1024, "height": 768}],
        videos=[_vid("ref_video_0", 90)],
        video_audios={"0": _aud(3.75)},
        audios=[_aud(5.0, name="ref_audio_0")],
    )
    lines = [l.strip() for l in text.splitlines()[1:]]
    assert lines[0].startswith("<Picture 1>")
    assert lines[1].startswith("<Audio 1>")    # the video's soundtrack
    assert lines[2].startswith("<Video 1>")
    assert lines[3].startswith("<Audio 2>")    # standalone audio continues the count
    assert text.splitlines()[0] == "H3 references: 1 picture(s), 1 video(s), 2 audio track(s)"


# ── trimming and warnings ─────────────────────────────────────────────────────

def test_reports_frames_used_after_the_17k_plus_5_rule_without_warning():
    # Routine rounding (96 -> 90) is information on the line, not a warning.
    text, warnings = describe_references(videos=[_vid("ref_video_0", 96)])
    assert "90 of 96 frames used" in text
    assert warnings == []


def test_exact_valid_length_has_no_trim_warning():
    _, warnings = describe_references(videos=[_vid("ref_video_0", 90)])
    assert warnings == []


def test_video_longer_than_generation_length_is_truncated_and_flagged():
    text, warnings = describe_references(videos=[_vid("ref_video_0", 192)], length=124)
    assert any("exceeds the generation length (124)" in w for w in warnings)
    assert "<Video 1>" in text and "124 of 192 frames used" in text   # 124 is itself a valid length


def test_too_short_video_is_reported_as_an_error_case():
    text, warnings = describe_references(videos=[_vid("ref_video_0", 4)])
    assert "TOO SHORT" in text
    assert any("at least 5" in w for w in warnings)


def test_duration_outside_recommended_range_is_flagged():
    _, short = describe_references(videos=[_vid("ref_video_0", 22)])      # 0.92s
    _, long_ = describe_references(videos=[_vid("ref_video_0", 175)])     # 7.3s -- fine
    _, very_long = describe_references(videos=[_vid("ref_video_0", 379)])  # 15.8s
    assert any("outside the recommended" in w for w in short)
    assert long_ == []
    assert any("outside the recommended" in w for w in very_long)


def test_video_without_soundtrack_is_marked():
    text, _ = describe_references(videos=[_vid("ref_video_0", 90)])
    assert "[no soundtrack]" in text


def test_orphan_soundtrack_is_warned_about():
    _, warnings = describe_references(videos=[_vid("ref_video_0", 90)], video_audios={"2": _aud(3.0)})
    assert any("ref_video_audio_2 has no matching ref_video_2" in w for w in warnings)


def test_nothing_connected_says_none():
    text, warnings = describe_references()
    assert text == "H3 references: none"
    assert warnings == []


def test_warnings_are_appended_to_the_text():
    text, warnings = describe_references(videos=[_vid("ref_video_0", 4)])
    assert warnings
    assert "Warnings:" in text
    assert all(w in text for w in warnings)
