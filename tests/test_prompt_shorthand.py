"""Tests for the S1/V1/A1 prompt shorthand expansion."""

from conftest import import_test_module

ps = import_test_module("utils/prompt_shorthand.py")
expand_prompt_shorthand = ps.expand_prompt_shorthand


def test_expands_subject_video_audio_tokens():
    text = "S1 is Luke, visible in V1. A1 is his voice reference."
    result = expand_prompt_shorthand(text)
    assert result == "<Subject 1> is Luke, visible in <Video 1>. <Audio 1> is his voice reference."


def test_expands_multi_digit_numbers():
    assert expand_prompt_shorthand("S12 and V10") == "<Subject 12> and <Video 10>"


def test_case_insensitive():
    assert expand_prompt_shorthand("s1 is here, v2 is there") == "<Subject 1> is here, <Video 2> is there"


def test_does_not_match_inside_larger_words():
    # "Subject" and "Visible" must not be mistaken for S<digits>/V<digits> tokens
    # since there's no digit run immediately after the S/V.
    text = "Subject matter is Visible in the scene."
    assert expand_prompt_shorthand(text) == text


def test_does_not_partially_match_leading_token():
    # xS1 / S1x must not expand -- the shorthand token must be its own word.
    assert expand_prompt_shorthand("xS1 and S1x") == "xS1 and S1x"


def test_already_expanded_text_is_a_safe_no_op():
    text = "<Subject 1> is visible in <Video 1>, with <Audio 1> as reference."
    assert expand_prompt_shorthand(text) == text


def test_empty_and_none_input():
    assert expand_prompt_shorthand("") == ""
    assert expand_prompt_shorthand(None) == ""


def test_multiple_subjects_and_videos_in_one_pass():
    text = "S1, S2 in V1; S3, S4, S5 in V2."
    expected = "<Subject 1>, <Subject 2> in <Video 1>; <Subject 3>, <Subject 4>, <Subject 5> in <Video 2>."
    assert expand_prompt_shorthand(text) == expected
