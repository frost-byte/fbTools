"""Tests for utils/node_output_tracker.py — pure logic only.

The monkeypatching glue in extension.py (installing patches on tracked node
classes, the PromptServer on_prompt hook) is deliberately not exercised here;
it can't be meaningfully unit-tested without a real ComfyUI execution.py, and
its own try/except isolation is the safety net for that code, not test
coverage.
"""
from conftest import import_test_module

tracker = import_test_module("utils/node_output_tracker.py")


def test_get_track_label_extracts_label():
    assert tracker.get_track_label("[track: Video Shift]") == "Video Shift"


def test_get_track_label_strips_whitespace():
    assert tracker.get_track_label("[track:   Audio Shift  ]") == "Audio Shift"


def test_get_track_label_no_tag_returns_none():
    assert tracker.get_track_label("ManualSigmas") is None
    assert tracker.get_track_label("") is None
    assert tracker.get_track_label(None) is None


def test_extract_tracked_nodes_finds_tagged_nodes():
    prompt = {
        "12": {"class_type": "ManualSigmas", "_meta": {"title": "[track: Video Shift]"}, "inputs": {}},
        "13": {"class_type": "KSampler", "_meta": {"title": "KSampler"}, "inputs": {}},
        "14": {"class_type": "ManualSigmas", "_meta": {"title": "[track: Audio Shift]"}, "inputs": {}},
    }
    assert tracker.extract_tracked_nodes(prompt) == {
        "12": "Video Shift",
        "14": "Audio Shift",
    }


def test_extract_tracked_nodes_empty_when_none_tagged():
    prompt = {"12": {"class_type": "KSampler", "_meta": {"title": "KSampler"}, "inputs": {}}}
    assert tracker.extract_tracked_nodes(prompt) == {}


def test_extract_tracked_nodes_handles_missing_meta():
    prompt = {"12": {"class_type": "KSampler", "inputs": {}}}
    assert tracker.extract_tracked_nodes(prompt) == {}


def test_extract_tracked_nodes_non_dict_input():
    assert tracker.extract_tracked_nodes(None) == {}
    assert tracker.extract_tracked_nodes({}) == {}


def test_stringify_capture_values_filters_none_and_blank():
    values = {"sigmas": "1, 0.87, 0.5", "empty": "", "blank": "   ", "none": None, "count": 3}
    assert tracker.stringify_capture_values(values) == {
        "sigmas": "1, 0.87, 0.5",
        "count": "3",
    }


def test_stringify_capture_values_empty_dict():
    assert tracker.stringify_capture_values({}) == {}
    assert tracker.stringify_capture_values(None) == {}


def test_stringify_capture_values_truncates_oversized_values():
    huge = "x" * 5000
    result = tracker.stringify_capture_values({"profile": huge, "short": "fine"})
    assert result["short"] == "fine"
    assert result["profile"].startswith("x" * tracker.MAX_CAPTURE_VALUE_LEN)
    assert result["profile"].endswith("(5000 chars total)")
    assert len(result["profile"]) < 5000


def test_stringify_capture_values_does_not_truncate_short_values():
    value = "x" * tracker.MAX_CAPTURE_VALUE_LEN
    result = tracker.stringify_capture_values({"key": value})
    assert result["key"] == value
