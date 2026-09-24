"""Source-level contract test for a regression in _h3_load_video_frames()
(nodes/compositions.py, moved out of extension.py in Plan 29): the duration -> frame_load_cap
conversion must divide by select_every_nth.

extension.py (and its split-out nodes/ modules) can't be imported directly in unit tests (its
~75 io.ComfyNode subclasses need a real base class, not conftest.py's bare MagicMock) — see
test_dataset_caption_api.py's own docstring for the same constraint. AST-parsing the function's
source is this repo's established way of covering a route/function under that constraint.

Bug history: duration_cap was computed as int(duration * target_fps), ignoring
select_every_nth entirely, while `frame_load_cap` (which this value feeds) is compared against
`sampled` — a count taken *after* the select_every_nth filter. With select_every_nth=2 this read
exactly 2x the intended duration's worth of post-filter frames from the source before stopping
(reported live: 4.7s at fps=24, select_every_nth=2 produced 112 frames instead of the correct 56,
then got ping-pong padded to 124 for not being a valid H3 17k+5 count — 56 already is one).
"""
import ast
from pathlib import Path

EXTENSION_PATH = Path(__file__).resolve().parents[1] / "nodes" / "compositions.py"
SOURCE = EXTENSION_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function: {name}")


def test_duration_cap_divides_by_select_every_nth():
    func = _find_function("_h3_load_video_frames")
    src = ast.get_source_segment(SOURCE, func) or ""
    assert "int(duration * target_fps / select_every_nth)" in src, (
        "duration_cap must divide by select_every_nth — without it, a bundle/clip with "
        "select_every_nth > 1 silently reads select_every_nth times too much of the source "
        "for a given `duration`, since frame_load_cap is compared against post-filter frames."
    )


def test_duration_cap_no_longer_has_the_unguarded_formula():
    func = _find_function("_h3_load_video_frames")
    src = ast.get_source_segment(SOURCE, func) or ""
    assert "int(duration * target_fps))" not in src


def test_select_every_nth_is_normalized_to_at_least_one_before_the_division():
    # Guards against a ZeroDivisionError if this ever regresses: select_every_nth must be
    # coerced away from 0/None before duration_cap's division uses it.
    func = _find_function("_h3_load_video_frames")
    src = ast.get_source_segment(SOURCE, func) or ""
    nth_idx = src.index('select_every_nth  = int(load_params.get("select_every_nth"')
    cap_idx = src.index("duration_cap = max(1, int(duration")
    assert nth_idx < cap_idx, "select_every_nth must be normalized before duration_cap uses it"
    assert 'int(load_params.get("select_every_nth",    1)) or 1' in src
