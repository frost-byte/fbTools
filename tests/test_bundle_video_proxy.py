"""Source-level contract tests for Plan 16: proxy-caching bundle video references.

extension.py can't be imported directly in tests (its ~75 io.ComfyNode subclasses need a real
base class, not conftest.py's bare MagicMock) — see test_dataset_caption_api.py's own docstring
for the same constraint. AST-parsing is this repo's established way of covering a route/function
under that constraint (also used by tests/test_source_profile_proxy_stream_route.py and
tests/test_h3_load_video_frames_duration_cap.py earlier this session).

The functions this test covers moved out of extension.py in Plan 29, split across three files
(bundles.py owns the routes; composition_shared.py and compositions.py each own one function the
other needs, since neither is a leaf relative to the other -- see nodes/composition_shared.py's
own docstring) -- so this searches all three rather than a single hardcoded path.
"""
import ast
from pathlib import Path

_NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"
_CANDIDATE_PATHS = [
    _NODES_DIR / "composition_shared.py",
    _NODES_DIR / "compositions.py",
    _NODES_DIR / "bundles.py",
]
_PARSED = [(p.read_text(encoding="utf-8"), ast.parse(p.read_text(encoding="utf-8"))) for p in _CANDIDATE_PATHS]


def _find_function(name: str):
    for source, tree in _PARSED:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                node._source = source  # stashed for _src()
                return node
    raise AssertionError(f"Missing function: {name}")


def _src(name: str) -> str:
    node = _find_function(name)
    return ast.get_source_segment(node._source, node) or ""


def _decorator_route_paths(func_node) -> list[str]:
    paths = []
    for decorator in func_node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr in {"get", "post"} and decorator.args:
                arg = decorator.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    paths.append(arg.value)
    return paths


# ── eligibility guard ────────────────────────────────────────────────────────

def test_bundle_proxy_eligible_requires_positive_duration_and_24_or_native_fps():
    src = _src("_bundle_proxy_eligible")
    assert "duration > 0.0" in src
    assert "force_rate in (0, 24)" in src


# ── generation-time fallback in _resolve_cast_media ────────────────────────────

def test_resolve_cast_media_want_video_branch_uses_proxy_when_eligible():
    src = _src("_resolve_cast_media")
    assert "_bundle_proxy_eligible(entry_load_params[\"force_rate\"], entry_load_params[\"duration\"])" in src
    assert "_ensure_bundle_proxy(" in src


def test_resolve_cast_media_proxy_call_is_guarded_by_try_except():
    func = _find_function("_resolve_cast_media")
    found_try_wrapping_ensure_call = False
    for node in ast.walk(func):
        if isinstance(node, ast.Try):
            try_src = ast.get_source_segment(func._source, node) or ""
            if "_ensure_bundle_proxy(" in try_src:
                found_try_wrapping_ensure_call = True
                assert node.handlers, "the _ensure_bundle_proxy call must be inside a try/except"
    assert found_try_wrapping_ensure_call, "_ensure_bundle_proxy call not found inside any try block"


def test_resolve_cast_media_only_resets_start_time_on_proxy_swap_not_duration_or_select_every_nth():
    # The proxy is pre-trimmed (so start_time resets to 0) but NOT pre-decimated — duration and
    # select_every_nth must survive unchanged so decimation still happens at load time.
    src = _src("_resolve_cast_media")
    swap_idx = src.index("video_file_for_entry = proxy_path")
    window = src[swap_idx:swap_idx + 400]
    assert 'entry_load_params["start_time"] = 0.0' in window
    assert 'entry_load_params["duration"]' not in window
    assert 'entry_load_params["select_every_nth"]' not in window


def test_resolve_cast_media_swaps_video_file_field_not_the_original_vfile():
    src = _src("_resolve_cast_media")
    assert '"video_file":       video_file_for_entry,' in src
    # Audio extraction (extract_from_visual) must keep using the real original file — a proxy is
    # silent (-an) and must never become an audio source.
    assert 'entry_audio_path   = abs_vfile' in src


# ── preview_sampled: fire-and-forget, not awaited ──────────────────────────────

def test_preview_sampled_fires_proxy_build_without_awaiting():
    src = _src("_bundles_preview_sampled")
    assert "_fire_bundle_proxy_build(bundle_id, path, start_time, duration, force_rate)" in src
    assert "await _fire_bundle_proxy_build" not in src


def test_fire_bundle_proxy_build_itself_never_awaits_the_executor_future():
    src = _src("_fire_bundle_proxy_build")
    assert "loop.run_in_executor(None, _build)" in src
    assert "await loop.run_in_executor(None, _build)" not in src


# ── bundles/save: fires proxy build after a successful upsert ─────────────────

def test_bundles_save_fires_proxy_build_after_upsert():
    src = _src("_bundles_save")
    upsert_idx = src.index("registry.upsert(data)")
    fire_idx = src.index("_fire_bundle_proxy_build(")
    assert fire_idx > upsert_idx


# ── new proxy_status route ──────────────────────────────────────────────────

def test_bundle_proxy_status_route_exists_at_expected_path():
    func = _find_function("_bundles_proxy_status")
    assert "/fbtools/bundles/proxy_status" in _decorator_route_paths(func)


def test_bundle_proxy_status_reports_ineligible_bundles_without_checking_freshness():
    src = _src("_bundles_proxy_status")
    assert "if not eligible:" in src
    assert '"fresh": False, "proxy_path": None, "eligible": False' in src
