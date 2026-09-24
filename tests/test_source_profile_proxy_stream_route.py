"""Source-level contract test for the /fbtools/source_profiles/proxy_stream route in
nodes/source_profiles.py (Plan 15: SceneCastBuild clip video preview; moved out of
extension.py in Plan 28).

Follows tests/test_dataset_caption_api.py's pattern: this module depends on full ComfyUI
runtime modules unavailable in unit tests, so this AST-parses the route handler's source
instead of importing it, and asserts on the safety checks that matter — the same
allow-listed-root pattern as the sibling /fbtools/bundles/audio_cache/stream route (itself
untested; this is the first test of that whole pattern in this repo, not a duplicate of an
existing one).
"""
import ast
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "source_profiles.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing async function: {name}")


def _decorator_route_paths(func_node: ast.AsyncFunctionDef) -> list[str]:
    paths = []
    for decorator in func_node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr in {"get", "post"} and decorator.args:
                arg = decorator.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    paths.append(arg.value)
    return paths


def test_proxy_stream_route_exists_at_expected_path():
    func = _find_function("_source_profiles_proxy_stream")
    assert "/fbtools/source_profiles/proxy_stream" in _decorator_route_paths(func)


def test_proxy_stream_rejects_missing_path_param():
    func = _find_function("_source_profiles_proxy_stream")
    src = ast.get_source_segment(SOURCE, func) or ""
    assert 'status=400' in src
    assert "path required" in src


def test_proxy_stream_scopes_to_the_source_profile_proxy_cache_root_via_realpath():
    func = _find_function("_source_profiles_proxy_stream")
    src = ast.get_source_segment(SOURCE, func) or ""
    # Same allow-listed-root shape as /fbtools/bundles/audio_cache/stream: resolve both sides
    # with realpath (defeats a ../ traversal or symlink escape) and require the resolved path to
    # sit strictly inside the proxy cache directory before ever touching the filesystem for it.
    assert '"proxies", "source_profiles"' in src
    assert "os.path.realpath(path)" in src
    assert "real_path.startswith(allowed_root + os.sep)" in src
    assert "status=403" in src


def test_proxy_stream_404s_a_missing_file_and_serves_a_real_one():
    func = _find_function("_source_profiles_proxy_stream")
    src = ast.get_source_segment(SOURCE, func) or ""
    assert "os.path.isfile(real_path)" in src
    assert "status=404" in src
    assert "web.FileResponse(real_path)" in src


def test_proxy_stream_never_generates_a_proxy():
    # ensure_source_profile_proxy() can run ffmpeg for minutes (prebuild_proxies exists as a
    # background job specifically because of that cost) — this route must only ever serve a proxy
    # that already exists, never build one inline in the request.
    func = _find_function("_source_profiles_proxy_stream")
    src = ast.get_source_segment(SOURCE, func) or ""
    assert "ensure_source_profile_proxy" not in src
    assert "_ensure_proxy" not in src
