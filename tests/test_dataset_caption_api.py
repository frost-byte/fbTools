"""
Source-level contract tests for dataset caption API handlers in extension.py.

These tests intentionally avoid importing extension.py because it depends on
full ComfyUI runtime modules that are not available in unit-test environments.
"""

import ast
from pathlib import Path


EXTENSION_PATH = Path(__file__).resolve().parents[1] / "extension.py"
SOURCE = EXTENSION_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.AsyncFunctionDef:
    for node in TREE.body:
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


def test_dataset_caption_routes_exist_with_expected_paths():
    route_expectations = {
        "serve_image": "/fbtools/dataset_caption/image",
        "list_dataset": "/fbtools/dataset_caption/list",
        "save_caption": "/fbtools/dataset_caption/save",
        "recaption_single": "/fbtools/dataset_caption/recaption",
    }

    for func_name, expected_path in route_expectations.items():
        func = _find_function(func_name)
        paths = _decorator_route_paths(func)
        assert expected_path in paths, f"{func_name} missing route {expected_path}"


def test_list_dataset_handler_contains_pagination_contract():
    func = _find_function("list_dataset")
    func_source = ast.get_source_segment(SOURCE, func) or ""

    assert 'params.get("path"' in func_source
    assert 'params.get("output_dir"' in func_source
    assert 'params.get("page", 1)' in func_source
    assert 'params.get("page_size", 10)' in func_source
    assert 'params.get("recursive", "false")' in func_source

    assert '"rows"' in func_source
    assert '"total"' in func_source
    assert '"page"' in func_source
    assert '"page_size"' in func_source
    assert '"total_pages"' in func_source


def test_serve_image_has_safety_checks():
    func = _find_function("serve_image")
    func_source = ast.get_source_segment(SOURCE, func) or ""

    assert "SAFE_EXTS" in func_source
    assert "Missing path parameter" in func_source
    assert "Forbidden file extension" in func_source
    assert "File not found" in func_source


def test_save_caption_writes_trimmed_caption():
    func = _find_function("save_caption")
    func_source = ast.get_source_segment(SOURCE, func) or ""

    assert 'body["txt_path"]' in func_source
    assert 'body["caption"].strip()' in func_source
    assert "txt_path.parent.mkdir(parents=True, exist_ok=True)" in func_source
    assert "txt_path.write_text(caption, encoding=\"utf-8\")" in func_source
    assert 'web.json_response({"ok": True' in func_source


def test_recaption_single_includes_expected_payload_fields():
    func = _find_function("recaption_single")
    func_source = ast.get_source_segment(SOURCE, func) or ""

    required_fields = [
        "image_path",
        "txt_path",
        "captioner_type",
        "instruction",
        "trigger_word",
        "device",
        "use_8bit",
        "clean_caption",
        "gemini_api_key",
    ]
    for field in required_fields:
        assert f'body.get("{field}"' in func_source or f'body["{field}"]' in func_source

    assert "caption_image(" in func_source
    assert "get_model(" in func_source
    assert "txt_path.write_text(caption, encoding=\"utf-8\")" in func_source
