"""Import smoke test for the route modules split out of extension.py.

Each nodes/<module>.py registers its /fbtools/* handlers on ComfyUI's PromptServer route table at import
time. This loads them under a synthetic package with folder_paths and server stubbed and checks the number
of routes each one registers, catching a missing import or a dropped handler without a running ComfyUI.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "fbt_route_test_pkg"

# module -> number of @routes.* handlers it must register
EXPECTED = {
    "llm_assistant": 40,
    "backgrounds_presets": 11,
    "media": 6,
    "registry_api": 15,
    "outfits": 7,
    "lora_info": 2,
    "prompt_collections": 4,
    "kdenlive_archive": 8,
    "libber": 13,
    "dataset_caption": 5,
}


class _Routes:
    def __init__(self):
        self.registered = []

    def _decorator(self, verb):
        def deco(path):
            def wrap(fn):
                self.registered.append((verb, path, fn.__module__))
                return fn
            return wrap
        return deco

    def __getattr__(self, name):
        if name in ("get", "post", "delete", "put", "patch"):
            return self._decorator(name)
        raise AttributeError(name)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    routes = _Routes()
    fp = types.ModuleType("folder_paths")
    fp.get_user_directory = lambda: str(tmp_path / "user")
    fp.get_input_directory = lambda: str(tmp_path / "input")
    fp.get_output_directory = lambda: str(tmp_path / "output")
    fp.base_path = str(tmp_path)
    fp.models_dir = str(tmp_path / "models")
    fp.get_folder_paths = lambda name: []

    class _PS:
        instance = types.SimpleNamespace(routes=routes, send_sync=lambda *a, **k: None)
    srv = types.ModuleType("server")
    srv.PromptServer = _PS

    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "server", srv)
    monkeypatch.setitem(sys.modules, PKG, pkg)
    for name in list(sys.modules):
        if name.startswith(PKG + "."):
            monkeypatch.delitem(sys.modules, name)
    return routes


@pytest.mark.parametrize("module,count", sorted(EXPECTED.items()))
def test_route_module_imports_and_registers_its_routes(env, module, count):
    importlib.import_module(f"{PKG}.nodes.{module}")
    # Importing a module can pull in sibling route modules (e.g. backgrounds -> llm_assistant); count only its own.
    own = [(v, p) for v, p, mod in env.registered if mod == f"{PKG}.nodes.{module}"]
    assert len(own) == count
    assert all(path.startswith("/fbtools/") for _, path in own)


def test_extension_imports_every_route_module():
    """A route module only registers its handlers when something imports it; extension.py must import each one."""
    ext = (ROOT / "extension.py").read_text(encoding="utf-8")
    missing = []
    for path in sorted((ROOT / "nodes").glob("*.py")):
        if path.stem in ("__init__", "shared"):
            continue
        if "@routes." not in path.read_text(encoding="utf-8"):
            continue
        stem = path.stem
        if f"from .nodes.{stem} import" not in ext and f"from .nodes import {stem}" not in ext:
            missing.append(stem)
    assert not missing, f"route modules never imported by extension.py: {missing}"


def _resolves(level: int, module: str | None, names: list[str]) -> bool:
    """Does `from <dots><module> import <names>` point at something that exists (level 1 = nodes/, 2 = package root)?"""
    base = ROOT / "nodes" if level == 1 else ROOT
    parts = (module or "").split(".") if module else []
    target = base.joinpath(*parts)
    if module:
        return target.with_suffix(".py").is_file() or target.is_dir()
    # "from . import x" / "from .. import x": each name must be a module or package there
    return all((base / f"{n}.py").is_file() or (base / n).is_dir() for n in names)


def test_relative_imports_in_nodes_modules_resolve():
    """Code moved from extension.py (package root) into nodes/ needs one more dot on every relative import,
    including imports inside function bodies that only run when a route is called."""
    import ast

    problems = []
    for path in sorted((ROOT / "nodes").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level > 2 or not _resolves(node.level, node.module, [a.name for a in node.names]):
                    problems.append(f"{path.name}:{node.lineno}: {'.' * node.level}{node.module or ''}")
    assert not problems, "unresolvable relative imports: " + "; ".join(problems)
